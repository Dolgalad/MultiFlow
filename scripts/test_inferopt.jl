ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["GKSwstype"] = "nul"



using Zygote
using CUDA
using MLUtils
using Flux, Statistics
using Graphs, LinearAlgebra
using SimpleWeightedGraphs
using GraphNeuralNetworks
using Plots
using InferOpt
using ProgressBars
using SparseArrays: sparse

using UMFSolver

# Goal: given an instance I=(G, K), learn a new cost function such that the instance solution
# can be computed as a collection of shortest o_k-t_k-paths for all k in K

# Graph cost encoder
#cost_encoder = MEGNetConv(3 => 1)


Zygote.@adjoint gpu(x) = gpu(x), dx -> (cpu(dx),)
Zygote.@adjoint cpu(x) = cpu(x), dx -> (gpu(dx),)

#device = CUDA.functional() ? Flux.gpu : Flux.cpu
unified_gpu(x) = fmap(x -> cu(x, unified=true), x; exclude = Flux._isbitsarray)

device = unified_gpu #Flux.cpu
#device = Flux.cpu


# classifier
struct TestClassifierModel
    node_embeddings
    edge_encoder
    graph_conv
    cost_l
    #_device
end

Flux.@functor TestClassifierModel

function TestClassifierModel(node_feature_dim::Int, 
			   edge_feature_dim::Int, 
			   n_layers::Int, 
			   nnodes::Int; 
			   drop_p::Float64=0.1,
			   #device=CUDA.functional() ? Flux.gpu : Flux.cpu,
			   )
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = Dense(edge_feature_dim => node_feature_dim)
    graph_conv      = [MEGNetConv(node_feature_dim => node_feature_dim) for _ in 1:n_layers]

    TestClassifierModel(node_embeddings, edge_encoder, graph_conv, Dense(node_feature_dim=>1, relu))
end

function (model::TestClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)
    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    ndemands = g.num_nodes - nnodes

    edge_features = model.edge_encoder(g.e)

    node_feature_dim = size(edge_features,1)

    full_node_embeddings = hcat(model.node_embeddings(1:nnodes), rand(node_feature_dim, ndemands) |> device)

    encoded_nodes,_encoded_edges = model.graph_conv[1](g, full_node_embeddings, edge_features)
    if length(model.graph_conv)>1
        for gnnlayer in model.graph_conv[2:end]
            encoded_nodes, _encoded_edges = gnnlayer(g, encoded_nodes, _encoded_edges)
            #println("\t", [minimum(encoded_nodes), maximum(encoded_nodes)], [minimum(_encoded_edges), maximum(_encoded_edges)])
        end
    end
    #println("\t", [minimum(encoded_nodes), maximum(encoded_nodes)], [minimum(_encoded_edges), maximum(_encoded_edges)])


    #println("encoded_edges : ", size(_encoded_edges))
    r=model.cost_l(_encoded_edges[:,g.edata.mask])
    #println("\tr: ", [minimum(r), maximum(r)])
    return 1.0 .+ r
    return repeat(model.cost_l(_encoded_edges[:,g.edata.mask]), ndemands)
end


function original_graph(instance)
    #println("in original graph : ", typeof(instance.e))
    nk = nv(instance) - sum(instance.ndata.mask)
    nnodes = nv(instance) - nk
    s, t = edge_index(instance)
    #println("in original graph : ", [size(s), size(t), size(instance.edata.e[1,:])])
    tg = SimpleWeightedDiGraph(s, t, instance.edata.e[1,:])
    return induced_subgraph(tg, 1:nnodes)[1]
end

function arc_matrix(g)
    src, dst = edge_index(g)
    #src, dst = [e.src for e in edges(g)], [e.dst for e in edges(g)]

    nnodes = size(src,1)
    am  = zeros(Int64, nnodes, nnodes)
    for a = 1:ne(g)
        am[src[a], dst[a]] = a
    end
    return am
end

function getcol(g, path; am=nothing)
    col::Vector{Int64} = []
    if isnothing(am)
        am = arc_matrix(g)
    end
    for j = 1:(size(path, 1)-1)
        push!(col, am[path[j], path[j+1]])
    end
    return col
end

function path_to_column(g, path; am=nothing)
    p = getcol(g, path, am=am)
    col = zeros(Float32, ne(g))
    col[p] .= 1
    return col
end

function edgeindex(g::SimpleWeightedDiGraph)
    return [e.src for e in edges(g)], [e.dst for e in edges(g)]
end


epoch_co_times = []
all_losses = []

function umf_maximizer(theta; instance)
    global epoch_co_times			      
    nind = graph_indicator(instance)
    eind = graph_indicator(instance, edges=true)
    nnodes = sum(instance.ndata.mask .& (nind .== 1))
    nk = sum(.!instance.ndata.mask .& (nind .== 1))

    cols = Float32[]
    th = reshape(theta, (nedges, nk, instance.num_graphs))

    t_co = @elapsed begin
    
        for gi in 1:instance.num_graphs
            tmp_g = getgraph(instance, gi)
            # graph without the demand nodes
            # set the new weightsA
            s,d = edge_index(tmp_g)
	    s,d = s[tmp_g.edata.mask], d[tmp_g.edata.mask]
	    ds,dt = UMFSolver.demand_endpoints(tmp_g)
            g = SimpleWeightedDiGraph(s, d, sigmoid(th[:,1,gi]))
	    am = arc_matrix(tmp_g)

            # for each demand get shortest path
            for k in 1:nk
		w = sparse(d,s,softplus(th[:,k,gi]),nnodes,nnodes)

                t_sp = @elapsed pe = a_star(g, ds[k], dt[k], w)

		if length(pe)==1
		    p = [src(pe[1]), dst(pe[1])]
		else
		    p = vcat([src(pe[1])], [dst(pe[i]) for i in 1:length(pe)])
		end

		if length(p)==0
		    println("No route : ", [ok, tk])
		end

                t_p2c = @elapsed col = path_to_column(g, p, am=am)

                cols = vcat(cols, col)
            end
        end
    end
    push!(epoch_co_times, t_co)
    return cols
end

train_losses = []
epoch_losses = []
co_times = []
all_losses = []

function umf_cost(ŷ; instance, theta )
    global epoch_losses
    _c = repeat(instance.e[1, instance.edata.mask], ndemands)
    r = dot(_c, vec(ŷ)) / instance.num_graphs

    return sum(dot(y, t) for (y,t) in zip(ŷ, theta))
end


perturbed_add = PerturbedMultiplicative(
    umf_maximizer;
    ε=.5, nb_samples=10
)

regret = Pushforward(
    perturbed_add, umf_cost
)

#include("model_inferopt.jl")

dataset_path = "/data1/schulz/datasets/dataset_prc_small_flexE_1_train"
dataset_path = "/data-a40/schulz/datasets/dataset_prc_small_flexE_1_train"

dataset = UMFSolver.load_dataset(dataset_path)[1:100]


nnodes = sum(dataset[1].ndata.mask)
ndemands = sum(map(!, dataset[1].ndata.mask))
nedges = sum(dataset[1].edata.mask)
println("Number of nodes: ", nnodes)
println("Number of demands: ", ndemands)
println("Number of edges: ", nedges)



println("Device: ", device)

#umf_encoder = ClassifierModel(64, 3, 4, nnodes)

ndemands = 60
#umf_encoder = Chain(Dense(3 => 1),
#		    x->repeat(x, ndemands)
#		    )|> device
#umf_encoder = TestClassifierModel(2,3,1,71) |> device
umf_encoder = UMFSolver.M9ClassifierModel(64, 3, 4, 71, device=device) |> device

lr = 1.0f-6
bs = 1
opt = Flux.Optimise.Optimiser(ClipNorm(1.0f0), Adam(bs* lr))
#opt = Flux.Optimise.Optimiser(Adam(bs* lr))

ps = Flux.params(umf_encoder)
println("Parameters: ")
println([size(p) for p in ps])

function pipeline_loss(x)
    tt = 1. .+ relu(umf_encoder(x))
    th, g = cpu(tt), cpu(x)
    return regret(th; instance=g)
end

train_graphs, test_graphs = MLUtils.splitobs(dataset, at=0.9)


train_loader = DataLoader(train_graphs,
                    batchsize=bs, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs,
                    batchsize=bs, shuffle=false, collate=true)

validation_losses = []
for epoch in 1:10000
    global epoch_losses, epoch_co_times
    #println("epoch: ", epoch)
    for g in ProgressBar(train_loader)
	g = g |> device
        grad = gradient(() -> pipeline_loss(g), ps)
        Flux.Optimise.update!(opt, ps, grad)
	for (i,gp) in enumerate(grad)
	    if !isnothing(gp)
	        if any(isnan.(gp))
		    println("NaN in grad of parameter $i")
		end
	    end
	end
	if CUDA.functional()
	    CUDA.reclaim();
	end
	GC.gc();
    end
    push!(train_losses, mean(epoch_losses))
    push!(co_times, mean(epoch_co_times))
    epoch_losses = []
    epoch_co_times = []

    #val_losses = []
    #for g in test_loader
    #    g = g |> device
    #    theta = reshape(umf_encoder(g.e[:,g.edata.mask]), nedges, ndemands)
    #    ŷ = umf_maximizer(theta; instance=g)
    #    push!(val_losses, Flux.Losses.tversky_loss(vec(ŷ), vec(g.targets); beta=0.1))
    #    if CUDA.functional()
    #        CUDA.reclaim()
    #    end
    #end
    #push!(validation_losses, mean(val_losses))
    #println("Train loss : ", mean(train_losses[end]))

    # plots
    p = plot(train_losses);
    savefig("inferopt2_train_loss_dbg.png")
    p = plot(co_times);
    savefig("inferopt2_train_co_times_dbg.png")

end
 
