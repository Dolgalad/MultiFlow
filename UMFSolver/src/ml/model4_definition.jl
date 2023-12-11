ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader
#using Flux.Losses: logitbinarycrossentropy, binary_focal_loss, mean, logsoftmax

#using UMFSolver
using Distributions
using TensorBoardLogger
using Logging
using LinearAlgebra

using BSON: @save

#device = CUDA.functional() ? Flux.gpu : Flux.cpu;

function MyMEGNetConv2(n; drop_p=0.1)
    phie = Chain([Flux.Dense(3*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n, relu)
		  ])
    phiv = Chain([Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n, relu)
		  ])
    return MEGNetConv(phie, phiv)
end


# message passing layer
struct M4MPLayer
    forward_conv
    drop1
    bn_n
    bn_e
    rev
    backward_conv
    #drop2
end

Flux.@functor M4MPLayer

function M4MPLayer(n::Int; drop_p::Float64=.1)
    return M4MPLayer(MyMEGNetConv2(n), Flux.Dropout(drop_p), Flux.BatchNorm(n), Flux.BatchNorm(n), M3EdgeReverseLayer(), MyMEGNetConv2(n))

end


function (l::M4MPLayer)(g, x, e)
    x,e = l.forward_conv(g, x, e)
    x = l.drop1(x)
    x = l.bn_n(x)
    e = l.bn_e(e)
    ng = l.rev(g)
    x,e = l.backward_conv(ng, x, e)
    return x, e
end


# classifier
struct M4ClassifierModel
    node_embeddings
    edge_encoder
    graph_conv
    scoring
end

Flux.@functor M4ClassifierModel

function M4ClassifierModel(node_feature_dim::Int, edge_feature_dim::Int, n_layers::Int, nnodes::Int; drop_p::Float64=0.1)
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder    = Chain(
			    Flux.Dense(edge_feature_dim => node_feature_dim, relu),
			    Flux.Dropout(drop_p),
			    Flux.BatchNorm(node_feature_dim),
			    #Flux.Dense(node_feature_dim => node_feature_dim, relu),
			    #Flux.Dropout(drop_p),
			    #Flux.BatchNorm(node_feature_dim),
			    Flux.Dense(node_feature_dim=>node_feature_dim),
			    Flux.Dropout(drop_p)
			    )
    graph_conv      = [M4MPLayer(node_feature_dim) for _ in 1:n_layers]

    scoring = Flux.Bilinear((2*node_feature_dim, 2*node_feature_dim) => 1)

    M4ClassifierModel(node_embeddings, edge_encoder, graph_conv, scoring)
end

minmax(x) = [minimum(x),maximum(x)]


function make_demand_codes(model::M4ClassifierModel, g::GNNGraph)
    nc, _= UMFSolver.compute_graph_embeddings(model, g)
    ds,dt = UMFSolver.demand_endpoints(g)
    return vcat(view(nc,:,ds), view(nc,:,dt))
end

function make_demand_codes(nc::AbstractMatrix, g::GNNGraph)
    ds,dt = UMFSolver.demand_endpoints(g)
    return vcat(view(nc,:,ds), view(nc,:,dt))
end

function (model::M4ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)
    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    ndemands = g.num_nodes - nnodes

    # same node encoding process as model2
    encoded_nodes, _encoded_edges = UMFSolver.compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(UMFSolver.concat_nodes, g, encoded_nodes, encoded_nodes)

    # encode demands
    encoded_demands = make_demand_codes(encoded_nodes, g)

    scores = UMFSolver.compute_edge_demand_scores(model, 
                                            encoded_edges[:,g.edata.mask], 
                                            encoded_demands,
                                            g)
    return scores



    # separate the encoded demands
    tmp_encoded_demands = make_demand_codes(encoded_nodes, g)
    demand_idx = repeat(1:ndemands, inner=nedges)
    encoded_demands = getobs(tmp_encoded_demands, demand_idx)

    #demand_idx = repeat(findall(==(0), g.ndata.mask), inner=nedges)
    #encoded_demands = getobs(encoded_nodes, demand_idx)
    #encoded_demands = getobs(ng.x, demand_idx)

    # separate the encoded real edges
    real_edge_idx = repeat(findall(==(1), g.edata.mask), ndemands)
    encoded_real_edges = getobs(encoded_edges, real_edge_idx)

    # score the edges 
    scores = model.scoring(encoded_real_edges, encoded_demands)

    return scores

end
