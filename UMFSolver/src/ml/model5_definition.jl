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

function printlayer(x)
    println(x)
    println([minimum(x), maximum(x) ], ", ", any(isnan.(x)), ", ", any(isinf.(x)))
    println("\t", length(unique(x)), ", ", size(x), ", ", ndims(x))
    return x
end



function MyMEGNetConv3(n; drop_p=0.1)
    phie = Chain([Flux.Dense(3*n=>2*n, relu),
		  Flux.Dropout(drop_p),
                  Flux.BatchNorm(2*n),
                  Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n)
		  ])
    phiv = Chain([Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
                  Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
		  Flux.BatchNorm(2*n),
		  Flux.Dense(2*n=>n)
		  ])
    return MEGNetConv(phie, phiv)
end

function MLP(in_dim, out_dim, hidden_dim; drop_p=0.1)
    return Chain([
                  Flux.Dense(in_dim=>hidden_dim, relu),
		  Flux.Dropout(drop_p),
                  Flux.Dense(hidden_dim=>hidden_dim, relu),
		  Flux.Dropout(drop_p),
                  Flux.Dense(hidden_dim=>out_dim),
		  Flux.Dropout(drop_p),
                 ])
end


# message passing layer
struct M5MPLayer
    forward_conv
    drop1
    bn_n
    bn_e
    rev
    backward_conv
    #drop2
end

Flux.@functor M5MPLayer

function M5MPLayer(n::Int; drop_p::Float64=.1)
    return M5MPLayer(MyMEGNetConv3(n), Flux.Dropout(drop_p), Flux.BatchNorm(n), Flux.BatchNorm(n), M3EdgeReverseLayer(), MyMEGNetConv3(n))

end


function (l::M5MPLayer)(g, x, e)
    x,e = l.forward_conv(g, x, e)
    x = l.drop1(x)
    x = l.bn_n(x)
    #x = Flux.normalise(x)
    #printlayer(x)
    e = l.bn_e(e)
    #e = Flux.normalise(e)
    #printlayer(e)
    ng = l.rev(g)
    x,e = l.backward_conv(ng, x, e)
    return x, e
end


# classifier
struct M5ClassifierModel
    node_embeddings
    edge_encoder
    demand_encoder
    graph_conv
    demand_mlp
    edge_mlp
    scoring
end

Flux.@functor M5ClassifierModel

#function m5_concat_nodes(xi, xj, e::Nothing)
#    return vcat(xi, xj)
#end

function sum_gxe(a, b)
    return [a[1], a[2]+b[2], a[3]+b[3]]
end

function M5ClassifierModel(node_feature_dim::Int, edge_feature_dim::Int, n_layers::Int, nnodes::Int; drop_p::Float64=0.1)
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = MLP(edge_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    demand_encoder = MLP(2*node_feature_dim+1, node_feature_dim, node_feature_dim, drop_p=drop_p)
    graph_conv      = [M5MPLayer(node_feature_dim, drop_p=drop_p) for _ in 1:n_layers]

    demand_mlp = MLP(3*node_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    edge_mlp = MLP(2*node_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((node_feature_dim, node_feature_dim) => 1)

    M5ClassifierModel(node_embeddings, edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring)
end


function compute_graph_embeddings(model::M5ClassifierModel, g::GNNGraph)
    if g.num_graphs==1
        nnodes = sum(g.ndata.mask)
        ndemands = g.num_nodes - nnodes
    else
        nnodes = sum(g.ndata.mask .& (graph_indicator(g).==1))
        ndemands = sum((.!g.ndata.mask) .& (graph_indicator(g).==1))
    end


    # first encode the edge features
    edge_features = model.edge_encoder(g.e)

    # dimension of node embeddings
    node_feature_dim = size(edge_features, 1)
   
    # stack node embeddings and demands
    #tmp = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands))
    
    # batch size support
    node_embedding_idx = repeat(1:nnodes, g.num_graphs)

    # initial node embeddings
    node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    #demand_embeddings_0 = zeros(Float32,node_feature_dim, ndemands * g.num_graphs) |> device

    demand_embeddings_0 = reduce(vcat, [UMFSolver.make_demand_codes(node_embeddings_0, g) , reshape(UMFSolver.demand_bandwidths(g), 1, ndemands * g.num_graphs)])
    demand_embeddings_0 = model.demand_encoder(demand_embeddings_0)

    full_node_embeddings = hcat(node_embeddings_0, demand_embeddings_0)
    #full_node_embeddings = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands) |> device)

    # debug
    #tmp_node_emb = zeros(Float32, node_feature_dim, nnodes) |> device
    #full_node_embeddings = hcat(tmp_node_emb, 0*zeros(Float32,node_feature_dim, ndemands) |> device)

    # apply the graph convolution
    s,t = edge_index(g)
    ng = GNNGraph(s, t, ndata=(;x=full_node_embeddings), edata=(;e=edge_features))
    #encoded_g = model.graph_conv(ng)
    encoded_nodes,_encoded_edges = model.graph_conv[1](ng, full_node_embeddings, edge_features)
    if length(model.graph_conv)>1
        for gnnlayer in model.graph_conv[2:end]
            encoded_nodes, _encoded_edges = gnnlayer(g, encoded_nodes, _encoded_edges)
        end
    end
    return encoded_nodes, _encoded_edges

end


function make_demand_codes(nc::AbstractMatrix, dnc::AbstractMatrix, g::GNNGraph)
    ds,dt = UMFSolver.demand_endpoints(g)
    return vcat(view(nc,:,ds), view(nc,:,dt), dnc)
end

function make_demand_codes(model::M5ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

function (model::M5ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)
    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    ndemands = g.num_nodes - nnodes

    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(UMFSolver.concat_nodes, g, encoded_nodes, encoded_nodes)
    #encoded_edges = apply_edges(vcat, g, encoded_nodes, encoded_nodes)

    # separate the encoded demands
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    
    # compute scores
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                                  model.edge_mlp(encoded_edges[:,g.edata.mask]), 
                                                  model.demand_mlp(encoded_demands),
                                                  g)
    return scores




    demand_idx = repeat(1:ndemands, inner=nedges)
    encoded_demands = getobs(tmp_encoded_demands, demand_idx)

    demand_node_idx = repeat(findall(==(0), g.ndata.mask), inner=nedges)
    encoded_demand_nodes = getobs(encoded_nodes, demand_node_idx)

    full_encoded_demands = reduce(vcat, [encoded_demands, encoded_demand_nodes])
    #encoded_demands = getobs(ng.x, demand_idx)

    # separate the encoded real edges
    real_edge_idx = repeat(findall(==(1), g.edata.mask), ndemands)
    encoded_real_edges = getobs(encoded_edges, real_edge_idx)

    # score the edges 
    scores = model.scoring(model.edge_mlp(encoded_real_edges), model.demand_mlp(full_encoded_demands))
    #scores = model.scoring(model.edge_mlp(encoded_real_edges), model.demand_mlp(encoded_demands))

    return scores

end
