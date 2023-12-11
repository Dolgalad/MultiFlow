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


# classifier
struct M3ClassifierModel
    node_embeddings
    edge_encoder
    graph_conv
    scoring
end

Flux.@functor M3ClassifierModel

#function m3_concat_nodes(xi, xj, e::Nothing)
#    return vcat(xi, xj)
#end

function M3ClassifierModel(node_feature_dim::Int, edge_feature_dim::Int, n_layers::Int, nnodes::Int; drop_p::Float64=0.1)
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder    = Chain(
			    Flux.Dense(edge_feature_dim => node_feature_dim, elu),
			    Flux.Dropout(drop_p),
			    Flux.BatchNorm(node_feature_dim),
			    Flux.Dense(node_feature_dim=>node_feature_dim),
			    Flux.Dropout(drop_p)
			    )
    graph_conv      = [M3MPLayer(node_feature_dim) for _ in 1:n_layers]

    scoring = Flux.Bilinear((2*node_feature_dim, node_feature_dim) => 1)

    M3ClassifierModel(node_embeddings, edge_encoder, graph_conv, scoring)
end

function (model::M3ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)
    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    ndemands = g.num_nodes - nnodes

    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(UMFSolver.concat_nodes, g, encoded_nodes, encoded_nodes)
    #encoded_edges = apply_edges(concat_nodes, g, encoded_g.x, encoded_g.x)
    #
    # scores with batching
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                            encoded_edges[:,g.edata.mask], 
                                            encoded_nodes[:,.!g.ndata.mask],
                                            g)
    return scores

   
    # separate the encoded demands
    demand_idx = repeat(findall(==(0), g.ndata.mask), inner=nedges)
    encoded_demands = getobs(encoded_nodes, demand_idx)
    #encoded_demands = getobs(ng.x, demand_idx)

    # separate the encoded real edges
    real_edge_idx = repeat(findall(==(1), g.edata.mask), ndemands)
    encoded_real_edges = getobs(encoded_edges, real_edge_idx)

    # score the edges 
    scores = model.scoring(encoded_real_edges, encoded_demands)

    return scores

end
