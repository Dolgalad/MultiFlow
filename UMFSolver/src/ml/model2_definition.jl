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


# graph opposite layer
struct EdgeReverseLayer <: GraphNeuralNetworks.GNNLayer
end

function (l::EdgeReverseLayer)(g::GNNGraph)
    return g
    s, t = edge_index(g)
    return GNNGraph(t, s, ndata=(;x=g.ndata.x), edata=(;e=g.edata.e))
end

# message passing layer
struct MPLayer
    forward_conv
    #drop1
    bn_n
    bn_e
    rev
    backward_conv
    #drop2
end

Flux.@functor MPLayer

function MPLayer(n::Int)
    return MPLayer(MEGNetConv(n=>n), Flux.BatchNorm(n), Flux.BatchNorm(n), EdgeReverseLayer(), MEGNetConv(n=>n))
end

function (l::MPLayer)(g, x, e)
    x,e = l.forward_conv(g, x, e)
    #x = l.drop1(x)
    x = l.bn_n(x)
    e = l.bn_e(e)
    ng = l.rev(g)
    x,e = l.backward_conv(ng, x, e)
    return x, e
end

# classifier
struct ClassifierModel
    node_embeddings
    edge_encoder
    graph_conv
    scoring
    _device
end

Flux.@functor ClassifierModel

function concat_nodes(xi, xj, e::Nothing)
    return vcat(xi, xj)
end

function ClassifierModel(node_feature_dim::Int, 
			 edge_feature_dim::Int, 
			 n_layers::Int, 
			 nnodes::Int;
			 device=CUDA.functional() ? Flux.gpu : Flux.cpu,
			 )
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder    = Chain(
			    Flux.Dense(edge_feature_dim => node_feature_dim, relu),
			    Flux.Dropout(.1),
			    Flux.BatchNorm(node_feature_dim),
			    Flux.Dense(node_feature_dim=>node_feature_dim, relu),
			    Flux.Dropout(.1)
			    )
    edge_encoder = Dense(edge_feature_dim => node_feature_dim, relu)
    #graph_conv      = [MEGNetConv(node_feature_dim => node_feature_dim) for _ in 1:n_layers]
    graph_conv      = [MPLayer(node_feature_dim) for _ in 1:n_layers]

    scoring = Flux.Bilinear((2*node_feature_dim, node_feature_dim) => 1)

    ClassifierModel(node_embeddings, edge_encoder, graph_conv, scoring, device)
end

function compute_graph_embeddings(model, g::GNNGraph)
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
    demand_embeddings_0 = zeros(Float32,node_feature_dim, ndemands * g.num_graphs) |> model._device
    full_node_embeddings = hcat(node_embeddings_0, demand_embeddings_0)
    #full_node_embeddings = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

    # debug
    #tmp_node_emb = zeros(Float32, node_feature_dim, nnodes) |> model._device
    #full_node_embeddings = hcat(tmp_node_emb, 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

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

function make_demand_codes(model, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    return encoded_nodes[:,.!g.ndata.mask]

end

function (model::ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)
    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    ndemands = g.num_nodes - nnodes


    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(concat_nodes, g, encoded_nodes, encoded_nodes)
    #encoded_edges = apply_edges(concat_nodes, g, encoded_g.x, encoded_g.x)

    # scores with batching
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                            encoded_edges[:,g.edata.mask], 
                                            encoded_nodes[:,.!g.ndata.mask],
                                            g)
    return scores
end
