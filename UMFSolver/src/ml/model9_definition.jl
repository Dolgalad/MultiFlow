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
# DEBUG
unified_gpu(x) = fmap(x -> cu(x, unified=true), x; exclude = Flux._isbitsarray)
#device = unified_gpu #Flux.cpu
#device = Flux.cpu


# classifier
struct M9ClassifierModel
    #node_embeddings
    edge_encoder
    demand_encoder
    graph_conv
    demand_mlp
    edge_mlp
    scoring
    _device
end

Flux.@functor M9ClassifierModel

#function sum_gxe(a, b)
#    return [a[1], a[2]+b[2], a[3]+b[3]]
#end

function M9ClassifierModel(node_feature_dim::Int, 
			   edge_feature_dim::Int, 
			   n_layers::Int, 
			   nnodes::Int; 
			   drop_p::Float64=0.1,
			   device=CUDA.functional() ? Flux.gpu : Flux.cpu,
			   )
    #node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = UMFSolver.MLP(edge_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    demand_encoder = UMFSolver.MLP(2*node_feature_dim+1, node_feature_dim, node_feature_dim, drop_p=drop_p)
    graph_conv      = [UMFSolver.M8MPLayer(node_feature_dim, drop_p=drop_p) for _ in 1:n_layers]

    demand_mlp = UMFSolver.MLP(3*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)
    edge_mlp = UMFSolver.MLP(2*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((node_feature_dim, node_feature_dim) => 1)

    #println("in M9ClassifierModel device :", device)

    M9ClassifierModel(edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring, device)
end


function compute_graph_embeddings(model::M9ClassifierModel, g::GNNGraph)
    #println("model9 compute_graph_embeddings device : ", model._device)
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
    
    # initial node embeddings
    # DEBUG
    node_embeddings_0 = rand(Float32, node_feature_dim, nnodes*g.num_graphs) |> model._device
    #node_embeddings_0 = CUDA.rand(Float32, node_feature_dim, nnodes*g.num_graphs) |> model._device

    demand_embeddings_0 = reduce(vcat, [UMFSolver.make_demand_codes(node_embeddings_0, g) , reshape(UMFSolver.demand_bandwidths(g), 1, ndemands * g.num_graphs)])
    demand_embeddings_0 = model.demand_encoder(demand_embeddings_0)

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


#function make_demand_codes(nc::AbstractMatrix, dnc::AbstractMatrix, g::GNNGraph)
#    ds,dt = UMFSolver.demand_endpoints(g)
#    return vcat(view(nc,:,ds), view(nc,:,dt), dnc)
#end

function make_demand_codes(model::M9ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

function (model::M9ClassifierModel)(g::GNNGraph)
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
    #println("model9 forward device : ", model._device)
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                                  model.edge_mlp(encoded_edges[:,g.edata.mask]), 
                                                  model.demand_mlp(encoded_demands),
                                                  g)
    return scores
end
