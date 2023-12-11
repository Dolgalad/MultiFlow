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


# message passing layer
struct M8MPLayer
    forward_conv
    drop_n
    drop_e
    bn_n
    bn_e
    rev
    backward_conv
    _n
    #drop2
end

Flux.@functor M8MPLayer

function M8MPLayer(n::Int; drop_p::Float64=.1)
    return M8MPLayer(UMFSolver.MyMEGNetConv3(n), 
		      Flux.Dropout(drop_p), 
		      Flux.Dropout(drop_p),
		      Flux.BatchNorm(n), 
		      Flux.BatchNorm(n), 
		      UMFSolver.M3EdgeReverseLayer(), 
		      UMFSolver.MyMEGNetConv3(n),
                      n
                     )

end


function (l::M8MPLayer)(g, x, e)
    # forward message passing
    xf,ef = l.forward_conv(g, x[end-l._n+1:end,:], e[end-l._n+1:end,:])
    xf = l.drop_n(xf)
    xf = l.bn_n(xf)
    ef = l.drop_e(ef)
    ef = l.bn_e(ef)
    # backward message passing
    ng = l.rev(g)
    xb,eb = l.backward_conv(ng, x[end-l._n+1:end,:], e[end-l._n+1:end,:])
    xb = l.drop_n(xb)
    xb = l.bn_n(xb)
    eb = l.drop_e(eb)
    eb = l.bn_e(eb)

    on, oe = vcat(x, 0.5f0 .* (xf+xb)), vcat(e, 0.5f0 * (ef+eb))
 
    return on,oe
end


# classifier
struct M8ClassifierModel
    node_embeddings
    edge_encoder
    demand_encoder
    graph_conv
    demand_mlp
    edge_mlp
    scoring
    _device
end

Flux.@functor M8ClassifierModel

#function sum_gxe(a, b)
#    return [a[1], a[2]+b[2], a[3]+b[3]]
#end

function M8ClassifierModel(node_feature_dim::Int, 
			   edge_feature_dim::Int, 
			   n_layers::Int, 
			   nnodes::Int; 
                           drop_p::Float64=0.1,
			   device=CUDA.functional() ? Flux.gpu : Flux.cpu,
			   )
    node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = UMFSolver.MLP(edge_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    demand_encoder = UMFSolver.MLP(2*node_feature_dim+1, node_feature_dim, node_feature_dim, drop_p=drop_p)
    graph_conv      = [UMFSolver.M8MPLayer(node_feature_dim, drop_p=drop_p) for _ in 1:n_layers]

    demand_mlp = UMFSolver.MLP(3*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)
    edge_mlp = UMFSolver.MLP(2*((n_layers+1)*node_feature_dim), node_feature_dim, node_feature_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((node_feature_dim, node_feature_dim) => 1)

    M8ClassifierModel(node_embeddings, edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring, device)
end


function compute_graph_embeddings(model::M8ClassifierModel, g::GNNGraph)
    #println("1")
    if g.num_graphs==1
        nnodes = sum(g.ndata.mask)
        ndemands = g.num_nodes - nnodes
    else
        nnodes = sum(g.ndata.mask .& (graph_indicator(g).==1))
        ndemands = sum((.!g.ndata.mask) .& (graph_indicator(g).==1))
    end
    #println("2")


    # first encode the edge features
    edge_features = model.edge_encoder(g.e)
    #println("3")

    # dimension of node embeddings
    node_feature_dim = size(edge_features, 1)
    # println("4")
  
    # stack node embeddings and demands
    #tmp = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands))
    
    # batch size support
    node_embedding_idx = repeat(1:nnodes, g.num_graphs)
    #println("5")

    # initial node embeddings
    node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    #println("6")
    # DEBUG
    ###node_embedding_idx = repeat(1:nnodes, g.num_graphs)

    ### initial node embeddings
    ### DEBUG
    ###node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    ##node_embeddings_0 = repeat(model.node_embeddings(1:nnodes), 1, g.num_graphs)

    allK = sum(g.K)

    #println("61")

    full_bandwidths = UMFSolver.demand_bandwidths(g)
    #println("62")

    all_bandwidths = reshape(full_bandwidths, 1, allK)
    #println("66")

    dcodes = UMFSolver.make_demand_codes(node_embeddings_0, g)
    #println("666")
    demand_embeddings_0 = reduce(vcat, [dcodes , all_bandwidths])
    #demand_embeddings_0 = vcat(UMFSolver.make_demand_codes(node_embeddings_0, g) , all_bandwidths)

    #println("7")

    demand_embeddings_0 = model.demand_encoder(demand_embeddings_0)
    #println("8")

    full_node_embeddings = hcat(node_embeddings_0, demand_embeddings_0)
    #full_node_embeddings = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

    # debug
    #tmp_node_emb = zeros(Float32, node_feature_dim, nnodes) |> model._device
    #full_node_embeddings = hcat(tmp_node_emb, 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

    # apply the graph convolution
    encoded_nodes,_encoded_edges = full_node_embeddings, edge_features
    #encoded_nodes,_encoded_edges = model.graph_conv[1](g, full_node_embeddings, edge_features)
    if length(model.graph_conv)>0
        for gnnlayer in model.graph_conv
            encoded_nodes, _encoded_edges = gnnlayer(g, encoded_nodes, _encoded_edges)
        end
    end
    #println("8")


    return encoded_nodes, _encoded_edges

end


#function make_demand_codes(nc::AbstractMatrix, dnc::AbstractMatrix, g::GNNGraph)
#    ds,dt = UMFSolver.demand_endpoints(g)
#    return vcat(view(nc,:,ds), view(nc,:,dt), dnc)
#end

function make_demand_codes(model::M8ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

function (model::M8ClassifierModel)(g::GNNGraph)
    # number of real edges
    #println("a")
    nedges = sum(g.edata.mask)
    #println("b")

    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)
    #println("c")

    ndemands = g.num_nodes - nnodes
    #println("d")


    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    #println("e")

    # encode get the encoded edges
    encoded_edges = apply_edges(UMFSolver.concat_nodes, g, encoded_nodes, encoded_nodes)
    #println("f")

    # separate the encoded demands
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    # println("g")
  
    # compute scores
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                                  model.edge_mlp(encoded_edges[:,g.edata.mask]), 
                                                  model.demand_mlp(encoded_demands),
                                                  g)
    #println("h")

    return scores
end
