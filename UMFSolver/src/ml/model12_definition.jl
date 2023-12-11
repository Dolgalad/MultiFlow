ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader
#using Flux.Losses: logitbinarycrossentropy, binary_focal_loss, mean, logsoftmax

#using UMFSolver
using Distributions
using TensorBoardLogger
using Logging
using LinearAlgebra
using Zygote

Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)


# classifier
struct M12ClassifierModel
    #node_embeddings
    edge_encoder
    demand_encoder
    graph_conv
    demand_mlp
    edge_mlp
    scoring
    _device
end

Flux.@functor M12ClassifierModel

function make_m12_layer(((x_dim,e_dim),(out_x_dim,out_e_dim))::Pair{Tuple{Int64,Int64},Tuple{Int64,Int64}})
    return Parallel(+,
                    UMFSolver.MyMEGNetConv3(x_dim),
                    identity
                   )
end

function M12ClassifierModel(node_feature_dim::Int, 
			   edge_feature_dim::Int, 
			   n_layers::Int, 
			   nnodes::Int; 
                           drop_p::Float64=0.1,
			   device=CUDA.functional() ? Flux.gpu : Flux.cpu,
			   )
    #node_embeddings = Flux.Embedding(nnodes, node_feature_dim)
    edge_encoder = UMFSolver.MLP(edge_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    demand_encoder = UMFSolver.MLP(2*node_feature_dim+1, node_feature_dim, node_feature_dim, drop_p=drop_p)
    #graph_conv      = [UMFSolver.M12MPLayer(node_feature_dim, drop_p=drop_p) for _ in 1:n_layers]
    graph_conv      = VGNNNet((node_feature_dim,node_feature_dim)=>(node_feature_dim,node_feature_dim),n_layers, layer_type=make_m12_layer, aggr=mean)

    demand_mlp = UMFSolver.MLP(3*node_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)
    edge_mlp = UMFSolver.MLP(2*node_feature_dim, node_feature_dim, node_feature_dim, drop_p=drop_p)

    scoring = Flux.Bilinear((node_feature_dim, node_feature_dim) => 1)

    M12ClassifierModel(edge_encoder, demand_encoder, graph_conv, demand_mlp, edge_mlp, scoring, device)
end


function compute_graph_embeddings(model::M12ClassifierModel, g::GNNGraph)
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
    #node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    if CUDA.functional()
        node_embeddings_0 = CUDA.rand(Float32, node_feature_dim, nnodes * g.num_graphs) .- 0.5
    else
        node_embeddings_0 = rand(Float32, node_feature_dim, nnodes * g.num_graphs) .- 0.5
    end
    # DEBUG
    ###node_embedding_idx = repeat(1:nnodes, g.num_graphs)

    ### initial node embeddings
    ### DEBUG
    ###node_embeddings_0 = model.node_embeddings(node_embedding_idx)
    ##node_embeddings_0 = repeat(model.node_embeddings(1:nnodes), 1, g.num_graphs)

    allK = sum(g.K)


    full_bandwidths = UMFSolver.demand_bandwidths(g)

    all_bandwidths = reshape(full_bandwidths, 1, allK)

    dcodes = UMFSolver.make_demand_codes(node_embeddings_0, g)

    demand_embeddings_0 = reduce(vcat, [dcodes , all_bandwidths])
    #demand_embeddings_0 = vcat(UMFSolver.make_demand_codes(node_embeddings_0, g) , all_bandwidths)


    demand_embeddings_0 = model.demand_encoder(demand_embeddings_0)


    full_node_embeddings = hcat(node_embeddings_0, demand_embeddings_0)
    #full_node_embeddings = hcat(model.node_embeddings(1:nnodes), 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

    # debug
    #tmp_node_emb = zeros(Float32, node_feature_dim, nnodes) |> model._device
    #full_node_embeddings = hcat(tmp_node_emb, 0*zeros(Float32,node_feature_dim, ndemands) |> model._device)

    # apply the graph convolution
    encoded_nodes,_encoded_edges = full_node_embeddings, edge_features
    #encoded_nodes,_encoded_edges = model.graph_conv[1](g, full_node_embeddings, edge_features)
    #if length(model.graph_conv)>0
    #    for gnnlayer in model.graph_conv
    #        encoded_nodes, _encoded_edges = gnnlayer(g, encoded_nodes, _encoded_edges)
    #    end
    #end
    gt = GNNGraph(g, ndata=(;x=encoded_nodes), edata=(;e=_encoded_edges))

    ge = model.graph_conv(gt)

    return ge.x, ge.e


    return encoded_nodes, _encoded_edges

end


#function make_demand_codes(nc::AbstractMatrix, dnc::AbstractMatrix, g::GNNGraph)
#    ds,dt = UMFSolver.demand_endpoints(g)
#    return vcat(view(nc,:,ds), view(nc,:,dt), dnc)
#end

function make_demand_codes(model::M12ClassifierModel, g::GNNGraph)
    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
    return encoded_demands

end

function (model::M12ClassifierModel)(g::GNNGraph)
    # number of real edges
    nedges = sum(g.edata.mask)

    # stack the node embeddings and demand embeddings
    nnodes = sum(g.ndata.mask)

    ndemands = g.num_nodes - nnodes


    encoded_nodes, _encoded_edges = compute_graph_embeddings(model, g)

    # encode get the encoded edges
    encoded_edges = apply_edges(UMFSolver.concat_nodes, g, encoded_nodes, encoded_nodes)

    # separate the encoded demands
    encoded_demands = make_demand_codes(encoded_nodes, encoded_nodes[:,.!g.ndata.mask], g)
  
    # compute scores
    scores = UMFSolver.compute_edge_demand_scores(model, 
                                                  model.edge_mlp(encoded_edges[:,g.edata.mask]), 
                                                  model.demand_mlp(encoded_demands),
                                                  g)

    return scores
end

function compute_edge_demand_scores(model::UMFSolver.M12ClassifierModel, 
				    edge_codes::AbstractMatrix, 
				    demand_codes::AbstractMatrix, 
				    g::GNNGraph;
				    )
    ngind = graph_indicator(g)
    ninodes = sum(g.ndata.mask[ngind .== 1])
    nidemands = sum(.!g.ndata.mask[ngind .== 1])

    egind = graph_indicator(g, edges=true)
    regind = egind[g.edata.mask]
    niedges = sum(g.edata.mask[egind .== 1])

    # demand node graph indicator
    dgind = ngind[.!g.ndata.mask]

    # stack repeated demand codes
    dind = collect(1:size(dgind,1)) |> model._device
    #demand_stacked_idx = reduce(vcat,[repeat(dind[dgind .== i], inner=niedges) for i=1:g.num_graphs])
    demand_stacked_idx = reduce(vcat,[repeat(dind[dgind .== i], inner=g.E[i]) for i=1:g.num_graphs])

    demand_stacked = getobs(demand_codes,demand_stacked_idx)


    # stacked repeated edge codes
    reind = collect(1:size(regind,1)) |> model._device
    #edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], nidemands) for i=1:g.num_graphs])
    edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], g.K[i]) for i=1:g.num_graphs])

    edge_stacked = getobs(edge_codes, edge_stacked_idx)


    # if scoring layer is bilinear
    if model.scoring isa Flux.Bilinear
        scores= model.scoring(edge_stacked, demand_stacked)
    else
        scores= model.scoring(vcat(edge_stacked, demand_stacked))
    end
    return scores
end

