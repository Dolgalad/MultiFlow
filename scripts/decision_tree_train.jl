using UMFSolver
using DecisionTree
using GraphNeuralNetworks
using Statistics
using Graphs
using LinearAlgebra
using Random: seed!
using JLD2
using EvalMetrics
using ProgressBars

# set the seed
seed!(2023)

#dataset_path = "datasets/AsnetAm_0_1_1"

# load dataset
#dataset = UMFSolver.load_dataset(dataset_path)

# create the feature and label vectors representing (arc,commodity) pairs
function make_arc_commodity_data(g::GNNGraph; node_feature_size=1)
    x, y = [], []
    s, t = edge_index(g)
    ds, dt = UMFSolver.demand_endpoints(g)
    nedges = sum(g.edata.mask)
    nnodes = sum(g.ndata.mask)
    node_vec = rand(node_feature_size, nnodes)
    for k in 1:g.K
        for a in 1:ne(g)
            if g.edata.mask[a]
                outi = outneighbors(g, s[a])
                inj = inneighbors(g, t[a])
                # arcs leaving i
                arcs_out_i = findall(==(1), (s .== s[a]) .& (g.edata.mask))
                arcs_in_j = findall(==(1), (t .== t[a]) .& (g.edata.mask))

                push!(x, vcat(g.e[1:3,a], 
                              [length(inj), length(outi)], 
                              minimum(g.e[1:3,arcs_out_i], dims=2),
                              maximum(g.e[1:3,arcs_out_i], dims=2),
                              mean(g.e[1:3,arcs_out_i], dims=2),
                              minimum(g.e[1:3,arcs_in_j], dims=2),
                              maximum(g.e[1:3,arcs_in_j], dims=2),
                              mean(g.e[1:3,arcs_in_j], dims=2),
                              node_vec[:, s[a]],
                              node_vec[:, t[a]],
                              g.e[:, nedges+k],
                              node_vec[:, ds[k]],
                              node_vec[:, dt[k]]
                             ))
                push!(y, g.targets[a,k])
            end
        end
    end
    return hcat(x...), Bool.(y)
end
function make_arc_commodity_data(graph_list)
    x, y = make_arc_commodity_data(graph_list[1])
    for i in ProgressBar(2:length(graph_list))
        xt, yt = make_arc_commodity_data(graph_list[i])
        x, y = hcat(x, xt), vcat(y, yt)
    end
    return x, y
end
network_names = ["AsnetAm", "Ntt", "giul39", "AttMpls", "Oxford",
                 "Iij", "india35", "Chinanet", "zib54"]
for network in network_names
    dataset_name = "$(network)_0_1_1"
    train_dataset_path = "datasets/$(dataset_name)/test"
    test_dataset_path = "datasets/$(dataset_name)/test"
    
    # load dataset
    dataset = UMFSolver.load_dataset(train_dataset_path, batchable=false)[1:100]
    
    x_train, y_train = make_arc_commodity_data(dataset)
    
    # define model
    #model = DecisionTreeClassifier(max_depth=10)
    println("Fitting...")
    #@time fit!(model, transpose(x_train), y_train)
    
    n_subfeatures = 0
    n_trees = 500
    max_depth = 10
    min_samples_leaf = 50
    min_samples_split = 100
    min_purity_increase = 0.0
    
    # path to save the model
    save_path = joinpath("testing_outputs", "random_forest", dataset_name, join([n_trees, max_depth, min_samples_leaf, min_samples_split], "_")*".jld")
    mkpath(dirname(save_path))
    
    _model = build_forest(y_train, transpose(x_train),
                       n_subfeatures,
                       n_trees,
                       0.7,
                       max_depth,
                       min_samples_leaf,
                       min_samples_split,
                       min_purity_increase
                      )
    
    # test dataset
    x_test, y_test = make_arc_commodity_data(UMFSolver.load_dataset(test_dataset_path, batchable=false))
    pred = apply_forest(_model, transpose(x_test))
    
    println("Test metrics : ")
    println("\tprecision = ", EvalMetrics.precision(y_test, pred))
    println("\trecall    = ", EvalMetrics.recall(y_test, pred))
    println("\taccuracy  = ", EvalMetrics.accuracy(y_test, pred))
    println("\tf1 score  = ", EvalMetrics.f1_score(y_test, pred))
    
    
    
    # save the model
    println("Saving Random Forest to : $(save_path)")
    @save save_path _model
end
