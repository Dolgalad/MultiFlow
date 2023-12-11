using UMFSolver
using GraphNeuralNetworks
using Statistics
using Graphs
using LinearAlgebra
using Random: seed!
using JLD2
using EvalMetrics
using ProgressBars
using CUDA
using MLUtils
using Flux
using Flux: DataLoader

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

# training hyperparameters
tversky_beta = 0.1
lr = 1.0e-6
nhidden = 64
nlayers = 4
dropout = 0.1
epochs = 1000
batch_size = 64
patience = 10

device = CUDA.functional() ? Flux.gpu : Flux.cpu

function loss(pred::AbstractVector, labs::AbstractVector)
    return Flux.Losses.tversky_loss(pred, labs; beta=tversky_beta)
end

function loss(loader, m)
    r = mean(loss(sigmoid(vec(m(x |> device))), y |> device) for (x,y) in loader)
    return r
end


for network in network_names
    dataset_name = "$(network)_0_1_1"
    train_dataset_path = "datasets/$(dataset_name)/train"
    
    # load dataset
    dataset = UMFSolver.load_dataset(train_dataset_path, batchable=false)[1:10]
    train_graphs, val_graphs = MLUtils.splitobs(dataset, at=0.9)

    
    x_train, y_train = make_arc_commodity_data(train_graphs)
    x_val, y_val = make_arc_commodity_data(val_graphs)

    train_loader = DataLoader((x_train, y_train),
                    batchsize=batch_size, shuffle=true, collate=true)
    val_loader = DataLoader((x_val, y_val),
                   batchsize=batch_size, shuffle=false, collate=true)
    
    # define model
    println("Fitting...")
    
    # path to save the model
    save_path = joinpath("testing_outputs", "mlp", dataset_name, join([tversky_beta, lr, nhidden, nlayers, dropout], "_")*".jld")
    mkpath(dirname(save_path))
   
    layers = [Dense(size(x_train,1)=>nhidden, relu), Dropout(dropout), BatchNorm(nhidden)]
    for n in 1:nlayers
        push!(layers, Dense(nhidden=>nhidden, relu))
        push!(layers, Dropout(dropout))
        push!(layers, BatchNorm(nhidden))
    end
    push!(layers, Dense(nhidden=>1))
    _model = Chain(layers...) |> device

    # optimizer
    opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam(batch_size* lr))

    # model parameters
    ps = Flux.params(_model)

    # early stopping
    es = Flux.early_stopping(() -> loss(val_loader, _model), patience);

    println("starting training")

    state = Flux.setup(opt, _model)

    for epoch in 1:epochs
        # epoch progress bar
        bar = ProgressBar(train_loader)
        set_description(bar, "Epoch $epoch")

        epoch_losses = []

        for (x,y) in bar
            # send batch to device
            # debug
            x = x |> device
            y = y |> device

            # compute gradient
            batch_loss, grad = Flux.withgradient(_model) do m
                loss(sigmoid(vec(m(x))), y)
            end

            Flux.update!(state, _model, grad[1])

            push!(epoch_losses, batch_loss)
        end

        val_loss = loss(val_loader, _model)
        println("Epoch loss : train = ", mean(epoch_losses), ", val = ", val_loss)
        val_acc,val_rec,val_prec,val_f1 = [],[],[],[]
        for (x,y) in val_loader
            x = x |> device
            pred = Flux.cpu(vec(_model(x))) .> 0
            #println([size(pred), size(y)])
            push!(val_acc,  UMFSolver.accuracy(pred, y))
            push!(val_rec,  UMFSolver.recall(pred, y))
            push!(val_prec, UMFSolver.precision(pred, y))
            push!(val_f1,   UMFSolver.f_beta_score(pred, y))
        end
        println("\tval acc,rec,prec,f1 = ", [mean(val_acc),mean(val_rec),mean(val_prec),mean(val_f1)])


        es() && break

        # early stopping
        #if !isnothing(early_stopping)
        #    if early_stopping(epoch, history)
        #        println("Early stopping")
        #        break
        #    end
        #end
    end
end
