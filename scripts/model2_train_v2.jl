ENV["JULIA_CUDA_MEMORY_POOL"] = "none"


using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader
using Flux.Losses: logitbinarycrossentropy, binary_focal_loss, mean, logsoftmax

using UMFSolver
using Distributions
using TensorBoardLogger
using Logging
using LinearAlgebra

using BSON: @save

device = CUDA.functional() ? Flux.gpu : Flux.cpu;

println("CUDA.functional: ", CUDA.functional())

workdir = "."
mkpath(workdir)


accuracy(y_pred, y_true)     = sum(y_pred .== y_true)/size(y_pred, 1)
tpcount(y_pred, y_true)      = sum( Bool.(y_pred .== y_true) .& Bool.(y_true)  )
precision(y_pred, y_true)    = sum(y_pred)>0 ? tpcount(y_pred, y_true)/sum(y_pred) : 0
recall(y_pred, y_true)       = sum(y_true)>0 ? tpcount(y_pred, y_true)/sum(y_true) : 0
f_beta_score(y_pred, y_true, beta=1.) = precision(y_pred,y_true)>0. && recall(y_pred,y_true)>0. ? sum(1 .+ (beta^2)) * (precision(y_pred,y_true)*recall(y_pred,y_true))/(((beta^2) * precision(y_pred,y_true)) + recall(y_pred,y_true)) : 0
graph_reduction(y_pred)      = 1. - sum(y_pred)/prod(size(y_pred))


function train_model(epochs, nhidden, 
		      nlayers, 
		      device, 
		      lr, 
		      dataset_path, 
		      model_name, 
		      tb_log_dir,
		      eval_dataset,
		      es_patience,
		      weight
		      )
    dataset_name = basename(dataset_path)
    eval_dataset_name = basename(eval_dataset)

    println("\tnhidden            = ", nhidden)
    println("\tnlayers            = ", nlayers)
    println("\tdevice             = ", device)
    println("\tlr                 = ", lr)
    println("\tdataset            = ", dataset_path)
    println("\tdataset name       = ", dataset_name)
    println("\tmodel name         = ", model_name)
    println("\tlog dir            = ", tb_log_dir)
    println("\tevaluation dataset = ", eval_dataset)
    println("\tevaluation dataset name = ", eval_dataset_name)
    println("\tES patience        = ", es_patience)
    println("\tweight             = ", weight)
    save_path = joinpath(workdir, "models", dataset_name, model_name)
    mkpath(save_path)
    println("\tmodel save path    = ", save_path)


    best_f1 = -1.0f8
   
    # list directories in dataset path
    all_graph_paths = [dir for dir in readdir(dataset_path, join=true) if UMFSolver.is_instance_path(dir)]

    # count number of nodes in the graph
    inst = UMFData(all_graph_paths[1])
    nnodes = nv(inst)
    println("\tnnodes             = ", nnodes)

    model = ClassifierModel(nhidden, 3, nlayers, nnodes) |> device
    
    ps = Flux.params(model)
    
    opt = Flux.Optimise.Optimiser(ClipNorm(1.0f-4), Adam(lr))
    opt = Adam(lr)

    
    train_graphs, test_graphs = MLUtils.splitobs(all_graph_paths, at=0.9)
    
    train_loader = DataLoader(train_graphs,
                    batchsize=1, shuffle=true, collate=true)
    test_loader = DataLoader(test_graphs,
                   batchsize=1, shuffle=false, collate=true)
    
    
    function metrics(loader)
        preds = [(Int64.(cpu((vec(model(load_instance(f[1]) |> device)) .> 0))),Int64.(vec(load_instance(f[1]).targets))) for f in loader]
        acc = mean([accuracy(pred, lab) for (pred,lab) in preds])
        rec = mean([recall(pred, lab) for (pred,lab) in preds])
        prec = mean([precision(pred, lab) for (pred,lab) in preds])
        f1 = mean([f_beta_score(pred, lab) for (pred,lab) in preds])
        gr = mean([graph_reduction(pred) for (pred,lab) in preds])
    
        return (acc=acc, rec=rec, prec=prec, f1=f1, gr=gr)
    end
    
    
    logger = TBLogger(tb_log_dir, tb_overwrite)
    
    
    function TBCallback(epoch)
        train_loss = loss(train_loader)
        test_loss = loss(test_loader)
        train_metrics = metrics(train_loader)
        test_metrics = metrics(test_loader)

        #println("train_loss: ", typeof(train_loss))
        #println("test loss : ", typeof(test_loss))
        #println("train_metrics: ", typeof(train_metrics))
        #println("test metrics : ", typeof(test_metrics))

        with_logger(logger) do
            @info "train" loss=train_loss train_metrics... log_step_increment=0
            @info "test" loss=test_loss test_metrics...
        end
    
    end
    
    
    #loss(g::GNNGraph) = logitbinarycrossentropy(vec(model(g)), vec(g.y))
    loss(g::GNNGraph) = Flux.Losses.tversky_loss(sigmoid(vec(model(g))), vec(g.targets); beta=0.7)
    loss(loader) = mean(loss(load_instance(f[1]) |> device) for f in loader)

    function load_instance(f; scale_instance=true)
        inst = UMFData(f)
        # scale cost and capacities
        if scale_instance
            inst = UMFSolver.scale(inst)
        end
        # check if a solution file exists
        solution_file_path = joinpath(f, "sol.jdl")
        if isfile(solution_file_path)
            sol = UMFSolver.load_solution(f)
        else
            ssol, stats = solveUMF(f*"/", "CG", "highs", "./output.txt")
            sol = ssol.x
        end
        y = (sol .> 0)
        return g = UMFSolver.to_gnngraph(inst, y, feature_type=Float32)
    end

    
    # Early stopping
    function es_metric()
        return loss(test_loader)
        return -metrics(test_loader).f1
    end
    es = Flux.early_stopping(es_metric, es_patience, min_dist=1.0f-8, init_score=1.0f8);
    
    
    println("starting training")
    for epoch in 1:epochs
        for f in train_loader
            # load the graph from its directory
            g = load_instance(f[1])
            g = g |> device
            grad = gradient(() -> loss(g), ps)
            Flux.Optimise.update!(opt, ps, grad)
        end

        test_metrics = metrics(test_loader)
        @info (; epoch, train_loss=loss(train_loader), test_loss=loss(test_loader), test_metrics=test_metrics)
        TBCallback(epoch)
        # checkpoints
        _model = model |> Flux.cpu
    	@save joinpath(save_path, "checkpoint_e$(epoch).bson") _model
        # early stopping
        es() && break
    end
    
    # testing: load the testing dataset
    function solve_dataset(dataset_dir::String, output_dir::String, pr::String)
        for f in readdir(dataset_dir, join=true)
            link_f = joinpath(f, "link.csv")
            if !isfile(link_f)
                continue
            end
            sol, stats = solveUMF(f*"/", "CG", "highs", "./output.txt", "", pr)
            stats_path = joinpath(output_dir, basename(f))
            
            UMFSolver.save(stats, stats_path)
            #inst = UMFData(f)
        end
    end
    
    
    #println("Starting evaluation")
    
    #mkpath(joinpath(workdir,"outputs/$(eval_dataset_name)/$(model_name)/default"))
    #solve_dataset(eval_dataset, joinpath(workdir,"outputs/$(eval_dataset_name)/$(model_name)/default"), "dijkstra")
    #mkpath(joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/kspfilter1"))
    #solve_dataset(eval_dataset, joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/kspfilter1"), "kspfilter 1")
    #mkpath(joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/kspfilter2"))
    #solve_dataset(eval_dataset, joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/kspfilter2"), "kspfilter 2")
    #mkpath(joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/clssp1"))
    #solve_dataset(eval_dataset, joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/clssp1"), "clssp $(save_path) 1")
    ## move the reduction file
    #mv(basename(save_path)*"_reduction.csv", joinpath(workdir, "outputs/$(eval_dataset_name)/$(model_name)/clssp1_reduction.csv"), force=true)

end 



epochs_ = [100]
nhiddens = [64]
nlayers_ = [4]
devices = [device]
lrs = [1.0e-6]
train_datasets = [joinpath(workdir, "datasets/dataset_prc_small_flexE_1_train")]
eval_datasets = [joinpath(workdir, "datasets/dataset_prc_small_flexE_1_test")]
weights = [.95]
es_patiences = [10]

# call train_model on a dummy case first to force compilation
train_model(1, 1, 1, device, 0.0001, eval_datasets[1], "dummy2_model5", "dummy2_model_log5", eval_datasets[1], 1, .5)


dataset_names = ["dataset_prc_small_flexE_1",
	         #"dataset_prc_small_flexE_2",
                 #"dataset_prc_small_flexE_3",
                 #"dataset_prc_small_flexE_4",
                 #"dataset_prc_small_flexE_5",
                 #"dataset_prc_small_mixed_1",
	         #"dataset_prc_small_mixed_2",
                 #"dataset_prc_small_mixed_3",
                 #"dataset_prc_small_mixed_4",
                 #"dataset_prc_small_mixed_5",
                 #"dataset_prc_small_vlan_1",
	         #"dataset_prc_small_vlan_2",
                 #"dataset_prc_small_vlan_3",
                 #"dataset_prc_small_vlan_4",
                 #"dataset_prc_small_vlan_5",
                 #"dataset_prc_middle_flexE_1",
	         #"dataset_prc_middle_flexE_2",
                 #"dataset_prc_middle_flexE_3",
                 #"dataset_prc_middle_flexE_4",
                 #"dataset_prc_middle_flexE_5",
	    ]

for name in dataset_names
    train_dataset_path = joinpath(workdir, "datasets", name*"_train")
    test_dataset_path = joinpath(workdir, "datasets", name*"_test")
    # DEBUG
    #train_dataset_path = joinpath(dirname(workdir), "datasets", name*"_train")
    #test_dataset_path = joinpath(dirname(workdir), "datasets", name*"_test")

    if !isdir(train_dataset_path) || !isdir(test_dataset_path)
        println("skipping ", name)
        continue
    end
    #model_name = "model2_l"*string(nlayers_[1])*"_lr"*string(lrs[1])*"_h"*string(nhiddens[1])*"_tversky.2"
    # DEBUG
    model_name = "model2_l"*string(nlayers_[1])*"_lr"*string(lrs[1])*"_h"*string(nhiddens[1])*"_tversky.7_emb0"

    tb_log_dir = joinpath(workdir, "tb_logs", "model2_$(name)_logs", model_name)

    train_model(epochs_[1], nhiddens[1], nlayers_[1], device, lrs[1], train_dataset_path, model_name, tb_log_dir, test_dataset_path, 10, .5)

    # reclaim memory
    println("Reclaiming memory")
    @sync GC.gc()
    CUDA.reclaim()
    println("done")

end

#exit()


## call train_model on a dummy case first to force compilation
#train_model(1, 1, 1, device, 0.0001, eval_datasets[1], "dummy2_model", "dummy2_model_log", eval_datasets[1], 1, .5)
##rm("dummy2_model.bson", force=true)
##rm("dummy2_model_log", recursive=true)
#
#for tprms in Iterators.product(epochs_,
#				nhiddens, 
#				nlayers_, 
#				devices, 
#				lrs, 
#				train_datasets, 
#				eval_datasets, 
#				es_patiences, 
#				weights)
#    model_name = "model2_l"*string(tprms[3])*"_lr"*string(tprms[5])*"_h"*string(tprms[2])*"_tversky.2"
#    #model_name = "model2_l"*string(tprms[3])*"_lr"*string(tprms[5])*"_h"*string(tprms[2])*"_focal"
#
#    tb_log_dir = joinpath(workdir, "tb_logs", "model2_"*basename(tprms[6])*"_logs", model_name)
#    train_model(tprms[1],tprms[2], 
#		 tprms[3], 
#		 tprms[4], 
#		 tprms[5], 
#		 tprms[6], 
#		 model_name, 
#		 tb_log_dir, 
#		 tprms[7], 
#		 tprms[8], 
#		 tprms[9])
#end
