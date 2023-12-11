"""Train M8ClassifierModel
Arguments : 
    - train dataset path
    - test dataset path
    - batch size
"""


ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["GKSwstype"] = "nul"

using Revise

using Plots
using GraphNeuralNetworks, Graphs, Flux, CUDA, Statistics, MLUtils
using Flux: DataLoader
using Flux.Losses: logitbinarycrossentropy, binary_focal_loss, mean, logsoftmax

using UMFSolver
using TensorBoardLogger
using Logging
using LinearAlgebra

using BSON: @save
using Random: seed!

# set the seed
seed!(2023)

device = CUDA.functional() ? Flux.gpu : Flux.cpu;

println("CUDA.functional: ", CUDA.functional())

# working directory : environment variable 
if !("TRAIN_WORKDIR" in keys(ENV))
    println("The TRAIN_WORKDIR environment variable is not set. Exiting.")
    exit()
end
workdir = ENV["TRAIN_WORKDIR"]

# training parameters
epochs = 10000
if length(ARGS)>=2
    bs = parse(Int64, ARGS[2])
else
    bs = 20
end
lr = 1.0e-6
nhidden = 64
nlayers = 0
tversky_beta = 0.1
es_patience = 100

if length(ARGS)>=1
    dataset_path = joinpath(ARGS[1], "train")
    dataset_name = basename(ARGS[1])
    test_dataset_path = joinpath(ARGS[1], "test")
else
    dataset_path = joinpath(workdir, "datasets", "Oxford_0_1_1", "train")
    dataset_name = "Oxford_0_1_1"
    test_dataset_path = joinpath(workdir, "datasets", "Oxford_0_1_1", "test")
end

# model name and save path
model_name = "model8_l"*string(nlayers)*"_lr"*string(lr)*"_h"*string(nhidden)*"_bs"*string(bs)*"_e"*string(epochs)*"_tversky"*string(tversky_beta)
save_path = joinpath(workdir, "models", dataset_name, model_name)
mkpath(save_path)

# tensorboard logging directory
tb_log_dir = joinpath(workdir, "tb_logs", dataset_name, model_name)

# testing solve output dir
test_solve_output_dir = joinpath(workdir, "solve_outputs", dataset_name, model_name)
# default solver
output_dir_default = joinpath(test_solve_output_dir, "default")
mkpath(output_dir_default)


# print summary
println("Training summary")
println("\tWorkdir                     = ", workdir)
println("\tepochs                      = ", epochs)
println("\tbatch size                  = ", bs)
println("\tlearning rate               = ", lr)
println("\thidden dimension            = ", nhidden)
println("\tGCN layers                  = ", nlayers)
println("\ttversky beta                = ", tversky_beta)
println("\tearly stopping patience     = ", es_patience)
println("\ttrain dataset               = ", dataset_path)
println("\tdataset name                = ", dataset_name)
println("\ttest dataset                = ", test_dataset_path)
println("\tmodel name                  = ", model_name)
println("\tcheckpoint directory        = ", save_path)
println("\tTensorBoard directory       = ", tb_log_dir)
println("\ttest solve output directory = ", test_solve_output_dir)


# run once for compilation
inst = UMFSolver.scale(UMFData(joinpath(test_dataset_path, "1")))
#solveUMF(inst,"CG","highs","./output.txt")
#solveUMF(inst,"CG","highs","./output.txt","","clssp model5_test_checkpoint.bson 1")
for inst_dir in readdir(test_dataset_path, join=true)
    if UMFSolver.is_instance_path(inst_dir)
        local inst = UMFSolver.scale(UMFData(inst_dir))
        s1,ss1 = solveUMF(inst,"CG","cplex",joinpath(output_dir_default, basename(inst_dir))*".json")
    end
end

print("Loading dataset...")
all_graphs = UMFSolver.load_dataset(dataset_path)
all_graphs = map(UMFSolver.aggregate_demand_paths, all_graphs)

nnodes = sum(all_graphs[1].ndata.mask)
println("nnodes : ", nnodes)

train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.9)

train_loader = DataLoader(train_graphs,
                batchsize=bs, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs,
               batchsize=bs, shuffle=false, collate=true)
println("done")

# model
model = UMFSolver.M8ClassifierModel(nhidden, 4, nlayers, nnodes) |> device

# optimizer
opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam(bs* lr))

# debug
function loss(pred::AbstractVector, labs::AbstractVector)
    #pred = sigmoid(vec(model(g)))
    #labels = vec(g.targets[g.target_mask])
    return Flux.Losses.tversky_loss(pred, labs; beta=tversky_beta)
end

function get_labs(g)
    g = g |> device
    return vec(g.targets[g.target_mask])
end
function loss(loader, m)
    #r = mean(loss(sigmoid(vec(model(g))), vec(g.targets[g.target_mask])) for g in loader)
    r = mean(loss(sigmoid(vec(model(g |> device))), get_labs(g)) for g in loader)
    return r
end




# tensorboard callback
logger = TBLogger(tb_log_dir, tb_overwrite)


function TBCallback(epoch, history)
    train_metrics = UMFSolver.last_metrics(history, prefix="train")
    test_metrics = UMFSolver.last_metrics(history, prefix="test")

    with_logger(logger) do
        @info "train" train_metrics... log_step_increment=0
        @info "test" test_metrics...
        @info "plot" UMFSolver.make_plots(model, test_loader.data[1])... log_step_increment=0

    end

    #@info "train" train_metrics
    @info "test" test_metrics

end

# save model checkpoint callback
function save_model_checkpoint(epoch, history)
    _model = model |> Flux.cpu
    @save joinpath(save_path, "checkpoint_e$(epoch).bson") _model
end


# use model to solve test instances
function solve_test_dataset(epoch, history)
    if epoch % 10 == 0 || epoch==1
        println("Testing the trained model on $(test_dataset_path)")
        # create a directory for this epochs test solve outputs for K=0
        output_dir_K0 = joinpath(test_solve_output_dir, "K0",string(epoch))
        mkpath(output_dir_K0)
        output_dir_K1 = joinpath(test_solve_output_dir, "K1",string(epoch))
        mkpath(output_dir_K1)
        for inst_dir in readdir(test_dataset_path, join=true)
            if UMFSolver.is_instance_path(inst_dir)
                local inst = UMFSolver.scale(UMFData(inst_dir))
             	checkpoint_path = joinpath(save_path, "checkpoint_e$epoch.bson")
                try s0,ss0 = solveUMF(inst,"CG","cplex",joinpath(output_dir_K0, basename(inst_dir))*".json", "", "clssp $checkpoint_path 0") catch e end
                s1,ss1 = solveUMF(inst,"CG","cplex",joinpath(output_dir_K1, basename(inst_dir))*".json", "", "clssp $checkpoint_path 1")
            end
        end
    end
end

# Early stopping
function es_metric(epoch, history)
    return UMFSolver.last_value(history, "test_loss")
end
es = Flux.early_stopping(es_metric, es_patience, min_dist=1.0f-8, init_score=1.0f8);

# start training
UMFSolver.train_model(model,opt,loss,train_loader,test_loader,
                      callbacks=[TBCallback, save_model_checkpoint, solve_test_dataset],
                      early_stopping=es,
                      epochs=epochs,
                     )

