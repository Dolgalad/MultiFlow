#ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["GKSwstype"] = "nul"

println("Updated1")
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

device = CUDA.functional() ? Flux.gpu : Flux.cpu;

println("CUDA.functional: ", CUDA.functional())

# working directory
workdir = "/data1/schulz"

# training parameters
epochs = 100
bs = 10
lr = 1.0e-6
nhidden = 64
nlayers = 4
tversky_beta = 0.2
es_patience = 100
dataset_path = joinpath(workdir, "datasets/dataset_prc_small_flexE_1_train")
dataset_name = basename(dataset_path)
test_dataset_path = joinpath(workdir, "datasets/dataset_prc_small_flexE_1_test")

# model name and save path
model_name = "model5_l"*string(nlayers)*"_lr"*string(lr)*"_h"*string(nhidden)*"_bs"*string(bs)*"_tversky"*string(tversky_beta)
save_path = joinpath(workdir, "models", dataset_name, model_name)
mkpath(save_path)

# tensorboard logging directory
tb_log_dir = joinpath(workdir, "tb_logs", dataset_name, model_name)

# training and validation datasets
print("Loading dataset...")
all_graphs = UMFSolver.load_dataset(dataset_path)
train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.9)

train_loader = DataLoader(train_graphs,
                batchsize=bs, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs,
               batchsize=bs, shuffle=false, collate=true)
println("done")

# model
model = UMFSolver.M5ClassifierModel(nhidden, 3, nlayers, 71) |> device

# optimizer
opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam(bs* lr))

# loss function
function loss(g::GNNGraph)
    #println("in loss : ", [typeof(p) for p in Flux.params(model)])
    return Flux.Losses.tversky_loss(sigmoid(vec(model(g))), vec(g.targets); beta=tversky_beta)
end
loss(loader) = mean(loss(g |> device) for g in loader)

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
        inst = UMFData(joinpath(test_dataset_path, "1"))
	checkpoint_path = joinpath(save_path, "checkpoint_e$epoch.bson")
	s1,ss1 = solveUMF(inst,"CG","highs","./model5_test_output.txt", "", "clssp $checkpoint_path 1")
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

