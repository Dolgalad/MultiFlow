ENV["JULIA_CUDA_MEMORY_POOL"] = "none"
ENV["GKSwstype"] = "nul"

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
epochs = 1000
batchsize = 30
lr = 1.0e-4
nhidden = 64
nlayers = 4
tversky_beta = 0.2
es_patience = 100
dataset_path = joinpath(workdir, "datasets/dataset_prc_small_flexE_1_train")
dataset_name = basename(dataset_path)

# model name and save path
model_name = "model2_l"*string(nlayers)*"_lr"*string(lr)*"_h"*string(nhidden)*"_bs"*string(batchsize)*"_tversky"*string(tversky_beta)
save_path = joinpath(workdir, "models", dataset_name, model_name)
mkpath(save_path)

# tensorboard logging directory
tb_log_dir = joinpath(workdir, "tb_logs", dataset_name, model_name)

# training and validation datasets
all_graphs = UMFSolver.load_dataset(dataset_path)
train_graphs, test_graphs = MLUtils.splitobs(all_graphs, at=0.9)

train_loader = DataLoader(train_graphs,
                batchsize=batchsize, shuffle=true, collate=true)
test_loader = DataLoader(test_graphs,
               batchsize=batchsize, shuffle=false, collate=true)

# model
model = UMFSolver.ClassifierModel(nhidden, 3, nlayers, 71) |> device

# optimizer
opt = Flux.Optimise.Optimiser(ClipNorm(1.0), Adam(batchsize * lr))

# loss function
loss(g::GNNGraph) = Flux.Losses.tversky_loss(sigmoid(vec(model(g))), vec(g.targets); beta=tversky_beta)
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

# Early stopping
function es_metric(epoch, history)
    return UMFSolver.last_value(history, "test_loss")
end
es = Flux.early_stopping(es_metric, es_patience, min_dist=1.0f-8, init_score=1.0f8);

# start training
UMFSolver.train_model(model,opt,loss,train_loader,test_loader,
                      callbacks=[TBCallback, save_model_checkpoint],
                      early_stopping=es,
                      epochs=epochs,
                     )

