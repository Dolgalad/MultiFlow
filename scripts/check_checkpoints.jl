using UMFSolver
using BSON: @load
using HDF5
using Statistics

checkpoint_directory = "models/Oxford_0_1_1/model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1"

test_instance_directory = "datasets/Oxford_0_1_1/test"

# sptable path
sptable_path = "sptables/Oxford_0_1_1.sptable"

# load the test instances
test_instances = [UMFData(f) for f in readdir(test_instance_directory, join=true) if is_instance_path(f)]

# test checkpoint function
function test_checkpoint(checkpoint_path, instances)
    @load checkpoint_path _model
    optimalities, speedups, graph_reductions= [],[],[]

    for inst in instances
        sinst = UMFSolver.scale(inst)
        gnng = UMFSolver.to_gnngraph(sinst, feature_type=Float32)
        t_prediction::Float64 = @elapsed scores = _model(gnng)
        h5open("prediction.jld", "w") do file
            write(file, "pred", scores)
        end

        # solve with default solver
        dij_s, dij_ss = solveCG(inst, 
                                set_config("CG","cplex","./output.txt","linear","dijkstra")
                               )

        # solve with ML solver
        dij_ml_s, dij_ml_ss = solveCG(inst, 
                                      set_config("CG","cplex","./output.txt","linear","clssp prediction.jld 1 $(sptable_path)")
                                     )
        # set aside the solver stats
        opt = (dij_ml_ss.stats["val"] - dij_ss.stats["val"]) / dij_ss.stats["val"]
        if opt >= 0
            push!(optimalities, opt)
            push!(graph_reductions, dij_ml_ss.stats["graph_reduction"])
            push!(speedups, dij_ss.stats["timetot"] / dij_ml_ss.stats["timetot"])
        end
    end
    if !isempty(optimalities)
        println("Checkpoint  : $(checkpoint_path)")
        println("\topt       : ", [minimum(optimalities), mean(optimalities), maximum(optimalities)])
        println("\tspeedups  : ", [minimum(speedups), mean(speedups), maximum(speedups)])
        println("\tgraph red : ", [minimum(graph_reductions), mean(graph_reductions), maximum(graph_reductions)])
    end
    return optimalities, speedups, graph_reductions

end

function get_epoch_num(p)
    r = parse(Int64, replace(replace(split(p,"_")[end], "e"=>""), ".bson"=>""))
    return r
end
checkpoint_paths = [joinpath(checkpoint_directory,f) for f in readdir(checkpoint_directory) if startswith(f, "checkpoint")]
checkpoint_paths = sort(checkpoint_paths, by=p->get_epoch_num(p))
for chp in checkpoint_paths
    test_checkpoint(chp, test_instances)
end
