using UMFSolver
using Graphs
using Plots
using ROC
using BSON: @load


function kspclassifier(inst, K=1)
    am = UMFSolver.get_arc_matrix(inst)
    cm = UMFSolver.get_cost_matrix(inst)
    g = UMFSolver.get_graph(inst)
    predictions = []
    for k in 1:nk(inst)
        ys = yen_k_shortest_paths(g, inst.srcdemands[k], inst.dstdemands[k], cm, K)
        col = zeros(ne(inst))
        for p in ys.paths
            col[UMFSolver.path_arc_indexes(p, am)] .= 1
        end
        push!(predictions, col)
    end
    return vcat(predictions...)
end

output_dir = "Iij_0_1_1_roc_imgs"
mkpath(output_dir)

test_instance_dir = "datasets/Iij_0_1_1/test/"

model_path = "best_models/Iij_0_1_1/best_checkpoint.bson"

sptable = "sptables/Iij_0_1_1.sptable"



props = LinRange(0.02, .25, 100)

function optimality(x, x0)
    return abs(x-x0)/x0
end

for test_instance_path = readdir(test_instance_dir, join=true)
    if !UMFSolver.is_instance_path(test_instance_path)
        println("skip ", test_instance_path, " ",UMFSolver.is_instance_path(test_instance_path)
)
        continue
    end
    println(test_instance_path)
    inst = UMFData(test_instance_path)
    default_config = set_config("CG","cplex","./output","linear","dijkstra")
    s0,ss0 = solveCG(inst, default_config)

    optims, grs, times = [],[],[]
    for p in props
        config = set_config("CG","cplex","./output","linear","clssp $(model_path) 0 $(sptable) keep_proportion:$p")
        s,ss = solveCG(inst, config)
        println([p, optimality(ss.stats["val"],ss0.stats["val"]), ss.stats["graph_reduction"], ss.stats["timetot"]])
        push!(optims, optimality(ss.stats["val"], ss0.stats["val"]))
        push!(grs, ss.stats["graph_reduction"])
        push!(times, ss0.stats["timetot"]/ss.stats["timetot"])
    end
    
    p=plot([plot(props, optims, label=false, ylabel="optimality"), plot(props, grs, ylabel="graph_reduction", label=false), plot(props, times, ylabel="solve speedup", xlabel="proportion arcs kept", label=false)]..., layout=(3,1), link=:x, size=(500, 500), bottom_margin=50Plots.px);
    savefig(p, joinpath(output_dir, basename(test_instance_path)*"_opt.png"))
    
    # ROC curves compare classifier with threshold 0, classifier with 10% arcs kept, k shortest paths
    
    # classifier
    @load model_path _model
    scores = _model(UMFSolver.to_gnngraph(UMFSolver.scale(inst), feature_type=Float32))
    
    # prediction standard
    pred_standard = vec(scores)
    
    # prediction with 10% arcs kept
    config = set_config("CG","cplex","./output","linear","clssp $(model_path) 0 $(sptable) keep_proportion:0.05")
    s_,ss_ = solveCG(inst, config)
    pred_kept_05 = vcat([Vector{Float64}(ss_.stats["pr"].filter.masks[k]) for k in 1:nk(inst)]...) .- 0.5
    config = set_config("CG","cplex","./output","linear","clssp $(model_path) 0 $(sptable) keep_proportion:0.02")
    s_,ss_ = solveCG(inst, config)
    pred_kept_02 = vcat([Vector{Float64}(ss_.stats["pr"].filter.masks[k]) for k in 1:nk(inst)]...) .- 0.5
    
    println(typeof(pred_kept_05))
    
    # K shortest paths
    pred_ksp1 = kspclassifier(inst, 1) .- 0.5
    pred_ksp3 = kspclassifier(inst, 3) .- 0.5
    println([typeof(pred_ksp1), typeof(pred_ksp3)])
    
    labels = vec(s0.x .> 0)
    
    roc_standard = roc(pred_standard, labels)
    roc_kept_02 = roc(pred_kept_02, labels)
    roc_kept_05 = roc(pred_kept_05, labels)
    #roc_kept = roc(pred_kept, labels)
    
    roc_ksp1 = roc(pred_ksp1, labels)
    roc_ksp3 = roc(pred_ksp3, labels)
    
    println("AUD standard   : ", AUC(roc_standard))
    println("AUC keeping 2% : ", AUC(roc_kept_02))
    println("AUC keeping 5% : ", AUC(roc_kept_05))
    println("AUC ksp K=1    : ", AUC(roc_ksp1))
    println("AUC ksp K=3    : ", AUC(roc_ksp3))
    
    p=plot(roc_standard, label="standard");
    plot!(roc_kept_02, label="kept 2%")
    plot!(roc_kept_05, label="kept 5%")
    plot!(roc_ksp1, label="ksp 1")
    plot!(roc_ksp3, label="ksp 3")

    roc_plot_path = joinpath(output_dir, basename(test_instance_path)*"_roc.png")
    savefig(p, roc_plot_path)
end


