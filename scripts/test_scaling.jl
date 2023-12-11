"""Todo : measure to total amount of bandwidth that is effectively routed sum_k b_k (1-y_k)

"""


using UMFSolver
using Plots

instance_dir = "datasets/AsnetAm_0_1_1/test"
model_names = readdir("testing_outputs/checkpoints/best_models/AsnetAm_0_1_1")

scaling_factors = LinRange(0.1, 1000.0, 10)

# default solve configuration
default_config = set_config("CG","cplex","./output","linear","dijkstra")
# test configurations
configs = Dict(
                  "2-ksp" => set_config("CG","cplex","./output","linear","kspfilter 2"),
                  "4-ksp" => set_config("CG","cplex","./output","linear","kspfilter 4"),
                  "8-ksp" => set_config("CG","cplex","./output","linear","kspfilter 8"),
              )
for (i,m) in enumerate(model_names)
    if m == "best_checkpoint.bson"
        continue
    end
    model_path = "testing_outputs/checkpoints/best_models/AsnetAm_0_1_1/$(m)/best_checkpoint.bson"
    println(model_path)
    ml_config = set_config("CG","cplex","./output","linear","clssp $(model_path) K:0 postprocessing:2")
    configs["model_$i"] = ml_config
end

optimality(v_orig::Float64, v_new::Float64) = abs(v_orig - v_new) / v_orig
speedup(t_orig::Float64, t_new::Float64) = t_orig / t_new
optimality(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = optimality(ss_orig.stats["val"], ss_new.stats["val"])
speedup(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = speedup(ss_orig.stats["timetot"], ss_new.stats["timetot"])


opt = Dict(k=>[] for k in keys(configs))
grs = Dict(k=>[] for k in keys(configs))
sps = Dict(k=>[] for k in keys(configs))

for (i,scaling_factor) in enumerate(scaling_factors)
    println("$i")
    # 
    opts, spss, grss = Dict(k=>[] for k in keys(configs)),
                       Dict(k=>[] for k in keys(configs)),
                       Dict(k=>[] for k in keys(configs))

    for inst_path in readdir(instance_dir, join=true)
        if !UMFSolver.is_instance_path(inst_path)
            continue
        end
        # scale instance
        inst = UMFSolver.scale_bandwidths(UMFData(inst_path), scaling_factor)
        # solve with default solver
        s0,ss0 = solveCG(inst, default_config)

        for k in keys(configs)
            s,ss = solveCG(inst, configs[k])
            push!(opts[k], optimality(ss0, ss))
            push!(spss[k], speedup(ss0, ss))
            push!(grss[k], ss.stats["graph_reduction"])
        end
    end
    for k in keys(configs)
        push!(opt[k], opts[k])
        push!(sps[k], spss[k])
        push!(grs[k], grss[k])
    end
end

#println([size(sizes), size(opt_ml), size(opt_ksp)])
#p=plot(sizes, [opt_ml, opt_ksp], label=["ml" "ksp"], seriestype=:scatter);
#savefig(p,"generalize_test.png")
ks = collect(keys(configs))
plots=[]
p = plot(scaling_factors, [[mean(x) for x in opt[k]] for k in keys(configs)], labels=reshape(ks, (1,size(ks,1))), xscale=:log);
for (i,k) in enumerate(ks)
    ss1 = [mean(x) .- std(x) for x in opt[k]]
    ss2 = [mean(x) .+ std(x) for x in opt[k]]
    plot!(scaling_factors, ss1, fillrange = ss2, fillalpha = 0.2, c = i, label=false, alpha=0)
end
push!(plots, p)
push!(plots, plot(scaling_factors, [[mean(x) for x in grs[k]] for k in keys(configs)], xscale=:log, labels=false))
push!(plots, plot(scaling_factors, [[mean(x) for x in sps[k]] for k in keys(configs)], xscale=:log, labels=false))
plot(plots..., layout=(3,1), link=:x, size=(400,800), margin_bottom=10Plots.px)
savefig("haha.png")
