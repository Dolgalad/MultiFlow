using UMFSolver
using Plots



inst = UMFData("/mnt/DataRaid/Documents/Documents_2023/julia/datasets/AsnetAm_0_1_1/test/1")
#instance_path = "/mnt/DataRaid/Documents/Documents_2023/julia/datasets/$(network)_0_1_1/test"
model_names = readdir("/mnt/DataRaid/Documents/Documents_2023/julia/AsnetAm_0_1_1")[1]
model_path = "/mnt/DataRaid/Documents/Documents_2023/julia/AsnetAm_0_1_1/$(model_names)/best_checkpoint.bson"
sptable = "/mnt/DataRaid/Documents/Documents_2023/julia/sptables/AsnetAm_0_1_1.sptable"

default_config = set_config("CG","cplex","./output","linear","dijkstra")
ksp_config = set_config("CG","cplex","./output","linear","kspfilter 4")
ml_config = set_config("CG","cplex","./output","linear","clssp $(model_path) K:0 postprocessing:2")

optimality(v_orig::Float64, v_new::Float64) = abs(v_orig - v_new) / v_orig
speedup(t_orig::Float64, t_new::Float64) = t_orig / t_new
optimality(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = optimality(ss_orig.stats["val"], ss_new.stats["val"])
speedup(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = speedup(ss_orig.stats["timetot"], ss_new.stats["timetot"])

sizes = []
opt_ml = []
opt_ksp = []

for (i,delta) in enumerate(LinRange(0.0, 2.0, 10))
    println("$i")
    try
        ninst = UMFSolver.generate_example_3(inst, demand_delta_p=delta)
        println("delta = $delta, instance_size = ", nv(ninst)*nk(ninst))
        s,ss = solveCG(ninst, default_config)
        s_ml,ss_ml = solveCG(ninst, ml_config)
        s_ksp,ss_ksp = solveCG(ninst, ksp_config)
        push!(sizes, nv(ninst)*nk(ninst))
        push!(opt_ml, optimality(ss, ss_ml))
        push!(opt_ksp, optimality(ss, ss_ksp))
    catch ex
        println("caught something")
        if isa(ex, InterruptException)
            throw(ex)
        end
    end
end

println([size(sizes), size(opt_ml), size(opt_ksp)])
p=plot(sizes, [opt_ml, opt_ksp], label=["ml" "ksp"], seriestype=:scatter);
savefig(p,"generalize_test.png")

