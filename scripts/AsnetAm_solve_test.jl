using UMFSolver
using BenchmarkTools
using Statistics

networks = ["AsnetAm","AttMpls","Chinanet","giul39","Iij","india35","Ntt","zib54"]

for network in networks
    println("Network : ", network)

    instance_path = "/mnt/DataRaid/Documents/Documents_2023/julia/datasets/$(network)_0_1_1/test"
    model_names = readdir("/mnt/DataRaid/Documents/Documents_2023/julia/$(network)_0_1_1")
    for model_name in model_names
        println("\tModel name : ", model_name)
        model_path = "/mnt/DataRaid/Documents/Documents_2023/julia/$(network)_0_1_1/$(model_name)/best_checkpoint.bson"
        sptable = "/mnt/DataRaid/Documents/Documents_2023/julia/sptables/$(network)_0_1_1.sptable"
        
        
        global opts = Dict("m1"=>[], "m2"=>[])
        global grs = Dict("m1"=>[], "m2"=>[])
        global tprs = Dict("m1"=>[], "m2"=>[])
        global ttot = Dict("m1"=>[], "m2"=>[])
        
        optimality(v_orig::Float64, v_new::Float64) = abs(v_orig - v_new) / v_orig
        speedup(t_orig::Float64, t_new::Float64) = t_orig / t_new
        optimality(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = optimality(ss_orig.stats["val"], ss_new.stats["val"])
        speedup(ss_orig::UMFSolver.SolverStatistics, ss_new::UMFSolver.SolverStatistics) = speedup(ss_orig.stats["timetot"], ss_new.stats["timetot"])
        
        config0 = set_config("CG","cplex","./output","linear","dijkstra")
        config1 = set_config("CG","cplex","./output","linear","kspfilter 8")
        config2 = set_config("CG","cplex","./output","linear","clssp $(model_path) K:0 postprocessing:2")
            
        
        for inst_path in readdir(instance_path, join=true)
            if !UMFSolver.is_instance_path(inst_path)
                continue
            end
            inst = UMFData(inst_path)
            s0,ss0 = solveCG(inst, config0);
            s1,ss1 = solveCG(inst, config1);
            s2,ss2 = solveCG(inst, config2);
        
            push!(opts["m1"], optimality(ss0, ss1))
            push!(opts["m2"], optimality(ss0, ss2))
            push!(grs["m1"], ss1.stats["graph_reduction"])
            push!(grs["m2"], ss2.stats["graph_reduction"])
            push!(tprs["m1"], ss1.stats["t_create_pricing"])
            push!(tprs["m2"], ss2.stats["t_create_pricing"])
            push!(ttot["m1"], speedup(ss0,ss1))
            push!(ttot["m2"], speedup(ss0,ss2))
        
        
        end
        
        println("\t\tMethod 1")
        println("\t\t\topt  = ", mean(opts["m1"]))
        println("\t\t\tgr   = ", mean(grs["m1"]))
        println("\t\t\ttprs = ", mean(tprs["m1"]))
        println("\t\t\tttot = ", mean(ttot["m1"]))
        
        println("\t\tMethod 2")
        println("\t\t\topt  = ", mean(opts["m2"]))
        println("\t\t\tgr   = ", mean(grs["m2"]))
        println("\t\t\ttprs = ", mean(tprs["m2"]))
        println("\t\t\tttot = ", mean(ttot["m2"]))
    end
end
