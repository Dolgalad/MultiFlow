using UMFSolver
using Statistics
using Plots
using Random

Random.seed!(2023)

function remove_random_arcs(path, a)
    inst = UMFData(path)
    p = rand(ne(inst))
    return UMFData(
                      inst.name*"_p$a",
                      inst.srcnodes[p .>= a],
                      inst.dstnodes[p .>= a],
                      inst.capacities[p .>= a],
                      inst.costs[p .>= a],
                      inst.latencies[p .> a],
                      inst.srcdemands,
                      inst.dstdemands,
                      inst.bandwidths,
                      inst.demand_latencies,
                      nk(inst),
                      nv(inst),
                      ne(inst)-sum(p .< a),
                  )
end



tol = 1e-8

alphas = LinRange(0.0, 0.1, 10)

network_names = ["Ntt_0_1_1" "AttMpls_0_1_1" "Chinanet_0_1_1" "giul39_0_1_1" "zib54_0_1_1" "Iij_0_1_1" "AsnetAm_0_1_1" "india35_0_1_1"]
model_paths = [
              "models/Ntt_0_1_1/best_checkpoint.bson"
              "models/AttMpls_0_1_1/best_checkpoint.bson"
              "models/Chinanet_0_1_1/best_checkpoint.bson"
              "models/giul39_0_1_1/best_checkpoint.bson"
              "models/zib54_0_1_1/best_checkpoint.bson"
              "models/Iij_0_1_1/best_checkpoint.bson"
              "models/AsnetAm_0_1_1/model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1/best_checkpoint.bson"
              "models/india35_0_1_1/best_checkpoint.bson"
              ]

optimality = []
optimality_std = []
graph_reduction = []

for (network,model_path) in zip(network_names, model_paths)
    println("Network = ", network)
    println("\tmodel = ", model_path)
    dataset_path = "datasets/$(network)/test"
    
    default_config = set_config("CG","cplex","./output","linear","dijkstra")
 
    config = set_config("CG","cplex","./output","linear","clssp $(model_path) postprocessing:2")

    model_opt, model_opt_std, model_gr = [],[],[]

    for alpha in alphas
        println("\tAlpha value : $alpha")
        alpha_optimality = []
        alpha_gr = []
        for instance_path in readdir(dataset_path, join=true)
            if !UMFSolver.is_instance_path(instance_path)
                continue
            end
            inst = remove_random_arcs(instance_path, alpha)
            _,ss0 = solveCG(inst, default_config)
            _,ss1 = solveCG(inst, config)
            opt = (ss1.stats["val"] - ss0.stats["val"]) / ss0.stats["val"]
            if opt >= -tol && opt < 0
                opt = 0.0
            end
    
            if opt >= -tol
                push!(alpha_optimality, opt)
                push!(alpha_gr, ss1.stats["graph_reduction"])
            end
        end
        push!(model_opt, mean(alpha_optimality))
        push!(model_opt_std, std(alpha_optimality))
        push!(model_gr, mean(alpha_gr))
        println("\t\topt = ", model_opt[end], ", std = ", model_opt_std[end], ", gr = ", model_gr[end])
    end
    push!(optimality, model_opt)
    push!(optimality_std, model_opt_std)
    push!(graph_reduction, model_gr)

end

plot(alphas, optimality, labels = network_names)
