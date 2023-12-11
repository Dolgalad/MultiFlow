using GraphPlot
using Compose
import Cairo
using Plots
using StatsPlots
using LinearAlgebra
using DataFrames
using VegaLite
using FileIO

using UMFSolver
using Graphs
using BSON: @load
using Flux
using JLD
using HDF5
using Statistics
using BenchmarkTools

include("scripts/make_shortest_path_table.jl")

model_name = "model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1/best_checkpoint.bson"
model_name = "best_checkpoint.bson"

model_dir = "best_models"

make_plots = false

network_suffix = "_frod"

image_dir = "best_checkpoint_test_imgs"
if isempty(network_suffix)
    image_dir = joinpath(image_dir, "default")
else
    image_dir = joinpath(image_dir, network_suffix)
end
if isdir(image_dir)
    rm(image_dir, force=true, recursive=true)
end
mkpath(image_dir)

# some networks are badly trained and can be excluded
excluded_networks = ["india35", "AsnetAm"]

# solver configs
dijconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "dijkstra")
laracconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "larac")
mldijconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "clssp prediction.jld 0")
mllaracconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "clslarac prediction.jld 0")

# list of network names
network_names = ["Oxford", "AsnetAm", "Ntt", "AttMpls", "Chinanet", "giul39", "india35", "zib54", "Iij"]

# store testing metrics
dij_speedups, dij_optimalities, dij_graph_reductions=Dict(n=>[] for n in network_names), 
                                         Dict(n=>[] for n in network_names), 
                                         Dict(n=>[] for n in network_names)
larac_speedups, larac_optimalities, larac_graph_reductions=Dict(n=>[] for n in network_names), 
                                         Dict(n=>[] for n in network_names), 
                                         Dict(n=>[] for n in network_names)

# utility function
ms_pr_time_prop(x) = 100 * [x.stats["time_ms_sol"]/x.stats["timetot"], 
                            x.stats["time_pr_sol"]/x.stats["timetot"]]

# first run
first_run = true

print("Running on Oxford_0_1_1...")
@load "models/Oxford_0_1_1/best_checkpoint.bson" _model
instance_path_t = "datasets/Oxford_0_1_1_delay/test/1"
inst_t = UMFData(instance_path_t)
sinst_t = UMFSolver.scale(inst_t)
gnng_t = UMFSolver.load_instance(instance_path_t)
scores_t = _model(gnng_t)
println([minimum(scores_t), maximum(scores_t)])
#jldopen("prediction.jld", "w") do file
#    write(file, "pred", scores_t)
#end
h5open("prediction.jld", "w") do file
    write(file, "pred", scores_t)
end

dij_s1_t, dij_ss1_t = solveCG(sinst_t, mldijconfig)
larac_s1, larac_ss1_t = solveCG(sinst_t, mllaracconfig)
dij_s_t, dij_ss_t = solveCG(sinst_t, dijconfig)
larac_s_t, larac_ss_t = solveCG(sinst_t, laracconfig)
println("done")

# master and pricing time bar plot
dij_ms_pr_times = []
dij_ms_pr_labels = []
ms_pr_data = []
ms_pr_data_constrained = []

for network_name in network_names
    model_checkpoint_path = joinpath(model_dir, "$(network_name)_0_1_1$(network_suffix)", model_name)
    if !isfile(model_checkpoint_path)
        println("Model checkpoint $(model_checkpoint_path) does not exist.")
        continue
    end
    if network_name in excluded_networks
        println("Skipping network $(network_name)")
        continue
    end

    println("Network $(network_name)$(network_suffix)")

    inst = UMFData("instances_sndlib/delay_corrected/$(network_name)_0_1_1")
    sinst = UMFSolver.scale(inst)
    g = UMFSolver.get_graph(inst)
    # save the shortest path table
    sptable_path = joinpath("sptables", "$(network_name)_0_1_1.sptable")
    Tc = create_shortest_path_table(inst, dstmx=inst.costs)
    Td = create_shortest_path_table(inst, dstmx=inst.latencies)

    #jldopen(sptable_path, "w") do file
    #    write(file, "cost", Tc)
    #    write(file, "delay", Td)
    #end
    h5open(sptable_path, "w") do file
        write(file, "cost", Tc)
        write(file, "delay", Td)
    end



    # solving unconstrained problem with default dijkstra pricing
    @load "$(model_dir)/$(network_name)_0_1_1$(network_suffix)/best_checkpoint.bson" _model
    t_prediction::Float64 = @elapsed scores = _model(UMFSolver.to_gnngraph(sinst, feature_type=Float32))
    println("t_prediction = $(t_prediction)")
    #jldopen("prediction.jld", "w") do file
    #    write(file, "pred", scores)
    #end
    h5open("prediction.jld", "w") do file
        write(file, "pred", scores)
    end

    n_solve = 1

    dij_stats, dij_ml_stats, larac_stats, larac_ml_stats = [],[],[],[]
    for i in 1:n_solve
        Base.GC.enable(false)
        dij_s, dij_ss = solveCG(sinst, 
                            set_config("CG","cplex","./output.txt","linear","dijkstra")
                                 )
        Base.GC.enable(true)
        Base.GC.gc()

        # solving unconstrained problem with ML dijkstra pricing
        Base.GC.enable(false)
        dij_s_ml, dij_ss_ml = solveCG(sinst,
                                      #set_config("CG","cplex","./output.txt","linear","clssp $(model_checkpoint_path) 0")
                                      #set_config("CG","cplex","./output.txt","linear","clssp prediction.jld 0")
                                      set_config("CG","cplex","./output.txt","linear","clssp prediction.jld 0 $(sptable_path)")
                                     )
        Base.GC.enable(true)
        Base.GC.gc()


        # solving constrained problem with default larac pricing
        Base.GC.enable(false)
        larac_s, larac_ss = solveCG(sinst, 
                                set_config("CG","cplex","./output.txt","linear","larac")
                                     )
        Base.GC.enable(true)
        Base.GC.gc()

        # solving constrained problem with ML LARAC pricing
        Base.GC.enable(false)
        larac_s_ml, larac_ss_ml = solveCG(sinst,
                                      #set_config("CG","cplex","./output.txt","linear","clslarac $(model_checkpoint_path) 0")
                                      #set_config("CG","cplex","./output.txt","linear","clslarac prediction.jld 0")
                                      set_config("CG","cplex","./output.txt","linear","clslarac prediction.jld 0 $(sptable_path)")
                                   )
        Base.GC.enable(true)
        Base.GC.gc()

        push!(dij_stats, dij_ss)
        push!(dij_ml_stats, dij_ss_ml)
        push!(larac_stats, larac_ss)
        push!(larac_ml_stats, larac_ss_ml)
        println("Values : ", [dij_ss.stats["val"], dij_ss_ml.stats["val"], larac_ss.stats["val"], larac_ss_ml.stats["val"]])
    end
    #push!(dij_ms_pr_times, 1000 * [dij_ss.stats["time_ms_sol"], dij_ss.stats["time_pr_sol"]])
    #push!(dij_ms_pr_times, 1000 * [dij_ss_ml.stats["time_ms_sol"], dij_ss_ml.stats["time_pr_sol"]])
    push!(dij_ms_pr_times, 1000 * [mean(ss.stats["time_ms_sol"] for ss in dij_stats), mean(ss.stats["time_pr_sol"] for ss in dij_stats)])
    push!(dij_ms_pr_times, 1000 * [mean(ss.stats["time_ms_sol"] for ss in dij_ml_stats), mean(ss.stats["time_pr_sol"] for ss in dij_ml_stats)])

    push!(dij_ms_pr_labels, network_name)
    push!(dij_ms_pr_labels, network_name*"_ml")

    push!(ms_pr_data, [network_name "dijkstra" "ms" 1000*mean(ss.stats["time_ms_sol"] for ss in dij_stats)])
    push!(ms_pr_data, [network_name "dijkstra" "pr" 1000*mean(ss.stats["time_pr_sol"] for ss in dij_stats)])
    push!(ms_pr_data, [network_name "dijkstra" "addcol" 1000* mean(ss.stats["time_ms_addcol"]  for ss in dij_stats)])
    push!(ms_pr_data, [network_name "dijkstra" "ms init" 1000*mean(ss.stats["t_create_master"] for ss in dij_stats)])
    push!(ms_pr_data, [network_name "dijkstra" "pr init" 1000*mean(ss.stats["t_create_pricing"] for ss in dij_stats)])
    #println(dij_ss.stats["t_create_pricing"] / dij_ss.stats["timetot"])

    push!(ms_pr_data, [network_name "clssp" "ms" 1000*     mean(ss.stats["time_ms_sol"]     for ss in dij_ml_stats)])
    push!(ms_pr_data, [network_name "clssp" "pr" 1000*     mean(ss.stats["time_pr_sol"]     for ss in dij_ml_stats)])
    push!(ms_pr_data, [network_name "clssp" "addcol" 1000* mean(ss.stats["time_ms_addcol"]  for ss in dij_ml_stats)])
    push!(ms_pr_data, [network_name "clssp" "ms init" 1000*mean(ss.stats["t_create_master"] for ss in dij_ml_stats)])
    push!(ms_pr_data, [network_name "clssp" "pr init" 1000*mean(ss.stats["t_create_pricing"] for ss in dij_ml_stats)])
    #println(dij_ss_ml.stats["t_create_pricing"] / dij_ss_ml.stats["timetot"])

    push!(ms_pr_data_constrained, [network_name "larac" "ms" 1000*     mean(ss.stats["time_ms_sol"]      for ss in larac_stats)])
    push!(ms_pr_data_constrained, [network_name "larac" "pr" 1000*     mean(ss.stats["time_pr_sol"]      for ss in larac_stats)])
    push!(ms_pr_data_constrained, [network_name "larac" "addcol" 1000* mean(ss.stats["time_ms_addcol"]   for ss in larac_stats)])
    push!(ms_pr_data_constrained, [network_name "larac" "ms init" 1000*mean(ss.stats["t_create_master"]  for ss in larac_stats)])
    push!(ms_pr_data_constrained, [network_name "larac" "pr init" 1000*mean(ss.stats["t_create_pricing"] for ss in larac_stats)])


    push!(ms_pr_data_constrained, [network_name "clslarac" "ms" 1000*     mean(ss.stats["time_ms_sol"]      for ss in larac_ml_stats)])
    push!(ms_pr_data_constrained, [network_name "clslarac" "pr" 1000*     mean(ss.stats["time_pr_sol"]      for ss in larac_ml_stats)])
    push!(ms_pr_data_constrained, [network_name "clslarac" "addcol" 1000* mean(ss.stats["time_ms_addcol"]   for ss in larac_ml_stats)])
    push!(ms_pr_data_constrained, [network_name "clslarac" "ms init" 1000*mean(ss.stats["t_create_master"]  for ss in larac_ml_stats)])
    push!(ms_pr_data_constrained, [network_name "clslarac" "pr init" 1000*mean(ss.stats["t_create_pricing"] for ss in larac_ml_stats)])



    #println("ms_t, pr_t, tot_t : ", [dij_ss.stats["time_ms_sol"], dij_ss.stats["time_pr_sol"], dij_ss.stats["timetot"]])
end
dij_ms_pr_times = transpose(hcat(dij_ms_pr_times...))
println(size(dij_ms_pr_times))

#x_pos = 1:size(dij_ms_pr_times,1)
groupedbar(dij_ms_pr_times, xticks=(1:size(dij_ms_pr_labels,1), dij_ms_pr_labels), bar_position=:stack, labels=["ms" "pr"] )

# make dataframe for solver times
df_unconstrained = DataFrame(vcat(ms_pr_data...), [:Network, :Pricing, :Step, :Time])
df_unconstrained[!,:Network] = convert.(String,df_unconstrained[!,:Network])
df_unconstrained[!,:Pricing] = convert.(String,df_unconstrained[!,:Pricing])
df_unconstrained[!,:Step] = convert.(String,df_unconstrained[!,:Step])
df_unconstrained[!,:Time] = convert.(Float64,df_unconstrained[!,:Time])

p = df_unconstrained |> @vlplot(:bar, x=:Pricing, y=:Time, color=:Step, column=:Network, title="Unconstrained solve times")
p |> FileIO.save(joinpath(image_dir, "grouped_solve_times_unconstrained.png"))

df_constrained = DataFrame(vcat(ms_pr_data_constrained...), [:Network, :Pricing, :Step, :Time])
df_constrained[!,:Network] = convert.(String,df_constrained[!,:Network])
df_constrained[!,:Pricing] = convert.(String,df_constrained[!,:Pricing])
df_constrained[!,:Step] = convert.(String,df_constrained[!,:Step])
df_constrained[!,:Time] = convert.(Float64,df_constrained[!,:Time])

p = df_constrained |> @vlplot(:bar, x=:Pricing, y=:Time, color=:Step, column=:Network, title="Constrained solve times")
p |> FileIO.save(joinpath(image_dir, "grouped_solve_times_constrained.png"))

for network_name in network_names
    # checkpoint path 
    model_checkpoint_path = joinpath(model_dir, "$(network_name)_0_1_1$(network_suffix)", model_name)
    if !isfile(model_checkpoint_path)
        println("Checkpoint $(model_checkpoint_path) not found")
        continue
    end
    if network_name in excluded_networks
        println("Skipping network $(network_name)")
        continue
    end

    # load model
    @load model_checkpoint_path _model

    # test instance directory
    test_instance_path = "datasets/$(network_name)_0_1_1_delay/test"
    for f in readdir(test_instance_path)
        if !UMFSolver.is_instance_path(joinpath(test_instance_path, f))
            println("Skipping ", joinpath(test_instance_path))
            continue
        end
    
        instance_path = joinpath(test_instance_path, f)
        println("Instance : $(instance_path)")
    
        # load instance
        inst = UMFData(instance_path)
        sinst = UMFSolver.scale(inst)
        gnng = UMFSolver.load_instance(instance_path)
    
        # prediction
        scores = _model(gnng)
    
        # save prediction
        h5open("prediction.jld", "w") do file
            write(file, "pred", scores)
        end
    
        # solve problem
        dij_s1, dij_ss1 = solveCG(sinst, mldijconfig)
        larac_s1, larac_ss1 = solveCG(sinst, mllaracconfig)
        dij_s, dij_ss = solveCG(sinst, dijconfig)
        larac_s, larac_ss = solveCG(sinst, laracconfig)
   
        #println("\tvals : ", [ss.stats["val"], ss1.stats["val"]])
        #println("\tGraph reduction   : ", round(100 * ss1.stats["graph_reduction"], digits=4), " %")
        #println("\tSpeedup           : ", round(100 * (ss.stats["timetot"] - ss1.stats["timetot"]) / ss1.stats["timetot"], digits=4), " %")
        #println("\tOptimality        : ", round(100 * (ss1.stats["val"] - ss.stats["val"]) / ss.stats["val"], digits=4), " %")
        #println("\tTime prop origin  : ", ms_pr_time_prop(ss))
        #println("\tTime prop ML      : ", ms_pr_time_prop(ss1))
        #println("\tMaster speedup    : ", round(100 * (ss.stats["time_ms_sol"] - ss1.stats["time_ms_sol"]) / ss.stats["time_ms_sol"], digits=4), " %")
        #println("\tPricing speedup   : ", round(100 * (ss.stats["time_pr_sol"] - ss1.stats["time_pr_sol"]) / ss.stats["time_pr_sol"], digits=4), " %")
    
        if dij_ss1.stats["val"] >= dij_ss.stats["val"]
            push!(dij_speedups[network_name], dij_ss.stats["timetot"] / dij_ss1.stats["timetot"])
            push!(dij_optimalities[network_name], (dij_ss1.stats["val"] - dij_ss.stats["val"]) / dij_ss.stats["val"])
            push!(dij_graph_reductions[network_name], dij_ss1.stats["graph_reduction"])
        else
            println("Optimality bad")
        end
        if larac_ss1.stats["val"] >= larac_ss.stats["val"]
            push!(larac_speedups[network_name], larac_ss.stats["timetot"] / larac_ss1.stats["timetot"])
            push!(larac_optimalities[network_name], (larac_ss1.stats["val"] - larac_ss.stats["val"]) / larac_ss.stats["val"])
            push!(larac_graph_reductions[network_name], larac_ss1.stats["graph_reduction"])
        else
            println("Larac optimality bad")
        end
    end
end

ls = [n for n in network_names if !isempty(dij_speedups[n])]
p=boxplot([dij_graph_reductions[n] for n in network_names if !isempty(dij_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Graph reduction");
savefig(p, joinpath(image_dir, "dijkstra_graph_reductions.png"))
p=boxplot([dij_speedups[n] for n in network_names if !isempty(dij_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Speedup", outliers=false);
savefig(p, joinpath(image_dir, "dijkstra_speedups.png"))
p=boxplot([dij_optimalities[n] for n in network_names if !isempty(dij_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Optimality", outliers=false);
savefig(p, joinpath(image_dir, "dijkstra_optimalities.png"))

ls = [n for n in network_names if !isempty(larac_speedups[n])]
p=boxplot([larac_graph_reductions[n] for n in network_names if !isempty(larac_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Graph reduction");
savefig(p, joinpath(image_dir, "larac_graph_reductions.png"))
p=boxplot([larac_speedups[n] for n in network_names if !isempty(larac_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Speedup", outliers=false);
savefig(p, joinpath(image_dir, "larac_speedups.png"))
p=boxplot([larac_optimalities[n] for n in network_names if !isempty(larac_speedups[n])], xticks=(1:size(ls,1),ls), labels=false, ylabel="Optimality", outliers=false);
savefig(p, joinpath(image_dir, "larac_optimalities.png"))
