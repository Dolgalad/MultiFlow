using GraphPlot
using Compose
import Cairo
using Plots
using StatsPlots
using LinearAlgebra
using DataFrames
using VegaLite
using FileIO
using Statistics
using CSV

include("read_test_outputs.jl")

output_dir = "testing_outputs/solver_outputs/test_outputs_nolats_temp"
#output_dir = "test_outputs_temp_best_1"
#output_dir = "test_outputs_constrained_best"
#output_dir = "test_outputs_best"
#output_dir = "test_outputs_nolats"
#output_dir = "testing_outputs/solver_outputs/test_outputs_temp_scaled_2"
#output_dir = "testing_outputs/solver_outputs/test_outputs_temp_postprocessing2"

problem_version = "unconstrained"
if problem_version == "unconstrained"
    default_solver = "dijkstra"
else
    default_solver = "larac"
end

network_prefixes = ["AsnetAm","AttMpls", "Chinanet", "giul39", "Iij", "india35", "Ntt", "zib54"]
#network_prefixes = ["AsnetAm"]

exclude_networks = ["Oxford"]

data = read_test_outputs(output_dir)

network_labels = []

ms_pr_data = []
ms_pr_data_2 = []

# optims, graph_reduction, tpricing, tinit, ttot
optims, graph_red, tpricing, tinit, ttot = Dict(), Dict(), Dict(), Dict(), Dict()

for k1 in keys(data)
    #println("k1 = ", k1, " ", typeof(data))
    network_name = k1 #split(k1,"_")[1]
    if network_name in exclude_networks
        continue
    end

    for k2 in keys(data[k1])
        #println("k2 = ", k2)
        models = keys(data[k1][k2][problem_version])
        println([k1, k2, models])
        if length(models)==1
            continue
        end
        model_field_data = Dict(m=>[] for m in models)
        model_field_data_2 = Dict(m=>[] for m in models)

        for m in models
            pricing_type = split(m,"_")[1]
            #println("m = ", m, ", pricing_type = ", pricing_type)
            field_data = []
            for k in keys(data[k1][k2][problem_version][m])
                opt = (data[k1][k2][problem_version][m][k]["val"]-data[k1][k2][problem_version][default_solver][k]["val"] ) / data[k1][k2][problem_version][default_solver][k]["val"]
                gr = data[k1][k2][problem_version][m][k]["graph_reduction"]
                dtpr = (data[k1][k2][problem_version][m][k]["time_pr_sol"] - data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]) / data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]

                ttott = data[k1][k2][problem_version][m][k]["time_pr_sol"]+data[k1][k2][problem_version][m][k]["time_ms_sol"]+data[k1][k2][problem_version][m][k]["time_ms_addcol"]
                ttot0 = data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]+data[k1][k2][problem_version][default_solver][k]["time_ms_sol"]+data[k1][k2][problem_version][default_solver][k]["time_ms_addcol"]

                dtpr = (data[k1][k2][problem_version][m][k]["time_pr_sol"] - data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]) / data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]
                dttot = (ttott - ttot0) / ttot0
                tii = data[k1][k2][problem_version][m][k]["t_create_pricing"]


                if opt > -tol
                    push!(field_data, (data[k1][k2][problem_version][m][k], opt))
                    if (network_name, m) in keys(optims)
                        push!(optims[(network_name,m)], opt)
                        push!(graph_red[(network_name,m)], gr)
                        push!(tpricing[(network_name,m)], dtpr)
                        push!(tinit[(network_name,m)], tii)
                        push!(ttot[(network_name,m)], dttot)
                    else
                        optims[(network_name,m)] = [ opt]
                        graph_red[(network_name,m)] = [gr]
                        tpricing[(network_name,m)] = [ dtpr]
                        tinit[(network_name,m)] =  [tii]
                        ttot[(network_name,m)] = [dttot]

                    end
                end
            end
            if !isempty(field_data)
                push!(model_field_data[m], [network_name pricing_type "ms" 1000*mean(ss["time_ms_sol"]     for (ss,_) in field_data)])
                push!(model_field_data[m], [network_name pricing_type "pr" 1000*     mean(ss["time_pr_sol"]     for (ss,_) in field_data)])
                push!(model_field_data[m], [network_name pricing_type "addcol" 1000* mean(ss["time_ms_addcol"]  for (ss,_) in field_data)])
                push!(model_field_data[m], [network_name pricing_type "ms init" 1000*mean(ss["t_create_master"] for (ss,_) in field_data)])
                push!(model_field_data[m], [network_name pricing_type "pr init" 1000*mean(ss["t_create_pricing"] for (ss,_) in field_data)])


                push!(model_field_data_2[m], [
                                              network_name 
                                              pricing_type 
                                              1000*mean(ss["t_create_pricing"] for (ss,_) in field_data) 
                                              1000*mean(ss["t_create_master"] for (ss,_) in field_data) 
                                              1000*mean(ss["time_pr_sol"]  for (ss,_) in field_data) 
                                              1000*mean(ss["time_ms_sol"] for (ss,_) in field_data) 
                                              1000*mean(ss["time_ms_addcol"] for (ss,_) in field_data) 
                                              mean(ss["graph_reduction"] for (ss,_) in field_data) 
                                              mean(ss["val"] for (ss,_) in field_data) 
                                              mean(o for (_,o) in field_data)])
            end
        end
        println(k2, ", ", [length(model_field_data[m]) for m in models])
        println("\t", any(isempty(model_field_data[m]) for m in models))
        if any(isempty(model_field_data[m]) for m in models)
            continue
        end
        for m in models
            for i in 1:length(model_field_data[m])
                push!(ms_pr_data, model_field_data[m][i])
            end
            for i in 1:length(model_field_data_2[m])
                push!(ms_pr_data_2, model_field_data_2[m][i])
            end

        end
    end
end

df = DataFrame(vcat(ms_pr_data...), [:Network, :Pricing, :Step, :Time])
df[!,:Network] = convert.(String,df[!,:Network])
df[!,:Pricing] = convert.(String,df[!,:Pricing])
df[!,:Step] = convert.(String,df[!,:Step])
df[!,:Time] = convert.(Float64,df[!,:Time])

df2 = DataFrame(vcat(ms_pr_data_2...), [:Network, :Pricing, :PrInit, :MsInit, :PrSol, :MsSol, :AddCol, :GR, :Val, :Opt])
df2[!,:Network] = convert.(String,df2[!,:Network])
df2[!,:Pricing] = convert.(String,df2[!,:Pricing])
df2[!,:PrInit] =  round.(convert.(Float64,df2[!,:PrInit]), digits=3)
df2[!,:MsInit] =  round.(convert.(Float64,df2[!,:MsInit]), digits=3)
df2[!,:PrSol] =   round.(convert.(Float64,df2[!,:PrSol]), digits=3)
df2[!,:MsSol] =   round.(convert.(Float64,df2[!,:MsSol]), digits=3)
df2[!,:AddCol] =  round.(convert.(Float64,df2[!,:AddCol]), digits=3)
df2[!,:GR] =  convert.(Float64,df2[!,:GR])
df2[!,:Val] =  convert.(Float64,df2[!,:Val])
df2[!,:Opt] =  convert.(Float64,df2[!,:Opt])

CSV.write(joinpath(output_dir, "solve_times_2.csv"), df2)


is_network_of_interest(net::String) = net in network_prefixes
df = filter(:Network=>is_network_of_interest, df)
println(df)
CSV.write(joinpath(output_dir, "solve_times.csv"), df)
p = df |> @vlplot(:bar, x=:Pricing, y=:Time, color=:Step, column=:Network, title="Solve times")
p |> FileIO.save(joinpath(output_dir, "grouped_solve_times_$(problem_version).svg"))
p |> FileIO.save(joinpath(output_dir, "grouped_solve_times_$(problem_version).png"))

p |> VegaLite.save(joinpath(output_dir, "grouped_solve_times_$(problem_version).vegalite"))


p = metric_boxplot3(output_dir, outliers=false,
       problem_version=problem_version, short_labels=false,
       network_prefixes=network_prefixes
      )

savefig(p, joinpath(output_dir, "boxplot_optimality_graph_reduction_$(problem_version).png"))

# compute performance metrics
df2.Ttot = df2.PrSol .+ df2.MsSol .+ df2.AddCol
methods = ["4sp","8sp","clssp"]
for m in methods
    println([size(df2[df2.Pricing .== m,:PrSol]), size(df2[df2.Pricing .== "dijkstra",:PrSol])])
    dtpr = (df2[df2.Pricing .== m,:PrSol] .- df2[df2.Pricing .== "dijkstra",:PrSol]) ./ df2[df2.Pricing .== "dijkstra", :PrSol]
    mdtpr, vdtpr = mean(dtpr), std(dtpr)
    dtot = (df2[df2.Pricing .== m,:Ttot] .- df2[df2.Pricing .== "dijkstra",:Ttot]) ./ df2[df2.Pricing .== "dijkstra", :Ttot]
    mdtot, vdtot = mean(dtot), std(dtot)
    gr = df2[df2.Pricing .== m,:GR]
    mgr, vgr = mean(gr), std(gr)
    opt = (df2[df2.Pricing .== m,:Val] .- df2[df2.Pricing .== "dijkstra",:Val]) ./ df2[df2.Pricing .== "dijkstra", :Val]
    mopt, vopt = mean(opt), std(opt)

    println(m, join(round.([100*mdtpr, 100*vdtpr, 100*mdtot, 100*vdtot, 100*mgr, 100*vgr, 100*mopt, 100*vopt], digits=3), " "))
end
