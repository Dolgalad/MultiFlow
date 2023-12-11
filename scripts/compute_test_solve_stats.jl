using GraphPlot
using Compose
import Cairo
using StatsPlots
using LinearAlgebra
using DataFrames
using FileIO
using Statistics
using CSV

include("read_test_outputs.jl")

output_dir = "testing_outputs/solver_outputs/test_outputs_nolats_temp"
output_dir = "testing_outputs/solver_outputs/test_outputs_temp_best_1"

#output_dir = "test_outputs_temp_best_1"
#output_dir = "test_outputs_constrained_best"
#output_dir = "test_outputs_best"
#output_dir = "test_outputs_nolats"
#output_dir = "testing_outputs/solver_outputs/test_outputs_temp_scaled_2"
#output_dir = "testing_outputs/solver_outputs/test_outputs_temp_postprocessing2"

problem_version = "constrained"
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

for k1 in keys(data)
    #println("k1 = ", k1, " ", typeof(data))
    network_name = split(k1,"_")[1]
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

        for m in models
            pricing_type = split(m,"_")[1]
            #println("m = ", m, ", pricing_type = ", pricing_type)
            field_data = []
            for k in keys(data[k1][k2][problem_version][m])
                opt = (data[k1][k2][problem_version][m][k]["val"]-data[k1][k2][problem_version][default_solver][k]["val"] ) / data[k1][k2][problem_version][default_solver][k]["val"]
                pr_speedup = (data[k1][k2][problem_version][m][k]["time_pr_sol"]-data[k1][k2][problem_version][default_solver][k]["time_pr_sol"] ) / data[k1][k2][problem_version][default_solver][k]["time_pr_sol"]
                ms_speedup = (data[k1][k2][problem_version][m][k]["time_ms_sol"]-data[k1][k2][problem_version][default_solver][k]["time_ms_sol"] ) / data[k1][k2][problem_version][default_solver][k]["time_ms_sol"]
                addcol_speedup = (data[k1][k2][problem_version][m][k]["time_ms_addcol"]-data[k1][k2][problem_version][default_solver][k]["time_ms_addcol"] ) / data[k1][k2][problem_version][default_solver][k]["time_ms_addcol"]

                dij_tott = data[k1][k2][problem_version][default_solver][k]["time_ms_sol"] + data[k1][k2][problem_version][default_solver][k]["time_pr_sol"] + data[k1][k2][problem_version][default_solver][k]["t_create_pricing"] + data[k1][k2][problem_version][default_solver][k]["t_create_master"] + data[k1][k2][problem_version][default_solver][k]["time_ms_addcol"] 
                ttot = data[k1][k2][problem_version][m][k]["time_ms_sol"] + data[k1][k2][problem_version][m][k]["time_pr_sol"] + data[k1][k2][problem_version][m][k]["t_create_pricing"] + data[k1][k2][problem_version][m][k]["t_create_master"] + data[k1][k2][problem_version][m][k]["time_ms_addcol"] 
                ttot_speedup = (ttot - dij_tott) / dij_tott

                ss = data[k1][k2][problem_version][m][k]

                if opt > -tol
                    push!(model_field_data[m], [network_name pricing_type opt ss["t_create_pricing"] ss["t_create_master"] ss["time_pr_sol"] pr_speedup ss["time_ms_sol"] ms_speedup ss["time_ms_addcol"] addcol_speedup ss["graph_reduction"] ss["val"] ttot ttot_speedup])

                end
            end
        end
        if any(isempty(model_field_data[m]) for m in models)
            continue
        end
        for m in models
            for i in 1:length(model_field_data[m])
                push!(ms_pr_data, model_field_data[m][i])
            end
        end
    end
end

df = DataFrame(vcat(ms_pr_data...), [:Network, :Pricing, :Opt, :PrInit, :MsInit, :PrSol, :PrSU, :MsSol, :MsSU, :AddCol, :AddColSU, :GR, :Val, :TotT, :TotSU])
df[!,:Network] = convert.(String,df[!,:Network])
df[!,:Pricing] = convert.(String,df[!,:Pricing])
df[!,:Opt] =  convert.(Float64,df[!,:Opt])
df[!,:PrInit] =  round.(convert.(Float64,df[!,:PrInit]), digits=3)
df[!,:MsInit] =  round.(convert.(Float64,df[!,:MsInit]), digits=3)
df[!,:PrSol] =   round.(convert.(Float64,df[!,:PrSol]), digits=3)
df[!,:PrSU] =   round.(convert.(Float64,df[!,:PrSU]), digits=3)
df[!,:MsSol] =   round.(convert.(Float64,df[!,:MsSol]), digits=3)
df[!,:MsSU] =   round.(convert.(Float64,df[!,:MsSU]), digits=3)
df[!,:AddCol] =  round.(convert.(Float64,df[!,:AddCol]), digits=3)
df[!,:AddColSU] =  round.(convert.(Float64,df[!,:AddColSU]), digits=3)
df[!,:GR] =  convert.(Float64,df[!,:GR])
df[!,:Val] =  convert.(Float64,df[!,:Val])
df[!,:TotT] =  convert.(Float64,df[!,:TotT])
df[!,:TotSU] =  convert.(Float64,df[!,:TotSU])

CSV.write(joinpath(output_dir, "solve_times_3.csv"), df)

