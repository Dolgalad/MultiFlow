using JSON
using Plots
using StatsPlots
using Base: startswith

tol = 1e-8

function Base.startswith(s::String, pl::AbstractVector)
    if isempty(pl)
        return true
    end
    return any(Base.startswith(s, p) for p in pl)
end

function read_test_outputs(output_dir)
    data = Dict()
    for f in readdir(output_dir, join=true)
        if isdir(f)
            data[basename(f)] = read_test_outputs(f)
        else
            if endswith(f, ".json")
                k = replace(basename(f), ".json"=>"")
                data[k] = JSON.parsefile(f)
            end
        end
    end
    return data
end


function optimality_data(output_dir; problem_version="unconstrained", network_prefixes=[])
    optimality = Dict()
    data = read_test_outputs(output_dir)
    default = problem_version=="unconstrained" ? "dijkstra" : "larac"
    for k1 in keys(data)
        if !startswith(k1, network_prefixes)
            continue
        end
        for k2 in keys(data[k1])
            models = keys(data[k1][k2][problem_version])
            for m in models
                if m!="dijkstra" && m!="larac"
                    model_optimality = Float64[]
                    for k in keys(data[k1][k2][problem_version][m])
                        opt = (data[k1][k2][problem_version][m][k]["val"]-data[k1][k2][problem_version][default][k]["val"] ) / data[k1][k2][problem_version][default][k]["val"]

                        if opt > -tol
                            push!(model_optimality, opt>=0 ? opt : 0)
                        else
                            println("Strange opt val : ", [k1, k2, problem_version, m, k, data[k1][k2][problem_version][m][k]["val"], data[k1][k2][problem_version][default][k]["val"], data[k1][k2][problem_version][m][k]["val"]-data[k1][k2][problem_version][default][k]["val"]])
                        end
                    end
                    if !isempty(model_optimality)
                        optimality[join([k2,m],"__")] = model_optimality
                    end

                end
            end
        end
    end
    return optimality
end

function speedup_data(output_dir; problem_version="unconstrained", network_prefixes=[])
    speedup = Dict()
    data = read_test_outputs(output_dir)
    default = problem_version=="unconstrained" ? "dijkstra" : "larac"
    for k1 in keys(data)
        if !startswith(k1, network_prefixes)
            continue
        end
        for k2 in keys(data[k1])
            models = keys(data[k1][k2][problem_version])
            for m in models
                if m!="dijkstra" && m!="larac"
                    model_speedups = Float64[]
                    for k in keys(data[k1][k2][problem_version][m])
                        spup = data[k1][k2][problem_version][default][k]["timetot"]  / data[k1][k2][problem_version][m][k]["timetot"]
                        push!(model_speedups, spup)
                    end
                    if !isempty(model_speedups)
                        speedup[join([k2,m],"__")] = model_speedups
                    end

                end
            end
        end
    end
    return speedup
end

function graph_reduction_data(output_dir; problem_version="unconstrained", network_prefixes=[])
    graph_reduction = Dict()
    data = read_test_outputs(output_dir)
    for k1 in keys(data)
        if !startswith(k1, network_prefixes)
            continue
        end
        for k2 in keys(data[k1])
            models = keys(data[k1][k2][problem_version])
            for m in models
                if m!="dijkstra" && m!="larac"
                    model_graph_reduction = []
                    for k in keys(data[k1][k2][problem_version][m])
                        spup = data[k1][k2][problem_version][m][k]["graph_reduction"]
                        push!(model_graph_reduction, spup)
                    end
                    if !isempty(model_graph_reduction)
                        graph_reduction[join([k2,m],"__")] = model_graph_reduction
                    end

                end
            end
        end
    end
    return graph_reduction
end


function metric_boxplot(output_dir, metric; outliers=true, problem_version="unconstrained", size=(1000,700), yticks=true, short_labels=false, network_prefixes=[])
    if metric=="optimality"
        data = optimality_data(output_dir, problem_version=problem_version, network_prefixes=network_prefixes)
    elseif metric=="speedup"
        data = speedup_data(output_dir, problem_version=problem_version, network_prefixes=network_prefixes)
    elseif metric=="graph_reduction"
        data = graph_reduction_data(output_dir, problem_version=problem_version, network_prefixes=network_prefixes)
    else
    end
    idx=sortperm(collect(keys(data)), rev=true)
    if short_labels
        labels = map(x->split(x,"_")[1], collect(keys(data)))
    else
        labels = collect(keys(data))
    end
    println("Plotting metric : ", metric)
    for i in 1:length(labels)
        println("\t", labels[idx[i]], " ", length(data[collect(keys(data))[idx[i]]]))
    end
    println()

    xscale=:identity
    if metric=="optimality"
        xscale=:identity
        for k in keys(data)
            data[k] = data[k].+tol
        end
    end

    if yticks
        boxplot(collect(values(data))[idx], yticks=(1:length(data), labels[idx]), labels=:none, size=size, orientation=:horizontal, outliers=outliers, xlabel=metric, ylim=[0,length(data)+1], xscale=xscale)
    else
        boxplot(collect(values(data))[idx], labels=:none, size=size, orientation=:horizontal, outliers=outliers, yticks=false, xlabel=metric, grid=true, ylim=[0,length(data)+1], xscale=xscale)
    end

end

function metric_boxplot2(output_dir; outliers=true, problem_version="unconstrained", short_labels=false, network_prefixes=[])
    plots = []
    for (i,k) in enumerate(["optimality","speedup","graph_reduction"])
        if i==1
            push!(plots, metric_boxplot(output_dir, k, outliers=outliers, problem_version=problem_version, yticks=true, short_labels=short_labels, network_prefixes=network_prefixes))
        else
            push!(plots, metric_boxplot(output_dir, k, outliers=outliers, problem_version=problem_version, yticks=false, network_prefixes=network_prefixes))
        end
    end
    # bar plot
    data = optimality_data(output_dir, problem_version=problem_version, network_prefixes=network_prefixes)
    idx=sortperm(collect(keys(data)), rev=true)
    vals = [sum(val .>= 0) for val in values(data)]
    push!(plots, bar(vals[idx], label=false, orientation=:horizontal, ylim=[0, length(vals)+1], yticks=false))

    plot(plots..., link=:y, layout=(1,4), size=(2000,800), bottom_margin=50Plots.px, top_margin=50Plots.px)
end


function metric_boxplot3(output_dir; outliers=true, problem_version="unconstrained", short_labels=false, network_prefixes=[])
    plots = []
    for (i,k) in enumerate(["optimality","graph_reduction"])
        if i==1
            push!(plots, metric_boxplot(output_dir, k, outliers=outliers, problem_version=problem_version, yticks=true, short_labels=short_labels, network_prefixes=network_prefixes))
        else
            push!(plots, metric_boxplot(output_dir, k, outliers=outliers, problem_version=problem_version, yticks=false, network_prefixes=network_prefixes))
        end
    end
    plot(plots..., link=:y, layout=(1,2), size=(800,500), bottom_margin=50Plots.px, top_margin=50Plots.px)
end
