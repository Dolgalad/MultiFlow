using Plots 
using StatsPlots
using JSON

function load_solve_data(data_directory)
    instance_data = Dict()
    for data_file in readdir(data_directory, join=true)
        # instance name
        instance_name = parse(Int64, replace(basename(data_file), ".json"=>""))
        solve_data = JSON.parsefile(data_file)
        instance_data[instance_name] = solve_data
    end
    return instance_data

end

function load_epoch_solve_data(data_directory)
    # first level of directories are epochs
    epoch_data = Dict()
    for epoch_dir in readdir(data_directory, join=true)
        # second level of files are solve data files
        epoch = parse(Int64, basename(epoch_dir))
        epoch_data[epoch] = load_solve_data(epoch_dir)
    end
    return epoch_data
end

cat = "mixed"

graph_categories = ["flexE", "vlan", "mixed"]
plots = []

with_outliers = true

model_type = "model6b"
for cat in graph_categories
    default_solve_data = load_solve_data("dataset_prc_small_$(cat)_1_train/$(model_type)_l4_lr1.0e-6_h64_bs20_tversky0.1/default/")
    epoch_K0_solve_data = load_epoch_solve_data("dataset_prc_small_$(cat)_1_train/$(model_type)_l4_lr1.0e-6_h64_bs20_tversky0.1/K0/")
    epoch_K1_solve_data = load_epoch_solve_data("dataset_prc_small_$(cat)_1_train/$(model_type)_l4_lr1.0e-6_h64_bs20_tversky0.1/K1/")
    
    epochs = sort(Int64.(keys(epoch_K0_solve_data)))
    println("number of epochs : ", size(epochs))
    idx = 1:length(epochs)

    
    # graph reduction
    graph_reduction_epoch_data = [[d["graph_reduction"] for (k,d) in epoch_K0_solve_data[e]] for e in epochs]
    idx = [i for (i,r) in enumerate(graph_reduction_epoch_data) if size(r,1)!=0]
    if cat=="mixed"
        idx2 = 1:8:length(idx)
    else
        idx2 = 1:5:length(idx)
    end

    #graph_reduction_epoch_data = [r for r in graph_reduction_epoch_data if size(r,1)!=0]
    #epochs = epochs[1:size(graph_reduction_epoch_data,1)]

    gr_plot = boxplot(graph_reduction_epoch_data[idx][idx2], xlabel=cat=="mixed" ? "Epoch" : "", color="lightblue", legend=:none, xticks=(1:size(epochs[idx][idx2],1), epochs[idx][idx2]), left_margin=15Plots.mm, outliers=with_outliers, xrotation=55);
    ylabel!("Graph reduction")
    push!(plots, gr_plot)
    
    # speedup
    speedup_epoch_data = [[default_solve_data[k]["timetot"]/d["timetot"] for (k,d) in epoch_K0_solve_data[e]] for e in epochs]
    #speedup_epoch_data = [r for r in speedup_epoch_data if size(r,1)!=0]

    speedup_plot = boxplot(speedup_epoch_data[idx][idx2], xlabel=cat=="mixed" ? "Epoch" : "", ylabel="Speedup", color="lightblue", legend=:none, xticks=(1:size(epochs[idx][idx2],1), epochs[idx][idx2]), title=cat, left_margin=15Plots.mm, outliers=with_outliers, xrotation=55, top_margin=0Plots.mm);
    push!(plots, speedup_plot)
   
    # LoI
    loi_epoch_data = [[(d["val"] - default_solve_data[k]["val"])/default_solve_data[k]["val"] for (k,d) in epoch_K0_solve_data[e]] for e in epochs]
    #loi_epoch_data = [r for r in loi_epoch_data if size(r,1)!=0]
    loi_epoch_data = [r[r .> 0] for r in loi_epoch_data]
    println([size(r) for r in loi_epoch_data])

    loi_plot = boxplot(loi_epoch_data[idx][idx2], xlabel=cat=="mixed" ? "Epoch" : "", ylabel="LoI", color="lightblue", legend=:none, xticks=(1:size(epochs[idx][idx2],1), epochs[idx][idx2]), outliers=with_outliers, left_margin=15Plots.mm,  gridalpha=0.3 , xrotation=55, top_margin=0Plots.mm);
    push!(plots, loi_plot)
    #yaxis=(:log10, [0.0000001, :auto]),
end 
plot(plots..., size=(1000, 1000), layout=(3, 3))
savefig(with_outliers ? "$(model_type)_metrics_plot.png" : "$(model_type)_metrics_plot_no_outliers.png")


