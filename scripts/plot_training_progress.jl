using UMFSolver
using Plots, StatsPlots
using ProgressBars
using LaTeXStrings

#inst = UMFData("instances/prc/small/flexE/1")

N=10
j=5
K=0
cat = "flexE"

bw = 3
epochs = collect(range(1,230,step=10))

#model_checkpoint_dir = "model2_trained/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
#image_dir = "model2_trained_results_seuil.25/imgs/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
# DEBUG
model_checkpoint_dir = "debug_model4"
image_dir = "debug_model4/imgs"

mkpath(image_dir)

for (cat, j) in Iterators.product(["flexE", "vlan", "mixed"], 1:5)
    epoch_data = []
    
    ref_instance = "instances/prc/small/$cat/$j"
    # add directory for the reference instance
    current_img_dir = joinpath(image_dir, "prc_small_$(cat)_$j")
    # if directory exists skip test
    #if isdir(current_img_dir)
    #    println("Skipping test on $ref_instance")
    #    continue
    #end
    mkpath(current_img_dir)

    inst=UMFData(ref_instance);
    for e in ProgressBar(epochs)
        #println("$cat Epoch $e")
        opts=[]
        speedups=[]
        prspeedups=[]
        grs=[]
        for i in 1:N
            inst1 = UMFSolver.generate_example(inst)
            inst2 = UMFSolver.scale(inst1)
            try
                s0,ss0 = solveUMF(inst2, "CG","highs","./output0.txt")
                s1,ss1 = solveUMF(inst2, "CG","highs","./output1.txt","","clssp $model_checkpoint_dir/checkpoint_e$e.bson $K")
                println("clssp $model_checkpoint_dir/checkpoint_e$e.bson $K")
                println("ref instance : ", ref_instance)
                speedup = ss0.stats["timetot"] / ss1.stats["timetot"]
                opt = (ss1.stats["val"] - ss0.stats["val"]) / ss0.stats["val"]
                pr_speedup = ss0.stats["time_pr_sol"] / ss1.stats["time_pr_sol"]
                gr = ss1.stats["graph_reduction"]
                push!(opts,opt)
                push!(speedups, speedup)
                push!(prspeedups, pr_speedup)
                push!(grs, gr)
            catch e
                println("Error in solve")
                #push!(opts,-1)
                #push!(speedups, -1)
                #push!(prspeedups, -1)
                #push!(grs, -1)

            end
            #println("speedup: ", speedup, ", opt: ", opt, ", pr_speedup: ", pr_speedup, ", gr: ", gr)
        end
        if length(opts)==0
            push!(epoch_data, (-1, -1, -1, -1))

        else
            push!(epoch_data, (opts, speedups, prspeedups, grs))
        end
    end
    
    #optimality
    a=[e[1] for e in epoch_data]
    println(a)
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"|c_{\mbox{CG}} - c| / c_{\mbox{CG}}", bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_optimality.png"))
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"|c_{\mbox{CG}} - c| / c_{\mbox{CG}}", outliers=false, bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_optimality_nooutliers.png"))

    
    # speedup
    a=[e[2] for e in epoch_data]
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"t_{\mbox{CG}} / t", bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_speedup.png"))
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"t_{\mbox{CG}} / t", outliers=false, bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_speedup_nooutliers.png"))

    
    # pricing speedup
    a=[e[3] for e in epoch_data]
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"t_{\mbox{pr}_{\mbox{CG}}} / t_{\mbox{pr}}", bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_pr_speedup.png"))
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"t_{\mbox{pr}_{\mbox{CG}}} / t_{\mbox{pr}}",outliers=false, bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_pr_speedup_nooutliers.png"))

    
    # graph reduction
    a=[e[4] for e in epoch_data]
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"(\sum\limits_k |A_k|) / k|A|", bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_graph_reduction.png"))
    boxplot(reshape(epochs, 1, size(epochs,1)), a, legend=false, xlabel="epoch", ylabel=L"(\sum\limits_k |A_k|) / k|A|",outliers=false, bar_width=bw)
    savefig(joinpath(current_img_dir, "epoch_graph_reduction_nooutliers.png"))


    epoch_data = nothing

    @sync GC.gc()
end
