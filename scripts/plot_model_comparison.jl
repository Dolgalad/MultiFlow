using UMFSolver
using Plots, StatsPlots
using ProgressBars
using LaTeXStrings

#inst = UMFData("instances/prc/small/flexE/1")

K=0
N=10

epochs = collect(range(1,59,step=5))

#model_checkpoint_dir = "model2_trained/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
#image_dir = "model2_trained_results_seuil.25/imgs/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
# DEBUG
model_checkpoint_1 = "debug_model/checkpoint_e80.bson"
model_checkpoint_2 = "debug_model_emb0/checkpoint_e50.bson"

image_dir = "model_comparison/imgs"

mkpath(image_dir)

for (cat, j) in Iterators.product(["flexE", "vlan", "mixed"], 1:5)
    epoch_data = []
    
    ref_instance = "instances/prc/small/$cat/$j"
    println("instance : ", ref_instance)

    inst=UMFData(ref_instance);
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
            s1,ss1 = solveUMF(inst2, "CG","highs","./output1.txt","","clssp $model_checkpoint_1 $K")
            s2,ss2 = solveUMF(inst2, "CG","highs","./output1.txt","","clssp $model_checkpoint_2 $K")

            speedup = ss2.stats["timetot"] / ss1.stats["timetot"]
            opt1 = (ss1.stats["val"] - ss0.stats["val"]) / ss0.stats["val"]
            opt2 = (ss2.stats["val"] - ss0.stats["val"]) / ss0.stats["val"]
            opt = opt2 / opt1

            pr_speedup = ss2.stats["time_pr_sol"] / ss1.stats["time_pr_sol"]
            gr = ss2.stats["graph_reduction"] / ss1.stats["graph_reduction"]
            push!(opts,opt)
            push!(speedups, speedup)
            push!(prspeedups, pr_speedup)
            push!(grs, gr)
            println([1,2,3,4])
            #println([speedup, opt, pr_speedup, gr])
        catch e
            println("Error in solve")
        #    #println(e)
        end
    end
        #println("speedup: ", speedup, ", opt: ", opt, ", pr_speedup: ", pr_speedup, ", gr: ", gr)
    
end
