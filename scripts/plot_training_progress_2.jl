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
epochs = collect(range(1,59,step=5))

#model_checkpoint_dir = "model2_trained/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
#image_dir = "model2_trained_results_seuil.25/imgs/dataset_prc_small_flexE_1_train/model2_l4_lr1.0e-6_h64_tversky.2"
# DEBUG
model_checkpoint_dir = "debug_model_emb0"
image_dir = "debug_model_emb0/imgs_comp"
output_dir = "debug_model_emb0/outputs"

mkpath(image_dir)
mkpath(output_dir)

for (cat, j) in Iterators.product(["flexE", "vlan", "mixed"], 1:5)
    epoch_data = []
    
    ref_instance = "instances/prc/small/$cat/$j"
    # add directory for the reference instance
    current_img_dir = joinpath(image_dir, "prc_small_$(cat)_$j")
    current_output_dir = joinpath(output_dir, "prc_small_$(cat)_$j")
    # if directory exists skip test
    #if isdir(current_img_dir)
    #    println("Skipping test on $ref_instance")
    #    continue
    #end
    mkpath(current_img_dir)
    mkpath(current_output_dir)

    inst=UMFData(ref_instance);
    for e in ProgressBar(epochs)
        # make a directory for output files at epoch e
        epoch_output_dir = joinpath(current_output_dir, "epoch_$e")
        mkpath(epoch_output_dir)
        #println("$cat Epoch $e")
        opts=[]
        speedups=[]
        prspeedups=[]
        grs=[]
           for i in 1:N
            inst1 = UMFSolver.generate_example(inst)
            inst2 = UMFSolver.scale(inst1)
            try
                s0,ss0 = solveUMF(inst2, "CG","highs",joinpath(epoch_output_dir, "default_$i.txt"))
                s1,ss1 = solveUMF(inst2, "CG","highs",joinpath(epoch_output_dir, "output_$i.txt"),"","clssp $model_checkpoint_dir/checkpoint_e$e.bson $K")
            catch e
                println("Error in solve")
            end
        end
    end
end
