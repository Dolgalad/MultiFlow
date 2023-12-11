using JLD2

@load "testing_outputs/unconstrained_robustness_data.jld2" 

idx = [1, 2, 4, 6, 8, 10, 11, 12]

println("alpha ", join([split(m,"_")[1] for m in model_names[idx]], " "))
#println("alpha ", join([m for m in model_names if !occursin("bs50", m)], " "))

for i in 1:length(alphas)
    println(round(alphas[i], digits=3), " ", join([round(100*optimality[j][i],digits=3) for j in idx], " "))
end
