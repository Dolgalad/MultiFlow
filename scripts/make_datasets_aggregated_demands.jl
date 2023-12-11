"""
Use this script to generate datasets where all identical demands are aggregated into one. 

The selection of bandwidth values has also been changed: rather than taking bandwidth values from 
a trucated normal distribution we select them randomly from the set of original bandwidth values.
"""

using UMFSolver


base_instance_dir = "./instances/"

dataset_dir = "/data1/schulz/datasets"
# DEBUG
dataset_dir = "./datasets"

if !isdir(dataset_dir)
    mkpath(dataset_dir)
end

n_train, n_test = 1000, 100

exclude = ["N1000","N1200","N1400","N1600","N1800","N2000","N800", "test", "middle", "large", "2","3","4","5"]

instances = []
instance_paths = []
# get list of base instances and sort them by increasing size (number of arcs)
for (root, dirs, files) in walkdir(base_instance_dir)
    for dir in dirs
        full_dir_path = joinpath(root, dir)
        if is_instance_path(full_dir_path) && !any(contains(dir, excl) for excl in exclude)
            push!(instances, UMFData(full_dir_path))
            push!(instance_paths, full_dir_path)
        end
    end
end

# sort by increasing number of demands
idx = sortperm(instances, by=x -> nk(x))

for (inst, full_dir_path) in zip(instances[idx], instance_paths[idx])
    # dataset name
    dataset_name = "dataset_"*replace(replace(full_dir_path, base_instance_dir => ""), "/" => "_")*"_agg"
    println("dataset name : ", dataset_name)
    println("dataset path : ", joinpath(dataset_dir, dataset_name))
    # train dataset
    dataset_path = joinpath(dataset_dir, dataset_name*"_train")
    UMFSolver.make_dataset_with_aggregated_demands(inst, n_train, dataset_path)
    # test dataset
    dataset_path = joinpath(dataset_dir, dataset_name*"_test")

    UMFSolver.make_dataset_with_aggregated_demands(inst, n_test, dataset_path)
    println()
end
