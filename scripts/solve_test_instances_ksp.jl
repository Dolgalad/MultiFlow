"""
Solve test instances with baseline and ML solvers
"""

using UMFSolver
using HDF5
using BenchmarkTools

include("make_shortest_path_table.jl")

output_dir = "testing_outputs/test_outputs_temp_scaled10"
#if isdir(output_dir)
#    rm(output_dir, recursive=true, force=true)
#end
mkpath(output_dir)
checkpoint_dir = "/mnt/DataRaid/Documents/Documents_2023/julia"

dataset_dir = "/mnt/DataRaid/Documents/Documents_2023/julia/datasets"

sptable_dir = "/mnt/DataRaid/Documents/Documents_2023/julia/sptables"

network_names = ["AsnetAm_0_1_1", "Ntt_0_1_1", "giul39_0_1_1", "AttMpls_0_1_1", "Oxford_0_1_1",
                 "Iij_0_1_1", "india35_0_1_1", "Chinanet_0_1_1", "zib54_0_1_1"]

function solve_dataset(datasetdir, outputdir, config; scale_demands::Float64=1.0)
    println("Solving dataset : ", datasetdir)
    println("\toutputdir = ", outputdir)
    #println("\tconfig = ", config)
    mkpath(outputdir)
    for inst_name in readdir(datasetdir)
        inst_path = joinpath(datasetdir, inst_name)
        if UMFSolver.is_instance_path(inst_path)
            inst = UMFSolver.scale_bandwidths(UMFData(inst_path), scale_demands)
            _config = UMFSolver.set_output_file(config, joinpath(outputdir, inst_name*".json"))
            try
                s,ss = solveCG(inst, _config)
                UMFSolver.save(ss, joinpath(outputdir, inst_name*".json"))
            catch e
                println("Error on $(inst_path)")
                throw(e)
                if isa(e,InterruptException)
                    throw(e)
                end
            end
        end
    end
end

function output_already_done(path)
    if !isdir(path)
        return false
    end
    output_files = readdir(path)
    return length(output_files)==100
end
for network in network_names
    network_output_dir = joinpath(output_dir, network)
    mkpath(network_output_dir)
    println("Network $network dir: $(network_output_dir)")
    for dataset_name in readdir(dataset_dir)
        if !startswith(dataset_name, network)
            continue
        end
        dataset_path = joinpath(dataset_dir, dataset_name, "test")
        println("\tDataset $(dataset_path)")
        # load first instance in dataset for shortest path table creation
        inst = UMFData(joinpath(dataset_path, "1"))

	# solve with dijkstra pricing
        dijconfig = set_config("CG","cplex","","linear","dijkstra")
	dij_output_dir = joinpath(network_output_dir, dataset_name,  "unconstrained", "dijkstra")
        # solve unconstrained
	if !output_already_done(dij_output_dir)
	    println("SSSSSSSSSSSSSS")
            solve_dataset(dataset_path, dij_output_dir, dijconfig, scale_demands=10.0)
	end

	# solve with larac pricing
        laracconfig = set_config("CG","cplex","","linear","larac")
	larac_output_dir = joinpath(network_output_dir, dataset_name,  "constrained", "larac")
        # solve constrained
	if !output_already_done(larac_output_dir)
            solve_dataset(dataset_path, larac_output_dir, laracconfig, scale_demands=10.0)
	end


	# solving with kspfilter
        for K in [1,2,4,8]
	    # solve unconstrained problem
            ksp_output_dir = joinpath(network_output_dir, dataset_name,  "unconstrained", "$(K)sp_sb10")
	    if !output_already_done(ksp_output_dir)
                # make shortest path table
                println("\tMaking shortest path table $K")
                @time sptable_c = create_shortest_path_table(inst, K=K)
                # save the table
                h5open("sptable.sptable", "w") do file
                    write(file, "cost", sptable_c)
                end

                kspconfig = set_config("CG","cplex","","linear","kspfilter sptable.sptable")
                # solve unconstrained
                solve_dataset(dataset_path, ksp_output_dir, kspconfig, scale_demands=10.0)
	    end

	    # solve constrained problem
            ksp_output_dir = joinpath(network_output_dir, dataset_name,  "constrained", "$(K)sp_sb10")
	    if !output_already_done(ksp_output_dir)
                # make shortest path table
                println("\tMaking shortest path table $K")
                @time sptable_c = create_shortest_path_table(inst, K=K)
                @time sptable_d = create_shortest_path_table(inst, K=K, dstmx=inst.latencies)

                # save the table
                h5open("sptable.sptable", "w") do file
                    write(file, "cost", sptable_c)
		    write(file, "delay", sptable_d)
                end

                kspconfig = set_config("CG","cplex","","linear","kspfilter sptable.sptable")
                # solve constrained
                solve_dataset(dataset_path, ksp_output_dir, kspconfig, scale_demands=10.0)
	    end

        end

	# solving with trained models
        @time sptable_c = create_shortest_path_table(inst, K=1)
        @time sptable_d = create_shortest_path_table(inst, K=1, dstmx=inst.latencies)

        # save the table
        h5open("sptable.sptable", "w") do file
            write(file, "cost", sptable_c)
            write(file, "delay", sptable_d)

        end

        model_checkpoint_dir = joinpath(checkpoint_dir, dataset_name)
        if isdir(model_checkpoint_dir)
            for model_dir in readdir(model_checkpoint_dir)
                checkpoint_path = joinpath(model_checkpoint_dir, model_dir, "best_checkpoint.bson")
                if isfile(checkpoint_path)
                    println("\t\tModel : $(model_dir)")
                    # copy checkpoint to current directory
                    cp(checkpoint_path, "best_checkpoint.bson", force=true)

                    _dijmlconfig = set_config("CG","cplex","","linear","clssp best_checkpoint.bson 0 sptable.sptable")
                    _laracmlconfig = set_config("CG","cplex","","linear","clslarac best_checkpoint.bson 0 sptable.sptable")

		    clssp_output_dir = joinpath(network_output_dir, dataset_name, "unconstrained", "clssp_"*model_dir)
		    clslarac_output_dir = joinpath(network_output_dir, dataset_name, "constrained", "clslarac_"*model_dir)

                    # solve unconstrained
		    if !output_already_done(clssp_output_dir)
                        solve_dataset(dataset_path, clssp_output_dir, _dijmlconfig, scale_demands=10.0)
	            end
                    # solve constrained
		    if !output_already_done(clslarac_output_dir)
                        solve_dataset(dataset_path, clslarac_output_dir, _laracmlconfig, scale_demands=10.0)
		    end
                end
            end
        end
    end
end
