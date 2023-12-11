using UMFSolver
using ProgressBars
using Statistics

config = set_config("CG","cplex","./output","linear","kspfilter 4")

network_names = ["AsnetAm","AttMpls", "Chinanet", "giul39", "Iij", "india35", "Ntt", "zib54"]

function solve_dataset(datasetdir, outputdir, config; scale_demands::Float64=1.0)
    println("Solving dataset : ", datasetdir)
    println("\toutputdir = ", outputdir)
    #println("\tconfig = ", config)
    mkpath(outputdir)

    stats = []
    for inst_name in ProgressBar(readdir(datasetdir))
        inst_path = joinpath(datasetdir, inst_name)
        if UMFSolver.is_instance_path(inst_path)
            inst = UMFSolver.scale_bandwidths(UMFData(inst_path), scale_demands)
            _config = UMFSolver.set_output_file(config, joinpath(outputdir, inst_name*".json"))
            try
                s,ss = solveCG(inst, _config)
                UMFSolver.save(ss, joinpath(outputdir, inst_name*".json"))
                push!(stats, ss)
            catch e
                println("Error on $(inst_path)")
                throw(e)
                if isa(e,InterruptException)
                    throw(e)
                end
            end
        end
    end
    return stats
end

for network in network_names
    dataset_path = "datasets/$(network)_0_1_1/test"
    solver_stats = solve_dataset(dataset_path, "./temp_solve", config)
    println(network,  " ", 1000*mean(s.stats["t_create_pricing"] for s in solver_stats))
end
