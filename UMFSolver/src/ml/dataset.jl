using ProgressBars

"""
    make_dataset(inst::UMFData, n::Int64, path::String)

Create a dataset by applying perturbations to a reference instance. Saves resulting instance and solution files to path directory.
"""
function make_dataset(inst::UMFData, n::Int64, path::String; perturbation_func=UMFSolver.generate_example, show_progress=true, prtype="dijkstra")
    c = 0
    mkpath(path)
    # directory for storing the solve information
    solve_output_dir = joinpath(path, "solve_outputs")
    mkpath(solve_output_dir)
    if show_progress
        progress = ProgressBar(1:n)
    else
        progress = 1:n
    end
    for c in progress
        success = false
        failure_count = 0
        max_failure = 10
        while !success && (failure_count<max_failure)
            try
                # generate an instance
                ninst = perturbation_func(inst)
                # try to solve this instance
                outf= joinpath(solve_output_dir, "output_"*string(c)*".txt")
                #sol,s0 = solveUMF(ninst, "CG", "cplex",outf)
                config = set_config("CG", "cplex", outf, "linear", prtype)
                sol,s0 = solveCG(ninst, config)
                save(s0, outf)
               
                # save the instance and the solution
                instance_dir = joinpath(path, string(c))
                mkpath(instance_dir)

                # add all columns to the solution
                ms = s0.stats["ms"]
                for k in 1:nk(ninst)
                    columns = columns_k(ms, k)
                    for i in eachindex(columns)
                        sol.x[columns[i], k] .= 1
                    end
                end

                save(ninst, instance_dir, verbose=false)
                save(sol, joinpath(instance_dir, "sol.jld"), verbose=false)
                success = true
            catch e
                println("Error ", e)
                failure_count += 1
            end
        end
    end
end


"""
    make_dataset_with_aggregated_demands(inst::UMFData, n::Int64, path::String)

Create a dataset by applying perturbations to a reference instance. New bandwidth values are taken from the set of original bandwidth values and all identical demands are aggregated into one.
"""
function make_dataset_with_aggregated_demands(inst::UMFData, n::Int64, path::String; perturbation_func=UMFSolver.generate_example_with_aggregated_demands, show_progress=true)
    c = 0
    mkpath(path)
    # directory for storing the solve information
    solve_output_dir = joinpath(path, "solve_outputs")
    mkpath(solve_output_dir)
    if show_progress
        progress = ProgressBar(1:n)
    else
        progress = 1:n
    end
    for c in progress
        # generate an instance
        ninst = perturbation_func(inst)
        # try to solve this instance
        outf= joinpath(solve_output_dir, "output_"*string(c)*".txt")
	#println("solve output dir : ", outf)
        sol,s0 = solveUMF(ninst, "CG", "highs",outf)
        
        # save the instance and the solution
        instance_dir = joinpath(path, string(c))
        mkpath(instance_dir)

        save(ninst, instance_dir, verbose=false)
        save(sol, joinpath(instance_dir, "sol.jld"), verbose=false)
    end
end


"""Make graphs batchable
"""
function make_batchable(gl)
    max_k = maximum(g.K for g in gl)
    max_ne = maximum(size(g.targets,1) for g in gl)
    println("max k : ", max_k)
    println("max ne : ", max_ne)
    #nedges = size(gl[1].targets, 1)
    new_graphs = []
    for g in gl
        nedges = size(g.targets,1)
        new_targets = zeros(Bool, (max_ne, max_k))
        new_targets[1:nedges, 1:size(g.targets, 2)] .= g.targets
        target_mask = zeros(Bool, (max_ne, max_k))
        target_mask[1:nedges,1:size(g.targets,2)] .= 1
        ng = GNNGraph(g, gdata=(;K=g.K, E=g.E, targets=new_targets, target_mask=target_mask))
        push!(new_graphs, ng)
    end
    new_graphs
end
"""
    load_dataset(dataset_dir::String)

Load dataset from directory.
"""
function load_dataset(dataset_dir::String; scale_instances=true, batchable=true, edge_dir=:double)
    graphs = []
    bar = ProgressBar(readdir(dataset_dir, join=true))
    set_description(bar, "Loading from $(dataset_dir)")
    for f in bar
        # must be a directory containing a link.csv and service.csv 
        if !is_instance_path(f)
            continue
        end

        inst = UMFData(f, edge_dir=edge_dir)
        # scale cost and capacities
        if scale_instances
            inst = scale(inst)
        end
        # check if a solution file exists
        solution_file_path = joinpath(f, "sol.jld")
        if isfile(solution_file_path)
            sol = load_solution(solution_file_path)
        else
            ssol, stats = solveUMF(f*"/", "CG", "highs", "./output.txt")
            sol = ssol.x
        end
        y = (sol .> 0)
        g = UMFSolver.to_gnngraph(inst, y, feature_type=Float32)
        
        push!(graphs, g)
    end
    if batchable
        graphs = make_batchable(graphs)
    end
    return graphs
end

"""
Load single instance
"""
function load_instance(inst_path; scale_instance=true)
    inst = UMFData(inst_path)
    # scale cost and capacities
    if scale_instance
        inst = UMFSolver.scale(inst)
    end
    # check if a solution file exists
    solution_file_path = joinpath(inst_path, "sol.jld")
    if isfile(solution_file_path)
        sol = UMFSolver.load_solution(solution_file_path)
    else
        ssol, stats = solveUMF(inst_path, "CG", "highs", "./output.txt")
        sol = ssol.x
    end
    y = (sol .> 0)
    return g = UMFSolver.to_gnngraph(inst, y, feature_type=Float32)
end


