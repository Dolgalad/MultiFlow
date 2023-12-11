using UMFSolver
using DataFrames
using CSV
using Graphs
using Distributions

dataset_dir = "datasets"

target_instance_dir = "instances_sndlib/delay_corrected"

function load_instance(filename::String)
    name = splitext(basename(filename))[1]
    nnodes,nedges,ndemands=0,0,0
    srcnodes, dstnodes= Int64[], Int64[]
    costs, capacities, delays = Float64[], Float64[], Float64[]
    srcdemands, dstdemands = Int64[], Int64[]
    bandwidths, delaydemands = Float64[], Float64[]
    edge_capacities = Dict()
    linenum = 0
    f=open(filename, "r")
    while !eof(f)
        l = readline(f)
        if linenum==0
            v = map(e->parse(Int64, e), split(strip(l), " "))
            nnodes,nedges,ndemands = v[1], v[2], v[3]

        end
        if linenum>0 && linenum<=nedges
            edge_data = split(strip(l), " ")
            e = (parse(Int64, edge_data[1]), parse(Int64, edge_data[2]))
            if e in keys(edge_capacities)
                edge_index = edge_capacities[e][1]
                capacities[edge_index] += parse(Float64, edge_data[3])
            else
                push!(srcnodes, parse(Int64, edge_data[1]))
                push!(dstnodes, parse(Int64, edge_data[2]))
                push!(capacities, parse(Float64, edge_data[3]))
                push!(costs, parse(Float64, edge_data[4]))
                push!(delays, parse(Float64, edge_data[6]))
                edge_capacities[e] = (size(srcnodes,1), parse(Float64, edge_data[3]))
            end

        elseif linenum>nedges
            edge_data = split(strip(l), " ")
            push!(srcdemands, parse(Int64, edge_data[1]))
            push!(dstdemands, parse(Int64, edge_data[2]))
            push!(bandwidths, parse(Float64, edge_data[3]))
            push!(delaydemands, parse(Float64, edge_data[4]))
        end
        linenum += 1
    end
    close(f)

    if minimum(srcnodes)==0 || minimum(dstnodes)==0
        srcnodes .+= 1
        dstnodes .+= 1
        srcdemands .+= 1
        dstdemands .+= 1
    end

    nedges = size(srcnodes, 1)

    return UMFData(name,
               srcnodes,
               dstnodes,
               capacities,
               costs,
               delays,
               srcdemands,
               dstdemands,
               bandwidths,
               delaydemands,
               ndemands,
               nnodes,
               nedges
              )
end

instance_dir = "/data/aschulz/Documents2023/sndlib_problems/instances/internship-2022-jason-instances888/instances888"

dijconfig = set_config("CG", "cplex", "./output.txt", "linear", "dijkstra")
laracconfig = set_config("CG", "cplex", "./output.txt", "linear", "larac")

cnames, dijsolvetimes, dijvals, dijmssolvetimes, dijprsolvetimes = [], [], [], [], []
dijmssolveprop, dijprsolveprop, dijrejected = [],[],[]
laracsolvetimes, laracvals, laracmssolvetimes, laracprsolvetimes = [], [], [], [], []
laracmssolveprop, laracprsolveprop, laracrejected = [],[],[]
nnodes, nedges, ndemands = [], [], []

graph_categories = []
for f in readdir(instance_dir, join=true)
    if !endswith(f, ".txt")
        continue
    end
    try
        local inst = load_instance(f)
        println("Name : $(inst.name) : nv=$(nv(inst)) ne=$(ne(inst)) nk=$(nk(inst))")
        s0,ss0 = solveCG(inst, laracconfig)
        nrejected= sum(gety(ss0.stats["ms"]) .> 0)
        println("\trejected: ", nrejected)

        # build the graph
        g = UMFSolver.get_graph(inst)
        latency_matrix = UMFSolver.get_latency_matrix(inst)
        cost_matrix = UMFSolver.get_cost_matrix(inst)

        # new delay bounds
        delay_bounds = Float64[]
        # check demands
        for k in 1:nk(inst)
            ub = inst.demand_latencies[k]
            # minimum delay path
            ds = dijkstra_shortest_paths(g, inst.srcdemands[k], latency_matrix)
            pd = enumerate_paths(ds, inst.dstdemands[k])
            pd_delay = UMFSolver.path_cost(pd, latency_matrix)
            # minimum cost path
            ds = dijkstra_shortest_paths(g, inst.srcdemands[k], cost_matrix)
            pc = enumerate_paths(ds, inst.dstdemands[k])
            pc_delay = UMFSolver.path_cost(pc, latency_matrix)

            if pd_delay > ub
                #println("\tDemand $k is not feasible ", [ub, pd_delay, pc_delay])
                if pd_delay < pc_delay
                    udist = Uniform(pd_delay, pc_delay)
                else
                    udist = Uniform(pd_delay, 2*pd_delay)
                end
                nub = rand(udist)
                #println("\tNew upper bound : ", nub)
                push!(delay_bounds, nub)
            else
                push!(delay_bounds, inst.demand_latencies[k])
            end

        end

        ninst = UMFData(inst.name,
                        inst.srcnodes,
                        inst.dstnodes,
                        inst.capacities,
                        inst.costs,
                        inst.latencies,
                        inst.srcdemands,
                        inst.dstdemands,
                        inst.bandwidths,
                        delay_bounds,
                        nk(inst),
                        nv(inst),
                        ne(inst)
              )
        s1,ss1 = solveCG(ninst, laracconfig)
        nrejected = sum(gety(ss1.stats["ms"]) .> 0)
        println("\tmodified rejected: ", nrejected)

        save(ninst, joinpath(target_instance_dir, ninst.name))

        # make train dataset
        graph_cat = split(ninst.name, "_")[1]
        if graph_cat in graph_categories
            continue
        else
            local train_dataset_path = joinpath(dataset_dir, ninst.name*"_delay", "train")
            if !isdir(train_dataset_path)
                UMFSolver.make_dataset(ninst, 1000, train_dataset_path; perturbation_func=UMFSolver.generate_example_2, prtype="larac")
            end
            # make test dataset
            local test_dataset_path = joinpath(dataset_dir, ninst.name*"_delay", "test")
            if !isdir(test_dataset_path)
                UMFSolver.make_dataset(ninst, 100, test_dataset_path; perturbation_func=UMFSolver.generate_example_2, prtype="larac")
            end
            local frod_train_dataset_path = joinpath(dataset_dir, ninst.name*"_delay_frod", "train")
            if !isdir(frod_train_dataset_path)
                UMFSolver.make_dataset(ninst, 1000, frod_train_dataset_path; perturbation_func=UMFSolver.generate_example_3, prtype="larac")
            end
            # make test dataset
            local frod_test_dataset_path = joinpath(dataset_dir, ninst.name*"_delay_frod", "test")
            if !isdir(frod_test_dataset_path)
                UMFSolver.make_dataset(ninst, 100, frod_test_dataset_path; perturbation_func=UMFSolver.generate_example_3, prtype="larac")
            end

            push!(graph_categories, graph_cat)
        end


    catch e
        println("Error solving ", f, e)
        #throw(e)
    end
end

#df = DataFrame(name=cnames, 
#               nv=nnodes,
#               ne=nedges,
#               nk=ndemands,
#               dijsolvetime=dijsolvetimes,
#               dijmssolvetime=dijmssolvetimes, 
#               dijprsolvetime=dijprsolvetimes, 
#               dijmssolveprop=dijmssolveprop, 
#               dijprsolveprop=dijprsolveprop, 
#               dijval=dijvals,
#               dijrejected=dijrejected,
#               laracsolvetime=  laracsolvetimes,
#               laracmssolvetime=laracmssolvetimes, 
#               laracprsolvetime=laracprsolvetimes, 
#               laracmssolveprop=laracmssolveprop, 
#               laracprsolveprop=laracprsolveprop, 
#               laracval=laracvals,
#               laracrejected=laracrejected,
#
#              )
#sort!(df,[:dijsolvetime, :dijmssolveprop, :dijprsolveprop], rev=true )
#println(df)
#CSV.write("read_instances_sebastien.csv", df)
