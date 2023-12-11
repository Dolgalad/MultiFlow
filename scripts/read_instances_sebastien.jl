using UMFSolver
using DataFrames
using CSV

dataset_dir = "datasets"

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
        # solving with dijkstra
        local sol, ss = solveCG(inst, dijconfig)
        push!(cnames, inst.name)
        push!(dijsolvetimes, 1000 * ss.stats["timetot"])
        push!(dijvals, ss.stats["val"])
        push!(dijmssolvetimes, 1000 * ss.stats["time_ms_sol"])
        push!(dijprsolvetimes, 1000 * ss.stats["time_pr_sol"])
        push!(dijmssolveprop, 100 * ss.stats["time_ms_sol"]/ss.stats["timetot"])
        push!(dijprsolveprop, 100 * ss.stats["time_pr_sol"]/ss.stats["timetot"])
        num_rejected = sum(gety(ss.stats["ms"]) .> 0)
        push!(dijrejected, 100 * round(num_rejected / nk(inst), digits=3))

        # solving with larac
        sol, ss = solveCG(inst, laracconfig)

        push!(laracsolvetimes, 1000 * ss.stats["timetot"])
        push!(laracvals, ss.stats["val"])
        push!(laracmssolvetimes, 1000 * ss.stats["time_ms_sol"])
        push!(laracprsolvetimes, 1000 * ss.stats["time_pr_sol"])
        push!(laracmssolveprop, 100 * ss.stats["time_ms_sol"]/ss.stats["timetot"])
        push!(laracprsolveprop, 100 * ss.stats["time_pr_sol"]/ss.stats["timetot"])
        num_rejected = sum(gety(ss.stats["ms"]) .> 0)
        push!(laracrejected, 100 * round(num_rejected / nk(inst), digits=3))

        push!(nnodes, nv(inst))
        push!(nedges, ne(inst))
        push!(ndemands, nk(inst))


        println("$(inst.name) solve time=$(ss.stats["timetot"]) val=$(ss.stats["val"]) ms time=$(round(ss.stats["time_ms_sol"] / ss.stats["timetot"], digits=3)) pr time=$(round(ss.stats["time_pr_sol"] / ss.stats["timetot"], digits=3))")

        # make train dataset
        graph_cat = split(inst.name, "_")[1]
        if graph_cat in graph_categories
            continue
        else
            local train_dataset_path = joinpath(dataset_dir, inst.name, "train")
            if !isdir(train_dataset_path)
                UMFSolver.make_dataset(inst, 1000, train_dataset_path; perturbation_func=UMFSolver.generate_example_2)
            end
            # make test dataset
            local test_dataset_path = joinpath(dataset_dir, inst.name, "test")
            if !isdir(test_dataset_path)
                UMFSolver.make_dataset(inst, 100, test_dataset_path; perturbation_func=UMFSolver.generate_example_2)
            end
            local frod_train_dataset_path = joinpath(dataset_dir, inst.name*"_frod", "train")
            if !isdir(frod_train_dataset_path)
                UMFSolver.make_dataset(inst, 1000, frod_train_dataset_path; perturbation_func=UMFSolver.generate_example_3)
            end
            # make test dataset
            local frod_test_dataset_path = joinpath(dataset_dir, inst.name*"_frod", "test")
            if !isdir(frod_test_dataset_path)
                UMFSolver.make_dataset(inst, 100, frod_test_dataset_path; perturbation_func=UMFSolver.generate_example_3)
            end

            push!(graph_categories, graph_cat)
        end


    catch e
        println("Error solving ", f, e)
    end
end

df = DataFrame(name=cnames, 
               nv=nnodes,
               ne=nedges,
               nk=ndemands,
               dijsolvetime=dijsolvetimes,
               dijmssolvetime=dijmssolvetimes, 
               dijprsolvetime=dijprsolvetimes, 
               dijmssolveprop=dijmssolveprop, 
               dijprsolveprop=dijprsolveprop, 
               dijval=dijvals,
               dijrejected=dijrejected,
               laracsolvetime=  laracsolvetimes,
               laracmssolvetime=laracmssolvetimes, 
               laracprsolvetime=laracprsolvetimes, 
               laracmssolveprop=laracmssolveprop, 
               laracprsolveprop=laracprsolveprop, 
               laracval=laracvals,
               laracrejected=laracrejected,

              )
sort!(df,[:dijsolvetime, :dijmssolveprop, :dijprsolveprop], rev=true )
println(df)
CSV.write("read_instances_sebastien.csv", df)
