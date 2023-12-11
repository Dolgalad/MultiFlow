using UMFSolver
using SNDlib

ENV["CPLEX_STUDIO_BINARIES"] = "/opt/ibm/ILOG/CPLEX_Studio221/cplex/bin/x86-64_linux/"

name = "germany50"
sndlib_network = "/home/aschulz/.cache/sndlib/instances/sndlib-instances-native/$name/$name.txt"
sndlib_solution = "/data/aschulz/Documents2023/sndlib_problems/sndlib-solutions-native/germany50--D-B-S-N-C-A-N-N/sol1--germany50--D-B-S-N-C-A-N-N.txt"
bidirected = true

# load network
net = SNDlib.load_network(sndlib_network)
println(net)

# load solution
sol = SNDlib.load_solution(sndlib_solution)

# convert SNDlib to UMFData object
function SNDNetwork_to_UMFData(network::SNDNetwork, solution::SNDSolution)
    n_nodes = length(network.nodes)
    n_edges = length(network.links)
    n_demands = length(network.demands)
    node_ids = [n.id for n in network.nodes]
    nodemap = Dict(node_ids .=> 1:n_nodes)
    # link data
    src_nodes, dst_nodes, costs, capacities = Int64[], Int64[], Float64[], Float64[]
    for e in 1:n_edges
        link = network.links[e]
        link_conf = get_link_configuration(solution, link.id)
        if length(link_conf.data)>0
            coeff = link_conf.data[2]
        else
            coeff = 0.0
        end
        push!(src_nodes, nodemap[link.src_id])
        push!(dst_nodes, nodemap[link.dst_id])
        push!(costs, link.data[end])
        push!(capacities, link.data[end-1] * coeff)
        if bidirected
            push!(src_nodes, nodemap[link.dst_id])
            push!(dst_nodes, nodemap[link.src_id])
            push!(costs, link.data[end])
            push!(capacities, link.data[end-1] * coeff)
        end

    end
    if bidirected
        n_edges = n_edges * 2
    end
    # demand data
    src_demands, dst_demands, bandwidths = Int64[], Int64[], Float64[]
    for k in 1:n_demands
        demand = network.demands[k]
        push!(src_demands, nodemap[demand.src_id])
        push!(dst_demands, nodemap[demand.dst_id])
        push!(bandwidths, demand.value)
    end

    return UMFData(
                   network.name,
                   src_nodes,
                   dst_nodes,
                   capacities,
                   costs,
                   src_demands,
                   dst_demands,
                   bandwidths,
                   n_demands,
                   n_nodes,
                   n_edges
    )
end

pb = SNDNetwork_to_UMFData(net, sol)
#config = set_config("CG", "cplex", "./output.txt", "linear", "dijkstra")
config = set_config("CG", "cplex", "./output.txt", "linear", "dijkstra")
s,ss = solveCG(pb, config)

println("Total solve time : ", ss.stats["timetot"])
println("Value : ", ss.stats["val"])
println("Times : ms = ", round(100 * ss.stats["time_ms_sol"]/ss.stats["timetot"], digits=3), "%, pr = ", round(100*ss.stats["time_pr_sol"]/ss.stats["timetot"], digits=3), "%")
println(sum(gety(ss.stats["ms"])))
