using UMFSolver
using Graphs

inst = UMFData("instances/prc/large/flexE/5")

g = SimpleGraph(nv(inst))
for i in 1:ne(inst)
    add_edge!(g, inst.srcnodes[i], inst.dstnodes[i])
end

function cost_matrix(inst)
    m = zeros(nv(inst), nv(inst))
    for i in 1:ne(inst)
        m[inst.srcnodes[i], inst.dstnodes[i]] = inst.costs[i]
    end
    return m
end
function path_cost(inst, path)
    m = cost_matrix(inst)
    ans = 0.0
    for i in 2:length(path)
        ans = ans + m[path[i-1], path[i]]
    end
    return ans
end
function check_path(inst, path)
    m = cost_matrix(inst)
    for i in 2:length(path)
        if m[path[i-1], path[i]] == 0
            return false
        end
    end
    return true
end

m = cost_matrix(inst)
for k in 1:nk(inst)
    ds = dijkstra_shortest_paths(g, inst.srcdemands[k], m)
    p = enumerate_paths(ds, inst.dstdemands[k])
    p1 = bidijkstra_shortest_path(g, inst.srcdemands[k], inst.dstdemands[k], m)
    if p != p1
        println(p, ", ", path_cost(inst, p), ", ", check_path(inst, p))
        println(p1, ", ", path_cost(inst, p1), ", ", check_path(inst, p1))
        println(path_cost(inst, p) == path_cost(inst, p1))
        println()
    end
end

