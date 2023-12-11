using Plots
using Statistics
using JLD2
using FileIO
using UMFSolver
using Graphs

function node_capacity(n::Int64, inst::UMFData; dir=:out, g=UMFSolver.get_graph(inst))
    am = UMFSolver.get_arc_matrix(inst)
    if dir==:out
        return sum(inst.capacities[am[n,v]] for v in outneighbors(g, n))
    else
        return sum(inst.capacities[am[v,n]] for v in inneighbors(g, n))
    end
end

function node_bandwidth(n::Int64, inst::UMFData; dir=:out)
    if dir==:out
        return sum(inst.bandwidths[findall(==(n), inst.srcdemands)])
    else
        return sum(inst.bandwidths[findall(==(n), inst.dstdemands)])
    end
end

function has_demand(inst::UMFData, u::Int64, v::Int64)
    return v in inst.dstdemands[findall(==(u), inst.srcdemands)]
end

function check_ksp_coverage(inst, K)
    cm = UMFSolver.get_cost_matrix(inst)
    g = UMFSolver.get_graph(inst)
    for u in 1:nv(g)
        for v in 1:nv(g)
            if u!=v
                ys = yen_k_shortest_paths(g, u, v, cm, K)
                outn = Set(outneighbors(g, u))
                inn = Set(inneighbors(g,v))
                # list of out neighbors of u in K shortest paths
                out_nodes = Set([p[2] for p in ys.paths])
                # list of in neighbors of v in K shortest paths
                in_nodes = Set([p[end-1] for p in ys.paths])
                if any( !(n in out_nodes) for n in outn) && has_demand(inst,u,v)
                    println("$u -> $v : kSP (k=$K) missing out neighbor from $u")
                    println("\toutn      = ", outn)
                    println("\tout_nodes = ", out_nodes)
                    println("\t instance has demand $u -> $v : ", has_demand(inst,u,v))

                elseif any( !(n in in_nodes) for n in inn) && has_demand(inst,u,v)
                    println("$u -> $v : kSP (k=$K) missing in neighbor from $u")
                    println("\tinn      = ", inn)
                    println("\tin_nodes = ", in_nodes)
                    println("\t instance has demand $u -> $v : ", has_demand(inst,u,v))


                end
            end
        end
    end
end


function plot_optimality(filename, scaling_factors; ignore=[])
    data = FileIO.load(filename)
    data = Dict(k=>data[k] for k in keys(data) if !(k in ignore))
    ks = collect(keys(data))

    p = plot(scaling_factors, [[mean(x)+1e-8 for x in data[k]] for k in keys(data)], xscale=:log, xlabel="Bandwidth scale factor", ylabel="Graph reduction", right_margin=10Plots.px, left_margin=10Plots.px, bottom_margin=10Plots.px, labels=reshape(ks, (1,size(ks,1))), yscale=:log)
    ymin=1e-8
    ymax=10
    for (i,k) in enumerate(ks)
        ss1 = [mean(x) .- std(x) for x in data[k]]
        ss1[findall(<(1e-8),ss1)].=1e-8
        ss2 = [mean(x) .+ std(x) for x in data[k]]
        if 1.1 * maximum(ss2) > ymax
            ymax = 1.1 * maximum(ss2)
        end

        plot!(scaling_factors, ss1, fillrange = ss2, fillalpha = 0.2, c = i, label=false, alpha=0, ylim=[ymin,ymax])
    end
    return p
end
