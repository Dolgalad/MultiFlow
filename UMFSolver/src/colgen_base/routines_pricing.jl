# Functions for the pricing problem in the column generation.

# Columns (corresponding to paths) are considered as the lists of arcs in the path.


"""
    set_demand!(pr::UMFShortestPathPricingData, k::Int64)

Set the pricing problem `pr` for demand `k`, by selecting the origin and destination of such demand.

See also [`UMFShortestPathPricingData`](@ref).
"""
function set_demand!(pr::UMFShortestPathPricingData, k::Int64)
    set_k!(pr, k)
    set_src!(pr, demandorigin(data(pr), k))
    set_dst!(pr, demanddest(data(pr), k))
    return k
end

"""
    chgcost!(pr::UMFShortestPathPricingData, newcost::Vector{<:Real})

Update the costs in the graph of the pricing problem `pr` with `newcost`.

See also [`UMFShortestPathPricingData`](@ref).

"""
function chgcost!(pr::UMFShortestPathPricingData, newcost::Vector{<:Real})
    dat = data(pr)
    for a = 1:size(newcost, 1)
        if pr.filter.masks[pr.cur_k][a]
            add_edge!(graph(pr), arcsource(dat, a), arcdest(dat, a), newcost[a])
        end
    end
    return
end

"""
    getcol(pr::UMFShortestPathPricingData, path::Vector{Int64})

Obtain the column (list of arcs) given the list of nodes `path`, output of shortest-path algorithm, for a pricing problem `pr`.

See also [`UMFShortestPathPricingData`](@ref).

"""
function getcol(pr::UMFShortestPathPricingData, path::Vector{Int64})
    col::Vector{Int64} = []
    for j = 1:(size(path, 1)-1)
        push!(col, arc(pr, path[j], path[j+1]))
    end
    return col
end

"""
    getcols(pr::UMFShortestPathPricingData, paths::Vector{Vector{Int64}})

Obtain all columns (list of arcs) from each list of nodes contained in vector `paths`, for a pricing problem `pr`.

See also [`getcol`](@ref), [`UMFShortestPathPricingData`](@ref).

"""
function getcols(pr::UMFShortestPathPricingData, paths::Vector{Vector{Int64}})
    cols::Vector{Vector{Int64}} = []
    for i = 1:size(paths, 1)
        push!(cols, getcol(pr, paths[i]))
    end
    return cols
end

"""
    generateNfeasiblecols_k(
        pr::UMFShortestPathPricingData,
        k::Int64
        N::Int64 = 1
    )

Generate `N` primal-feasible columns with the Yen K-shortest path algorithm, for demand `k` for pricing `pr`. Default `N` =1.

Return them as lists of arcs.

See also [`UMFShortestPathPricingData`](@ref).

"""
function generateNfeasiblecols_k(pr::UMFShortestPathPricingData, k::Int64, N::Int64 = 1)
    #set subprob k:
    set_demand!(pr, k)
    # call yen_k_shortest_path
    ys = yen_k_shortest_paths(
        SimpleDiGraph(graph(pr)),
        src(pr),
        dst(pr),
        weights(graph(pr)),
        N,
    )
    #find columns
    return getcols(pr, ys.paths)
end


"""
    solve!(pr::UMFShortestPathPricingData, config::DefaultDijkstraPricingConfig)

Solve the pricing problem with the Dijkstra algorithm, specified by `config`.
Called in the column generation by the function `solveCG`.


See also [`solveCG`](@ref), [`UMFShortestPathPricingData`](@ref), [`DefaultDijkstraPricingConfig`](@ref).
"""
function solve!(pr::UMFShortestPathPricingData, config::PrColGenconfig)
#function solve!(pr::UMFShortestPathPricingData, config::DefaultDijkstraPricingConfig)
    k = pr.cur_k
    g = graph(pr)
    w = Graphs.weights(g)

    if !has_vertex(g, src(pr)) || !has_vertex(g, dst(pr))
        setsol!(pr, Inf)
        return 
    elseif config isa UMFSolver.LARACPricingConfig || config isa UMFSolver.ClassifierAndLARACConfig || config isa UMFSolver.kSPLARACPricingConfig
        # solve the problem with LARAC
        path = larac_shortest_path(g, src(pr), dst(pr), pr.dat.demand_latencies[k], w, pr.latency_matrix[1:nv(g),1:nv(g)])
        #path = larac_shortest_path(g, src(pr), dst(pr), pr.dat.demand_latencies[k], Graphs.weights(g), pr.latency_matrix)

    else
        # solve the problem with dijkstra
        k = pr.cur_k
        ds = dijkstra_shortest_paths(g, src(pr))
        path = enumerate_paths(ds, dst(pr))
    end

    #set xopt
    col = getcol(pr, path)
    setxopt!(pr, col)
    #set sol
    if isempty(col)
        pcost = Inf
    else
        pcost = path_cost(path, w)
    end

    # DEBUG
    setsol!(pr, pcost)

    return
end

function path_cost(path, w)
    if length(path)<2
        return 0.0
    else
        return sum(w[path[i],path[i+1]] for i in 1:length(path)-1)
    end
end


"""
    solve_pricing!(
        pr::UMFShortestPathPricingData,
        config::PrColGenconfig, 
        duals_arcs::Vector{<:Real}, 
        duals_demands::Vector{<:Real}, 
        tol::Float64
        )

Solve the pricing problem in a column generation algorithm.
Return the columns with negative reduced cost.


See also [`chgcost`](@ref), [`solve!`](@ref), [`UMFShortestPathPricingData`](@ref), [`DefaultDijkstraPricingConfig`](@ref).
"""
function solve_pricing!(
    pr::UMFShortestPathPricingData,
    config::PrColGenconfig,
    duals_arcs::Vector{<:Real},
    duals_demands::Vector{<:Real},
    tol::Float64,
    #filter=ArcDemandFilter(data(pr), config),
)
    pb::UMFData = data(pr)
    narcs::Int64 = numarcs(pb)
    nd::Int64 = numdemands(pb)
    newcost::Vector{Float64} = zeros(narcs)
    csts::Vector{Float64} = costs(pb)
    newcols::Vector{Vector{Vector{Int64}}} = [[] for k = 1:nd]
    solk::Float64 = 0
    xk::Vector{Int64} = []
    #println("New costs : ")
    #println(csts - duals_arcs)
    bbb = csts - duals_arcs

    for k = 1:nd
        set_demand!(pr, k)
        newcost = bdw(pb, k) * (csts - duals_arcs)
        # apply masking
        #offset = sum(newcost)
        #newcost[.!filter.masks[k]] .= offset
        chgcost!(pr, newcost)
        solve!(pr, config)
        solk = sol(pr)
        xk = xopt(pr)
        #if isempty(xk)
        #    continue
        #end
        #println("\tcol selection : ", solk, " - ", duals_demands[k], "=", solk-duals_demands[k])
        #println("\t\t", solk-duals_demands[k] < -tol )
        if solk - duals_demands[k] < -tol
            if isempty(xk)
                println(xk,", ", solk - duals_demands[k] , [solk, duals_demands[k]])
            end
            push!(newcols[k], xk)
        end
        #pcs = pcs*string(solk/bdw(pb,k))*"\n"
    end
    #write("graph_paths/"*string(gfi), pcs)
    return newcols
end
