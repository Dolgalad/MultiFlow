using SparseArrays
#using Graphs, SimpleWeightedGraphs


"""
    UMFShortestPathPricingData

    Data structure for the pricing problem for column generation on UMCF instance.

    It is a sub-type of `AbstractPricingPbData`.
    It can be used in a column generation algorithm for the UMCF. 
    It allows to solve the pricing problem with the Dijkstra algorithm.
        
    # Constructor
        
        UMFShortestPathPricingData(
            pb::UMFData,
            config::ColGenConfigBase
        )
        
    Construct a `UMFShortestPathPricingData` with information from the UMCF instance and the column generation configuration.
        
    # Arguments
    - `pb`: the UMCF instance data
    - `config`: the configuration type for the column generation algorithm.
        
    See also [`UMFData`](@ref), [`ColGenConfigBase`](@ref), [`AbstractPricingPbData`](@ref). 
        
"""
mutable struct UMFShortestPathPricingData <: AbstractPricingPbData
    dat::UMFData
    narcs::Int64
    nnodes::Int64
    ndemands::Int64
    graph::SimpleWeightedDiGraph{Int64,Float64} # original graph
    graph0::SimpleWeightedDiGraph{Int64,Float64} # graph that can be modified 
    arcs_matrix::Matrix{Int64}
    cur_k::Int64 #current demand
    cur_src::Int64 #current source index
    cur_dst::Int64 #current destination index
    xopt::Vector{Int64}# the optimal vector
    sol::Real
    filter::AbstractArcDemandFilter
    cost_matrix
    latency_matrix
    #function UMFShortestPathPricingData(pb::UMFData, config::ColGenConfigBase; filter=ArcDemandFilter(pb, config))
    function UMFShortestPathPricingData(pb::UMFData, config::ColGenConfigBase)

        dat = pb
        narcs = numarcs(pb)
        ndemands = numdemands(pb)
        nnodes = numnodes(pb)
        graph = SimpleWeightedDiGraph(arcsources(dat), arcdests(dat), 1.0 * costs(dat))
        graph0 = deepcopy(graph)
        #arcs_matrix = zeros(nnodes, nnodes)
        #cost_matrix = zeros(nnodes, nnodes)
        #latency_matrix = zeros(nnodes, nnodes)
        #for a = 1:narcs
        #    arcs_matrix[arcsource(pb, a), arcdest(pb, a)] = a
        #    cost_matrix[arcsource(pb, a), arcdest(pb, a)] = pb.costs[a]
        #    latency_matrix[arcsource(pb, a), arcdest(pb, a)] = pb.latencies[a]
        #end
        arcs_matrix = sparse(pb.srcnodes, pb.dstnodes, 1:ne(pb))
        cost_matrix = sparse(pb.srcnodes, pb.dstnodes, pb.costs)
        latency_matrix = sparse(pb.srcnodes, pb.dstnodes, pb.latencies)

        cur_src = 0
        cur_dst = 0
        cur_k = 0
        xopt = []
        sol = 0
        # filter
        t_create_filter = 0.
        if typeof(config.prconfig)==kSPFilterPricingConfig
            t_create_filter =
                @elapsed filter = kSPFilter(pb, config, config.prconfig.K)  
        elseif typeof(config.prconfig)==ClassifierAndSPConfig
            t_create_filter = @elapsed filter = ClassifierSPFilter(pb, config)
        elseif typeof(config.prconfig)==SVMAndSPConfig
            t_create_filter = @elapsed filter = SVMSPFilter(pb, config)
        elseif typeof(config.prconfig)==RFAndSPConfig
            t_create_filter = @elapsed filter = RFSPFilter(pb, config)
        elseif typeof(config.prconfig)==MLPAndSPConfig
            t_create_filter = @elapsed filter = MLPSPFilter(pb, config)
        elseif typeof(config.prconfig)==ClassifierAndLARACConfig
            #t_create_filter = @elapsed filter = ClassifierLARACFilter(pb, config, config.prconfig.model_path, sptable_path=config.prconfig.sptable_path)
            t_create_filter = @elapsed filter = ClassifierLARACFilter(pb, config)
        else        
            t_create_filter =
                @elapsed filter = ArcDemandFilter(pb, config)
        end
        #println("t_create_filter = $(t_create_filter)")

        new(
            dat,
            narcs,
            nnodes,
            ndemands,
            graph,
            graph0,
            arcs_matrix,
            cur_k,
            cur_src,
            cur_dst,
            xopt,
            sol,
            filter,
            cost_matrix,
            latency_matrix,
        )
    end
end

"""
"""
function full_filter_mask(pr::UMFShortestPathPricingData)
    f = hcat([pr.filter.masks[k] for k in 1:numdemands(pr.dat)]...)
    return transpose(f)
end


"""
    data(pr::UMFShortestPathPricingData)

Get the `UMFData` structure considered by `pr`.

"""
function data(pr::UMFShortestPathPricingData)
    return pr.dat
end

"""
    k(pr::UMFShortestPathPricingData)

Get the current demand index considered by `pr`.

"""
function k(pr::UMFShortestPathPricingData)
    return pr.cur_k
end

"""
    src(pr::UMFShortestPathPricingData)

Get the current demand source node considered by `pr`.

"""
function src(pr::UMFShortestPathPricingData)
    return pr.cur_src
end

"""
    dst(pr::UMFShortestPathPricingData)

Get the current demand destination node considered by `pr`.

"""
function dst(pr::UMFShortestPathPricingData)
    return pr.cur_dst
end

"""
    set_k!(pr::UMFShortestPathPricingData, k::Int64)

Set the current demand index for `pr` to `k`.

"""
function set_k!(pr::UMFShortestPathPricingData, k::Int64)
    pr.cur_k = k
    return k
end

"""
    set_src!(pr::UMFShortestPathPricingData, src::Int64)

Set the current demand source node for `pr` to `src`.

"""
function set_src!(pr::UMFShortestPathPricingData, src::Int64)
    pr.cur_src = src
    return src
end

"""
    set_dst!(pr::UMFShortestPathPricingData, dst::Int64)

Set the current demand destination node for `pr` to `dst`.

"""
function set_dst!(pr::UMFShortestPathPricingData, dst::Int64)
    pr.cur_dst = dst
    return dst
end

"""
    graph(pr::UMFShortestPathPricingData)

Get the graph for the instance in `pr`.

"""
function graph(pr::UMFShortestPathPricingData)
    #if !isnothing(mask)
    #    g = copy(pr.graph)
    #    edge_idx_2_remove = collect(1:ne(g))[.!mask]
    #    #println("edge_i : ", edge_idx_2_remove)
    #    edge_2_remove = [(arcsource(pr.dat, i), arcdest(pr.dat, i)) for i in edge_idx_2_remove]
    #    for (s,t) in edge_2_remove
    #        rem_edge!(g, s, t)
    #    end
    #    return g
    #end
    return pr.filter.graphs[pr.cur_k]
    return pr.graph
end

"""
    arc(pr::UMFShortestPathPricingData, src::Int64, dst::Int64)

Get the index of the arc, in the graph of `pr`, that links `src` to `dst`. Return ``0`` if the arc does not exist.

"""
function arc(pr::UMFShortestPathPricingData, src::Int64, dst::Int64)
    return pr.arcs_matrix[src, dst]
end

"""
    setxopt!(pr::UMFShortestPathPricingData, vec::Vector{Int64})

Set the optimal point in `pr` to `vec`.

"""
function setxopt!(pr::UMFShortestPathPricingData, vec::Vector{Int64})
    pr.xopt = vec
    return pr.xopt
end

"""
    xopt(pr::UMFShortestPathPricingData)

Get the current optimal point for the pricing problem `pr`.

"""
function xopt(pr::UMFShortestPathPricingData)
    return pr.xopt
end

"""
    setsol!(pr::UMFShortestPathPricingData, val::Real)

Set the optimal value of the pricing problem `pr` to the value `val`.

"""
function setsol!(pr::UMFShortestPathPricingData, val::Real)
    pr.sol = val
    return pr.sol
end

"""
    sol(pr::UMFShortestPathPricingData)

Get the current optimal value of the pricing problem `pr`.

"""
function sol(pr::UMFShortestPathPricingData)
    return pr.sol
end
