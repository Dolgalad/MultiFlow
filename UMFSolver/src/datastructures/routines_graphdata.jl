"""
    out_and_in_arcs(
        nnodes::Int64,
        narcs::Int64,
        arcsrcs::Vector{Int64},
        arcdsts::Vector{Int64}
    )

Obtain lists of outgoing and incoming arcs for each node of a graph.

# Arguments
    - `nnodes`: number of nodes
    - `narcs`: number of arcs
    - `arcsrcs`: list of source nodes of the arcs
    - `arcdsts`: list of destination nodes of the arcs

# Return values
    - `outgoing::Vector{Vector{Int64}}`: list of outgoing arcs of the nodes
    - `incoming::Vector{Vector{Int64}}`: list of incoming arcs of the nodes

"""
function out_and_in_arcs(
    nnodes::Int64,
    narcs::Int64,
    arcsrcs::Vector{Int64},
    arcdsts::Vector{Int64},
)
    outgoing::Vector{Vector{Int64}} = [[] for i = 1:nnodes]
    incoming::Vector{Vector{Int64}} = [[] for i = 1:nnodes]
    for a = 1:narcs
        s = arcsrcs[a]
        d = arcdsts[a]
        push!(outgoing[s], a)
        push!(incoming[d], a)
    end
    return outgoing, incoming
end

"""
    are_arcs_doubled(
        nnodes::Int64,
        outgoing::Vector{Vector{Int64}},
        arcdsts::Vector{Int64}
    )

Find if all arcs in a directed graph have their inverse arc.

Return `true` if all arcs have their inverse arc, `false` otherwise.

# Arguments:
    - `nnodes`: number of nodes 
    - `outgoing`: lists of outgoing arcs from the nodes,
    - `arcdsts`: lists of destination nodes of the arcs.
"""
function are_arcs_doubled(
    nnodes::Int64,
    outgoing::Vector{Vector{Int64}},
    arcdsts::Vector{Int64},
)
    exists::Bool = false
    for u = 1:nnodes
        for a in outgoing[u]
            d = arcdsts[a]
            exists = false
            for e in outgoing[d]
                if u == arcdsts[e]
                    exists = true
                    break
                end
            end
            if !exists
                return false
            end
        end
    end
    return true
end

"""

    checkindices4!(
        srcs1::Vector{Int64},
        dsts1::Vector{Int64},
        srcs2::Vector{Int64},
        dsts2::Vector{Int64},
    )

Make sure that the 4 indices of nodes start from 1: increase all indices by 1 if at least one entry is 0.

"""
function checkindices4!(
    srcs1::Vector{Int64},
    dsts1::Vector{Int64},
    srcs2::Vector{Int64},
    dsts2::Vector{Int64},
)
    if minimum(srcs1) == 0 ||
       minimum(dsts1) == 0 ||
       minimum(srcs2) == 0 ||
       minimum(dsts2) == 0
        srcs1 .+= 1
        dsts1 .+= 1
        srcs2 .+= 1
        dsts2 .+= 1
    end
end

"""
    getnummodes(srcs::Vector{Int64}, dsts::Vector{Int64})

Get the number of nodes from the lists of origins and destinations of all arcs.

"""
function getnummodes(srcs::Vector{Int64}, dsts::Vector{Int64})
    size1 = maximum(srcs)
    size2 = maximum(dsts)
    nnodes = max(size1, size2)
    return nnodes
end


"""
    doublearclist!(
        srcnodes::Vector{Int64},
        dstnodes::Vector{Int64},
        capacities::Vector{<:Real},
        costs::Vector{<:Real},
        latencies::Vector{<:Number},

    )

Extend the arcs lists by adding all the reverse arcs.

# Arguments

    - `srcnodes`: list of origins of the arcs
    - `dstnodes`: list of destinations of the arcs
    - `capacities`: list of capacities of the arcs
    - `costs`: lit of costs of the arcs.
    - `latencies` : list of latencies of the arcs

"""
function doublearclist!(
    srcnodes::Vector{Int64},
    dstnodes::Vector{Int64},
    capacities::Vector{<:Real},
    costs::Vector{<:Real},
    latencies::Vector{<:Number},
)
    tmpsrcs = copy(srcnodes)
    srcnodes = append!(srcnodes, dstnodes)
    dstnodes = append!(dstnodes, tmpsrcs)
    costs = append!(costs, costs)
    capacities = append!(capacities, capacities)
    latencies = append!(latencies, latencies)
    return srcnodes, dstnodes, capacities, costs, latencies
end


"""
    doublearcs!(
        srcnodes::Vector{Int64},
        dstnodes::Vector{Int64},
        capacities::Vector{<:Real},
        costs::Vector{<:Real},
        latencies::Vector{<:Number},
        narcs::Int64,
    )

Double the number of arcs and extend the arcs lists by adding all the reverse arcs.

# Arguments

    - `srcnodes`: list of origins of the arcs
    - `dstnodes`: list of destinations of the arcs
    - `capacities`: list of capacities of the arcs
    - `costs`: lit of costs of the arcs
    - `latencies` : list of latencies of the arcs
    - `narcs`: the number of arcs.

"""
function doublearcs!(
    srcnodes::Vector{Int64},
    dstnodes::Vector{Int64},
    capacities::Vector{<:Real},
    costs::Vector{<:Real},
    latencies::Vector{<:Number},
    narcs::Int64,
)
    doublearclist!(srcnodes, dstnodes, capacities, costs, latencies)
    newnarcs = 2 * narcs
    return newnarcs
end
