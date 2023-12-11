"""Create the table T where element T[i][j] is the shortest path from i to j
"""

using Graphs
using SimpleWeightedGraphs

function path_arc_indexes(path::Vector{Int64}, arc_matrix::Matrix{Int64})
    col::Vector{Int64} = []
    for j = 1:(size(path, 1)-1)
        push!(col, arc_matrix[path[j], path[j+1]])
    end
    return col
end

function get_arc_matrix(pb)
    arc_matrix = zeros(Int64, (nv(pb), nv(pb)))
    for a in 1:ne(pb)
        arc_matrix[arcsource(pb, a), arcdest(pb, a)] = a
    end
    return arc_matrix
end


function create_shortest_path_table(pb::UMFData; dstmx=costs(pb), K=1)
    g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)

    arc_matrix = get_arc_matrix(pb)
    #T = Dict{Int64, Dict{Int64, Vector{Int64}}}()
    T = zeros(Bool, nv(pb), nv(pb), ne(pb))
    for i in 1:nv(g)
        ds = dijkstra_shortest_paths(g, i)
        #T[i] = Dict{Int64, Vector{Int64}}(j=>[] for j in 1:nv(g))
        for j in 1:nv(g)
            if i!=j
	        if K==1
                    #T[i][j] = enumerate_paths(ds, j)
                    col = path_arc_indexes(enumerate_paths(ds, j), arc_matrix)
                    T[i,j,col] .= 1
		else
		    ys = yen_k_shortest_paths(g, i, j, weights(g), K)
		    for path in ys.paths
                        col = path_arc_indexes(path, arc_matrix)
                        T[i,j,col] .= 1
		    end
		end
            end
        end
    end
    return T
end
