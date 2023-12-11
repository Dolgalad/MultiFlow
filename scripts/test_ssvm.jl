using UMFSolver

function prediction_to_column(g, theta)
    ds = dijkstra_shortest_paths(g, 1)
    p = enumerate_paths(ds, 5)
    return path_to_column(g, p)
end


function arc_matrix(g)
           #src, dst = edge_index(g)
           src, dst = [e.src for e in edges(g)], [e.dst for e in edges(g)]

           nnodes = size(src,1)
           am  = zeros(Int64, nnodes, nnodes)
           for a = 1:ne(g)
               am[src[a], dst[a]] = a
           end
           return am
end

function getcol(g, path)
           col::Vector{Int64} = []
           am = arc_matrix(g)
           for j = 1:(size(path, 1)-1)
               push!(col, am[path[j], path[j+1]])
           end
           return col
end

function path_to_column(g, path)
           p = getcol(g, path)
           col = zeros(Float32, ne(g))
           col[p] .= 1
           return col
end

umf_encoder = UMFSolver.M9ClassifierModel(64, 3, 4, 71) |> device

