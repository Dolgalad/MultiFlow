using LinearAlgebra

function node_distance_matrix(codes; dist_func=(x,y) -> norm(x-y))
    nnodes = size(codes, 2)
    m = zeros(nnodes, nnodes)
    for i in 1:nnodes
        for j in 1:nnodes
            if i!=j
                m[i,j] = dist_func(codes[:,i], codes[:,j])
            end
        end
    end
    return m
end
