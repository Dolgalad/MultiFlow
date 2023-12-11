# metrics
accuracy(y_pred, y_true)     = sum(y_pred .== y_true)/size(y_pred, 1)
tpcount(y_pred, y_true)      = sum( Bool.(y_pred .== y_true) .& Bool.(y_true)  )
precision(y_pred, y_true)    = sum(y_pred)>0 ? tpcount(y_pred, y_true)/sum(y_pred) : 0
recall(y_pred, y_true)       = sum(y_true)>0 ? tpcount(y_pred, y_true)/sum(y_true) : 0
f_beta_score(y_pred, y_true, beta=1.) = precision(y_pred,y_true)>0. && recall(y_pred,y_true)>0. ? sum(1 .+ (beta^2)) * (precision(y_pred,y_true)*recall(y_pred,y_true))/(((beta^2) * precision(y_pred,y_true)) + recall(y_pred,y_true)) : 0
graph_reduction(y_pred)      = 1. - sum(y_pred)/prod(size(y_pred))

function metrics(g::GNNGraph, model; device=CUDA.functional() ? Flux.gpu : Flux.cpu)
    # debug
    pred, lab = Int64.(cpu((vec(model(g |> device)) .> 0))),Int64.(cpu(vec(g.targets[g.target_mask])))
    #pred, lab = Int64.(cpu((vec(model(g)) .> 0))),Int64.(cpu(vec(g.targets[g.target_mask])))

    acc = accuracy(pred, lab)
    rec = recall(pred, lab)
    prec = precision(pred, lab)
    f1 = f_beta_score(pred, lab)
    gr = graph_reduction(pred)
    #nd = node_embedding_distance(g, model)
    #return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr, "nd"=>nd)
    return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr)

end

function metrics(pred, labs)
    # debug
    #pred, lab = Int64.(cpu(pred .> 0)),Int64.(cpu(labs))
    pred, lab = Int64.(pred .> 0),Int64.(labs)

    #pred, lab = Int64.(cpu((vec(model(g)) .> 0))),Int64.(cpu(vec(g.targets[g.target_mask])))

    acc = accuracy(pred, lab)
    rec = recall(pred, lab)
    prec = precision(pred, lab)
    f1 = f_beta_score(pred, lab)
    gr = graph_reduction(pred)
    #nd = node_embedding_distance(g, model)
    #return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr, "nd"=>nd)
    return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr)

end


#function metrics(loader, model; device=CUDA.functional() ? Flux.gpu : Flux.cpu)
#    # debug
#    preds = [(Int64.(cpu((vec(model(g |> device)) .> 0))),Int64.(vec(g.targets[g.target_mask]))) for g in loader]
#    #preds = [(Int64.(cpu((vec(model(g)) .> 0))),Int64.(vec(g.targets[g.target_mask]))) for g in loader]
#
#    acc = mean([accuracy(pred, lab) for (pred,lab) in preds])
#    rec = mean([recall(pred, lab) for (pred,lab) in preds])
#    prec = mean([precision(pred, lab) for (pred,lab) in preds])
#    f1 = mean([f_beta_score(pred, lab) for (pred,lab) in preds])
#    gr = mean([graph_reduction(pred) for (pred,lab) in preds])
#    # debug
#    #nd = mean(node_embedding_distance(g |> device, model) for g in loader)
#    #nd = mean(node_embedding_distance(g, model) for g in loader)
#
#    return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr, "nd"=>nd)
#end

function metrics_loader(loader, model; device=CUDA.functional() ? Flux.gpu : Flux.cpu)
    # debug
    preds = [(Int64.(cpu((vec(model(g |> device)) .> 0))),Int64.(vec(g.targets[g.target_mask]))) for g in loader]
    #preds = [(Int64.(cpu((vec(model(g |> device)) .> 0))),Int64.(vec(g.targets[g.target_mask]))) for g in loader]

    #preds = [(Int64.(cpu((vec(model(g)) .> 0))),Int64.(vec(g.targets[g.target_mask]))) for g in loader]

    acc = mean([accuracy(pred, lab) for (pred,lab) in preds])
    rec = mean([recall(pred, lab) for (pred,lab) in preds])
    prec = mean([precision(pred, lab) for (pred,lab) in preds])
    f1 = mean([f_beta_score(pred, lab) for (pred,lab) in preds])
    gr = mean([graph_reduction(pred) for (pred,lab) in preds])
    # debug
    #nd = mean(node_embedding_distance(g |> device, model) for g in loader)
    #nd = mean(node_embedding_distance(g, model) for g in loader)

    #return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr, "nd"=>nd)
    return Dict("acc"=>acc, "rec"=>rec, "prec"=>prec, "f1"=>f1, "gr"=>gr)

end


"""
Computes distances between node embeddings, returns matrix
"""
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


embedding_diff(xi,xj,e) = xi.-xj
"""
Mean embedding distance
"""
function node_embedding_distance(g::GNNGraph, model)
    node_embeddings, edge_embeddings = compute_graph_embeddings(model, g)
    return norm(apply_edges(embedding_diff, g, node_embeddings, node_embeddings))
end
