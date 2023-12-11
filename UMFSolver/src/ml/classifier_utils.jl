using CUDA
using Flux

function compute_edge_demand_scores(model, 
				    edge_codes::AbstractMatrix, 
				    demand_codes::AbstractMatrix, 
				    g::GNNGraph;
				    )
    ngind = graph_indicator(g)
    ninodes = sum(g.ndata.mask[ngind .== 1])
    nidemands = sum(.!g.ndata.mask[ngind .== 1])

    egind = graph_indicator(g, edges=true)
    regind = egind[g.edata.mask]
    niedges = sum(g.edata.mask[egind .== 1])

    # demand node graph indicator
    dgind = ngind[.!g.ndata.mask]

    # stack repeated demand codes
    dind = collect(1:size(dgind,1)) |> model._device
    demand_stacked_idx = reduce(vcat,[repeat(dind[dgind .== i], inner=niedges) for i=1:g.num_graphs])
    demand_stacked = getobs(demand_codes,demand_stacked_idx)


    # stacked repeated edge codes
    reind = collect(1:size(regind,1)) |> model._device
    #edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], nidemands) for i=1:g.num_graphs])
    edge_stacked_idx = reduce(vcat, [repeat(reind[regind .== i], g.K[i]) for i=1:g.num_graphs])

    edge_stacked = getobs(edge_codes, edge_stacked_idx)


    # if scoring layer is bilinear
    if model.scoring isa Flux.Bilinear
        scores= model.scoring(edge_stacked, demand_stacked)
    else
        scores= model.scoring(vcat(edge_stacked, demand_stacked))
    end
    return scores
end


