using Plots

function make_plots(model, g; device=CUDA.functional() ? Flux.gpu : Flux.cpu)
    # image on test graph 1
    node_codes,_ = UMFSolver.compute_graph_embeddings(model, g |> device)

    nedm = UMFSolver.node_distance_matrix(Flux.cpu(node_codes))
    dne = Flux.cpu(UMFSolver.make_demand_codes(model, g |> device))
    nd_plt=plot(heatmap(nedm, title="Node embedding distances"), size=(500,500));
    
    ds,dt = UMFSolver.demand_endpoints(g)
    demand_ep = hcat(ds, dt)
    demand_sorted_idx = sortperm(view.(Ref(demand_ep), 1:size(demand_ep, 1), :))
    
    nnodes = sum(g.ndata.mask)
    ndemands = nv(g) - nnodes
    m = zeros(Bool, ndemands, nnodes)
    for k in 1:ndemands
        m[k, ds[demand_sorted_idx[k]]] = 1
        m[k, dt[demand_sorted_idx[k]]] = 1
    end
    dedm = UMFSolver.node_distance_matrix(dne[:,demand_sorted_idx])
    dnd_plt=plot(heatmap(m, title="Demand-node adjacency", legend=:none),heatmap(dedm,title="Demand embedding distances", cmap=cgrad(:lighttest,categorical=false), showaxis=:x), size=(1000,500));

    return (ndm=nd_plt, dndm=dnd_plt)
end

