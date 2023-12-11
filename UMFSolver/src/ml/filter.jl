using BSON
using BSON: @load
using CUDA
using GraphNeuralNetworks
using Flux
using HDF5
using SparseArrays
using Graphs
using LIBSVM
using DecisionTree
using SimpleWeightedGraphs

import Base: reverse

abstract type AbstractArcDemandFilter end


function get_edge_masked_graph(pb::UMFData, mask::AbstractVector)
    if !any(mask)
        return SimpleWeightedDiGraph()
    end
    return SimpleWeightedDiGraph(pb.srcnodes[mask], pb.dstnodes[mask], pb.costs[mask])

    g = SimpleWeightedDiGraph(arcsources(pb), arcdests(pb), 1.0 * costs(pb))
    for e in 1:ne(g)
        if mask[e]==0
            rem_edge!(g, arcsource(pb, e), arcdest(pb, e))
        end
    end
    return g
end
function path_arc_indexes(path::Vector{Int64}, arc_matrix::AbstractMatrix{Int64})
    if length(path)<2
        return Int64[]
    end
    return [arc_matrix[path[i],path[i+1]] for i in 1:length(path)-1]
    col::Vector{Int64} = []
    for j = 1:(size(path, 1)-1)
        push!(col, arc_matrix[path[j], path[j+1]])
    end
    return col
end

function get_arc_matrix(pb::UMFData)
    return sparse(pb.srcnodes, pb.dstnodes, 1:ne(pb))
    arc_matrix = spzeros(Int64, (numnodes(pb), numnodes(pb)))
    for a = 1:numarcs(pb)
        arc_matrix[arcsource(pb, a), arcdest(pb, a)] = a
    end
    return arc_matrix

end



struct ArcDemandFilter <: AbstractArcDemandFilter
    #masks::Matrix{Bool}
    masks::Dict{Int64, Vector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    function ArcDemandFilter(pb::UMFData, config::AbstractConfiguration)

        m = Dict{Int64, Vector{Bool}}()
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))
        for k in 1:numdemands(pb)
            # default is to keep all arcs
            m[k] = ones(Bool, ne(pb))
            gs[k] = get_edge_masked_graph(pb, m[k])
        end
        new(m, gs)
    end
end

function get_ksp_demand_mask(k::Int64, 
    pb::UMFData, 
    config::ColGenConfigBase, 
    g::SimpleWeightedDiGraph, 
    K::Int,
    arc_matrix::AbstractMatrix{Int64};
    dstmx = Graphs.weights(g)
    )
    m = spzeros(Bool, numarcs(pb))

    if K!=0
        ds = yen_k_shortest_paths(g, pb.srcdemands[k], pb.dstdemands[k], dstmx ,K)
        
        for p in ds.paths
            col = path_arc_indexes(p, arc_matrix)
            m[col] .= 1
        end
    end
    return m
end

struct kSPFilter <: AbstractArcDemandFilter
    #masks::SparseMatrixCSC{Bool,Int64}
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}

    function kSPFilter(pb::UMFData, config::ColGenConfigBase, K=1; dstmx=costs(pb))
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
        arc_matrix = get_arc_matrix(pb)
	sptable = config.prconfig.sptable
        m = Dict{Int64, SparseVector{Bool}}()
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))
        for k in 1:numdemands(pb)
            m[k] = get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
	    if !isnothing(sptable)
                m[k] .|= sptable[pb.srcdemands[k], pb.dstdemands[k], :]
	    end

            gs[k] = get_edge_masked_graph(pb, m[k])
        end    
        new(m, gs)
    end
end

struct kSPLARAC<: AbstractArcDemandFilter
    #masks::SparseMatrixCSC{Bool,Int64}
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}

    function kSPLARAC(pb::UMFData, config::ColGenConfigBase, K=1; dstmx=costs(pb))
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
	wd = UMFSolver.get_latency_matrix(pb)
        arc_matrix = get_arc_matrix(pb)
	sptable = config.prconfig.sptable
        m = Dict{Int64, SparseVector{Bool}}()
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))
        for k in 1:numdemands(pb)
            m[k] = get_ksp_demand_mask(k, pb, config, g, K, arc_matrix) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix, dstmx=wd)
	    if !isnothing(sptable)
                m[k] .|= sptable[pb.srcdemands[k], pb.dstdemands[k], :]
	    end

            gs[k] = get_edge_masked_graph(pb, m[k])
        end    
        new(m, gs)
    end
end



function clssp_compute_masks2(k::Int64, pb::UMFData, g::SimpleWeightedDiGraph{Int64,Float64}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real})
    # forward MST
    mst_f = dijkstra_shortest_paths(g, pb.srcdemands[k])
    # backward MST
    mst_b = dijkstra_shortest_paths(g, pb.dstdemands[k])
    #m_tmp = spzeros(Bool, ne(g))
    m_tmp = sparsevec(scores[k,:].>0)

    #idx = @views findall(>(0), scores[k,:])
    feidx = @views reduce(vcat, path_arc_indexes(enumerate_paths(mst_f,i),arc_matrix) for i in pb.srcnodes[m_tmp])
    beidx = @views reduce(vcat, path_arc_indexes(reverse(enumerate_paths(mst_b,i)),arc_matrix) for i in pb.dstnodes[m_tmp])
    m_tmp[feidx] .= 1
    m_tmp[beidx] .= 1
    return m_tmp
    m_tmp[feidx] .= 1
    m_tmp[beidx] .= 1
    return m_tmp

    for i in idx
        # arc has been predicted
        # get path to origin
        p = enumerate_paths(mst_f, arcsource(pb, i))
        if length(p)>0
            col = path_arc_indexes(p, arc_matrix)
            m_tmp[col] .= 1
        end
        # get path to target
        #p = enumerate_paths(mst_b, arcsource(pb, i))
        p = enumerate_paths(mst_b, arcdest(pb, i))
        if length(p)>0
            col = path_arc_indexes(reverse(p), arc_matrix)
            m_tmp[col] .= 1
        end
        m_tmp[i] = 1
    end
    return m_tmp
end
function clssp_compute_masks3(k::Int64, pb::UMFData, g::SimpleWeightedDiGraph{Int64,Float64}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real})
    # forward MST
    mst_f = dijkstra_shortest_paths(g, pb.srcdemands[k])
    # backward MST
    mst_b = dijkstra_shortest_paths(g, pb.dstdemands[k])
    m_tmp = @views sparsevec(scores[k,:].>0)
    for (i,j) in @views zip(pb.srcnodes[m_tmp], pb.dstnodes[m_tmp])
        # arc has been predicted
        # get path to origin
        m_tmp[path_arc_indexes(enumerate_paths(mst_f, i), arc_matrix)] .= 1
        m_tmp[path_arc_indexes(reverse(enumerate_paths(mst_b, j)), arc_matrix)] .= 1
    end
    return m_tmp
end

function reverse(g::SimpleWeightedDiGraph)
    ng = SimpleWeightedDiGraph(nv(g))
    for e in edges(g)
        add_edge!(ng, e.dst, e.src, e.weight)
    end
    return ng

end

function clssp_compute_masks(k::Int64, pb::UMFData, g::SimpleWeightedDiGraph{Int64,Float64}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real}; threshold::Float64=0.0, keep_proportion::Float64=0.0)
    # forward MST
    mst_f = dijkstra_shortest_paths(g, pb.srcdemands[k])
    # backward MST
    #mst_b = dijkstra_shortest_paths(g, pb.dstdemands[k])
    mst_b = dijkstra_shortest_paths(reverse(g), pb.dstdemands[k])

    m_tmp = spzeros(Bool, ne(g))
    if !any(scores[k,:] .> threshold)
        p = enumerate_paths(mst_f, pb.dstdemands[k])
        if length(p)>0
            col = path_arc_indexes(p, arc_matrix)
            m_tmp[col] .= 1
        end
        return m_tmp

    end
    for i in 1:ne(g)
        if scores[k, i] .> threshold
            # arc has been predicted
            # get path to origin
            p = enumerate_paths(mst_f, arcsource(pb, i))
            if length(p)>0
                col = path_arc_indexes(p, arc_matrix)
                m_tmp[col] .= 1
            end
            # get path to target
            #p = enumerate_paths(mst_b, arcsource(pb, i))
            p = enumerate_paths(mst_b, arcdest(pb, i))
            if length(p)>0
                col = path_arc_indexes(Base.reverse(p), arc_matrix)
                m_tmp[col] .= 1
            end
            m_tmp[i] = 1
        end
    end
    return m_tmp
end

#function clssp_compute_precomputed_masks(k::Int64, pb::UMFData, sptable::AbstractArray{Bool}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real})
#    m_tmp = spzeros(Bool, numarcs(pb))
#    for i in 1:numarcs(pb)
#        if scores[k, i] .> 0
#            # arc has been predicted
#            # get path to origin
#            m_tmp .|= sptable[pb.srcdemands[k], arcsource(pb,i), :]
#            m_tmp .|= sptable[pb.dstdemands[k], arcdest(pb,i), :]
#            m_tmp[i] = 1
#        end
#    end
#    return m_tmp
#end
function clssp_compute_precomputed_masks2(k::Int64, pb::UMFData, sptable::AbstractArray{Bool}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real})
    idx = @views scores[k,:].>0
    if !any(scores[k,:] .> 0)
        return sptable[pb.srcdemands[k],pb.dstdemands[k],:]
    end

    return @views reduce(.|, eachrow(sptable[pb.srcdemands[k],pb.srcnodes[idx],:])) .| @views reduce(.|,eachrow(sptable[pb.dstdemands[k], pb.dstnodes[idx],:])) .| idx
end
function clssp_compute_precomputed_masks(k::Int64, pb::UMFData, sptable::AbstractArray{Bool}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real}, threshold::Float64=0., keep_proportion::Float64=0.)
    if keep_proportion>0.0
        to_keep = trunc(Int64, ne(pb) * keep_proportion)
        sort_idx = sortperm(scores[k,:], rev=true)
        idx = @views scores[k,:].>=scores[k,sort_idx[to_keep]]
    else
        idx = @views scores[k,:].>threshold
    end

    if any(idx)
        return @views vec(maximum(sptable[pb.srcdemands[k],pb.srcnodes[idx],:],dims=1)) .| @views vec(maximum(sptable[pb.dstdemands[k],pb.dstnodes[idx],:],dims=1)) .| idx
    else
        return sptable[pb.srcdemands[k],pb.dstdemands[k],:]
    end
end

function clssp_compute_precomputed_masks_3(k::Int64, pb::UMFData, sptable::AbstractArray{Bool}, arc_matrix::AbstractMatrix{Int64}, scores::AbstractMatrix{<:Real}, threshold::Float64=0., keep_proportion::Float64=0.)
    if keep_proportion>0.0
        to_keep = trunc(Int64, ne(pb) * keep_proportion)
        sort_idx = sortperm(scores[k,:], rev=true)
        idx = @views scores[k,:].>=scores[k,sort_idx[to_keep]]
    else
        idx = @views scores[k,:].>threshold
    end

    if any(idx)
        # create a graph
        g = SimpleWeightedDiGraph(pb.srcnodes[idx], pb.dstnodes[idx], vec(scores[k,idx]))
        # vertex queue
        q, r = Set(vertices(g)), Set()
        # first vertex in queue
        while !isempty(q)
            u = first(q)
            inn, outn = inneighbors(g,u), outneighbors(g,u)
            # if node has no outneighbors and no inneighbors skip it
            if isempty(inn) && isempty(outn)
                # it might be necessary to come back to this vertex in the future
                pop!(q)
            elseif !isempty(inn) && !isempty(outn)
                # it is never necessary to come back to this vertex
                push!(r, pop!(q))
            elseif isempty(inn)
                # add v with highest score (v,u)
                # all candidate in neighbors
                candidates = pb.srcnodes[findall(==(u), pb.dstnodes)]
                candidate_scores = scores[k, [arc_matrix[v,u] for v in candidates]]
                (s,vi) = findmax(candidate_scores)
                v = candidates[vi]
                idx[arc_matrix[v,u]] = 1

                if !has_vertex(g, v)
                    add_vertices!(g, v-nv(g))
                end
                add_edge!(g, v, u, s) 
                if !(v in r)
                    push!(q, v)
                end
            else
                # add v with highest score (u,v)
                candidates = pb.dstnodes[findall(==(u), pb.srcnodes)]
                candidate_scores = scores[k, [arc_matrix[u,v] for v in candidates]]
                (s,vi) = findmax(candidate_scores)
                v = candidates[vi]

                idx[arc_matrix[u,v]] = 1
                if !has_vertex(g, v)
                    add_vertices!(g, v-nv(g))
                end
                add_edge!(g, u, v, s) 

                if !(v in r)
                    push!(q, v)
                end
            end
        end
        return idx

        return @views vec(maximum(sptable[pb.srcdemands[k],pb.srcnodes[idx],:],dims=1)) .| @views vec(maximum(sptable[pb.dstdemands[k],pb.dstnodes[idx],:],dims=1)) .| idx
    else
        return sptable[pb.srcdemands[k],pb.dstdemands[k],:]
    end
end


struct ClassifierSPFilter <: AbstractArcDemandFilter
    #masks::SparseMatrixCSC{Bool,Int64}
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    #function ClassifierSPFilter(pb::UMFData, 
    #        config::ColGenConfigBase, 
    #        model_path::String; 
    #        K::Int64=1, 
    #        dstmx=costs(pb), 
    #        sptable_path::String="", 
    #        sptable_category="cost",
    #        threshold::Float64=0.,
    #        keep_proportion::Float64=0.
    #    )
    function ClassifierSPFilter(pb::UMFData, config::ColGenConfigBase; K::Int64=config.prconfig.K, dstmx=costs(pb), sptable_path::String="", sptable_category="cost")
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
        arc_matrix = get_arc_matrix(pb)

	if !isempty(methods(config.prconfig.model))
	    # model
            _g = UMFSolver.to_gnngraph(UMFSolver.scale(pb), feature_type=Float32)
            if CUDA.functional()
	        _g = _g |> Flux.gpu
                pred = Flux.cpu(config.prconfig.model(_g))
            else
                pred = config.prconfig.model(_g)
            end

	else
	    # prediction
	    pred = config.prconfig.model
	end



        # reshaping and sending to CPU
        y = transpose(reshape(pred, numarcs(pb), numdemands(pb)))

        # sparsified graphs
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))

        # shortest path table
        use_sptable::Bool = false
        sptable = nothing
	if !isnothing(config.prconfig.sptable)
	    sptable = config.prconfig.sptable
	    use_sptable = true
	end
        
        m = Dict{Int64, SparseVector{Bool}}()


        postprocessing_fct = clssp_compute_precomputed_masks
        if config.prconfig.postprocessing_method == 2
            postprocessing_fct = clssp_compute_precomputed_masks_3
        end

	demands_done = Dict()

        for k in 1:nk(pb)
	    key = (pb.srcdemands[k], pb.dstdemands[k])
	    if key in keys(demands_done)
	        m[k] = m[demands_done[key]]
		gs[k] = gs[demands_done[key]]
	    else
                if !use_sptable
                    m[k] = clssp_compute_masks(k, pb, g, arc_matrix, y, threshold=config.prconfig.threshold, keep_proportion=config.prconfig.keep_proportion) .| get_ksp_demand_mask(k, pb, config, g, config.prconfig.K, arc_matrix)
                else
                    m[k] = postprocessing_fct(k, pb, sptable, arc_matrix, y, config.prconfig.threshold, config.prconfig.keep_proportion) .| get_ksp_demand_mask(k, pb, config, g, config.prconfig.K, arc_matrix)
                end
                gs[k] = get_edge_masked_graph(pb, m[k])
		demands_done[key] = k
	    end
        end
        new(m, gs)
    end

end


# create the feature and label vectors representing (arc,commodity) pairs
function make_arc_commodity_data(g::GNNGraph; node_feature_size=1, with_labels=true)
    x, y = [], []
    s, t = edge_index(g)
    ds, dt = UMFSolver.demand_endpoints(g)
    nedges = sum(g.edata.mask)
    nnodes = sum(g.ndata.mask)
    node_vec = rand(node_feature_size, nnodes)
    for k in 1:g.K
        for a in 1:ne(g)
            if g.edata.mask[a]
                outi = outneighbors(g, s[a])
                inj = inneighbors(g, t[a])
                # arcs leaving i
                arcs_out_i = findall(==(1), (s .== s[a]) .& (g.edata.mask))
                arcs_in_j = findall(==(1), (t .== t[a]) .& (g.edata.mask))

                push!(x, vcat(g.e[1:3,a], 
                              [length(inj), length(outi)], 
                              minimum(g.e[1:3,arcs_out_i], dims=2),
                              maximum(g.e[1:3,arcs_out_i], dims=2),
                              mean(g.e[1:3,arcs_out_i], dims=2),
                              minimum(g.e[1:3,arcs_in_j], dims=2),
                              maximum(g.e[1:3,arcs_in_j], dims=2),
                              mean(g.e[1:3,arcs_in_j], dims=2),
                              node_vec[:, s[a]],
                              node_vec[:, t[a]],
                              g.e[:, nedges+k],
                              node_vec[:, ds[k]],
                              node_vec[:, dt[k]]
                             ))
                if with_labels
                    push!(y, g.targets[a,k])
                end
            end
        end
    end
    if with_labels
        return hcat(x...), Bool.(y)
    else
        return hcat(x...)
    end
end


struct SVMSPFilter <: AbstractArcDemandFilter
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    function SVMSPFilter(pb::UMFData, config::ColGenConfigBase; K::Int64=1, dstmx=costs(pb), sptable_path::String="", sptable_category="cost", scale=true)
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
        arc_matrix = get_arc_matrix(pb)

        # create the features for classification
        if scale
            _g = UMFSolver.to_gnngraph(UMFSolver.scale(pb))
        else
            _g = UMFSolver.to_gnngraph(pb)
        end

        features = make_arc_commodity_data(_g, with_labels=false)

        _, pred = svmpredict(config.prconfig.model, features)
        pred = -pred[1,:]


        # reshaping and sending to CPU
        y = transpose(reshape(pred, numarcs(pb), numdemands(pb)))

        # sparsified graphs
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))

        # shortest path table
        use_sptable::Bool = false
        sptable = nothing
	#if !isnothing(config.prconfig.sptable)
	#    sptable = config.prconfig.sptable
	#    use_sptable = true
	#end
        
        m = Dict{Int64, SparseVector{Bool}}()


        #postprocessing_fct = clssp_compute_precomputed_masks
        #if config.prconfig.postprocessing_method == 2
        #    postprocessing_fct = clssp_compute_precomputed_masks_3
        #end
        postprocessing_fct = clssp_compute_precomputed_masks_3


	demands_done = Dict()

        for k in 1:nk(pb)
	    key = (pb.srcdemands[k], pb.dstdemands[k])
	    if key in keys(demands_done)
	        m[k] = m[demands_done[key]]
		gs[k] = gs[demands_done[key]]
	    else
                if !use_sptable
                    m[k] = clssp_compute_masks(k, pb, g, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                else
                    m[k] = postprocessing_fct(k, pb, sptable, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                end
                gs[k] = get_edge_masked_graph(pb, m[k])
		demands_done[key] = k
	    end
        end
        new(m, gs)
    end

end

struct RFSPFilter <: AbstractArcDemandFilter
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    function RFSPFilter(pb::UMFData, config::ColGenConfigBase; K::Int64=1, dstmx=costs(pb), sptable_path::String="", sptable_category="cost", scale=true)
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
        arc_matrix = get_arc_matrix(pb)

        # create the features for classification
        if scale
            _g = UMFSolver.to_gnngraph(UMFSolver.scale(pb))
        else
            _g = UMFSolver.to_gnngraph(pb)
        end

        features = make_arc_commodity_data(_g, with_labels=false)

        pred = apply_forest(config.prconfig.model, transpose(features))

        # reshaping and sending to CPU
        y = transpose(reshape(pred, numarcs(pb), numdemands(pb)))

        # sparsified graphs
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))

        # shortest path table
        use_sptable::Bool = false
        sptable = nothing
	#if !isnothing(config.prconfig.sptable)
	#    sptable = config.prconfig.sptable
	#    use_sptable = true
	#end
        
        m = Dict{Int64, SparseVector{Bool}}()


        #postprocessing_fct = clssp_compute_precomputed_masks
        #if config.prconfig.postprocessing_method == 2
        #    postprocessing_fct = clssp_compute_precomputed_masks_3
        #end
        postprocessing_fct = clssp_compute_precomputed_masks_3


	demands_done = Dict()

        for k in 1:nk(pb)
	    key = (pb.srcdemands[k], pb.dstdemands[k])
	    if key in keys(demands_done)
	        m[k] = m[demands_done[key]]
		gs[k] = gs[demands_done[key]]
	    else
                if !use_sptable
                    m[k] = clssp_compute_masks(k, pb, g, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                else
                    m[k] = postprocessing_fct(k, pb, sptable, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                end
                gs[k] = get_edge_masked_graph(pb, m[k])
		demands_done[key] = k
	    end
        end
        new(m, gs)
    end

end

struct MLPSPFilter <: AbstractArcDemandFilter
    masks::Dict{Int64,SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    function MLPSPFilter(pb::UMFData, config::ColGenConfigBase; K::Int64=1, dstmx=costs(pb), sptable_path::String="", sptable_category="cost", scale=true)
        g = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, dstmx)
        arc_matrix = get_arc_matrix(pb)

        # create the features for classification
        if scale
            _g = UMFSolver.to_gnngraph(UMFSolver.scale(pb))
        else
            _g = UMFSolver.to_gnngraph(pb)
        end

        features = make_arc_commodity_data(_g, with_labels=false)

        if CUDA.functional()
            model = config.prconfig.model |> Flux.gpu
            features = features |> Flux.gpu
            pred = model(features) |> Flux.cpu
        else
            pred = config.prconfig.model(features)
        end

        # reshaping and sending to CPU
        y = transpose(reshape(pred, numarcs(pb), numdemands(pb)))

        # sparsified graphs
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))

        # shortest path table
        use_sptable::Bool = false
        sptable = nothing
	#if !isnothing(config.prconfig.sptable)
	#    sptable = config.prconfig.sptable
	#    use_sptable = true
	#end
        
        m = Dict{Int64, SparseVector{Bool}}()


        #postprocessing_fct = clssp_compute_precomputed_masks
        #if config.prconfig.postprocessing_method == 2
        #    postprocessing_fct = clssp_compute_precomputed_masks_3
        #end
        postprocessing_fct = clssp_compute_precomputed_masks_3


	demands_done = Dict()

        for k in 1:nk(pb)
	    key = (pb.srcdemands[k], pb.dstdemands[k])
	    if key in keys(demands_done)
	        m[k] = m[demands_done[key]]
		gs[k] = gs[demands_done[key]]
	    else
                if !use_sptable
                    m[k] = clssp_compute_masks(k, pb, g, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                else
                    m[k] = postprocessing_fct(k, pb, sptable, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, g, K, arc_matrix)
                end
                gs[k] = get_edge_masked_graph(pb, m[k])
		demands_done[key] = k
	    end
        end
        new(m, gs)
    end

end



struct ClassifierLARACFilter <: AbstractArcDemandFilter
    #masks::SparseMatrixCSC{Bool,Int64}
    masks::Dict{Int64, SparseVector{Bool}}
    graphs::Vector{SimpleWeightedDiGraph{Int64,Float64}}
    #function ClassifierLARACFilter(pb::UMFData, config::ColGenConfigBase, model_path::String, K::Int64=1; sptable_path::String="")
    function ClassifierLARACFilter(pb::UMFData, config::ColGenConfigBase, K::Int64=config.prconfig.K; sptable_path::String="")
        gc = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, pb.costs)
        gd = SimpleWeightedDiGraph(pb.srcnodes, pb.dstnodes, pb.latencies)
        arc_matrix = get_arc_matrix(pb)

        # shortest path table
        use_sptable::Bool = false
        sptable_c,sptable_d = nothing,nothing
	if !isnothing(config.prconfig.sptable)
	    sptable_c,sptable_d = config.prconfig.sptable
	    use_sptable = true
	end

	if !isempty(methods(config.prconfig.model))
	    # model
            _g = UMFSolver.to_gnngraph(UMFSolver.scale(pb), feature_type=Float32)
            if CUDA.functional()
	        _g = _g |> Flux.gpu
                pred = Flux.cpu(config.prconfig.model(_g))
            else
                pred = config.prconfig.model(_g)
            end

	else
	    # prediction
	    pred = config.prconfig.model
	end


        # reshaping and sending to CPU
        y = transpose(reshape(pred, numarcs(pb), numdemands(pb)))

        # sparsified graphs
        gs = Vector{SimpleWeightedDiGraph{Int64,Float64}}(undef, nk(pb))

        m = Dict{Int64, SparseVector{Bool}}()

        for k in 1:nk(pb)
            #m[k] = get_ksp_demand_mask(k, pb, config, gd, config.prconfig.K, arc_matrix)

            if !use_sptable
                m[k] = clssp_compute_masks(k, pb, gc, arc_matrix, y) .| clssp_compute_masks(k, pb, gd, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, gc, config.prconfig.K, arc_matrix)
            else
                m[k] = clssp_compute_precomputed_masks(k,pb, sptable_c, arc_matrix, y) .| clssp_compute_precomputed_masks(k,pb, sptable_d, arc_matrix, y) .| get_ksp_demand_mask(k, pb, config, gc, config.prconfig.K, arc_matrix)
            end
            gs[k] = get_edge_masked_graph(pb, m[k])
        end

        new(m, gs)
    end
end
