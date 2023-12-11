using Graphs
using GraphNeuralNetworks
"""
    AugmentedGraph
"""
struct AugmentedGraph
    #graph::SimpleDiGraph
    edge_src::Vector{Int64}
    edge_dst::Vector{Int64}
    edge_features::Matrix{Float64}
    node_mask::Vector{Bool}
    edge_mask::Vector{Bool}
end


"""
    digraph(inst::UMFData)

Converts a UMFData object to a SimpleDiGraph
"""
function digraph(inst::UMFData)
    g = SimpleDiGraph(numnodes(inst))
    for i in 1:size(inst.srcnodes,1)
        add_edge!(g, inst.srcnodes[i], inst.dstnodes[i])
    end
    return g
end

"""
    augmented_graph(inst::UMFData)

Converts a UMFData object to DemandGraph i.e. a SimpleDiGraph with a node added for every demand, an edge linking its source node to this demand node and an edge emanating from the demand node towards the destination of the demand.
"""
function augmented_graph(inst::UMFData)
    # create a SimpleDiGraph
    #g = digraph(inst)
    
    # create the DemandGraph
    n,m,k = numnodes(inst),numarcs(inst),numdemands(inst)
    redge_features_size = (4, m)
    redge_features = zeros(Float64, redge_features_size)
    node_mask = zeros(Bool, n+k)
    node_mask[1:n] .= 1
    edge_mask = zeros(Bool, m+2*k)
    edge_mask[1:m] .= 1
    
    redge_features[1,:] = inst.costs
    redge_features[2,:] = inst.capacities
    redge_features[3,:] = inst.latencies

    # demand nodes -> demand source
    demand_nodes = collect(1:k) .+ n
    dnto_src, dnto_dst = demand_nodes, inst.srcdemands
    dnto_features = zeros(Float64, (4, k))
    #dnto_features[3,:] .= inst.demand_latencies
    dnto_features[4,:] .= inst.bandwidths
    # demand target -> demand node
    dntt_src, dntt_dst = inst.dstdemands, demand_nodes
    dntt_features = zeros(Float64, (4, k))
    #dntt_features[3,:] .= -inst.demand_latencies
    dntt_features[4,:] .= -inst.bandwidths

    edge_src = vcat(inst.srcnodes, dnto_src, dntt_src)
    edge_dst = vcat(inst.dstnodes, dnto_dst, dntt_dst)

    edge_features = hcat(redge_features, dnto_features, dntt_features)

    return AugmentedGraph(edge_src, edge_dst, edge_features, node_mask, edge_mask)
end

"""
    get_instance(g::AugmentedGraph)

Convert an AugmentedGraph object to UMFData
"""
function get_instance(ag::AugmentedGraph)
    edge_src, edge_dst = [],[]
    demand_src, demand_dst = [],[]
    costs, capacities = [],[]
    bandwidths = []
    bandwidth_map = Dict() # keep dictionary of demand bandwidths

    nrealnodes = sum(ag.node_mask)
    nrealedges = sum(ag.edge_mask)
    ndemands = size(ag.node_mask,1) - nrealnodes

    srcnodes = ag.edge_src[1:nrealedges]
    dstnodes = ag.edge_dst[1:nrealedges]
    costs = ag.edge_features[1,1:nrealedges]
    capacities = ag.edge_features[2,1:nrealedges]
    latencies = ag.edge_features[3,1:nrealedges]

    demandsrc = ag.edge_dst[nrealedges+1:nrealedges+ndemands]
    demanddst = ag.edge_src[nrealedges+1+ndemands:end]
    bandwidths = ag.edge_features[4,nrealedges+1:nrealedges+ndemands]
    demand_latencies = ag.edge_features[3,nrealedges+1:nrealedges+ndemands]
    return UMFData("",srcnodes,dstnodes,capacities,costs,latencies,demandsrc,demanddst,bandwidths,demand_latencies,
                   ndemands,nrealnodes,nrealedges)
end

"""
    to_gnngraph(inst::UMFData)

Convert a UMFData instance to GNNGraph object
"""
function to_gnngraph(inst::UMFData; feature_type=eltype(inst.costs))
    # start by creating the augmented graph
    ag = augmented_graph(inst)
    
    k = nk(inst)
    demand_bandwidths_mask = .!ag.edge_mask
    demand_bandwidths_mask[findall(==(1), demand_bandwidths_mask)[k+1:end]] .= 0

    # demand to source edge mask
    demand_to_source_mask = zeros(Bool, ne(inst)+2*nk(inst))
    target_to_demand_mask = zeros(Bool, ne(inst)+2*nk(inst))

    demand_to_source_mask[ne(inst)+1:ne(inst)+nk(inst)] .= 1
    target_to_demand_mask[ne(inst)+nk(inst)+1:end] .= 1

    return GNNGraph(ag.edge_src, ag.edge_dst,
               ndata=(;mask=ag.node_mask),
               edata=(;e=feature_type.(ag.edge_features), 
		       mask=ag.edge_mask,
		       demand_bandwidths_mask=demand_bandwidths_mask,
		       demand_to_source_mask=demand_to_source_mask,
		       target_to_demand_mask=target_to_demand_mask
		     ),
               gdata=(;K=nk(inst),
                       E=ne(inst),
		      )
           )
end

function tempf(k)
    dev = CUDA.functional() ? Flux.gpu : Flux.cpu
    return vcat(ones(Bool, k), zeros(Bool, k)) |> dev
end

"""
    to_gnngraph(inst::UMFData, y::AbstractMatrix)

Convert a UMFData instance to GNNGraph object with target labels
"""
function to_gnngraph(inst::UMFData, y::AbstractMatrix; feature_type=eltype(inst.costs))
    # start by creating the augmented graph
    ag = augmented_graph(inst)

    k = nk(inst)
    demand_bandwidths_mask = .!ag.edge_mask
    demand_bandwidths_mask[findall(==(1), demand_bandwidths_mask)[k+1:end]] .= 0

    # demand to source edge mask
    demand_to_source_mask = zeros(Bool, ne(inst)+2*nk(inst))
    target_to_demand_mask = zeros(Bool, ne(inst)+2*nk(inst))

    demand_to_source_mask[ne(inst)+1:ne(inst)+nk(inst)] .= 1
    target_to_demand_mask[ne(inst)+nk(inst)+1:end] .= 1

    g=GNNGraph(ag.edge_src, ag.edge_dst,
               ndata=(;mask=ag.node_mask),
               edata=(;e=feature_type.(ag.edge_features), 
		       mask=ag.edge_mask,
		       demand_bandwidths_mask=demand_bandwidths_mask,
		       demand_to_source_mask=demand_to_source_mask,
		       target_to_demand_mask=target_to_demand_mask
		     ),
               gdata=(;targets=y, K=nk(inst), E=ne(inst))
           )
    return g
end




function demand_endpoints(g::GNNGraph)
    s,t = edge_index(g)

    # offset due to demand nodes
    offsets = cumsum(g.K) .- g.K
    gie = graph_indicator(g, edges=true)
    if g.num_graphs==1
        ds_t,dt_t = t[g.demand_to_source_mask], s[g.target_to_demand_mask]
    else
        ds_t,dt_t = (t .- offsets[gie])[g.demand_to_source_mask], (s .- offsets[gie])[g.target_to_demand_mask]
    end
    return ds_t,dt_t

    if g.num_graphs==1
        ndemands = sum(.!g.ndata.mask)
        ds,dt = t[.!g.edata.mask][1:ndemands], s[.!g.edata.mask][ndemands+1:end]
	#println("ds == ds_t : ", ds == ds_t)
	#println("dt == dt_t : ", dt == dt_t)
	return ds,dt
    else
        gin = graph_indicator(g)
        ninodes = sum(gin .== 1)

        ndemands = sum((.!g.ndata.mask) .& (gin .== 1)) # same number of demands in each instance
        #ds,dt = vcat([t[(.!g.edata.mask) .& (gie .== i)][1:ndemands] .- (i-1)*ndemands for i in 1:g.num_graphs]...),
        #vcat([s[(.!g.edata.mask) .& (gie .== i)][ndemands+1:end] .- (i-1)*ndemands for i in 1:g.num_graphs]...)
        ds,dt = vcat([t[(.!g.edata.mask) .& (gie .== i)][1:g.K[i]] .- sum(g.K[1:(i-1)]) for i in 1:g.num_graphs]...),
        vcat([s[(.!g.edata.mask) .& (gie .== i)][g.K[i]+1:end] .- sum(g.K[1:(i-1)]) for i in 1:g.num_graphs]...)

	#println("ds == ds_t : ", ds == ds_t)
	#println("dt == dt_t : ", dt == dt_t)

        return ds, dt
    end
end

function demand_bandwidths(g::GNNGraph)
    return g.e[4, g.edata.demand_bandwidths_mask]
    if g.num_graphs == 1
        ndemands = sum(.!g.ndata.mask)

        #return g.e[:,.!g.edata.mask][4,1:g.K[1]]
        #idx = .!g.edata.mask
        #idx[findall(==(1), idx)[ndemands+1:end]] .= 0
        #return view(g.e,4,idx)
        return view(g.e,4,.!g.edata.mask)[1:ndemands] # best

        return g.e[:,.!g.edata.mask][4,1:ndemands] # best

    else
        #gin = graph_indicator(g)
        gie = graph_indicator(g, edges=true)
        #return vcat([g.e[:,(.!g.edata.mask) .& (gie .== i)][4,1:ndemands] for i in 1:g.num_graphs]...)

        #r = vcat([g.e[:,(.!g.edata.mask) .& (gie .== i)][4,1:g.K[i]] for i in 1:g.num_graphs]...)
        #r = reduce(vcat,[g.e[:,(.!g.edata.mask) .& (gie .== i)][4,1:g.K[i]] for i in 1:g.num_graphs])
	#println(typeof([4,1:g.K[1]])," ", typeof(g.K[1]), " ", typeof(1:g.K[1]))
        r = reduce(vcat,[g.e[4,(.!g.edata.mask) .& (gie .== i)] for i in 1:g.num_graphs])

	aaa = vcat([tempf(g.K[i]) for i in 1:g.num_graphs]...)
	#r_ind = reduce(vcat, aaa)
	re = getobs(r, aaa)
	return re

    end
end


# combine labels for demands sharing the same origin and destination vertices
function aggregate_demand_paths(g::GNNGraph)
    ndemands = sum(.!g.ndata.mask)
    ds,dt = UMFSolver.demand_endpoints(g)
    new_labels = Dict()
    for k in 1:ndemands
        if haskey(new_labels, (ds[k], dt[k]))
            new_labels[(ds[k], dt[k])] .|= g.targets[:,k]
        else
            new_labels[(ds[k], dt[k])] = g.targets[:,k]
        end
    end
    for k in 1:ndemands
        g.targets[:,k] .= new_labels[(ds[k], dt[k])]
    end
    return g
end

