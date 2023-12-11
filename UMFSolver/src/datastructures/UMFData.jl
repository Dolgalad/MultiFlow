include("routines_graphdata.jl")
#using Random
#using Distributions

"""
    UMFData

Data structure for an instance of Unsplittable MCF.

It has three constructors:

    UMFData(
        name::String,
        srcnodes::Vector{Int64},
        dstnodes::Vector{Int64},
        capacities::Vector{<:Real},
        costs::Vector{<:Real},
        latencies::Vector{<:Real},
        srcdemands::Vector{Int64},
        dstdemands::Vector{Int64},
        bandwidths::Vector{<:Real},
        demand_latencies::Vector{<:Real},
        ndemands::Int64,
        nnodes::Int64,
        narcs::Int64,
    )

    Default constructor.

    # Arguments

    - `name`: name of the instance
    - `srcnodes`: vector of source nodes for the arcs
    - `dstnodes`: vector of destination nodes for the arcs
    - `capacities`: vector of arc capacities
    - `costs`: vector of arc costs
    - `latencies` : vector of arc latencies
    - `srcdemands`: vector of source nodes for the demands
    - `dstdemands`: vector of destination nodes for the demands
    - `bandwidths`: vector of demand bandwidths
    - `demand_latencies` : vector of demand latencies
    - `ndemands`: number of demands
    - `nnodes`: number of nodes of the graph
    - `narcs`: number of arcs of the graph.


    UMFData(
        linkcsvfile::String,
        servicecsvfile::String,
        namedata::String
    )

    Construct the data starting from the two .csv files.

    # Arguments

    - `linkcsvfile`: csv file with data information on the graph
    - `servicecsvfile`: csv file with data information on the demands
    - `namedata`: name for the UMFData instance.

    UMFData(instance_path::String)

    Construct the data from <instance_path>/link.csv and <instance_path>/service.csv.

"""
struct UMFData <: AbstractProblemData
    name::String
    srcnodes::Vector{Int64}
    dstnodes::Vector{Int64}
    capacities::Vector{Float64}
    costs::Vector{Float64}
    latencies::Vector{Float64}
    srcdemands::Vector{Int64}
    dstdemands::Vector{Int64}
    bandwidths::Vector{Float64}
    demand_latencies::Vector{Float64}
    ndemands::Int64
    nnodes::Int64
    narcs::Int64
    #constructor from directory
    function UMFData(instance_dir::String; edge_dir=:double)
        linkpath=joinpath(instance_dir, "link.csv")
        servicepath=joinpath(instance_dir, "service.csv")
        name=dirname(linkpath)
        return UMFData(linkpath,servicepath,name,edge_dir=edge_dir)
    end
    #constructor from provided csv files
    function UMFData(linkcsvfile::String, servicecsvfile::String, namedata::String; edge_dir=:double)
        dflinks = CSV.read(linkcsvfile, DataFrame)
        rename!(dflinks, strip.(lowercase.(names(dflinks))))
        dfservices = CSV.read(servicecsvfile, DataFrame)
        rename!(dfservices, strip.(lowercase.(names(dfservices))))

        # store arrays for arcs
        #srcnodes = dflinks." srcNodeId"
        #dstnodes = dflinks." dstNodeId"
        srcnodes = dflinks.srcnodeid
        dstnodes = dflinks.dstnodeid

        capacities = dflinks.bandwidth
        costs = dflinks.cost
        latencies = dflinks.latency
        narcs = size(dflinks, 1) # number of arcs
        name = namedata

        #store arrays for demands
        srcdemands = dfservices.srcnodeid
        dstdemands = dfservices.dstnodeid
        bandwidths = dfservices.bandwidth
        demand_latencies = dfservices.latency
        ndemands = size(dfservices, 1)# number of demands

        #adjust indices for arcs and demands (assert not to start with 0)
        if !(srcdemands isa Vector{Int64}) || !(dstdemands isa Vector{Int64})
            srcdemands = Int64[s for s in srcdemands]
            dstdemands = Int64[s for s in dstdemands]
        end
        checkindices4!(srcnodes, dstnodes, srcdemands, dstdemands)

        #find number of nodes (now they are starting from 1)
        nnodes = getnummodes(srcnodes, dstnodes)

        #check if arcs are single or double.
        outgoing, incoming = out_and_in_arcs(nnodes, narcs, srcnodes, dstnodes)
        doubled::Bool = are_arcs_doubled(nnodes, outgoing, dstnodes)
        # println(doubled)

        # if single, double arcs for they are given in both directions
        if !doubled && edge_dir==:double
            narcs = doublearcs!(srcnodes, dstnodes, capacities, costs, latencies, narcs)
        end
        #narcs = size(srcnodes, 1)#number of arcs
        # construct structure:
        new(
            name,
            srcnodes,
            dstnodes,
            capacities,
            costs,
            latencies,
            srcdemands,
            dstdemands,
            bandwidths,
            demand_latencies,
            ndemands,
            nnodes,
            narcs,
        )
    end

    #default constructor
    function UMFData(
        name::String,
        srcnodes::Vector{Int64},
        dstnodes::Vector{Int64},
        capacities::Vector{<:Real},
        costs::Vector{<:Real},
        latencies::Vector{<:Real},
        srcdemands::Vector{Int64},
        dstdemands::Vector{Int64},
        bandwidths::Vector{<:Real},
        demand_latencies::Vector{<:Real},
        ndemands::Int64,
        nnodes::Int64,
        narcs::Int64,
    )
        new(
            name,
            srcnodes,
            dstnodes,
            capacities,
            costs,
            latencies,
            srcdemands,
            dstdemands,
            bandwidths,
            demand_latencies,
            ndemands,
            nnodes,
            narcs,
        )
    end
end

"""
    nameprob(dat::UMFData)

Get the name from the instance `dat`.

"""
function nameprob(dat::UMFData)
    return dat.name
end

"""
    arcsource(dat::UMFData, a::Int64)

Get the source node of arc `a` for instance `dat`.

"""
function arcsource(dat::UMFData, a::Int64)
    return dat.srcnodes[a]
end

"""
    arcsources(dat::UMFData)

Get the vector of source nodes of instance `dat`.

"""
function arcsources(dat::UMFData)
    return dat.srcnodes
end

"""
    arcdest(dat::UMFData, a::Int64)

Get the destination node of arc `a` for instance `dat`.

"""
function arcdest(dat::UMFData, a::Int64)
    return dat.dstnodes[a]
end

"""
    arcdests(dat::UMFData)

Get the vector of destination nodes of instance `dat`.

"""
function arcdests(dat::UMFData)
    return dat.dstnodes
end

"""
    capacity(dat::UMFData, a::Int64)

Get the capacity of arc `a` for instance `dat`.
    
"""
function capacity(dat::UMFData, a::Int64)
    return dat.capacities[a]
end

"""
    capacities(dat::UMFData)

Get the vector of capacities of instance `dat`.

"""
function capacities(dat::UMFData)
    return dat.capacities
end

"""
    cost(dat::UMFData, a::Int64)

Get the cost of arc `a` for instance `dat`.
    
"""
function cost(dat::UMFData, a::Int64)
    return dat.costs[a]
end

"""
    costs(dat::UMFData)

Get the vector of costs of instance `dat`.

"""
function costs(dat::UMFData)
    return dat.costs
end

"""
    demandorigin(dat::UMFData, k::Int64)

Get the origin node of demand `k` for instance `dat`.

"""
function demandorigin(dat::UMFData, k::Int64)
    return dat.srcdemands[k]
end

"""
    demandorigins(dat::UMFData)

Get the vector of origin nodes of instance `dat`.

"""
function demandorigins(dat::UMFData)
    return dat.srcdemands
end

"""
    demanddest(dat::UMFData, k::Int64)

Get the destination node of demand `k` for instance `dat`.

"""
function demanddest(dat::UMFData, k::Int64)
    return dat.dstdemands[k]
end

"""
    demanddests(dat::UMFData)

Get the vector of destination nodes of instance `dat`.

"""
function demanddests(dat::UMFData)
    return dat.dstdemands
end

"""
    bdw(dat::UMFData, k::Int64)

Get the bandwidth of demand `k` for instance `dat`.

"""
function bdw(dat::UMFData, k::Int64)
    return dat.bandwidths[k]
end

"""
    bdws(dat::UMFData)

Get the vector of bandwidths of instance `dat`.

"""
function bdws(dat::UMFData)
    return dat.bandwidths
end

"""
   numdemands(dat::UMFData)

Get the number of demands of instance `dat`.

"""
function numdemands(dat::UMFData)
    return dat.ndemands
end

"""
   numnodes(dat::UMFData)

Get the number of nodes of instance `dat`.

"""
function numnodes(dat::UMFData)
    return dat.nnodes
end

"""
   numarcs(dat::UMFData)

Get the number of arcs of instance `dat`.

"""
function numarcs(dat::UMFData)
    return dat.narcs
end

"Get a new instance with capacities multiplied by f."

"""
    scale_capacity(dat::UMFData, f::Real)

Construct an `UMFData` instance which has the same characteristics of `dat`, except that capacities multiplied by the factor `f`.

"""
function scale_capacity(dat::UMFData, f::Real)
    dataname = nameprob(dat) * "_factor$f"
    srcns = arcsources(dat)
    dstns = arcdests(dat)
    caps::Vector{Float64} = capacities(dat)
    caps *= f
    csts = costs(dat)
    srcdems = demandorigins(dat)
    dstdems = demanddests(dat)
    bandws = bdws(dat)
    ndems = numdemands(dat)
    nnods = numnodes(dat)
    n_arcs = numarcs(dat)
    return UMFData(
        dataname,
        srcns,
        dstns,
        caps,
        csts,
        dat.latencies,
        srcdems,
        dstdems,
        bandws,
        dat.demand_latencies,
        ndems,
        nnods,
        n_arcs,
    )
end

function scale_bandwidths(dat::UMFData, f::Real)
    dataname = nameprob(dat) * "_factorbwd$f"
    srcns = arcsources(dat)
    dstns = arcdests(dat)
    caps = capacities(dat)
    csts = costs(dat)
    srcdems = demandorigins(dat)
    dstdems = demanddests(dat)
    bandws = bdws(dat)
    bandws *= f
    ndems = numdemands(dat)
    nnods = numnodes(dat)
    n_arcs = numarcs(dat)
    return UMFData(
        dataname,
        srcns,
        dstns,
        caps,
        csts,
        dat.latencies,
        srcdems,
        dstdems,
        bandws,
        dat.demand_latencies,
        ndems,
        nnods,
        n_arcs,
    )
end


function scale(dat::UMFData)
    dataname = nameprob(dat)*"_scaled"
    srcns = arcsources(dat)
    dstns = arcdests(dat)
    caps_f = maximum(dat.capacities)
    caps::Vector{Float64} = capacities(dat)
    bandws::Vector{Float64} = bdws(dat)
    latencies::Vector{Float64} = dat.latencies


    caps /= caps_f
    bandws /= caps_f
    csts = costs(dat)
    csts_f = maximum(csts)
    csts /= csts_f
    lats_f = maximum(latencies)
    srcdems = demandorigins(dat)
    dstdems = demanddests(dat)
    #bandws = bdws(dat)
    ndems = numdemands(dat)
    nnods = numnodes(dat)
    n_arcs = numarcs(dat)
    return UMFData(
        dataname,
        srcns,
        dstns,
        caps,
        csts,
        lats_f > 0 ? dat.latencies / lats_f : dat.latencies,
        srcdems,
        dstdems,
        bandws,
        dat.demand_latencies / lats_f,
        ndems,
        nnods,
        n_arcs,
    )
end

# TODO: make sure that the order is conserved for links and services and 
# make sure that arcs are always doubled ?
function save(inst::UMFData, dirname::String; verbose::Bool=true)
    link_filename = joinpath(dirname, "link.csv")
    service_filename = joinpath(dirname, "service.csv")
    # link dataframe
    link_df = DataFrame(srcNodeId=inst.srcnodes, dstNodeId=inst.dstnodes, cost=inst.costs, bandwidth=inst.capacities, latency=inst.latencies)
    #link_df = unique(link_df)
    service_df = DataFrame(srcNodeId=inst.srcdemands, dstNodeId=inst.dstdemands, bandwidth=inst.bandwidths, latency=inst.demand_latencies)
    if verbose
        println("Saving instance to $dirname")
    end
    mkpath(dirname)
    CSV.write(link_filename, link_df)
    CSV.write(service_filename, service_df)
end


"""
    is_instance_path(path::String)

Check if path contains an instance
"""
function is_instance_path(path::String)
    link_file=joinpath(path,"link.csv")
    service_file=joinpath(path,"service.csv")

    return isdir(path) && isfile(link_file) && isfile(service_file)
end

function summary(inst::UMFData)
    println("Number of arcs   : ", numarcs(inst))
    println("Number of nodes  : ", numnodes(inst))
    println("Number of demands: ", numdemands(inst))
end

function has_nan(inst::UMFData)
    return any(isnan.(inst.bandwidths)) || any(isnan.(inst.srcdemands)) || any(isnan.(inst.dstdemands))
end

function has_inf(inst::UMFData)
    return any(isinf.(inst.bandwidths)) || any(isinf.(inst.srcdemands)) || any(isinf.(inst.dstdemands))
end

"""
    has_latency_constraint(dat::UMFData)

Check if instance has a latency constraint
"""
function has_latency_constraint(dat::UMFData)
    return any(.! isinf.(dat.demand_latencies))
end

"""
    get_graph(dat::UMFData)

Construct graph
"""
function get_graph(dat::UMFData)
    g = Graphs.SimpleDiGraph(nv(dat))
    for a in 1:ne(dat)
        add_edge!(g, dat.srcnodes[a], dat.dstnodes[a])
    end
    return g
end

function get_cost_matrix(dat::UMFData)
    m = zeros(nv(dat), nv(dat))
    for a in 1:ne(dat)
        m[dat.srcnodes[a], dat.dstnodes[a]] = dat.costs[a]
    end
    return m
end

function get_latency_matrix(dat::UMFData)
    m = zeros(nv(dat), nv(dat))
    for a in 1:ne(dat)
        m[dat.srcnodes[a], dat.dstnodes[a]] = dat.latencies[a]
    end
    return m
end
