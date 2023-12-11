using Distributions#, Shuffle
using BenchmarkTools:@elapsed, @timed


"""
    random_demand_bandwidths(inst::UMFData)
random bandwidth values.
"""
function random_demand_bandwidths(inst::UMFData; factor=1., ndemands=numdemands(inst))
    bdw_type = eltype(inst.bandwidths)
    return bdw_type.(rand(inst.bandwidths, ndemands)) * factor


    bdw_mean = bdw_type(mean(inst.bandwidths))
    bdw_std = bdw_type(std(inst.bandwidths))
    bdw_min, bdw_max = bdw_type(minimum(inst.bandwidths)), bdw_type(maximum(inst.bandwidths))
    dist = Normal(bdw_mean, bdw_std)
    if bdw_std != 0
        dist = truncated(dist, bdw_min, bdw_max)
    end
    return bdw_type.(rand(dist, ndemands)) * factor
end

"""
    random_demand_latencies(inst::UMFData)
random latencies values.
"""
function random_demand_latencies(inst::UMFData; factor=1., ndemands=numdemands(inst))
    bdw_type = eltype(inst.demand_latencies)
    bdw_mean = bdw_type(mean(inst.demand_latencies))
    bdw_std = bdw_type(std(inst.demand_latencies))
    bdw_min, bdw_max = bdw_type(minimum(inst.demand_latencies)), bdw_type(maximum(inst.demand_latencies))
    dist = Normal(bdw_mean, bdw_std)
    if bdw_std != 0
        dist = truncated(dist, bdw_min, bdw_max)
    end
    return bdw_type.(rand(dist, ndemands)) * factor

end

"""
    random_demand_endpoints(inst::UMFData)

Select a new set of demand origins and targets
"""
function random_demand_endpoints(inst::UMFData; ndemands = numdemands(inst))
    new_origins = rand(inst.srcdemands, ndemands)
    new_targets = rand(inst.dstdemands, ndemands)
    idx = (new_origins .== new_targets)
    while any(idx)
        new_targets[idx] = rand(inst.dstdemands, sum(idx))
        idx = (new_origins .== new_targets)
    end
    return new_origins, new_targets
end

"""
    fully_random_demand_endpoints(inst::UMFData)

Select a new set of demand origins and targets
"""
function fully_random_demand_endpoints(inst::UMFData; ndemands = numdemands(inst))
    new_origins = rand(1:nv(inst), ndemands)
    new_targets = rand(1:nv(inst), ndemands)
    idx = (new_origins .== new_targets)
    while any(idx)
        new_targets[idx] = rand(1:nv(inst), sum(idx))
        idx = (new_origins .== new_targets)
    end
    return new_origins, new_targets
end



"""
    shake_instance(inst::UMFData)

Apply bandwidth and demand endpoint perturbation to an UMFData instance.
"""
function shake_instance(inst::UMFData; bdw_factor=1., latency_factor=1.0, ndemands=numdemands(inst))
    new_bdw = random_demand_bandwidths(inst; factor=bdw_factor, ndemands=ndemands)
    new_lat = random_demand_latencies(inst; factor=latency_factor, ndemands=ndemands)
    new_demand_src, new_demand_trg = random_demand_endpoints(inst; ndemands=ndemands)
    return UMFData(
        inst.name * "_shook",
        inst.srcnodes,
        inst.dstnodes,
        inst.capacities,
        inst.costs,
        inst.latencies,
        new_demand_src,
        new_demand_trg,
        new_bdw,
        new_lat,
        ndemands,
        numnodes(inst),
        numarcs(inst),
    )
end

"""
    shake_instance_2(inst::UMFData)

Apply bandwidth and demand endpoint perturbation to an UMFData instance. Fully random endpoints.
"""
function shake_instance_2(inst::UMFData; bdw_factor=1., latency_factor=1.0, ndemands=numdemands(inst))
    new_bdw = random_demand_bandwidths(inst; factor=bdw_factor, ndemands=ndemands)
    new_lat = random_demand_latencies(inst; factor=latency_factor, ndemands=ndemands)
    new_demand_src, new_demand_trg = fully_random_demand_endpoints(inst; ndemands=ndemands)
    return UMFData(
        inst.name * "_shook",
        inst.srcnodes,
        inst.dstnodes,
        inst.capacities,
        inst.costs,
        inst.latencies,
        new_demand_src,
        new_demand_trg,
        new_bdw,
        new_lat,
        ndemands,
        numnodes(inst),
        numarcs(inst),
    )
end


function shake_instance_with_aggregated_demands(inst::UMFData; bdw_factor=1., ndemands=numdemands(inst))
    new_bdw = rand(inst.bandwidths, ndemands)
    new_demand_src, new_demand_trg = random_demand_endpoints(inst; ndemands=ndemands)
    return UMFData(
        inst.name * "_shook_agg",
        inst.srcnodes,
        inst.dstnodes,
        inst.capacities,
        inst.costs,
        inst.latencies,
        new_demand_src,
        new_demand_trg,
        new_bdw,
        inst.demand_latencies,
        ndemands,
        numnodes(inst),
        numarcs(inst),
    )
end


function available_capacities(inst::UMFData, ms)
    columns = UMFSolver.allcolumns(ms)
    flows = UMFSolver.getx(ms)
    caps = copy(inst.capacities)
    for k in 1:numdemands(inst)
        for i in eachindex(columns[k])
            for a in columns[k][i]
                caps[a] -= flows[k][i] * inst.bandwidths[k]
            end
        end
    end
    return caps
end

path_capacity(path, capacities) = minimum(capacities[a] for a in path)

function non_saturated_path_exists(available_capacities, columns; tol=1e-3)
    for k in eachindex(columns)
        for p in eachindex(columns[k])
            if path_capacity(columns[k][p], available_capacities) > tol
                return true
            end
        end
    end
    return false
end

function saturate_instance(inst::UMFData; tol=1e-3)
    # solve with column generation
    (sol, ss) = solveUMF(inst, "CG", "cplex", "./output.txt")
    ms = ss.stats["ms"]
    # all columns
    columns = UMFSolver.allcolumns(ms)
    # fractional flow per column
    flows = UMFSolver.getx(ms)
    # available capacities 
    available_caps = available_capacities(inst, ms)
    # new bandwidths
    new_bdw = copy(inst.bandwidths)
    # keep list of demands 
    demand_paths = []
    for k in eachindex(columns)
        for p in eachindex(columns[k])
            push!(demand_paths, (k,p))
        end
    end
    demand_paths = Set(demand_paths)

    #while non_saturated_path_exists(available_caps, columns, tol=tol)
    while !isempty(demand_paths)
        (k,p) = rand(demand_paths)
        if size(columns[k],1)==0
            println("Poping ", (k,p))
            pop!(demand_paths, (k,p))
            continue
        end
        #p = rand(1:size(columns[k], 1))
        # check if path capacity if greater than 0
        s = path_capacity(columns[k][p], available_caps)
        if s > tol
            if s < 1
                bdw_delta = s
            else
                dist = Uniform(0, s)
                bdw_delta = rand(dist)
            end
            new_bdw[k] += bdw_delta

            # update available capacities
            for a in columns[k][p]
                available_caps[a] -= bdw_delta
            end
        end
        # pop (k,p) if paths are saturated
        if path_capacity(columns[k][p], available_caps) < tol
            #println("poping : ", (k,p))
            pop!(demand_paths, (k,p))
        end
    end
    return UMFData(
        inst.name * "_saturated",
        inst.srcnodes,
        inst.dstnodes,
        inst.capacities,
        inst.costs,
        inst.latencies,
        inst.srcdemands,
        inst.dstdemands,
        new_bdw,
        inst.demand_latencies,
        numdemands(inst),
        numnodes(inst),
        numarcs(inst),
    )
end

function generate_example(inst::UMFData; demand_p=.05, bdw_factor=1.05)
    ndemands = numdemands(inst)
    # shake
    inst_t = shake_instance(inst)
    # saturate
    inst_s = saturate_instance(inst_t)

    # increase demand
    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
    new_bdw = copy(inst_s.bandwidths)
    new_bdw[demand_idx] *= bdw_factor
    return UMFData(
        inst_s.name * "_generated",
        inst_s.srcnodes,
        inst_s.dstnodes,
        inst_s.capacities,
        inst_s.costs,
        inst_s.latencies,
        inst_s.srcdemands,
        inst_s.dstdemands,
        new_bdw,
        inst_s.demand_latencies,
        numdemands(inst),
        numnodes(inst),
        numarcs(inst),
    )

end

"""Generates examples, changes the number of demands
"""
function generate_example_2(inst::UMFData; demand_p=.05, bdw_factor=1.05, demand_delta_p=0.1)
    delta_range = range(floor(Int64, -demand_delta_p * nk(inst)), floor(Int64, demand_delta_p*nk(inst)))
    ndemands = numdemands(inst) + rand(delta_range)
    # shake
    inst_t = shake_instance(inst, ndemands=ndemands)
    # saturate
    inst_s = saturate_instance(inst_t)

    # increase demand
    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
    new_bdw = copy(inst_s.bandwidths)
    new_bdw[demand_idx] *= bdw_factor
    return UMFData(
        inst_s.name * "_generated",
        inst_s.srcnodes,
        inst_s.dstnodes,
        inst_s.capacities,
        inst_s.costs,
        inst_s.latencies,
        inst_s.srcdemands,
        inst_s.dstdemands,
        new_bdw,
        inst_s.demand_latencies,
        numdemands(inst_t),
        numnodes(inst_t),
        numarcs(inst_t),
    )

end

"""Generates examples, changes the number of demands
"""
function generate_example_3(inst::UMFData; demand_p=.05, bdw_factor=1.05, demand_delta_p=0.1)
    delta_range = range(floor(Int64, -demand_delta_p * nk(inst)), floor(Int64, demand_delta_p*nk(inst)))
    ndemands = numdemands(inst) + rand(delta_range)
    # shake
    inst_t = shake_instance_2(inst, ndemands=ndemands)
    # saturate
    inst_s = saturate_instance(inst_t)

    # increase demand
    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
    new_bdw = copy(inst_s.bandwidths)
    new_bdw[demand_idx] *= bdw_factor
    return UMFData(
        inst_s.name * "_generated",
        inst_s.srcnodes,
        inst_s.dstnodes,
        inst_s.capacities,
        inst_s.costs,
        inst_s.latencies,
        inst_s.srcdemands,
        inst_s.dstdemands,
        new_bdw,
        inst_s.demand_latencies,
        numdemands(inst_t),
        numnodes(inst_t),
        numarcs(inst_t),
    )

end

"""Generates examples, changes the number of demands
"""
function generate_example_4(inst::UMFData; demand_p=.05, bdw_factor=1.05, ndemands=numdemands(inst))
    # shake
    inst_t = shake_instance_2(inst, ndemands=ndemands)
    # saturate
    inst_s = saturate_instance(inst_t)

    # increase demand
    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
    new_bdw = copy(inst_s.bandwidths)
    new_bdw[demand_idx] *= bdw_factor
    return UMFData(
        inst_s.name * "_generated",
        inst_s.srcnodes,
        inst_s.dstnodes,
        inst_s.capacities,
        inst_s.costs,
        inst_s.latencies,
        inst_s.srcdemands,
        inst_s.dstdemands,
        new_bdw,
        inst_s.demand_latencies,
        numdemands(inst_t),
        numnodes(inst_t),
        numarcs(inst_t),
    )

end


function generate_example_with_aggregated_demands(inst::UMFData; demand_p=.05, bdw_factor=1.05)
    # shake
    inst_t = shake_instance_with_aggregated_demands(inst)
    # saturate
    inst_s = saturate_instance(inst_t)

    # increase demand
    ndemands = numdemands(inst_s)
    nselecteddemands = Int64(max(1, trunc(demand_p * ndemands)))
    demand_idx = sample(1:ndemands, nselecteddemands, replace=false)
    new_bdw = copy(inst_s.bandwidths)
    new_bdw[demand_idx] *= bdw_factor
    return UMFData(
        inst_s.name * "_generated",
        inst_s.srcnodes,
        inst_s.dstnodes,
        inst_s.capacities,
        inst_s.costs,
        inst_s.srcdemands,
        inst_s.dstdemands,
        new_bdw,
        numdemands(inst_s),
        numnodes(inst_s),
        numarcs(inst_s),
    )

end


