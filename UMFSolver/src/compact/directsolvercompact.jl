
"""
    addUMFflowconstraints(dat::UMFData, model::Model)

Add flow constraints, using data contained in `dat`, to `model`.

"""
function addUMFflowconstraints(dat::UMFData, model::Model)
    nnodes = numnodes(dat)
    narcs = numarcs(dat)
    nd = numdemands(dat)
    outgoing, incoming = out_and_in_arcs(nnodes, narcs, arcsources(dat), arcdests(dat))

    x = model[:x]
    for v = 1:nnodes, k = 1:nd
        rhs = setflowconstrrhs(dat, v, k)
        @constraint(
            model,
            sum(x[a, k] for a in outgoing[v]) - sum(x[a, k] for a in incoming[v]) == rhs
        )
    end
end

"""
    setflowconstrrhs(dat::UMFData, v::Int64, k::Int64)

Get the righ-hand-side for the flow constraint for node `v` for demand `k` with data `dat`.

"""
function setflowconstrrhs(dat::UMFData, v::Int64, k::Int64)
    if (v == demandorigin(dat, k))
        rhs = 1
    elseif (v == demanddest(dat, k))
        rhs = -1
    else
        rhs = 0
    end
    return rhs
end


"""
    createUMFcompactdirectmodel(
        dat::UMFData,
        optimizer::String,
        relaxed::Bool
    )

Create the model for a compact UMCF problem.

#Arguments:
    - `dat`: data structure containing the instance data
    - `optimizer`: type of optimizer
    - `relaxed`: if `true`, write the linear relaxation; if `false`, keep integrality constraints.

"""
function createUMFcompactdirectmodel(dat::UMFData, optimizer::String, relaxed::Bool)
    narcs = numarcs(dat)
    nd = numdemands(dat)
    # define model:
    model = createdirectmodel(optimizer)
    # set silent
    set_silent(model)
    # add variables
    @variable(model, x[1:narcs, 1:nd], Bin)
    # flow constraints:
    addUMFflowconstraints(dat, model)
    # capacity constraint:
    for a = 1:narcs
        @constraint(model, sum(bdw(dat, k) * x[a, k] for k = 1:nd) <= capacity(dat, a))
    end
    # objective function:
    @objective(
        model,
        Min,
        sum(cost(dat, a) * bdw(dat, k) * x[a, k] for a = 1:narcs, k = 1:nd)
    )
    if relaxed
        relax_integrality(model)
    end
    return model
end

"""
    printmodel(model::Model, namefile::String)

Write `model` to a text file named `namefile`.

"""
function printmodel(model::Model, namefile::String)
    write_to_file(model, namefile)
    return
end

"""
    buildandprint(
        dat::UMFData,
        optimizer::String,
        relaxed::Bool,
        filename::String
    )

Create a model `model` and write it to a text file named `filename`.

See also [`createUMFcompactdirectmodel`](@ref), [`printmodel`](@ref).

"""
function buildandprint(dat::UMFData, optimizer::String, relaxed::Bool, filename::String)
    model = createUMFcompactdirectmodel(dat, optimizer, relaxed)
    printmodel(model, filename)
    return
end

"""
    directsolveUMFcompact(
        dat::UMFData,
        optimizer::String,
        relaxed::Bool,
        verbose::Bool,
)

Create and solve the compact formulation (or its linear relaxation) of the unsplittable multicommodity flow problem defined by `dat` using the `optimizer` optimizer.  

# Arguments
- `data`: structure containing the instance data
- `optimizer`: solver to be used to solve the ILP
- `relaxed`: if `true`, the linear relaxation of the problem is solved and if `false`, the ILP is solved to optimality
- `verbose`: if `true`, write the optimal solution.

See also [`UMFData`](@ref)

"""
function directsolveUMFcompact(
    dat::UMFData,
    optimizer::String,
    relaxed::Bool,
    verbose::Bool,
)
    narcs = numarcs(dat)
    nd = numdemands(dat)
    model = createUMFcompactdirectmodel(dat, optimizer, relaxed)
    optimize!(model)
    if termination_status(model) == OPTIMAL
        if verbose
            x = model[:x]
            for k = 1:nd
                print("xopt[k$k]:\t")
                for a = 1:narcs
                    if value(x[a, k]) > 0
                        print(a, "\t")
                    end
                end
                println()
            end
        end
    else
        d = dual_status(model)
        println("The dual status is $d")
    end
    return model
end

"""
    writeresults(
        outputname::String,
        model::Model,
        dat::UMFData,
        optimizer::String,
        type::String,
        tottime::Float64,
        tot_alloc::Int64,
    )

Write the results of a solved UMCF instance to a text file.

# Arguments
- `outputname`: name for the output file,
- `model`: solved JuMP model to consider
- `dat`: structure containing the instance data
- `optimizer`: solver to be used to solve the ILP
- `type`: selected type of solution; either linear relaxation or integral solution
- `tottime`: total solution time, obtained by `@timed`
- `tot_alloc`: total allocated memory, obtained by `@timed`.

The following data are obtained from the problem instance:
- `filename`: the name of the `dat` instance
- `narcs`: the number of arcs
- `nnodes`: the number of nodes
- `nd`: the number of demands
- `status`: the termination status after solving the problem
- `objval`: the optimal value of the problem, if solved to optimality; ``0`` otherwise
- `time`: the actual solution time given by the solver.

All the results are written on a single line, separated by ";", as follows:

"`filename`;`narcs`;`nnodes`;`nd`;`optimizer`;`type`;`status`;`objval`;`time`;`tottime`;`tot_alloc`"

"""
function writeresults(
    outputname::String,
    model::Model,
    dat::UMFData,
    optimizer::String,
    type::String,
    tottime::Float64,
    tot_alloc::Int64,
)
    filename = nameprob(dat)
    narcs = numarcs(dat)
    nd = numdemands(dat)
    nnodes = numnodes(dat)
    status = termination_status(model)
    objval = 0
    if termination_status(model) == OPTIMAL
        objval = objective_value(model)
    end
    time = solve_time(model)
    io = open(outputname, "w")
    write(
        io,
        "$(filename);$(narcs);$(nnodes);$(nd);$(optimizer);$(type);$(status);$(objval);$(time);$tottime;$tot_alloc",
    )
    write(io, "\n")
    close(io)
end


"""
    solve(pb::UMFData, config::CompactConfigBase)

Solve the instance `pb` on its compact formulation and write the results to file, whose name is specified in `config`.
Options are defined by `config`.
All the result values are written on a single line, separated by ";", as follows:

"`filename`;`narcs`;`nnodes`;`nd`;`optimizer`;`type`;`status`;`objval`;`time`;`tottime`;`tot_alloc`"

where the following data are obtained after solving the instance:
- `filename`: the name of the `pb` instance
- `narcs`: the number of arcs
- `nnodes`: the number of nodes
- `nd`: the number of demands
- `status`: the termination status after solving the problem
- `objval`: the optimal value of the problem, if solved to optimality; ``0`` otherwise
- `time`: the actual solution time given by the solver;
- `tottime`: total solution time, obtained with `@timed`
- `tot_alloc`: total allocated memory, obtained with `@timed`;

the following are obtained from the configuration file:
- `outputname`: name for the output file,
- `optimizer`: solver to be used to solve the ILP
- `type`: selected type of solution; either linear relaxation or integral solution.

Return the optimal value of the problem, if solved to optimality, and ``0`` otherwise

See also [`CompactConfigBase`](@ref), [`UMFData`](@ref)

"""
function solve(dat::UMFData, config::CompactConfigBase)

    t = @timed directsolveUMFcompact(dat, config.optimizer, config.relaxed, false)
    model = t.value
    tottime = t.time
    totalloc = t.bytes
    type = config.relaxed ? "relax" : "int"
    writeresults(config.outputname, model, dat, config.optimizer, type, tottime, totalloc)
    if termination_status(model) == OPTIMAL
        return objective_value(model)
    else
        return 0
    end
end
