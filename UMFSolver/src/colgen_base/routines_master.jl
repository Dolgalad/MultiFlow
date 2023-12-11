# Functions for the master problem in the column generation.

# Columns (corresponding to paths) are considered as the lists of arcs in the path.


"""
    computebigM(ms::UMFLinearMasterData)

Compute bigM value, to be used for artificial variables for master problem.

See also [`UMFLinearMasterData`](@ref).

"""
function computebigM(ms::UMFLinearMasterData)
    s::Float64 = sum(costs(data(ms)))
    return s
end

"""
    setbigM!(ms::UMFLinearMasterData)

Given bigM value, set it in the master problem structure, to be used for artificial variables.

See also [`computebigM`](@ref), [`UMFLinearMasterData`](@ref).

"""
function setbigM!(ms::UMFLinearMasterData)
    bm = computebigM(ms)
    setbigM!(ms, bm)
    for k = 1:numdemands(data(ms))
        setbigM_k!(ms, bm * bdw(data(ms), k), k)
    end
    return bm
end



"""
    initializemodel!(ms::UMFLinearMasterData, config::MsColGenconfig)

Initialize the master problem with only artificial variables and void constraints.

See also [`UMFLinearMasterData`](@ref), [`MsColGenconfig`](@ref).
"""
function initializemodel!(ms::UMFLinearMasterData, config::MsColGenconfig)
    # initialize model of type linear master
    dat = data(ms)
    nd = numdemands(dat)
    narcs = numarcs(dat)
    mod = model(ms)
    capacity = capacities(dat)
    M = setbigM!(ms)
    sym::Vector{Symbol} = []
    for k = 1:nd
        push!(sym, Symbol("x$k"))
        vref::Vector{VariableRef} =
            mod[sym[k]] = @variable(mod, [1:1], lower_bound = 0, base_name = "x$k")
        set_model_vars_k!(ms, k, vref)
        set_name(vref[1], "y_$k") # optional
    end

    # capacity constraints on arcs a in A, named cap[a]
    @constraint(mod, cap[a in 1:narcs], 0 <= capacity[a])
    constraints(ms)["capacity"] = cap

    #convexity constraints on k in K named conv[k]
    @constraint(mod, conv[k in 1:nd], model_vars_k(ms, k)[1] == 1)
    constraints(ms)["convexity"] = conv
    # objective
    @objective(mod, Min, sum(bigM_dems(ms)[k] * model_vars_k(ms, k)[1] for k = 1:nd))

    return
end


"""
    update_coeffs_model_master_k!(
        ms::UMFLinearMasterData,
        var::VariableRef,
        col::Vector{Int64},
        k::Int64,
        config::MsColGenconfig,
)

Set the coefficients in the model for master for the added column. Routine called by `addcol_k`.

See also [`addcol_k!`](@ref), [`UMFLinearMasterData`](@ref).

"""
function update_coeffs_model_master_k!(
    ms::UMFLinearMasterData,
    var::VariableRef,
    col::Vector{Int64},
    k::Int64,
    config::MsColGenconfig,
)

    ## debug
    #if isempty(col)
    #    return 
    #end

    mod = model(ms)
    # update constr - capacity
    for i = 1:size(col, 1)
        set_normalized_coefficient(
            constraints(ms)["capacity"][col[i]],
            var,
            bdw(data(ms), k),
        )
    end
    # update constr- convexity
    set_normalized_coefficient(constraints(ms)["convexity"][k], var, 1)
    # update obj.
    coef = bdw(data(ms), k) * sum(cost(data(ms), col[i]) for i = 1:size(col, 1))
    set_objective_coefficient(mod, var, coef)
    return
end


"""
    addcol_k!(
        ms::UMFLinearMasterData,
        newcol::Vector{Int64},
        k::Int64,
        config::DefaultLinearMasterConfig,
    )

Add the new column `newcol`, relative to demand `k`, to the master `ms` problem, a linear model, with default configuration `config`.

See also [`solveCG`](@ref), [`UMFLinearMasterData`](@ref), [`DefaultLinearMasterConfig`](@ref).
"""
function addcol_k!(
    ms::UMFLinearMasterData,
    newcol::Vector{Int64},
    k::Int64,
    config::DefaultLinearMasterConfig,
)
    mod = model(ms)
    vars::Vector{VariableRef} = model_vars_k(ms, k)
    push!(vars, @variable(mod, lower_bound = 0))
    sizes(ms)[k] += 1
    push!(columns_k(ms, k), newcol)
    update_coeffs_model_master_k!(ms, vars[end], newcol, k, config)
    push!(getx(ms)[k], 0.0)
    return
end


"""
    setxopt!(ms::UMFLinearMasterData)

Set the optimal vector in the master problem `ms` from the internal solution x_k_p. Called by function `solve!(ms, config)`.

See also [`solve!`](@ref).
"""
function setxopt!(ms::UMFLinearMasterData)
    nd = numdemands(data(ms))
    narcs = numarcs(data(ms))
    x_k_p::Vector{Vector{Float64}} = getx(ms)
    ms.xopt = [zeros(narcs) for k = 1:nd]
    for k = 1:nd
        for p = 1:size(x_k_p[k], 1)
            for a = 1:size(column_x_k_p(ms, k, p), 1)
                xopt_k(ms, k)[column_x_k_p(ms, k, p)[a]] += x_k_p[k][p]
            end
        end
    end
end


"""
    solve!(ms::UMFLinearMasterData, config::DefaultLinearMasterConfig)

Solve the master problem with the LP solver, as specified by `config`.
Called in the column generation by the function `solveCG`.

If the model is not solved to optimality, both a warning massage and an error are thrown.

See also [`solveCG`](@ref), [`DefaultLinearMasterConfig`](@ref).
"""
function solve!(ms::UMFLinearMasterData, config::DefaultLinearMasterConfig)
    optimize!(model(ms))
    # DEBUG
    st = termination_status(model(ms))
    if st != OPTIMAL && st != OTHER_ERROR && st != ALMOST_INFEASIBLE
        println("solve status : ", termination_status(model(ms)))
        @warn("Model not solved to optimality")
        error(
            "Master not solved. The termination status is ",
            termination_status(model(ms)),
        )
        return nothing
    end
    setduals_arcs!(ms, dual.(constraints(ms)["capacity"]))
    setduals_demands!(ms, dual.(constraints(ms)["convexity"]))

    #set x_k_p and y_k
    for k = 1:numdemands(data(ms))
        #x_k_p
        setx_k!(ms, k, value.(model_vars_k(ms, k)[2:end]))
        #y_k
        sety_k!(ms, k, value(model_vars_k(ms, k)[1]))
    end

    # set xopt
    setxopt!(ms)
    #set sol
    sol::Float64 = setsol!(ms, objective_value(model(ms)))
    setprimalsol!(ms, sol)
    return
end

"""
    setduals_arcs!(ms::UMFLinearMasterData, duals::Vector{<:Real})

Set the dual values in the master problem `ms` for arc constraints to the values `duals`.
Called by the function `solve!(ms, config)`, where `config` is of type `MsColGenconfig`.

See also [`solve!`](@ref), [`UMFLinearMasterData`](@ref), [`MsColGenconfig`](@ref).
"""
function setduals_arcs!(ms::UMFLinearMasterData, duals::Vector{<:Real})
    ms.sigma .= duals
    return ms.sigma
end

"""
    setduals_demands!(ms::UMFLinearMasterData, duals::Vector{<:Real})

Set the dual values in the master problem `ms` for demand constraints to the values `duals`.
Called by the function `solve!(ms, config)`, where `config` is of type `MsColGenconfig`.

See also [`solve!`](@ref), [`UMFLinearMasterData`](@ref), [`MsColGenconfig`](@ref).
"""
function setduals_demands!(ms::UMFLinearMasterData, duals::Vector{<:Real})
    ms.tau .= duals
    return ms.tau
end

"""
    check_acceptance(ms::UMFLinearMasterData)

Check whether all demands are accepted: that is, artificial variables have ``0`` value.
`ms` is the current master problem.

Return `true` if all demands are accepted, `false` otherwise.

Called at the end of the column generation by the function `solveCG`.

See also [`UMFLinearMasterData`](@ref).
"""
function check_acceptance(ms::UMFLinearMasterData)
    y_k = gety(ms)
    #if y_k != zeros(size(y_k, 1))
    if y_k > zeros(size(y_k, 1)) # some values can be < 0 (very small values)
        return false
    end
    return true
end



"""
    solve_master!(
        ms::UMFLinearMasterData,
        config::MsColGenconfig, 
        )

Solve the master problem in a column generation algorithm.
Return the dual vectors, respectively for the capacity constraints and the convexity constraints, and the solution value of the master.


See also [`solve!`](@ref), [`UMFLinearMasterData`](@ref), [`MsColGenconfig`](@ref).
"""
function solve_master!(ms::UMFLinearMasterData, config::MsColGenconfig)
    solve!(ms, config)
    duals_arcs, duals_demands = duals(ms)
    solval::Float64 = sol(ms)
    solprimal::Float64 = primalsol(ms)
    return duals_arcs, duals_demands, solval, solprimal
end



## ROUNDING

"""
    roundingsol(
        ndemands::Int64,
        cols::Vector{Vector{Vector{Int64}}},
        bws::Vector{<:Real}, 
        csts::Vector{<:Real},
        caps::Vector{<:Real},
        bigms::Vector{<:Real}, 
        duals::Vector{<:Real}
    )

Finds a feasible primal solution for the master with a heuristic algorithm.
Return the vector of actual primal variables, the vector of artificial variables, and the solution value of the master.


# Arguments

- `ndemands`: number of demands
- `cols`: list of extreme columns of the current restricted master
- `bws`: vector of bandwidths
- `csts`: vector of original costs on arcs of the problem
- `caps`: vector of arc capacities
- `bigms`: vector of costs of non-acceptance
- `duals`: vector of current dual variables on arcs of the problem (not multiplyed by the bandwidth).


Return:
- `x_k_p_feas`: the vector of the found primal solution
- `y_k_feas`: the vector of non accepted demands in the primal solution
- `solvalue`: the value of the primal solution found.

"""
function roundingsol(
    ndemands::Int64,
    cols::Vector{Vector{Vector{Int64}}},#columns
    bws::Vector{<:Real}, #b_k
    csts::Vector{<:Real}, #r_a (without b_k)
    caps::Vector{<:Real}, #c_a
    bigms::Vector{<:Real}, # bigM: costs of non-acceptance
    duals::Vector{<:Real}, # vector of dual variables on arc constraints.
)
    perm_bws::Vector{Int64} = sortperm(bws, rev = true)
    avail_capacities::Vector{Float64} = copy(caps)
    newcosts::Vector{Float64} = csts + duals # once for all!
    solvalue::Float64 = 0
    x_k_p_feas::Vector{Vector{Float64}} = [zeros(size(cols[k], 1)) for k = 1:ndemands]
    y_k_feas::Vector{Float64} = zeros(ndemands)
    z_k::Vector{Float64} = ones(ndemands) # remainder of convexity constraints = 1 - sum_p x_k_p - y_k, should be 0 at the end.
    maxl::Int64 = maximum(size(p, 1) for p in cols)
    lengths::Vector{Float64} = zeros(maxl)
    newlengths::Vector{Float64} = zeros(maxl)
    capmin::Float64 = 0.0
    for k in perm_bws
        resize!(newlengths, size(cols[k], 1))
        for p = 1:size(cols[k], 1)
            lengths[p] = bws[k] * compute_length_k_p(cols, csts, k, p)
            newlengths[p] = bws[k] * compute_length_k_p(cols, newcosts, k, p)
        end
        perm_lengths::Vector{Int64} = sortperm(newlengths)
        for p in perm_lengths
            if z_k[k] == 0
                break
            end
            capmin = minimum(avail_capacities[a] for a in cols[k][p])
            if capmin >= bws[k] # path entirely fits in solution
                # update x_k_p_feas
                x_k_p_feas[k][p] = min(1.0, z_k[k])
                # update z_k
                z_k[k] = 0
            else # not entirely fits
                # update x_k_p_feas
                x_k_p_feas[k][p] = min(capmin / bws[k], z_k[k])
                # update z_k
                z_k[k] -= x_k_p_feas[k][p]
            end
            # update avail_capacities
            for a in cols[k][p]
                avail_capacities[a] -= bws[k] * x_k_p_feas[k][p]
            end
            # update solvalue
            solvalue += x_k_p_feas[k][p] * lengths[p]
        end
        # update y_k_feas
        y_k_feas[k] = z_k[k]
        z_k[k] = 0
        #update solvalue
        solvalue += y_k_feas[k] * bigms[k]
    end
    return x_k_p_feas, y_k_feas, solvalue
end
