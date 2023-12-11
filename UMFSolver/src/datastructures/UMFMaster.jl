include("../model_functions/functions_models.jl")


"""
    UMFLinearMasterData

Data structure for the Linear master problem for column generation on UMCF instance.

It is a sub-type of `AbstractMasterPbData`.
It can be used in a column generation algorithm. 
It allows to solve the master problem with a linear optimizer.


# Constructor

    UMFLinearMasterData(
        pb::UMFData,
        config::ColGenConfigBase
    )

    Construct a `UMFLinearMasterData` with information from the UMCF instance and the column generation configuration.

    # Arguments
    - `pb`: the UMCF instance data
    - `config`: the configuration type for the column generation algorithm.

    See also [`UMFData`](@ref), [`ColGenConfigBase`](@ref), [`AbstractMasterPbData`](@ref). 

"""
mutable struct UMFLinearMasterData <: AbstractMasterPbData
    optimizer::String
    mod::Model
    constrs::Dict{String,Any}
    dat::UMFData
    sigma::Vector{Float64}
    tau::Vector{Float64}
    bigM0::Float64
    bigM_dems::Vector{Float64}
    sizes::Vector{Int64} # numbers of columns stored for demand k
    cols::Vector{Vector{Vector{Int64}}} #multidim array to store ALL cols!
    P_a_k::Vector{Vector{Vector{Int64}}} # multidim array to store indices of cols that contain arc a
    P_a_k_exists::Vector{BitVector} # for each a, k, true if P_a_k[a][k] is nonempty.
    xopt::Vector{Vector{Float64}}    # to store optimal vector in the original size.
    sol::Float64   # optimal value, lower bound
    primalsol::Float64 # optimal primal solution, upper bound
    #model variables
    mod_vars::Vector{Vector{VariableRef}} # to contain all the variables of the model. A vector for each demand
    # vectors of variables
    x_k_p::Vector{Vector{Float64}} # vector of x variables
    y_k::Vector{Float64} # vector of y variables.
    nit_tot::Int64 # total number of iterations of steps
    nit_avg::Int64 # average nuber of iterations for each RMP resolution
    q_a::Vector{Float64} # weight on capacity constraints
    q_k::Vector{Float64} # weight on convexity constraints
    function UMFLinearMasterData(pb::UMFData, config::ColGenConfigBase)
        dat = pb
        optimizer = config.optimizer
        mod = createdirectmodel(optimizer) #direct mode. See also ../model_functions/functions_models.jl
        set_silent(mod) # to avoid printing of the solver.
        constrs = Dict{String,Any}()
        narcs = numarcs(dat)
        ndemands = numdemands(dat)
        sigma = zeros(narcs)
        tau = zeros(ndemands)
        sizes = zeros(ndemands)
        cols = [[] for k = 1:ndemands]# 3-dimensional: a vector for each k
        P_a_k = [[[] for k = 1:ndemands] for a = 1:narcs] # 3-dimensional: a vector for each a to contain indices for eack k.
        P_a_k_exists = [falses(ndemands) for a = 1:narcs]
        mod_vars = [[] for k = 1:ndemands]
        xopt = [zeros(narcs) for k = 1:ndemands]
        sol = 0
        primalsol = sol
        bigM0 = 0
        bigM_dems = zeros(ndemands)
        x_k_p = [[] for k = 1:ndemands]
        y_k = zeros(ndemands)
        nit_avg = 0
        nit_tot = 0
        q_a = []
        q_k = []

        new(
            optimizer,
            mod,
            constrs,
            dat,
            sigma,
            tau,
            bigM0,
            bigM_dems,
            sizes,
            cols,
            P_a_k,
            P_a_k_exists,
            xopt,
            sol,
            primalsol,
            mod_vars,
            x_k_p,
            y_k,
            nit_tot,
            nit_avg,
            q_a,
            q_k,
        )
    end
end

"""
    model(ms::UMFLinearMasterData)

Get the LP model contained in `ms`.

"""
function model(ms::UMFLinearMasterData)
    return ms.mod
end

"""
    constraints(ms::UMFLinearMasterData)

Get the constraints of the LP model contained in `ms`.

"""
function constraints(ms::UMFLinearMasterData)
    return ms.constrs
end

"""
    optimizer(ms::UMFLinearMasterData)

Get the optimizer selected to solve the LP model contained in `ms`.

"""
function optimizer(ms::UMFLinearMasterData)
    return ms.optimizer
end

"""
    data(ms::UMFLinearMasterData)

Get the `UMFData` structure considered by `ms`.

"""
function data(ms::UMFLinearMasterData)
    return ms.dat
end

"""
    duals_arcs(ms::UMFLinearMasterData)

Get the dual variables on the arc capacity constraints.

"""
function duals_arcs(ms::UMFLinearMasterData)
    return ms.sigma
end

"""
    duals_demands(ms::UMFLinearMasterData)

Get the dual variables on the convexity constraints on the demands.

"""
function duals_demands(ms::UMFLinearMasterData)
    return ms.tau
end

"""
    duals(ms::UMFLinearMasterData)

Get the dual variables on the arc capacity constraints and on the convexity constraints on the demands.

"""
function duals(ms::UMFLinearMasterData)
    return ms.sigma, ms.tau
end

"""
    bigM(ms::UMFLinearMasterData)

Get the big M value for the instance `ms` (needed to manage the artificial variables).

See also [`bigM_dems`](@ref)

"""
function bigM(ms::UMFLinearMasterData)
    return ms.bigM0
end

"""
    bigM_dems(ms::UMFLinearMasterData)

Get the vector of big M values for the demands of `ms` (which are the big M of the instance multiplied by the bandwidths of the demands, thus different demands may have different big M values).

See also [`bigM`](@ref)

"""
function bigM_dems(ms::UMFLinearMasterData)
    return ms.bigM_dems
end

"""
    setbigM!(ms::UMFLinearMasterData, bm::Real)

Set the big M value of the instance `ms` to value `bm`.

"""
function setbigM!(ms::UMFLinearMasterData, bm::Float64)
    ms.bigM0 = bm
    return ms.bigM0
end

"""
    setbigM_k!(ms::UMFLinearMasterData, bm::Real, k::Int64)

Set the big M value of demand `k` of `ms` to value `bm`.

"""
function setbigM_k!(ms::UMFLinearMasterData, bm::Float64, k::Int64)
    ms.bigM_dems[k] = bm
    return bm
end

"""
    allcolumns(ms::UMFLinearMasterData)

Get the vector containing all columns currently stored in `ms`.

"""
function allcolumns(ms::UMFLinearMasterData)
    return ms.cols
end

"""
    columns_k(ms::UMFLinearMasterData, k::Int64)

Get the vector containing all columns relative to demand `k` currently stored in `ms`.

"""
function columns_k(ms::UMFLinearMasterData, k::Int64)
    return ms.cols[k]
end

"""
    column_x_k_p(ms::UMFLinearMasterData, k::Int64, p::Int64)

Get the column of index `p` relative to demand `k` currently stored in `ms`.

"""
function column_x_k_p(ms::UMFLinearMasterData, k::Int64, p::Int64)
    return ms.cols[k][p]
end

"""
    P_a_k(ms::UMFLinearMasterData)

Get the vector containing column indices of P_a_k, currently stored in `ms`.

"""
function P_a_k(ms::UMFLinearMasterData)
    return ms.P_a_k
end

"""
    P_a_k_exists(ms::UMFLinearMasterData)

Get the vector containing, for each arc and each demand, true whenever P_a_k is nonempty, currently stored in `ms`.

"""
function P_a_k_exists(ms::UMFLinearMasterData)
    return ms.P_a_k_exists
end


"""
    add_P_a_k!(ms::UMFLinearMasterData, a::Int64, k::INt64, p::Int64)

Add column p to the set of column indices for demand `k` that contain arc `a`.
Also set `ms.P_a_k_exists[a][k]` to `true`. 

"""
function add_P_a_k!(ms::UMFLinearMasterData, a::Int64, k::Int64, p::Int64)
    push!(ms.P_a_k[a][k], p)
    ms.P_a_k_exists[a][k] = true
    return
end

"""
    sizes(ms::UMFLinearMasterData)

Get the vector containing the sizes of the current sets of columns for all demands for `ms`.

"""
function sizes(ms::UMFLinearMasterData)
    return ms.sizes
end

"""
    xopt(ms::UMFLinearMasterData)

Get the current primal optimal point of `ms`, which has components on all arcs, for all demands.

See also [`xopt_k`](@ref)

"""
function xopt(ms::UMFLinearMasterData)
    return ms.xopt
end

"""
    xopt_k(ms::UMFLinearMasterData, k::Int64)

Get the current primal optimal point of `ms`, which has components on all arcs, for demand `k`.

"""
function xopt_k(ms::UMFLinearMasterData, k::Int64)
    return ms.xopt[k]
end


"""
    setxopt!(ms::UMFLinearMasterData, newxopt::Vector{Vector{Float64}})

Sets the internal optimal vector in the master to `newxopt`.
"""
function setxopt!(ms::UMFLinearMasterData, newxopt::Vector{Vector{Float64}})
    ms.xopt = newxopt
    return
end




"""
    model_vars(ms::UMFLinearMasterData)

Get all the variables of the model, as a vector of vectors of `VariableRef` type.

See also [`model_vars_k`](@ref)

"""
function model_vars(ms::UMFLinearMasterData)
    return ms.mod_vars
end

"""
    model_vars_k(ms::UMFLinearMasterData, k::Int64)

Get all the variables of the model for demand `k`, as a vector of `VariableRef` type.
The first element is (if it exists) the artificial variable `y_k`.
See also [`model_vars`](@ref)

"""
function model_vars_k(ms::UMFLinearMasterData, k::Int64)
    return ms.mod_vars[k]
end


"""
    set_model_vars!(ms::UMFLinearMasterData, v::Vector{Vector{VariableRef}})

Set all the variables of the model, as a vector of vectors of `VariableRef` type.

See also [`set_model_vars_k!`](@ref)

"""
function set_model_vars!(ms::UMFLinearMasterData, v::Vector{Vector{VariableRef}})
    ms.mod_vars = v
    return
end

"""
    set_model_vars_k!(ms::UMFLinearMasterData, k::Int64, v::Vector{VariableRef})

Set all the variables of the model for demand `k`, as a vector of `VariableRef` type.
The first element is (if it exists) the artificial variable `y_k`.
See also [`set_model_vars!`](@ref)

"""
function set_model_vars_k!(ms::UMFLinearMasterData, k::Int64, v::Vector{VariableRef})
    ms.mod_vars[k] = v
    return
end


"""
    sol(ms::UMFLinearMasterData)

Get the current optimal (dual) value of the master problem `ms`.

"""
function sol(ms::UMFLinearMasterData)
    return ms.sol
end

"""
    setsol!(ms::UMFLinearMasterData, val::Real)

Set the optimal (dual) value of the master problem `ms` to the value `val`.

"""
function setsol!(ms::UMFLinearMasterData, val::Float64)
    ms.sol = val
    return ms.sol
end

"""
    primalsol(ms::UMFLinearMasterData)

Get the current optimal primal value of the master problem `ms`.

"""
function primalsol(ms::UMFLinearMasterData)
    return ms.primalsol
end

"""
    setprimalsol!(ms::UMFLinearMasterData, val::Real)

Set the optimal primal value of the master problem `ms` to the value `val`.

"""
function setprimalsol!(ms::UMFLinearMasterData, val::Float64)
    ms.primalsol = val
    return ms.primalsol
end


"""
    getx(ms::UMFLinearMasterData)

Get the current optimal vector, in the master dimensions:
a vector that contains, for each demand, the vector of coefficients for each column.

"""
function getx(ms::UMFLinearMasterData)
    return ms.x_k_p
end

"""
    gety(ms::UMFLinearMasterData)

Get the current vector of the artificial, *non-acceptance* variables for each demand.

"""
function gety(ms::UMFLinearMasterData)
    return ms.y_k
end

"""
    setx!(ms::UMFLinearMasterData, x::Vector{Vector{Float64}})

Set the current optimal vector to `x`.
`x` is a vector that contains, for each demand, the vector of coefficients for each column.

"""
function setx!(ms::UMFLinearMasterData, x::Vector{Vector{Float64}})
    ms.x_k_p = x
    return
end

"""
    setx_k!(ms::UMFLinearMasterData, k::Int64, x::Vector{Float64})

Set the current optimal vector for demand `k` to `x`.
`x` is a vector that contains the vector of coefficients for each column for demand `k`.

"""
function setx_k!(ms::UMFLinearMasterData, k::Int64, x::Vector{Float64})
    ms.x_k_p[k] = x
    return
end

"""
    setx_k_p!(ms::UMFLinearMasterData, k::Int64, p::Int64, x::Float64)

Set the value corresponding to column `p` of demand `k` to `x`.

"""
function setx_k_p!(ms::UMFLinearMasterData, k::Int64, p::Int64, x::Float64)
    ms.x_k_p[k][p] = x
    return
end

"""
    sety!(ms::UMFLinearMasterData, y::Vector{Float64})

Set the vector of the artificial, *non-acceptance* variables to `y`.

"""
function sety!(ms::UMFLinearMasterData, y::Vector{Float64})
    ms.y_k = y
    return ms.y_k
end

"""
    sety_k!(ms::UMFLinearMasterData, k::Int64, y::Float64)

Set the *non-acceptance* variable for demand `k` to `y`.

"""
function sety_k!(ms::UMFLinearMasterData, k::Int64, y::Float64)
    ms.y_k[k] = y
    return ms.y_k
end


"""
    setnit_avg!(ms::UMFLinearMasterData, n::Int64)

Store the average number of iterations `n` of the master in the master structure.

"""
function setnit_avg!(ms::UMFLinearMasterData, n::Int64)
    ms.nit_avg = n
    return
end

"""
    setnit_tot!(ms::UMFLinearMasterData, n::Int64)

Store the total number of iterations `n` of the master in the master structure.

"""
function setnit_tot!(ms::UMFLinearMasterData, n::Int64)
    ms.nit_tot = n
    return
end


"""
    getnit_avg(ms::UMFLinearMasterData)

Get the average number of iterations of the master.

"""
function getnit_avg(ms::UMFLinearMasterData)
    return ms.nit_avg
end

"""
    getnit_tot(ms::UMFLinearMasterData)

Get the total number of iterations of the master.

"""
function getnit_tot(ms::UMFLinearMasterData)
    return ms.nit_tot
end


#weights q.

"""
    getweights_a(ms::UMFLinearMasterData)

Get the vector of weigths on arc capacity constraints.

"""
function getweights_a(ms::UMFLinearMasterData)
    return ms.q_a
end

"""
    getweights_k(ms::UMFLinearMasterData)

Get the vector of weigths on demand convexity constraints.

"""
function getweights_k(ms::UMFLinearMasterData)
    return ms.q_k
end


"""
    setweights_a!(ms::UMFLinearMasterData, q_a::Vector{Float64})

Set the vector of weigths on arc capacity constraints to `q_a`.

"""
function setweights_a!(ms::UMFLinearMasterData, q_a::Vector{Float64})
    ms.q_a = q_a
    return
end

"""
    setweights_k!(ms::UMFLinearMasterData, q_k::Vector{Float64})

Set the vector of weigths on demand convexity constraints to `q_k`.

"""
function setweights_k!(ms::UMFLinearMasterData, q_k::Vector{Float64})
    ms.q_k = q_k
    return
end
