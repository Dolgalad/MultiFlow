include("routines_master.jl")
include("routines_pricing.jl")
include("baseline_rounding.jl")

"""
    solveCG(
        pb::UMFData,
        config::ColGenConfigBase
    )

Solve the instance `pb` with column generation. Options are defined by `config`.

Called by `solve(pb::UMFData, config::ColGenConfigBase)`.

If a restricted master problem is not solved to optimality, a warning and an error message are thrown and the algorithm terminates.

# Arguments
- `pb`: the instance data
- `config`: the configuration type for column generation algorithm

# Return values
- `time`: the total time for the column generation
- `val`: the dual feasible solution value of the column generation
- `primal`: the primal feasible solution value found in the column generation
- `nits`: the number of column generation iterations
- `ncols`: the number of columns added in the column generation
- `t_create_master`: the time spent to create the restricted master problem
- `t_create_pricing`: the time spent to create the pricing problem
- `t_init_master`: the time spent to initialize the restricted master problem
- `time_ms_sol`: the total time spent to solve each restricted master problem
- `time_pr_sol`: the total time spent to solve each pricing problem
- `time_ms_addcol`: the total time spent to add columns to each restricted master problem
- `last_timetot`: the total time for the last iterations of exact column generation
- `last_val`: the exact dual feasible solution value 
- `last_primal`: the exact primal feasible solution value 
- `last_nits`: the number of iterations for the last iterations of exact column generation
- `last_ncols`: the number of columns added in the last iterations of exact column generation
- `last_time_ms_sol`: the total time spent to solve the restricted master problems in the last iterations of exact column generation
- `last_time_pr_sol`: the total time spent to solve the pricing problems in the last iterations of exact column generation
- `last_time_ms_addcol`:the total time spent to add columns to the restricted master problems in the last iterations of exact column generation
- `xoptfin`: the primal solution obtained at the end of the exact column generation 
- `ms::UMFLinearMasterData`: the master problem
- `pr::UMFShortestPathPricingData`: the pricing problem
- `accepteddemands::Int64`: an integer to indicate if all demands are accepted: ``1`` if all demands are accepted, ``0`` otherwise.
- `solrounding`: the value of the baseline rounding solution 

See also [`ColGenConfigBase`](@ref), [`UMFData`](@ref), [`UMFLinearMasterData`](@ref), [`UMFShortestPathPricingData`](@ref), [`CGcycle!`](@ref)

"""
function solveCG(pb::UMFData, config::ColGenConfigBase)

    # check for NaN or Inf values in problem
    if has_nan(pb) || has_inf(pb)
        println("NaN or Inf in problem definition")
        println(pb)
        println("Nan in bandwidths : ", any(isnan.(pb.bandwidths)))
        println("Inf in bandwidths : ", any(isinf.(pb.bandwidths)))

    end
    timetot::Float64 = 0.0
    val::Float64 = -1.0
    primal::Float64 = val
    nits::Int64 = 0
    ncols::Int64 = 0
    time_ms_sol::Float64 = 0.0
    time_pr_sol::Float64 = 0.0
    time_ms_addcol::Float64 = 0.0
    last_timetot::Float64 = 0.0
    last_val::Float64 = 0.0
    last_primal::Float64 = 0.0
    last_nits::Int64 = 0
    last_ncols::Int64 = 0
    last_time_ms_sol::Float64 = 0.0
    last_time_pr_sol::Float64 = 0.0
    last_time_ms_addcol::Float64 = 0.0
    maxit = 10000
    tol_pricing = 1e-4
    tol_val = 1e-8
    nd = numdemands(pb)

    # start column generation
    t_create_master::Float64 =
        @elapsed ms::UMFLinearMasterData = UMFLinearMasterData(pb, config)
    t_init_master::Float64 = @elapsed initializemodel!(ms, config.msconfig)

    ## pricing filter
    #t_create_filter = 0.
    #if typeof(config.prconfig)==kSPFilterPricingConfig
    #    t_create_filter =
    #        @elapsed filter = kSPFilter(pb, config, config.prconfig.K)  
    #elseif typeof(config.prconfig)==ClassifierAndSPConfig
    #    t_create_filter = @elapsed filter = ClassifierSPFilter(pb, config, config.prconfig.model_path)
    #else        
    #    t_create_filter =
    #        @elapsed filter = ArcDemandFilter(pb, config)
    #end

    t_create_pricing::Float64 =
        #@elapsed pr::UMFShortestPathPricingData = UMFShortestPathPricingData(pb, config; filter=filter)
        @elapsed pr::UMFShortestPathPricingData = UMFShortestPathPricingData(pb, config)

    #println("t_create_pricing = $(t_create_pricing)")

    # column generation cycle
    t = @timed CGcycle!(maxit, ms, pr, config, tol_val, nd, tol_pricing)
    val, primal, nits, ncols, time_ms_sol, time_pr_sol, time_ms_addcol = t.value
    timetot = t.time

    # last iterations to get exact solution
    if (primal - val) / abs(val) > 1e-3
        config2 = ColGenConfigBase(
            DefaultLinearMasterConfig(),
            config.prconfig,
            config.optimizer,
            config.outputname,
        )
        t = @timed CGcycle!(100, ms, pr, config2, tol_val, nd, tol_pricing)
        last_val,
        last_primal,
        last_nits,
        last_ncols,
        last_time_ms_sol,
        last_time_pr_sol,
        last_time_ms_addcol = t.value
        last_timetot = t.time
    end

    xoptfin = xopt(ms)

    # call rounding heuristic
    #println("Calling rounding heuristic ")
    #println("\tbdws : ", any(isnan.(bdws(pb))), ", ", any(isinf.(bdws(pb))))
    #println("\tcosts: ", any(isnan.(costs(pb))), ", ", any(isinf.(costs(pb))))
    #println("\tcapacities: ", any(isnan.(capacities(pb))), ", ", any(isinf.(capacities(pb))))

    x_k_p_feas, y_k_feas, solrounding = baselinerounding(
        nd,
        allcolumns(ms),
        bdws(pb),
        costs(pb),
        capacities(pb),
        pr,
        config.prconfig,
        getx(ms),
        bigM_dems(ms),
    )

    # check whether all demands are accepted or not
    accepteddemands = 1
    if !(check_acceptance(ms))
        #Not all demands are accepted
        accepteddemands = 0
    end

    # compute average number of iterations
    nit_tot::Int64 = getnit_tot(ms)
    nit_avg::Int64 = nits == 0 ? 0 : round(nit_tot / nits)

    # create the UMFSolutionData structure
    sol = UMFSolutionData(hcat(xoptfin...), x_k_p_feas)
    
    # compute graph reduction
    graph_reduction = 1. - sum([sum(pr.filter.masks[k]) for k in 1:nk(pb)]) / (numarcs(pb) * numdemands(pb))
    # create the SolverStatistics structure
    stats_keys = ["timetot","val","primal","nits","ncols","t_create_master","t_create_pricing",
                  "t_init_master","time_ms_sol","time_pr_sol","time_ms_addcol","last_timetot",
                  "last_val","last_primal","last_nits","last_ncols","last_time_ms_sol",
                  "last_time_pr_sol","last_time_ms_addcol","ms","pr","nit_avg","accepteddemands",
                  "graph_reduction",
                 ]
    stats_vals = [timetot,val,primal,nits,ncols,t_create_master,t_create_pricing,t_init_master,
                  time_ms_sol,time_pr_sol,time_ms_addcol,last_timetot,last_val,last_primal,
                  last_nits,last_ncols,last_time_ms_sol,last_time_pr_sol,last_time_ms_addcol,
                  ms,pr,nit_avg,accepteddemands,graph_reduction
                 ]
    solvestats = SolverStatistics(Dict(stats_keys .=> stats_vals))
    return sol, solvestats

    return timetot,
    val,
    primal,
    nits,
    ncols,
    t_create_master,
    t_create_pricing,
    t_init_master,
    time_ms_sol,
    time_pr_sol,
    time_ms_addcol,
    last_timetot,
    last_val,
    last_primal,
    last_nits,
    last_ncols,
    last_time_ms_sol,
    last_time_pr_sol,
    last_time_ms_addcol,
    xoptfin,
    ms,
    pr,
    nit_avg,
    accepteddemands,
    solrounding
end

"""
    solve(pb::UMFData, config::ColGenConfigBase)

Solve instance `pb` with the column generation algorithm, and write results on a text file, whose name is specified in `config`.

# Output

All the results are written on a single line, separated by ";", as follows:

"`pbname`;`narcs`;`nnodes`;`nd`;`typealg`;`optimizer`;`mstype`;`prtype`;`ttot_CG`;`val_CG`;`primal_CG`;`nits_CG`;`ncols_CG`;`t_ms_create`;`t_pr_create`;`t_ms_init`;`t_ms_sol_CG`;`t_pr_sol_CG`;`t_ms_addcol_CG`;`ttot_last`;`val_last`;`nits_last`;`ncols_last`;`t_ms_sol_last`;`t_pr_sol_last`;`t_ms_addcol_last`;`nit_avg`;`accepteddemands`;`rounding`;`ttime`"

where the meaning is as follows:
- `pbname`: name of the instance
- `narcs`: number of arcs
- `nnodes`: number of nodes
- `nd`: number of demands
- `typealg`: type of algorithm used (`CG` in this case)
- `optimizer`: chosen LP optiimzer which could be used
- `mstype`: type of master configuration
- `prtype`: type of pricing configuration
- `ttot_CG`: the total time for the column generation
- `val_CG`: the dual feasible solution value obtained at the end of the column generation
- `primal_CG`: the primal feasible solution value found in the column generation
- `nits_CG`: the number of column generation iterations
- `ncols_CG`: the number of columns added in the column generation
- `t_ms_create`: the time spent to create the restricted master problem
- `t_pr_create`: the time spent to create the pricing problem
- `t_ms_init`: the time spent to initialize the restricted master problem
- `t_ms_sol_CG`: the total time spent to solve each restricted master problem in the column generation
- `t_pr_sol_CG`: the total time spent to solve each pricing problem in the column generation
- `t_ms_addcol_CG`: the total time spent to add columns to each restricted master problem in the column generation
- `ttot_last`: the total time for the last iterations of exact column generation
- `val_last`: the exact dual feasible solution value 
- `primal_last`: the exact primal feasible solution value 
- `nits_last`: the number of iterations for the last iterations of exact column generation
- `ncols_last`: the number of columns added in the last iterations of exact column generation
- `t_ms_sol_last`: the total time spent to solve the restricted master problems in the last iterations of exact column generation
- `t_pr_sol_last`: the total time spent to solve the pricing problems in the last iterations of exact column generation
- `t_ms_addcol_last`:the total time spent to add columns to the restricted master problems in the last iterations of exact column generation
- `accepteddemands`: demand acceptance parameter: if all demands are accepted it is ``1``, otherwise ``0``
- `rounding`: solution value of the rounding heuristic
- `time`: the total time of the solution algorithm, with exact column generation and primal heuristic included.

Return the value of the column generation algorithm.

See also [`solveCG`](@ref), [`ColGenConfigBase`](@ref), [`UMFData`](@ref)

"""
function solve(pb::UMFData, config::ColGenConfigBase)
    pbname = nameprob(pb)
    pbname = replace(pbname, "data" => "data_")
    narcs = numarcs(pb)
    nnodes = numnodes(pb)
    nd = numdemands(pb)
    #all configuration data:
    typealg = "CG"
    mstype::String = ""
    prtype::String = ""
    if typeof(config.msconfig) == DefaultLinearMasterConfig
        mstype = "mslinear"
    else
        mstype = "msdefault"
    end
    #if typeof(config.prconfig) == DefaultDijkstraPricingConfig
    #    prtype = "dijkstra"
    #else
    #    prtype = "prdefault"
    #end
    if typeof(config.prconfig) == DefaultDijkstraPricingConfig
        prtype = "dijkstra"
    elseif typeof(config.prconfig) == LARACPricingConfig
        prtype = "larac"
    elseif typeof(config.prconfig) == kSPFilterPricingConfig
        prtype = "kspfilter "*string(config.prconfig.K)
    elseif typeof(config.prconfig) == ClassifierAndSPConfig
        #prtype = "clssp "*string(config.prconfig.model_path)*" "*string(config.prconfig.K)
    else
        prtype = "prdefault"
    end

    optimizer = config.optimizer
    t = @timed solveCG(pb, config)

    # dump solver statistics to output file
    save(t.value[2], config.outputname)


    return t.value
    return val_CG
end


"""
    CGcycle!(maxit::Int64,
        ms::UMFLinearMasterData,
        pr::UMFShortestPathPricingData,
        config::ColGenConfigBase,
        tol_val::Float64,
        nd::Int64,
        tol_pricing::Float64
    )

Execute the column generation iterations, given initial master and pricing structures `ms` and `pr` respectively, configuration options `config`, tolerances as described below and number of demands `nd`.

# Arguments
- `maxit`: the maximum number of column generation iterations
- `ms`: the master problem, already initialized
- `pr`: the pricing problem
- `config`: the configuration options, see also [`ColGenConfigBase`](@ref)
- `tol_val`: tolerance on the value of the master solutions: if the solution does not improve after some iterations, stop the column generation algorithm.
- `nd`: number of demands of the problem
- `tol_pricing`: tolerance on the reduced costs.

The column generation algorithm solves alternatively the retricted master and the pricing problems, until:
- the maximum number of iterations is reached, or
- the value has not improved for more than 5 iterations, or 
- no column with negative reduced cost is found.


Return the following values:

- `mssol`: the final solution value of the master problem, always dual feasible
- `mssol_primal`: the final primal solution value
- `it_nb`: the number of column generation iterations
- `cols_tot`: the number of columns added to the restricted master problems
- `time_ms_sol`: the total time spent to solve each restricted master problem
- `time_pr_sol`: the total time spent to solve each pricing problem
- `time_ms_addcol`: the total time spent to add columns to each restricted master problem.

"""
function CGcycle!(
    maxit::Int64,
    ms::UMFLinearMasterData,
    pr::UMFShortestPathPricingData,
    config::ColGenConfigBase,
    tol_val::Float64,
    nd::Int64,
    tol_pricing::Float64,
    #filter::AbstractArcDemandFilter,
)
    it_nb::Int64 = 0
    cols_tot::Int64 = 0
    time_ms_sol::Float64 = 0.0
    time_pr_sol::Float64 = 0.0
    time_ms_addcol::Float64 = 0.0
    mssol::Float64 = 1e30
    mssol_primal::Float64 = 0.0
    oldmssol::Float64 = 0.0
    num_unchanged::Int64 = 0
    max_unchanged::Int64 = 5
    newaddedcols::Int64 = 0
    newcols::Vector{Vector{Vector{Int64}}} = [[] for k = 1:nd]
    sigma::Vector{Float64} = zeros(numarcs(data(ms)))
    tau::Vector{Float64} = zeros(nd)

    for nit in Base.OneTo(maxit)
        #println("CG iteration $nit")
        #println("\tobjective : ", objective_function(ms.mod))
        #print(ms.mod)
        it_nb += 1
        # master
        oldmssol = mssol
        time_ms_sol +=
            @elapsed sigma, tau, mssol, mssol_primal = solve_master!(ms, config.msconfig)
        if mssol >= oldmssol - abs(oldmssol) * tol_val
            num_unchanged += 1
        else
            num_unchanged = 0
        end
        newaddedcols = 0 # counts the columnss added in the pricing phase
        # pricing
        time_pr_sol +=
            @elapsed newcols = solve_pricing!(pr, config.prconfig, sigma, tau, tol_pricing)
        for k = 1:nd
            for c in eachindex(newcols[k])
                # debug
                #if isempty(newcols[k][c])
                #    continue
                #end
                time_ms_addcol += @elapsed addcol_k!(ms, newcols[k][c], k, config.msconfig)
                newaddedcols += 1
                cols_tot += 1
            end
        end
        if newaddedcols == 0 || num_unchanged > max_unchanged
            if newaddedcols == 0
                #println("NO NEW COLUMNS")
            end
            if num_unchanged > max_unchanged
                #println("NO IMPROVEMENT")
            end
            #abc = readline()
            break
        end

    end
    #println("CGcycle done")
    return mssol, mssol_primal, it_nb, cols_tot, time_ms_sol, time_pr_sol, time_ms_addcol
end
