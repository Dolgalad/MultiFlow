using JLD2
using BSON

"""
readinstance(path::String, f::Real=1.0)

Read an instance by the path of the source files `link.csv` and `service.csv`.
Rescale arc capacities by factor `f`, which by default is ``1.0``.

Return the created data structure of `UMFData` type.

"""
function readinstance(path::String, f::Real = 1.0)
    linkname = path * "link.csv"
    servicename = path * "service.csv"
    dat = UMFData(linkname, servicename, path)
    dat2 = scale_capacity(dat, f)
    return dat2
end


"""
    set_config(form_type, optimizer, outputfile, mastertype, pricingtype, maxit, warmstart, betarule, stepfactor, alphastop, dastep, gradfactor)

Set the configuration structures to solve a UMCF instance.
Used as a subroutine in function `solveUMF`.
The arguments are as selected in `solveUMF`.

# Arguments:

    - `form_type::String`: label to choose the solution method.
    - `optimizer::String`: choice of the optimizer. 
    - `outputfile::String`: name of the text file to contain the results
    - `mastertype::String`: choice of the master type.
    - `pricingtype::String`: choice of the pricing type. 

Return the chosen configuration struture, to be used to solve the instance.

See also [`solveUMF`](@ref).

"""
function set_config(
    form_type::String,
    optimizer::String,
    outputfile::String,
    mastertype::String,
    pricingtype::String,
    maxit::Int64 = 1000,
    warmstart::Int64 = 0,
    betarule::Int64 = 1,
    stepfactor::Float64 = 1.0,
    alphastop::Float64 = 1e-4,
    dastep::Int64 = 0,
    gradfactor::Float64 = 10.0,
)

    if optimizer != "highs" && !(optimizer == "cplex" && isinstalled("CPLEX"))
        throw(ArgumentError("Not accepted optimizer. Use cplex or highs."))
        return
    end
    if form_type == "CG"
        ws::Bool = warmstart == 1 ? true : false
        das::Bool = dastep == 1 ? true : false
        if mastertype == "linear"
            msconfig = DefaultLinearMasterConfig()
        else
            msconfig = DefaultLinearMasterConfig()
        end
        if pricingtype == "dijkstra"
            prconfig = DefaultDijkstraPricingConfig()
        elseif pricingtype == "larac"
            prconfig = LARACPricingConfig()
        elseif startswith(pricingtype, "kspfilter")
            K = 0
            filter_arg = split(pricingtype," ")[2]
            if all(isdigit(c) for c in filter_arg)
                K = parse(Int64, filter_arg)
            end
            sptable = nothing
            if isfile(filter_arg)
                sptable = h5open(filter_arg, "r") do file
                           read(file,"cost")
                end 
            end
            prconfig = kSPFilterPricingConfig(K, sptable)
        elseif startswith(pricingtype, "ksplarac")
            K = 0
            filter_arg = split(pricingtype," ")[2]
            if all(isdigit(c) for c in filter_arg)
                K = parse(Int64, filter_arg)
            end
            sptable = nothing
            if isfile(filter_arg)
                sptable = h5open(filter_arg, "r") do file
                           read(file,"cost"),read(file,"delay")
                end 
            end
            prconfig = kSPLARACPricingConfig(K, sptable)

        elseif startswith(pricingtype, "clssp")
            vs = split(pricingtype, " ")
            model_path = string(vs[2])
            # load model or prediction
            if endswith(model_path, ".bson")
                @load model_path _model
                if CUDA.functional()
                    _model = _model |> Flux.gpu
                end
            else
                _model = h5open(model_path, "r") do file
                    read(file, "pred")
                end
            end

            K = 0
            sptable_path = ""
            sptable = nothing
            threshold = 0.
            keep_proportion = 0.
            postprocessing_method = 1

            if length(vs)>=3
                for val in vs[3:end]
                    if startswith(val, "threshold")
                        threshold = parse(Float64, split(val, ":")[2])
                    elseif startswith(val, "K")
                        K = parse(Int64, split(val,":")[2])
                    elseif startswith(val, "sptable_path")
                        sptable_path = split(val, ":")[2]
                        if isfile(sptable_path)
                            sptable = h5open(sptable_path, "r") do file
                                       read(file,"cost")
                            end 
                        end

                    elseif startswith(val, "keep_proportion")
                        keep_proportion = parse(Float64, split(val, ":")[2])
                    elseif startswith(val, "postprocessing")
                        postprocessing_method = parse(Int64, split(val,":")[2])
                    end
                end
            end
            prconfig = ClassifierAndSPConfig(_model, 
                                             K, 
                                             sptable,
                                             threshold,
                                             keep_proportion,
                                             postprocessing_method
                                            )
        elseif startswith(pricingtype, "svm")
            vs = split(pricingtype, " ")
            model_path = string(vs[2])
            JLD2.@load model_path _model
            prconfig = SVMAndSPConfig(_model)
        elseif startswith(pricingtype, "random_forest")
            vs = split(pricingtype, " ")
            model_path = string(vs[2])
            JLD2.@load model_path _model
            prconfig = RFAndSPConfig(_model)
        elseif startswith(pricingtype, "mlp")
            vs = split(pricingtype, " ")
            model_path = string(vs[2])
            BSON.@load model_path _model
            prconfig = MLPAndSPConfig(_model)
        elseif startswith(pricingtype, "clslarac")
            vs = split(pricingtype, " ")
            model_path = string(vs[2])
            # load model or prediction
            if endswith(model_path, ".bson")
                @load model_path _model
                if CUDA.functional()
                    _model = _model |> Flux.gpu
                end
            else
                _model = h5open(model_path, "r") do file
                    read(file, "pred")
                end
            end

            K = parse(Int64, vs[3])
            sptable_path = ""
            sptable = nothing
            threshold = 0.
            keep_proportion = 0.
            if length(vs)>3
                sptable_path = vs[4]
                if isfile(vs[4])
                    sptable = h5open(vs[4], "r") do file
                               read(file,"cost"), read(file,"delay")
                    end 
                end
            end

            if length(vs)>4
                if startswith(vs[5], "threshold")
                    threshold = parse(Float64, split(vs[5], ":")[2])
                elseif startswith(vs[5], "keep_proportion")
                    keep_proportion = parse(Float64, split(vs[5], ":")[2])
                end
            end

            prconfig = ClassifierAndLARACConfig(_model, K, sptable, threshold, keep_proportion)
        else
            prconfig = DefaultDijkstraPricingConfig()
        end
        config = ColGenConfigBase(msconfig, prconfig, optimizer, outputfile)
    elseif form_type == "compactLR"
        config = CompactConfigBase(true, optimizer, outputfile)
    elseif form_type == "compactINT"
        config = CompactConfigBase(false, optimizer, outputfile)
    else
        print("Not a correct type")
        return nothing
    end

    return config
end



"""
    solveUMF(dat, form_type, optimizer, outputfile, mastertype="", pricingtype="", maxit = 100, warmstart = 0, betarule =1, stepfactor = 1.0, alphastop = 1e-4, dastep = 0, gradfactor = 10.0)

Solve a UMF instance.

Solve the instance `dat` with one of the methods: column generation, compact formulation (integral solution), linear relaxation of compact formulation.
If column generation, information on the type of master and pricing are selected with `mastertype` and `pricingtype`.

# Arguments:

- `dat::UMFData`: data structure containing the instance data
- `form_type::String`: label to choose the solution method. Available choices are:
        - `"CG"`: for the column generation solver;
        - `"compactLR"`: for the linear relaxation of compact formulation;
        - `"compactINT"`: for the compact formulation (to integrality)
- `optimizer::String`: choice of the optimizer. Available choices are:
        - `highs`: to use the LP solver HiGHS
        - `cplex`: to use the LP solver CPLEX
- `outputfile::String`: name of the text file to contain the results
- `mastertype::String=""`: choice of the master type. Available choices are:
        - `linear`: to solve the master with the chosen optimizer
- `pricingtype::String=""`: choice of the pricing type. Available choices are:
        - `dijkstra`: to solve the shortest path problems with Dijkstra's algorithm.


See also [`solve`](@ref), [`ColGenConfigBase`](@ref), [`CompactConfigBase`](@ref).


"""
function solveUMF(
    dat::UMFData,
    form_type::String,
    optimizer::String,
    outputfile::String,
    mastertype::String = "",
    pricingtype::String = "",
    maxit::Int64 = 1000,
    warmstart::Int64 = 0,
    betarule::Int64 = 1,
    stepfactor::Float64 = 1.0,
    alphastop::Float64 = 1e-4,
    dastep::Int64 = 0,
    gradfactor::Float64 = 10.0,
)
    config = set_config(
        form_type,
        optimizer,
        outputfile,
        mastertype,
        pricingtype,
        maxit,
        warmstart,
        betarule,
        stepfactor,
        alphastop,
        dastep,
        gradfactor,
    )
    return solve(dat, config)
end

"""
    solveUMF(path, form_type, optimizer, outputfile, mastertype="", pricingtype="", maxit = 100, warmstart = 0, betarule =0, stepfactor = 1.0, alphastop = 1e-4, dastep = 0, gradfactor = 10.0)

Generate the instance `dat` with its path `path` and then
call the function `solveUMF(dat,form_type, optimizer, outputfile, mastertype, pricingtype, maxit, warmstart, betarule, stepfactor, alphastop, dastep, gradfactor)`.

# Examples 

```@example
include UMFSolver # hide
solveUMF("../instances/toytests/test1/", "compactLR", "cplex", "../output/julia/t1.txt")
```

```@example
include UMFSolver # hide
solveUMF("../instances/toytests/test1/", "CG", "cplex", "../output/julia/t1.txt")
```

"""
function solveUMF(
    path::String,
    form_type::String,
    optimizer::String,
    outputfile::String,
    mastertype::String = "",
    pricingtype::String = "",
    maxit::Int64 = 1000,
    warmstart::Int64 = 0,
    betarule::Int64 = 1,
    stepfactor::Float64 = 1.0,
    alphastop::Float64 = 1e-4,
    dastep::Int64 = 0,
    gradfactor::Float64 = 10.0,
)
    dat::UMFData = readinstance(path)
    return solveUMF(
        dat,
        form_type,
        optimizer,
        outputfile,
        mastertype,
        pricingtype,
        maxit,
        warmstart,
        betarule,
        stepfactor,
        alphastop,
        dastep,
        gradfactor,
    )
end

"""
    solveUMFrescaled(
        path::String,
        form_type::String,
        optimizer::String,
        outputfile::String,
        factor::Float64=1.0,
        mastertype::String = "",
        pricingtype::String = "",
        maxit::Int64 = 1000,
        warmstart::Int64 = 0,
        betarule::Int64 = 1,
        stepfactor::Float64 = 1.0,
        alphastop::Float64 = 1e-4,
        dastep::Int64 = 0,
        gradfactor::Float64 = 10.0
    )
Generate an instance with its path `path` and then
multiplies the arc capacities by the factor `f`; then 
call the function `solveUMF` with the modified instance and the parameters chosen at input.

# Examples 

```@example
include UMFSolver # hide
solveUMFrescaled("../instances/toytests/test1/", "compactLR", "highs", "../output/julia/test1.txt", 0.5)
```

```@example
include UMFSolver # hide
solveUMFrescaled("../instances/toytests/test1/", "CG", "cplex", "../output/julia/test1.txt", 0.5)
```

```@example
include UMFSolver # hide
solveUMFrescaled("../instances/toytests/test1/", "CG", "cplex", "../output/julia/test1.txt", 0.5, "linear", "dijkstra")
```

See also [`solveUMF`](@ref).

"""
function solveUMFrescaled(
    path::String,
    form_type::String,
    optimizer::String,
    outputfile::String,
    factor::Float64= 1.0,
    mastertype::String = "",
    pricingtype::String = "",
    maxit::Int64 = 1000,
    warmstart::Int64 = 0,
    betarule::Int64 = 1,
    stepfactor::Float64 = 1.0,
    alphastop::Float64 = 1e-4,
    dastep::Int64 = 0,
    gradfactor::Float64 = 10.0,
)
    dat::UMFData = readinstance(path)
    dat2 = UMFSolver.scale_capacity(dat, factor)
    return solveUMF(
        dat2,
        form_type,
        optimizer,
        outputfile,
        mastertype,
        pricingtype,
        maxit,
        warmstart,
        betarule,
        stepfactor,
        alphastop,
        dastep,
        gradfactor,
    )
end
