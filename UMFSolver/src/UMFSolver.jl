module UMFSolver

import Pkg

function isinstalled(pk::AbstractString)
    return pk in [v.name for v in values(Pkg.dependencies())]
end

export isinstalled

using JuMP

# add solvers:
# HiGHS
using HiGHS
# CPLEX
if isinstalled("CPLEX")
    @eval using CPLEX
end
# potentially, other solvers

using DataFrames, CSV
using Graphs, SimpleWeightedGraphs # only by pricing.
using StatsBase, Random
import Graphs: nv, ne
using GraphNeuralNetworks


# Data structures
include("otherstructures/Problemstructs.jl")
include("otherstructures/Configstructs.jl")
include("datastructures/UMFData.jl")
include("datastructures/UMFSolutionData.jl")
include("datastructures/SolverStatistics.jl")
include("datastructures/UMFMaster.jl")
# Machine learning
include("ml/metrics.jl")
include("ml/history.jl")
include("ml/training.jl")
include("ml/classifier_utils.jl")
include("ml/plots.jl")
include("ml/mem_utils.jl")
include("ml/filter.jl")
include("ml/augmented_graph.jl")
include("ml/instance_generation.jl")
include("ml/dataset.jl")
include("ml/model2_definition.jl")
include("ml/layer_utils.jl")
include("ml/model3_definition.jl")
include("ml/model4_definition.jl")
include("ml/model5_definition.jl")
include("ml/model6_definition.jl")
include("ml/model7_definition.jl")
include("ml/model8_definition.jl")
include("ml/model81_definition.jl")
include("ml/vnet.jl")
include("ml/model11_definition.jl")
include("ml/model12_definition.jl")

include("ml/model9_definition.jl")

# pricing
include("datastructures/UMFPricing.jl")
include("instance_reader/read_solve.jl")

# overloading nv, ne from Graphs.jl
#nv(inst::UMFData) = length(Set(vcat(inst.srcnodes, inst.dstnodes)))
nv(inst::UMFData) = numnodes(inst)
ne(inst::UMFData) = numarcs(inst) #length(inst.srcnodes)
nk(inst::UMFData) = numdemands(inst)


export arcsource,
    arcsources,
    arcdest,
    arcdests,
    demandorigin,
    demandorigins,
    demanddest,
    demanddests,
    numarcs,
    numdemands,
    numnodes,
    capacity,
    capacities,
    cost,
    costs,
    bdw,
    bdws,
    UMFData,
    has_nan,
    has_inf,
    UMFLinearMasterData,
    UMFShortestPathPricingData,
    ColGenConfigBase,
    CompactConfigBase,
    DefaultDijkstraPricingConfig,
    DefaultLinearMasterConfig,
    data,
    readinstance,
    solveUMF,
    solveUMFrescaled,
    set_config,
    model,
    optimizer,
    save,
    nv,ne,nk,
    load_solution,
    load_solverstats,
    is_instance_path,
    # ML
    accuracy,
    precision,    
    recall,
    f_beta_score,
    graph_reduction
    metrics,
    full_filter_mask,
    EdgeReverseLayer,
    MPLayer,
    ClassifierModel

# Compact solver
include("compact/directsolvercompact.jl")

export directsolveUMFcompact

# Column generation base

include("colgen_base/solverCG.jl")

export initializemodel!,
    solveCG,
    roundingsol,
    baselinerounding,
    addcol_k!,
    duals,
    sol,
    xopt,
    getx,
    gety,
    set_demand!,
    chgcost!,
    solve_master!,
    solve_pricing!


export solve, solve!

end #Module
