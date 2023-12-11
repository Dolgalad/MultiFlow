"Abstract configuration type"
abstract type AbstractConfiguration end

"Abstract configuration type for column generation solver"
abstract type AbstractColGenConfiguration <: AbstractConfiguration end

"Abstract configuration type for compact solver"
abstract type AbstractCompactConfiguration <: AbstractConfiguration end

"""
Abstract configuration type for master problem

See also [`DefaultLinearMasterConfig`](@ref)
"""
abstract type MsColGenconfig <: AbstractConfiguration end

"""
Abstract configuration type for pricing problem

See also [`DefaultDijkstraPricingConfig`](@ref)
"""
abstract type PrColGenconfig <: AbstractConfiguration end

"""
    DefaultLinearMasterConfig

Default configuration type for master sub-problem of a column generation, sub-type of `MsColGenconfig`.

Solve the master with the LP solver chosen in the configuration for the colunm generation.

See also [`ColGenConfigBase`](@ref)
"""
struct DefaultLinearMasterConfig <: MsColGenconfig end


"""
    DefaultDijkstraPricingConfig

Default configuration type for pricing sub-problem of a column generation, sub-type of `PrColGenconfig`.

Solve a pricing **shortest path** problem with the Dijkstra algorithm.

See also [`ColGenConfigBase`](@ref)
"""
struct DefaultDijkstraPricingConfig <: PrColGenconfig end


"""
    LARACPricingConfig

Configuration type for pricing sub-problem using LARAC constrained shortest-path algorithm.

See also [`ColGenConfigBase`](@ref)
"""
struct LARACPricingConfig <: PrColGenconfig end


"""
    kSPFilterPricingConfig
    
Solve a pricing **shortest path** problem with Dijkstra algorithm on a graph that have previously 
modified to include only the k-shortest paths satisfying the current demand.
"""
struct kSPFilterPricingConfig <: PrColGenconfig 
    K::Int64
    sptable
end

struct kSPLARACPricingConfig <: PrColGenconfig 
    K::Int64
    sptable
end


struct ClassifierAndSPConfig <: PrColGenconfig 
    model
    K::Int64
    sptable
    threshold::Float64
    keep_proportion::Float64
    postprocessing_method::Int64
end

struct SVMAndSPConfig <: PrColGenconfig 
    model
end

struct RFAndSPConfig <: PrColGenconfig 
    model
end

struct MLPAndSPConfig <: PrColGenconfig 
    model
end


struct ClassifierAndLARACConfig <: PrColGenconfig
    model
    K::Int64
    sptable
    threshold::Float64
    keep_proportion::Float64
end


"""
    ColGenConfigBase(
        msconfig::MsColGenconfig
        prconfig::PrColGenconfig
        optimizer::String
        outputname::String
    )

    Default configuration structure for column generation solver

# Arguments

- `msconfig::MsColGenconfig`: configuration type for master subproblem
- `prconfig::PrColGenconfig`: configuration type for pricing subproblem
- `optimizer::String`: default optimizer used in the column generation
- `outputname::String`: name of the output file.

See also [`MsColGenconfig`](@ref), [`PrColGenconfig`](@ref)

"""
struct ColGenConfigBase <: AbstractColGenConfiguration
    msconfig::MsColGenconfig
    prconfig::PrColGenconfig
    optimizer::String
    outputname::String
end

function set_output_file(config::ColGenConfigBase, filename::String)
    return ColGenConfigBase(config.msconfig, config.prconfig, config.optimizer, filename)
end


"""
    CompactConfigBase(
        relaxed::Bool
        optimizer::String
        outputname::String
    )

    Default configuration structure for compact solver

# Arguments

- `relaxed::Bool`: if `true`, solve the linear relaxation; if `false`, solve the original problem.
- `optimizer::String`: optimizer chosen to solve the problem
- `outputname::String`: name of the output file.

"""
struct CompactConfigBase <: AbstractCompactConfiguration
    relaxed::Bool
    optimizer::String
    outputname::String
end
