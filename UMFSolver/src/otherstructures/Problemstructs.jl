"Abstract type for generic problem"
abstract type AbstractProblemData end

"Abstract type for master problem"
abstract type AbstractMasterPbData <: AbstractProblemData end

"Abstract type for pricing problem"
abstract type AbstractPricingPbData <: AbstractProblemData end
