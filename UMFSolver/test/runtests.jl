using UMFSolver

using JuMP

using GraphNeuralNetworks

#include("../src/UMFSolver.jl") # needed until problem is fixed
#push!(LOAD_PATH, "../src")
#using UMFSolver # just the module (for now).
using Test

const testdir = dirname(@__FILE__)

tests=["data/test_data.jl",
       "data/solution.jl",
       "colgen_base/test_CG.jl",
       "compact/test_compact.jl",
       "readsolve/test_read_solve.jl",
       "ml/augmented_graph.jl",
       "ml/gnngraph.jl",
       "ml/dataset.jl",
      ]


@testset verbose = true "UMFSolver" begin
    for t in tests
        tp = joinpath(testdir, t)
        include(tp)
    end
end;
