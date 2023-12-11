using DataFrames
using CSV

input = "/data/aschulz/Documents2023/julia/multiflow/testing_outputs/solver_outputs/test_outputs_temp_postprocessing2/solve_times.csv"
df = DataFrame(CSV.File(input))

