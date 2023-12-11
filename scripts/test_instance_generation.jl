using UMFSolver

inst = UMFData("instances/prc/middle/flexE/1")

for _ in 1:10
    # generate example and solve
    t_gen = @timed inst_s = UMFSolver.generate_example(inst)
    # solve it
    sol, ss = solveUMF(inst_s, "CG", "highs", "./output.txt")
    ms = ss.stats["ms"]
    # check sol
    println("Generation time  : ", t_gen.time)
    println("Demands accepted : ", ss.stats["accepteddemands"])
    println("Value            : ", ss.stats["val"])
    println("Time             : ", ss.stats["timetot"])
    println("N iterations     : ", ss.stats["nits"])
    println("N columns        : ", ss.stats["ncols"])

    y = UMFSolver.gety(ms)
    println("sum y < 0        : ", sum(y .< 0))
    println()

end
