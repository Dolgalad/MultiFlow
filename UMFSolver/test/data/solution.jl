@testset "Save solution" begin
    test_instances = ["../../instances/toytests/test0/",
                      "../../instances/toytests/test1/",
                      "../../instances/toytests/test2/",
                      "../../instances/toytests/test3/",
                     ]
    for inst_path in test_instances
        inst = UMFData(inst_path)
        # solve and compare values, iterations and columns
        sol, ss = solveUMF(inst, "CG", "highs", "./output.txt")
        # save solution
        save(sol, "test_solution.jld")
        # save solver stats
        save(ss, "test_solverstats.json")
        # load the solution
        @test isfile("test_solution.jld")
        x = load_solution("test_solution.jld")
        @test sol.x == x
        # load the solver stats
        @test isfile("test_solverstats.json")
        ss1 = load_solverstats("test_solverstats.json")
        for k in keys(ss1.stats)
            @test ss.stats[k] == ss1.stats[k]
        end

        rm("test_solution.jld", force=true)
        rm("./output.txt", force=true)
        rm("test_solverstats.json", force=true)
    end
  
end
