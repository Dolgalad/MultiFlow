
@testset "test_read_solve" begin
    # Write your tests here.
    path = "../../instances/toytests/test1/"
    type = "compactLR"
    optim = "highs"
    outfile = "../../output/julia/test1.txt"
    master = ""
    pricing = ""

    if isinstalled("CPLEX")
        optim = "cplex"
    end
    val = solveUMF(path, type, optim, outfile)
    @test val ≈ 145.0
    type = "compactINT"
    optim = "highs"
    val = solveUMF(path, type, optim, outfile)
    @test val ≈ 145.0

    path = "../../instances/toytests/test2/"
    type = "compactLR"
    if isinstalled("CPLEX")
        optim = "cplex"
    end
    outfile = "../../output/julia/test2.txt"
    master = ""
    pricing = ""
    val = solveUMF(path, type, optim, outfile)
    @test val ≈ 513.0
    type = "compactINT"
    optim = "highs"
    val = solveUMF(path, type, optim, outfile)
    @test val ≈ 605.0

    type = "CG"
    if isinstalled("CPLEX")
        optim = "cplex"
    end
    sol,stats = solveUMF(path, type, optim, outfile)
    @test stats.stats["val"] ≈ 513.0

    master = "linear"
    pricing = "dijkstra"
    sol,stats = solveUMF(path, type, optim, outfile)
    @test stats.stats["val"] ≈ 513.0

end
