
@testset "UMFData" begin
    # Write your tests here.
    # toy instances
    dat_0 = readinstance("../../instances/toytests/test0/")
    dat_1 = readinstance("../../instances/toytests/test1/")
    dat_2 = readinstance("../../instances/toytests/test2/")
    dat_3 = readinstance("../../instances/toytests/test3/")
    @test arcsource(dat_0, 1) == 1
    @test arcdest(dat_0, 1) == 4
    @test numarcs(dat_0) == numarcs(dat_1) == numarcs(dat_2) == 20
    @test numnodes(dat_0) == numnodes(dat_1) == numnodes(dat_2) == 7
    @test numdemands(dat_0) == numdemands(dat_1) == numdemands(dat_2) == 3
    @test capacity(dat_0, 2) == capacity(dat_1, 2) == capacity(dat_2, 2) == 12
    @test capacity(dat_0, 7) == 20
    @test capacity(dat_1, 7) == 10
    @test capacity(dat_2, 7) == 6
    @test capacity(dat_3, 7) == 5
    @test cost(dat_0, 4) ≈ 8
    @test demandorigin(dat_1, 2) == 2
    @test demanddest(dat_1, 3) == 7
    @test bdws(dat_0) == [5, 5, 5]
    @test nv(dat_0) == numnodes(dat_0)
    @test ne(dat_0) == numarcs(dat_0)
    @test nk(dat_0) == numdemands(dat_0)



    # small instances
    dat_sm1 = readinstance("../../instances/prc/small/mixed/1/")
    dat_sm2 = readinstance("../../instances/prc/small/mixed/2/")
    @test arcsource(dat_sm1, 1) == 1
    @test arcdest(dat_sm1, 1) == 44
    @test numarcs(dat_sm1) == 1404
    @test numarcs(dat_sm2) == 1482
    @test numdemands(dat_sm1) == numdemands(dat_sm2) == 60
    @test numnodes(dat_sm1) == numnodes(dat_sm2) == 71
    @test capacity(dat_sm1, 2) == 10000
    @test capacity(dat_sm2, 2) == 10000
    @test cost(dat_sm1, 7) ≈ 1.3418049802315528
    @test cost(dat_sm2, 7) ≈ 1.3449100303706698
    @test demandorigin(dat_sm1, 60) == 6 # = 1+ number written in service file
    @test demanddest(dat_sm1, 60) == 71 # = 1+ number written in service file
    @test bdw(dat_sm1, 60) == 500

end

@testset "Save UMFData" begin
    test_instances = ["../../instances/toytests/test0/",
                      "../../instances/toytests/test1/",
                      "../../instances/toytests/test2/",
                      "../../instances/toytests/test3/",
                     ]
    for inst_path in test_instances
        inst = UMFData(inst_path)
        # save instance
        save(inst, "test_instance", verbose=false)
        # load instance
        inst1 = UMFData("test_instance")
        @test numnodes(inst) == numnodes(inst1)
        @test numarcs(inst) == numarcs(inst1)
        @test numdemands(inst) == numdemands(inst1)
        # solve and compare values, iterations and columns
        sol, ss = solveUMF(inst, "CG", "highs", "./output.txt")
        sol1, ss1 = solveUMF(inst1, "CG", "highs", "./output.txt")
        @test ss.stats["val"] == ss1.stats["val"]
        @test ss.stats["nits"] == ss1.stats["nits"]
        @test ss.stats["ncols"] == ss1.stats["ncols"]

    end
    #rm("test_instance", force=true, recursive=true)
    rm("./output.txt", force=true)
  
end
