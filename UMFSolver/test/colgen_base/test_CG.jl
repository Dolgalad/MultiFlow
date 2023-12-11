
@testset "columngen" begin
    # Write your tests here.
    # toy instances
    dat_0 = readinstance("../../instances/toytests/test0/")
    dat_1 = readinstance("../../instances/toytests/test1/")
    dat_2 = readinstance("../../instances/toytests/test2/")
    dat_3 = readinstance("../../instances/toytests/test3/")

    #colgensolve
    prconfig = DefaultDijkstraPricingConfig()
    msconfig = DefaultLinearMasterConfig()
    config = ColGenConfigBase(msconfig, prconfig, "highs", "outputfileNULL")
    mod_cg_0_sol, mod_cg_0_ss = solveCG(dat_0, config)
    mod_cg_1_sol, mod_cg_1_ss = solveCG(dat_1, config)
    if isinstalled("CPLEX")
        config = ColGenConfigBase(msconfig, prconfig, "cplex", "outputfileNULL")
    end
    mod_cg_2_sol, mod_cg_2_ss = solveCG(dat_2, config)
    mod_cg_3_sol, mod_cg_3_ss = solveCG(dat_3, config)

    #some tests on the solutions.
    #test on optimal values
    value_0 = mod_cg_0_ss.stats["val"]
    value_1 = mod_cg_1_ss.stats["val"]
    value_2 = mod_cg_2_ss.stats["val"]
    value_3 = mod_cg_3_ss.stats["val"]

    @test value_0 ≈ 135
    @test value_1 ≈ 145
    @test value_2 ≈ 513
    @test value_3 ≈ 605


    #test on optimal vectors

    vect_0 = mod_cg_0_sol.x
    vect_1 = mod_cg_1_sol.x
    vect_2 = mod_cg_2_sol.x
    vect_3 = mod_cg_3_sol.x

    @test vect_0[1,1] ≈ vect_0[7,1] ≈ vect_0[8,1] ≈ 1.0
    @test vect_0[3,2] ≈ vect_0[7,2] ≈ 1.0
    @test vect_0[3,3] ≈ vect_0[7,3] ≈ vect_0[8,3] ≈ vect_0[15,3] ≈ 1.0


    #maybe vect_1 is not unique.

    @test vect_2[1,1] ≈ vect_2[7,1] ≈ vect_2[8,1] ≈ 0.2
    @test vect_2[9,1] ≈ vect_2[10,1] ≈ 0.8
    @test vect_2[3,2] ≈ vect_2[7,2] ≈ 1.0
    @test vect_2[3,3] ≈ vect_2[4,3] ≈ vect_2[15,3] ≈ 1.0

    @test vect_3[9,1] ≈ vect_3[10,1] ≈ 1.0
    @test vect_3[3,2] ≈ vect_3[7,2] ≈ 1.0
    @test vect_3[3,3] ≈ vect_3[4,3] ≈ vect_3[15,3] ≈ 1.0



    #test on number of iterations and columns generated
    nit_0 = mod_cg_0_ss.stats["nits"]
    nit_1 = mod_cg_1_ss.stats["nits"]
    nit_2 = mod_cg_2_ss.stats["nits"]
    nit_3 = mod_cg_3_ss.stats["nits"]

    ncol_0 = mod_cg_0_ss.stats["ncols"]
    ncol_1 = mod_cg_1_ss.stats["ncols"]
    ncol_2 = mod_cg_2_ss.stats["ncols"]
    ncol_3 = mod_cg_3_ss.stats["ncols"]

    @test nit_0 == 1 + 1
    @test nit_1 == 3 + 1
    @test nit_2 == 3 + 1
    @test nit_3 == 3 + 1

    @test ncol_0 == 0 + numdemands(dat_0)
    @test ncol_1 == 6 + numdemands(dat_1)
    @test ncol_2 == 6 + numdemands(dat_2)
    @test ncol_3 == 6 + numdemands(dat_3)

    #some tests on first column generation iterations:
    #test that all columns are as expected.


    ms0 = mod_cg_0_ss.stats["ms"]
    ms1 = mod_cg_1_ss.stats["ms"]
    ms2 = mod_cg_2_ss.stats["ms"]
    ms3 = mod_cg_3_ss.stats["ms"]


    @test UMFSolver.column_x_k_p(ms0, 1, 1) == [1, 7, 8]
    @test UMFSolver.column_x_k_p(ms0, 2, 1) == [3, 7]
    @test UMFSolver.column_x_k_p(ms0, 3, 1) == [15, 3, 7, 8]

    @test UMFSolver.column_x_k_p(ms2, 1, 1) == [1, 7, 8]
    @test UMFSolver.column_x_k_p(ms2, 1, 2) == [1, 4]
    @test UMFSolver.column_x_k_p(ms2, 1, 3) == [9, 10]
    @test UMFSolver.column_x_k_p(ms2, 2, 1) == [3, 7]
    @test UMFSolver.column_x_k_p(ms2, 2, 2) == [3, 4, 18]
    @test UMFSolver.column_x_k_p(ms2, 2, 3) == [12, 9, 10, 18]
    @test UMFSolver.column_x_k_p(ms2, 3, 1) == [15, 3, 7, 8]
    @test UMFSolver.column_x_k_p(ms2, 3, 2) == [15, 3, 4]
    @test UMFSolver.column_x_k_p(ms2, 3, 3) == [15, 12, 9, 10]





    # small instances
    dat_sm1 = readinstance("../../instances/prc/small/mixed/1/")
    dat_sm2 = readinstance("../../instances/prc/small/mixed/2/")

    #colgensolve
    prconfig = DefaultDijkstraPricingConfig()
    msconfig = DefaultLinearMasterConfig()
    if isinstalled("CPLEX")
        config = ColGenConfigBase(msconfig, prconfig, "cplex", "outputfileNULL")
    else
        config = ColGenConfigBase(msconfig, prconfig, "highs", "outputfileNULL")
    end
    mod_cg_sm1_sol, mod_cg_sm1_ss = solveCG(dat_sm1, config)
    mod_cg_sm2_sol, mod_cg_sm2_ss = solveCG(dat_sm2, config)

    #some tests:
    #objective values
    value_sm1 = mod_cg_sm1_ss.stats["val"]
    value_sm2 = mod_cg_sm2_ss.stats["val"]

    @test value_sm1 ≈ 150369.34881516828
    @test value_sm2 ≈ 160264.5359289629

    #optimal solutions

    vect_sm1 = mod_cg_sm1_sol.x
    vect_sm2 = mod_cg_sm2_sol.x

    @test vect_sm1[4,1] ≈
          vect_sm1[441,1] ≈
          vect_sm1[649,1] ≈
          vect_sm1[655, 1] ≈
          vect_sm1[689, 1] ≈
          1.0
    @test vect_sm1[5,2] ≈
          vect_sm1[250,2] ≈
          vect_sm1[649,2] ≈
          vect_sm1[655,2] ≈
          vect_sm1[689,2] ≈
          1.0
    @test vect_sm1[2,3] ≈
          vect_sm1[644,3] ≈
          vect_sm1[649,3] ≈
          vect_sm1[655,3] ≈
          vect_sm1[689,3] ≈
          1.0
    @test vect_sm1[7,4] ≈
          vect_sm1[478,4] ≈
          vect_sm1[650,4] ≈
          vect_sm1[667,4] ≈
          vect_sm1[683,4] ≈
          1.0
    @test vect_sm1[7,5] ≈
          vect_sm1[478,5] ≈
          vect_sm1[650,5] ≈
          vect_sm1[667,5] ≈
          vect_sm1[683,5] ≈
          1.0
    @test vect_sm1[8,6] ≈ vect_sm1[709,6] ≈ vect_sm1[1166,6] ≈ 1.0


    @test vect_sm2[4,1] ≈
          vect_sm2[337,1] ≈
          vect_sm2[691,1] ≈
          vect_sm2[698,1] ≈
          vect_sm2[729,1] ≈
          1.0
    @test vect_sm2[5,2] ≈
          vect_sm2[611,2] ≈
          vect_sm2[692,2] ≈
          vect_sm2[712,2] ≈
          vect_sm2[729,2] ≈
          1.0
    @test vect_sm2[1,3] ≈
          vect_sm2[551,3] ≈
          vect_sm2[692,3] ≈
          vect_sm2[712,3] ≈
          vect_sm2[729,3] ≈
          1.0


    #number of iterations and columns generated

    nit_sm1 = mod_cg_sm1_ss.stats["nits"]
    nit_sm2 = mod_cg_sm2_ss.stats["nits"]

    ncol_sm1 = mod_cg_sm1_ss.stats["ncols"]
    ncol_sm2 = mod_cg_sm2_ss.stats["ncols"]

    @test nit_sm1 == 1 + 1
    @test nit_sm2 == 2 + 1

    @test ncol_sm1 == 0 + numdemands(dat_sm1)
    @test ncol_sm2 == 35 + numdemands(dat_sm2)


    #test that all columns are as expected.
    ms_sm2 = mod_cg_sm2_ss.stats["ms"]

    pr_sm1 = mod_cg_sm1_ss.stats["pr"]
    pr_sm2 = mod_cg_sm2_ss.stats["pr"]

    @test UMFSolver.column_x_k_p(ms_sm2, 4, 1) == [7, 290, 691, 698, 729]
    @test UMFSolver.column_x_k_p(ms_sm2, 4, 2) == [8, 689, 691, 698, 729]
    @test UMFSolver.column_x_k_p(ms_sm2, 5, 1) == [10, 290, 691, 698, 729]
    @test UMFSolver.column_x_k_p(ms_sm2, 5, 2) == [9, 677, 692, 712, 729]
    @test UMFSolver.column_x_k_p(ms_sm2, 6, 1) == [7, 290, 691, 698, 729]
    @test UMFSolver.column_x_k_p(ms_sm2, 6, 2) == [8, 689, 691, 698, 729]


    # test on rescaled:

    path = "../../instances/toytests/test1/"
    type = "CG"
    optimizer = isinstalled("CPLEX") ? "cplex" : "highs"
    outfile = "../../output/test_toy1_f0.5.txt"
    factor = 0.5

    sol, sstats = solveUMFrescaled(path, type, optimizer, outfile, factor)
    #@test sol ≈ 1239.0
    @test sstats.stats["val"] ≈ 1239.0


    dat_1_bis = readinstance(path, factor)
    mod_cg_1_bis_sol, mod_cg_1_bis_ss = solveCG(dat_1_bis, config)
    # optimal vector
    vect_1_bis = mod_cg_1_bis_sol.x
    #master problem
    ms1_bis = mod_cg_1_bis_ss.stats["ms"]
    @test vect_1_bis == hcat(UMFSolver.xopt(ms1_bis)...)

    @test vect_1_bis[9,1] ≈ vect_1_bis[10,1] ≈ 1.0
    @test vect_1_bis[3,2] ≈ vect_1_bis[7,2] ≈ 1.0
    @test vect_1_bis[3,3] ≈ vect_1_bis[15,3] ≈ 0.2
    @test vect_1_bis[4,3] ≈ 0.5
    @test vect_1_bis[6,3] ≈ 0.3


    x_k_p = UMFSolver.getx(ms1_bis)
    y_k = UMFSolver.gety(ms1_bis)

    @test x_k_p[1] ≈ [0.0, 0.0, 1.0]
    @test x_k_p[2] ≈ [1.0, 0.0, 0.0]
    @test x_k_p[3] ≈ [0.0, 0.2, 0.0, 0.3]
    @test y_k ≈ [0.0, 0.0, 0.5]


end
