
@testset "compact" begin
    # Write your tests here.
    # toy instances
    dat_0 = readinstance("../../instances/toytests/test0/")
    dat_1 = readinstance("../../instances/toytests/test1/")
    dat_2 = readinstance("../../instances/toytests/test2/")
    dat_3 = readinstance("../../instances/toytests/test3/")

    #compactsolve
    #only dat_2 should be enough now. Do all with dat_2. But before removing the others, see in colgen what happens.
    mod_comp_0 = directsolveUMFcompact(dat_0, "highs", true, false)
    mod_comp_1 = directsolveUMFcompact(dat_1, "highs", true, false)
    mod_comp_2 = directsolveUMFcompact(dat_2, "highs", true, false)
    mod_comp_2int = directsolveUMFcompact(dat_2, "highs", false, false)
    if isinstalled("CPLEX")
        mod_comp_3 = directsolveUMFcompact(dat_3, "cplex", true, false)
    else
        mod_comp_3 = directsolveUMFcompact(dat_3, "highs", true, false)
    end
    @test objective_value(mod_comp_0) ≈ 135
    @test objective_value(mod_comp_1) ≈ 145
    @test objective_value(mod_comp_2) ≈ 513
    @test objective_value(mod_comp_2int) ≈ 605
    @test objective_value(mod_comp_3) ≈ 605

    xopt_0 = value.(mod_comp_0[:x])

    @test xopt_0[1, 1] ≈ xopt_0[7, 1] ≈ xopt_0[8, 1] ≈ 1.0
    @test xopt_0[3, 2] ≈ xopt_0[7, 2] ≈ 1.0
    @test xopt_0[3, 3] ≈ xopt_0[7, 3] ≈ xopt_0[8, 3] ≈ xopt_0[15, 3] ≈ 1.0


end
