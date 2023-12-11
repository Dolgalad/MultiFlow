@testset "Augmented Graph" begin
    # Write your tests here.
    # toy instances
    test_instances = ["../../instances/toytests/test0/",
                      "../../instances/toytests/test1/",
                      "../../instances/toytests/test2/",
                      "../../instances/toytests/test3/",
                      "../../instances/prc/small/mixed/1/",
                      "../../instances/prc/small/mixed/2/",

                     ]

    for inst_path in test_instances
        inst = UMFData(inst_path)
        ag = UMFSolver.augmented_graph(inst)
        inst1 = UMFSolver.get_instance(ag)
        @test inst.costs == inst1.costs
        @test inst.capacities == inst1.capacities
        @test inst.bandwidths == inst1.bandwidths
        @test inst.srcnodes == inst1.srcnodes
        @test inst.dstnodes == inst1.dstnodes
        @test inst.srcdemands == inst1.srcdemands
        @test inst.dstdemands == inst1.dstdemands


    end
end
