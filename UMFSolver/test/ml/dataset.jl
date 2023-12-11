using GraphNeuralNetworks

@testset "Dataset" begin
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
        sol, ss = solveUMF(inst, "CG", "highs", "./output.txt")
        # make small dataset, do not scale instances
        UMFSolver.make_dataset(inst, 10, "dummy_dataset", show_progress=false)
        @test isdir("dummy_dataset")
        # load instance
        dataset = UMFSolver.load_dataset("dummy_dataset", scale_instances=false)
        g = dataset[1]

        @test length(dataset) == 10
        @test g isa GNNGraph
        @test nv(g) == nv(inst) + nk(inst)
        @test ne(g) == ne(inst) + 2*nk(inst)
        @test sum(g.edata.mask) == ne(inst)
        @test sum(g.ndata.mask) == nv(inst)
        # cost and capacities should be the same
        @test all([(g.e[:,g.edata.mask] == dataset[i].e[:,dataset[i].edata.mask]) for i in 2:length(dataset)])
        @test all([(size(_g.targets[:,:,1]) == size(sol.x)) for _g in dataset])

        dataset = UMFSolver.load_dataset("dummy_dataset", scale_instances=true)
        @test length(dataset) == 10
        @test dataset[1] isa GNNGraph
        # cost and capacities should be the same
        g = dataset[1]
        g1 = dataset[2]
        println(g.e[:,g.edata.mask] == g1.e[:,g1.edata.mask])
        @test all([(g.e[:,g.edata.mask] == dataset[i].e[:,dataset[i].edata.mask]) for i in 2:length(dataset)])
        @test all([(size(_g.targets[:,:,1]) == size(sol.x)) for _g in dataset])

        # batching
        batch_g = batch([_g for _g in dataset])
        @test batch_g isa GNNGraph
        @test nv(batch_g) == sum([nv(_g) for _g in dataset])
        @test ne(batch_g) == sum([ne(_g) for _g in dataset])
        @test batch_g.num_graphs == 10
        @test size(batch_g.targets,3) == 10
        

        #rm("dummy_dataset",recursive=true, force=true)
        #rm("./output.txt",recursive=true)


    end

end
