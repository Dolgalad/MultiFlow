@testset "GNNGraph" begin
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
        g = UMFSolver.to_gnngraph(inst)
        nnodes = sum(g.ndata.mask)
        nedges = sum(g.edata.mask)
        ndemands = size(g.ndata.mask, 1) - nnodes
        edge_src, edge_dst = edge_index(g)
        @test nnodes == numnodes(inst)
        @test nedges == numarcs(inst)
        @test ndemands == numdemands(inst)
        # graph edges
        @test edge_src[g.edata.mask] == inst.srcnodes
        @test edge_dst[g.edata.mask] == inst.dstnodes
        # demand origins
        @test edge_dst[.!g.edata.mask][1:ndemands] == inst.srcdemands
        # demand targets
        @test edge_src[.!g.edata.mask][ndemands+1:end] == inst.dstdemands
        # check features
        @test g.edata.e[1,1:nedges] == inst.costs # real edge costs
        @test g.edata.e[1,g.edata.mask] == inst.costs # real edge costs
        @test g.edata.e[2,1:nedges] == inst.capacities # real edge capacities
        @test g.edata.e[2,g.edata.mask] == inst.capacities # real edge capacities
        @test all(g.edata.e[3,1:nedges] .== 0) # demand edge feature
        @test all(g.edata.e[3,g.edata.mask] .== 0) # demand edge feature
        @test g.edata.e[3,nedges+1:nedges+ndemands] == inst.bandwidths # demand to origin feature
        @test g.edata.e[3,nedges+ndemands+1:end] == -inst.bandwidths # target to demand feature
        @test all(g.edata.e[1:2,nedges+1:end] .== 0) # no cost or capacity for demand edges
        @test all(g.edata.e[1:2,.!g.edata.mask] .== 0) # no cost or capacity for demand edges

        # for GPU compatibility
        g = UMFSolver.to_gnngraph(inst; feature_type=Float32)
        nnodes = sum(g.ndata.mask)
        nedges = sum(g.edata.mask)
        ndemands = size(g.ndata.mask, 1) - nnodes
        edge_src, edge_dst = edge_index(g)
        @test nnodes == numnodes(inst)
        @test nedges == numarcs(inst)
        @test ndemands == numdemands(inst)
        # graph edges
        @test edge_src[g.edata.mask] == inst.srcnodes
        @test edge_dst[g.edata.mask] == inst.dstnodes
        # demand origins
        @test edge_dst[.!g.edata.mask][1:ndemands] == inst.srcdemands
        # demand targets
        @test edge_src[.!g.edata.mask][ndemands+1:end] == inst.dstdemands
        # check features, doing approximate comparison since type of features goes from Float64 to Float32 (for gpu compatibility)
        @test isapprox(g.edata.e[1,1:nedges], inst.costs) # real edge costs
        @test isapprox(g.edata.e[1,g.edata.mask], inst.costs) # real edge costs
        @test isapprox(g.edata.e[2,1:nedges], inst.capacities) # real edge capacities
        @test isapprox(g.edata.e[2,g.edata.mask], inst.capacities) # real edge capacities
        @test all(g.edata.e[3,1:nedges] .== 0) # demand edge feature
        @test all(g.edata.e[3,g.edata.mask] .== 0) # demand edge feature
        @test isapprox(g.edata.e[3,nedges+1:nedges+ndemands], inst.bandwidths) # demand to origin feature
        @test isapprox(g.edata.e[3,nedges+ndemands+1:end], -inst.bandwidths) # target to demand feature
        @test all(g.edata.e[1:2,nedges+1:end] .== 0) # no cost or capacity for demand edges
        @test all(g.edata.e[1:2,.!g.edata.mask] .== 0) # no cost or capacity for demand edges

        # with target labels
        sol, ss = solveUMF(inst, "CG", "highs", "./output.txt")
        g = UMFSolver.to_gnngraph(inst, sol.x)
        @test size(g.targets[:,:,1]) == size(sol.x)
        @test g.targets[:,:,1] == sol.x
        rm("./output.txt", force=true)

      
        
    end
end
