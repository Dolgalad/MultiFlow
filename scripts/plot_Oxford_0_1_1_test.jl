using GraphPlot
using Compose
import Cairo
using Plots

using UMFSolver
using Graphs
using BSON: @load
using Flux
using JLD

model_checkpoint_path = "models/Oxford_0_1_1/model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1/best_checkpoint.bson"
model_checkpoint_path = "models/AsnetAm_0_1_1/model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1/best_checkpoint.bson"
model_checkpoint_path = "models/Chinanet_0_1_1/model8_l4_lr1.0e-6_h64_bs10_e10000_tversky0.1/best_checkpoint.bson"

make_plots = false

# list of network names
network_names = ["Oxford", "AsnetAm", "Ntt", "AttMpls", "Chinanet", "giul39", "india35", "zib54"]

# load model
@load model_checkpoint_path _model

# solver configs
config = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "dijkstra")
config = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "larac")

#mlconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "clssp $(model_checkpoint_path) 0")
mlconfig = UMFSolver.set_config("CG", "cplex", "./output.txt", "linear", "clslarac prediction.jld 0")


# output image directory
output_dir = joinpath(dirname(model_checkpoint_path), "test_imgs")

# test instance directory
test_instance_path = "datasets/Oxford_0_1_1/test"
test_instance_path = "datasets/AsnetAm_0_1_1/test"
test_instance_path = "datasets/Chinanet_0_1_1_delay/test"

ms_pr_time_prop(x) = 100 * [x.stats["time_ms_sol"]/x.stats["timetot"], 
                            x.stats["time_pr_sol"]/x.stats["timetot"]]

# store testing metrics
speedups, optimalities, graph_reductions=[],[],[]
for f in readdir(test_instance_path)
    if !UMFSolver.is_instance_path(joinpath(test_instance_path, f))
        println("$f is no instance")
        continue
    end

    instance_path = joinpath(test_instance_path, f)
    println("Instance : $f")

    # load instance
    inst = UMFData(instance_path)
    sinst = UMFSolver.scale(inst)
    gnng = UMFSolver.load_instance(instance_path)

    # prediction
    scores = _model(gnng)

    # save prediction
    jldopen("prediction.jld", "w") do file
        write(file, "pred", scores)
    end

    # solve problem
    s1, ss1 = solveCG(sinst, mlconfig)
    filter = ss1.stats["pr"].filter
    
    s, ss = solveCG(sinst, config)

    println("\tvals : ", [ss.stats["val"], ss1.stats["val"]])
    println("\tGraph reduction   : ", round(100 * ss1.stats["graph_reduction"], digits=4), " %")
    println("\tSpeedup           : ", round(100 * (ss.stats["timetot"] - ss1.stats["timetot"]) / ss1.stats["timetot"], digits=4), " %")
    println("\tOptimality        : ", round(100 * (ss1.stats["val"] - ss.stats["val"]) / ss.stats["val"], digits=4), " %")
    println("\tTime prop origin  : ", ms_pr_time_prop(ss))
    println("\tTime prop ML      : ", ms_pr_time_prop(ss1))
    println("\tMaster speedup    : ", round(100 * (ss.stats["time_ms_sol"] - ss1.stats["time_ms_sol"]) / ss.stats["time_ms_sol"], digits=4), " %")
    println("\tPricing speedup   : ", round(100 * (ss.stats["time_pr_sol"] - ss1.stats["time_pr_sol"]) / ss.stats["time_pr_sol"], digits=4), " %")

    if ss1.stats["val"] >= ss.stats["val"]
        push!(speedups, (ss.stats["timetot"] - ss1.stats["timetot"]) / ss1.stats["timetot"])
        push!(optimalities, (ss1.stats["val"] - ss.stats["val"]) / ss.stats["val"])
        push!(graph_reductions, ss1.stats["graph_reduction"])
    end

   
    #gnng = UMFSolver.to_gnngraph(sinst)
    #gnng = UMFSolver.load_instance(instance_path)
    g = UMFSolver.get_graph(inst)
    edge_idx = Dict(Edge(inst.srcnodes[i], inst.dstnodes[i]) => i for i in 1:ne(inst))
    
    posx,posy =spring_layout(g)
    
    # prediction
    #scores = _model(gnng)
    scores = reshape(scores, numarcs(inst), numdemands(inst))
    println("\tscores: ", size(scores))
    println("\tmin : ", minimum(scores))
    println("\tmax : ", maximum(scores))

    if !make_plots
        continue
    end
    
    # demand number
    for demand_n in 1:nk(inst)
        # image dir
        img_dir = joinpath(output_dir, f, string(demand_n))
        mkpath(img_dir)
        
        
        # node colors
        nodefillc = [colorant"turquoise" - colorant"rgb(0,0,0,0.5)" for _ in 1:nv(g)]
        nodefillc[inst.srcdemands[demand_n]] = colorant"blue" - colorant"rgb(0,0,0,0.5)" 
        nodefillc[inst.dstdemands[demand_n]] = colorant"red"- colorant"rgb(0,0,0,0.5)" 
        
        # edge colors
        graph_edges = collect(edges(g))
        edgestrokec = [colorant"lightgray" - colorant"rgb(0,0,0,0.5)"  for _ in 1:ne(g)]
        edgestrokec2 = [colorant"lightgray" - colorant"rgb(0,0,0,0.5)"  for _ in 1:ne(g)]
        edgestrokec3 = [colorant"lightgray" - colorant"rgb(0,0,0,0.5)"  for _ in 1:ne(g)]
        edgestrokec4 = [colorant"lightgray" - colorant"rgb(0,0,0,0.5)"  for _ in 1:ne(g)]
        
        edgelinewidth = [0.5 for _ in 1:ne(g)]
        sol_edge_idx = findall(>(0), s.x[:,demand_n])
        for v in sol_edge_idx
            e = Edge(arcsource(inst, v), arcdest(inst, v))
            edgestrokec[findfirst(==(e), graph_edges)] = colorant"green"
            edgelinewidth[findfirst(==(e), graph_edges)] = 1
            #println(e, ", ", edge_idx[e], ", ", graph_edges[edge_idx[e]])
        end
        
        
        # prediction edge width
        prededgelinewidth = zeros(ne(g))
        for v in findall(>(0), scores[:,demand_n])
            e = Edge(arcsource(inst, v), arcdest(inst, v))
            edgestrokec2[findfirst(==(e), graph_edges)] = colorant"red" - colorant"rgb(0,0,0,0.8)"
            prededgelinewidth[findfirst(==(e), graph_edges)] = sigmoid(scores[v,demand_n])
        end
        
        # filter
        #filter_m, filter_gs = UMFSolver.get_filter_masks(UMFSolver.scale(inst), model_checkpoint_path, K=0)
        filter_m, filter_gs = filter.masks, filter.graphs
        
        prededgelinewidth3 = zeros(ne(g))
        for v in findall(==(1), filter_m[demand_n])
            e = Edge(arcsource(inst, v), arcdest(inst, v))
            idx = findfirst(==(e), graph_edges)
            #println([e, graph_edges[idx], sigmoid(scores[v,1])])
            edgestrokec3[idx] = colorant"blue" - colorant"rgb(0,0,0,0.8)"
            prededgelinewidth3[idx] = 1#sigmoid(scores[v,1])
        
        end
        
        # labels
        prededgelinewidth4 = zeros(ne(g))
        #println("demand : ", [inst.srcdemands[demand_n], inst.dstdemands[demand_n]])
        for v in findall(==(1), gnng.targets[:,demand_n,1])
            e = Edge(arcsource(inst, v), arcdest(inst, v))
            idx = findfirst(==(e), graph_edges)
            #println("labeled : ", [e, graph_edges[idx], sigmoid(scores[v,demand_n])])
            edgestrokec4[idx] = colorant"blue" - colorant"rgb(0,0,0,0.8)"
            prededgelinewidth4[idx] = 1#sigmoid(scores[v,1])
        
        end
        
        ds,dt = UMFSolver.demand_endpoints(gnng)
        demand_ep = hcat(ds, dt)
        demand_sorted_idx = sortperm(view.(Ref(demand_ep), 1:size(demand_ep, 1), :))
         
        mscores = [-1.0e8 for _ in 1:ne(g)]
        cg = cgrad(:lighttest, 4, categorical=true)
        sigscore = (scores[:,demand_n] .- minimum(scores[:,demand_n])) ./ (maximum(scores[:,demand_n]) - minimum(scores[:,demand_n]))
        for v in 1:ne(g)
            e = Edge(arcsource(inst, v), arcdest(inst, v))
            re = Edge(arcdest(inst,v), arcsource(inst,v))
            idx = findfirst(==(e), graph_edges)
            idxr = findfirst(==(re), graph_edges)
        
            mscores[idx] = max(sigscore[v], mscores[idxr])
        
            if sigscore[v]>mscores[idxr]
                mscores[idxr] = sigscore[v]
            end
        end
        edgestrokec5 = map(x->cg[x] - colorant"rgb(0,0,0,0.7)", mscores)
        
        
        draw(PNG(joinpath(img_dir, "graph_original.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                          NODESIZE=0.02,
                                                          nodelabel=1:nv(g),
                                                          NODELABELSIZE=2,
                                                          arrowlengthfrac=0, 
                                                          nodefillc=nodefillc,
                                                          edgestrokec=edgestrokec,
                                                          edgelinewidth=edgelinewidth,
                                                         ))
        draw(PNG(joinpath(img_dir,"graph_prediction.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                          NODESIZE=0.02,
                                                          nodelabel=1:nv(g),
                                                          NODELABELSIZE=2,
                                                          arrowlengthfrac=0, 
                                                          nodefillc=nodefillc,
                                                          edgestrokec=edgestrokec2,
                                                          edgelinewidth=prededgelinewidth,
                                                         ))
        draw(PNG(joinpath(img_dir, "graph_masked.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                          NODESIZE=0.02,
                                                          nodelabel=1:nv(g),
                                                          NODELABELSIZE=2,
                                                          arrowlengthfrac=0, 
                                                          nodefillc=nodefillc,
                                                          edgestrokec=edgestrokec3,
                                                          edgelinewidth=prededgelinewidth3,
                                                         ))
        
        draw(PNG(joinpath(img_dir,"graph_labels.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                          NODESIZE=0.02,
                                                          nodelabel=1:nv(g),
                                                          NODELABELSIZE=2,
                                                          arrowlengthfrac=0, 
                                                          nodefillc=nodefillc,
                                                          edgestrokec=edgestrokec4,
                                                          edgelinewidth=prededgelinewidth4,
                                                         ))
        draw(PNG(joinpath(img_dir,"graph_pred_colors.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                          NODESIZE=0.02,
                                                          nodelabel=1:nv(g),
                                                          NODELABELSIZE=2,
                                                          arrowlengthfrac=0, 
                                                          nodefillc=nodefillc,
                                                          edgestrokec=edgestrokec5,
                                                          #edgelinewidth=prededgelinewidth4,
                                                         ))
    end
end

