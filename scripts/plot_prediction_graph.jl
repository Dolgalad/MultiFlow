using GraphPlot
using Compose
import Cairo
using Plots

using UMFSolver
using Graphs
using BSON: @load
using Flux

cat = "flexE"

model_checkpoint_path = "models_small_$(cat)_1/model6b_l4_lr1.0e-6_h64_bs20_tversky0.1/checkpoint_e865.bson"
model_checkpoint_path = "model8_checkpoint_e2000.bson"
model_checkpoint_path = "model9_small_flexE_checkpoint_e1000.bson"

img_dir = joinpath(dirname(model_checkpoint_path), "imgs")
mkpath(img_dir)

instance_path = "instances/prc/small/$cat/1"
#instance_path = "datasets/dataset_prc_small_flexE_1_train/1"
#instance_path = "test_dataset/1"

tmp_inst = UMFData(instance_path)
UMFSolver.make_dataset(tmp_inst, 1, "temp_dataset")

instance_path = "temp_dataset/1"

gnng = UMFSolver.load_instance(instance_path)
println(gnng)
# training and validation datasets
function aggregate_demand_paths(g)
    ndemands = sum(.!g.ndata.mask)
    ds,dt = UMFSolver.demand_endpoints(g)
    new_labels = Dict()
    for k in 1:ndemands
        if haskey(new_labels, (ds[k], dt[k]))
            new_labels[(ds[k], dt[k])] .|= g.targets[:,k]
        else
            new_labels[(ds[k], dt[k])] = g.targets[:,k]
        end
    end
    for k in 1:ndemands
        g.targets[:,k] .= new_labels[(ds[k], dt[k])]
    end
    return g
end

gnng = aggregate_demand_paths(gnng)

inst = UMFData(instance_path)

@load model_checkpoint_path _model

# demand number
demand_n = 1

function get_graph(inst)
    g = SimpleDiGraph(numnodes(inst))
    edge_list = [Edge(inst.srcnodes[i], inst.dstnodes[i]) for i in 1:ne(inst)]
    return SimpleDiGraph(edge_list), Dict(Edge(inst.srcnodes[i], inst.dstnodes[i]) => i for i in 1:ne(inst))
    for i in 1:ne(inst)
        add_edge!(g, inst.srcnodes[i], inst.dstnodes[i])
    end
    return g
end

s, ss = solveUMF(UMFSolver.scale(inst), "CG", "highs", "./output.txt", "", "clssp $(model_checkpoint_path) 0")
println("with ml : ", ss.stats["val"])
s, ss = solveUMF(UMFSolver.scale(inst), "CG", "highs", "./output.txt")
println("without ml : ", ss.stats["val"])

println("Graph reduction : ", ss.stats["graph_reduction"])

g, edge_idx = get_graph(inst)

posx,posy =spring_layout(g)

# prediction
scores = _model(UMFSolver.to_gnngraph(UMFSolver.scale(inst)))
scores = reshape(scores, numarcs(inst), numdemands(inst))
println("scores: ", size(scores))
println("\t", [minimum(scores[:,1]), maximum(scores[:,demand_n])])
println("\t", size(scores[scores[:,demand_n].>0, demand_n]))

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
    println(e, ", ", edge_idx[e], ", ", graph_edges[edge_idx[e]])
end


# prediction edge width
prededgelinewidth = zeros(ne(g))
for v in findall(>(0), scores[:,demand_n])
    e = Edge(arcsource(inst, v), arcdest(inst, v))
    edgestrokec2[findfirst(==(e), graph_edges)] = colorant"red" - colorant"rgb(0,0,0,0.8)"
    prededgelinewidth[findfirst(==(e), graph_edges)] = sigmoid(scores[v,demand_n])
end

# filter
filter_m, filter_gs = UMFSolver.get_filter_masks(UMFSolver.scale(inst), model_checkpoint_path, K=0)
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
println("demand : ", [inst.srcdemands[demand_n], inst.dstdemands[demand_n]])
for v in findall(==(1), gnng.targets[:,demand_n,1])
    e = Edge(arcsource(inst, v), arcdest(inst, v))
    idx = findfirst(==(e), graph_edges)
    println("labeled : ", [e, graph_edges[idx], sigmoid(scores[v,demand_n])])
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
