using BSON:@load
using UMFSolver
using Plots

include("smoothing_metrics.jl")

#model_path = "debug_model/checkpoint_e83.bson"
model_path = "debug_model4/checkpoint_e46.bson"

# debug model
#model_path = "debug_model_emb0/checkpoint_e59.bson"
instance_path = "instances/prc/small/flexE/1"

instance_name = "prc_small_flexE_1"
image_path = dirname(model_path)

# load model
@load model_path _model

# generate an instance
#inst = UMFData(instance_path)
inst = UMFSolver.scale(UMFSolver.generate_example(UMFData(instance_path)))
# keep an index of sorted demands
demand_endpoints = hcat(inst.srcdemands, inst.dstdemands, inst.bandwidths)
demand_sorted_idx = sortperm(view.(Ref(demand_endpoints), 1:size(demand_endpoints, 1), :))

# reorder demands
setindex!(inst.srcdemands, inst.srcdemands[demand_sorted_idx], 1:nk(inst))
setindex!(inst.dstdemands, inst.dstdemands[demand_sorted_idx], 1:nk(inst))
setindex!(inst.bandwidths, inst.bandwidths[demand_sorted_idx], 1:nk(inst))

# create GNNGraph
g = UMFSolver.to_gnngraph(inst)

# compute node embeddings
#node_codes, edge_codes = UMFSolver.compute_graph_embeddings(_model, g)
node_codes, edge_codes = UMFSolver.m4_compute_graph_embeddings(_model, g)

# DEBUG: only for model4
demand_codes = UMFSolver.make_demand_codes(node_codes, g)

# compute boolean adjacency matrix
adj_mat = (adjacency_matrix(g, dir=:in) .+ adjacency_matrix(g, dir=:out)) .> 0

# compute the node embedding distance matrix
nedm = node_distance_matrix(node_codes)

# plot adjacency matrices, in , out and both
in_adj_mat = adjacency_matrix(g, dir=:in)
in_adj_mat[in_adj_mat .== 0] .= -1
out_adj_mat = adjacency_matrix(g, dir=:out)
out_adj_mat[out_adj_mat .== 0] .= -1
both_adj_mat = adjacency_matrix(g, dir=:out) .+ adjacency_matrix(g, dir=:in)
both_adj_mat[both_adj_mat .== 0] .= -1

plot(heatmap(in_adj_mat, title="In adjacency matrix",cmap=cgrad(:lighttest, LinRange(0.01,1,10), rev=false, categorical=true,  scale=:log)),
     heatmap(out_adj_mat,title="Out adjacency matrix",cmap=cgrad(:lighttest, LinRange(0.01,1,10), rev=false, categorical=true,  scale=:log)),
     size=(2000,1000)
    )
savefig(joinpath(image_path, "adjacency_$instance_name.png"))

# plot adjacency matrix and distance matrix side by side
plot(heatmap(adj_mat, legend=:none,title="Augmented graph adjacency matrix"), heatmap(nedm, title="Node embedding distances"), size=(1000,500))
savefig(joinpath(image_path, "adjacency_distances_$instance_name.png"))

# same as before but logarithm of distance matrix
plot(heatmap(adj_mat, title="Augmented graph adjacency matrix"), heatmap(log.(nedm.+1), title="Node embedding log distances"), size=(2000,1000))
savefig(joinpath(image_path, "adjacency_log_distances_$instance_name.png"))

# plot only real nodes
nmask = g.ndata.mask
plot(heatmap(adj_mat[nmask,nmask], title="Graph adjacency matrix"), heatmap(nedm[nmask,nmask], title="Node embedding distances"), size=(2000,1000))
savefig(joinpath(image_path, "node_adjacency_distances_$instance_name.png"))

# same as before but logarithm of distance matrix
#plot(heatmap(adj_mat[nmask,nmask], title="Graph adjacency matrix"), heatmap(log.(nedm[nmask,nmask].+1), title="Node embedding log distances"), size=(2000,1000))

# plot only demand nodes embedding distances
#
nedm = node_distance_matrix(node_codes)
println(unique(nedm[.!nmask,.!nmask]))
println(length(unique(nedm[.!nmask,.!nmask])))
ncats = 2*length(unique(nedm[.!nmask,.!nmask]))
plot(heatmap(nedm[.!nmask,.!nmask], title="Demand embedding distances", cmap=cgrad(:lighttest, ncats, categorical=true, scale=:log)), size=(1000,1000))
savefig(joinpath(image_path, "demand_distances_$instance_name.png"))

# plot representation of demands
m = zeros(Bool, numdemands(inst), numnodes(inst))
for k in 1:numdemands(inst)
    m[k, inst.srcdemands[k]] = 1
    m[k, inst.dstdemands[k]] = 1
end
plot(heatmap(m, title="Demand-node adjacency", legend=:none), size=(1000,1000))
savefig(joinpath(image_path, "demand_node_adjacency_$instance_name.png"))

plot(heatmap(m, title="Demand-node adjacency", legend=:none), heatmap(nedm[.!nmask,.!nmask], title="Demand embedding distances", cmap=cgrad(:lighttest, ncats, categorical=true, scale=:log)), size=(2000,1000))
savefig(joinpath(image_path, "demand_node_adjacency_and_distances_$instance_name.png"))

# DEBUG: model4 demand codes
dedm = node_distance_matrix(demand_codes)
ncats = 2*length(unique(dedm))
plot(heatmap(dedm, title="Demand embedding distances", cmap=cgrad(:lighttest, ncats, categorical=true)), size=(1000,1000))
savefig(joinpath(image_path, "model4_demand_distances_$instance_name.png"))

plot(heatmap(m, title="Demand-node adjacency", legend=:none), heatmap(dedm, title="Demand embedding distances", cmap=cgrad(:lighttest, categorical=false)), size=(2000,1000))
savefig(joinpath(image_path, "model4_demand_node_adjacency_and_distances_$instance_name.png"))

