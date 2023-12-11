using GraphPlot
using Compose
import Cairo
using Plots

using UMFSolver
using Graphs

instance_dir = "instances/sndlib/delay_corrected"
instances = ["Oxford_0_1_1", "Ntt_0_1_1", "AttMpls_0_1_1", "Chinanet_0_1_1", "giul39_0_1_1", "zib54_0_1_1", "Iij_0_1_1", "AsnetAm_0_1_1", "india35_0_1_1"]

output_dir = "sndlib_networks"
mkpath(output_dir)

function isolated_vertices(g)
    l = Int64[]
    for n in 1:nv(g)
        if length(neighbors(g, n))==0
            push!(l, n)
        end
    end
    return l
end
function clean_graph(g)
    rem_vertices!(g, isolated_vertices(g))
end

for i in instances
    inst = UMFData(joinpath(instance_dir, i))
    g = UMFSolver.get_graph(inst)
    clean_graph(g)
    posx,posy =spring_layout(g, 0)

    println("$i $(nv(inst)) $(ne(inst)) $(nk(inst))")
    
    draw(PNG(joinpath(output_dir, "$i.png"), 16cm, 16cm), gplot(g, posx, posy, 
                                                  NODESIZE=0.04,
                                                  #nodelabel=1:nv(g),
                                                  NODELABELSIZE=4,
                                                  arrowlengthfrac=0, 
                                                  nodefillc=colorant"orange",
                                                  edgestrokec=colorant"black",
                                                  #edgelinewidth=edgelinewidth,
                                                 ))
end

