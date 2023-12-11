# graph opposite layer
struct M3EdgeReverseLayer <: GraphNeuralNetworks.GNNLayer
end

function (l::M3EdgeReverseLayer)(g::GNNGraph)
    return g
    s, t = edge_index(g)
    return GNNGraph(t, s, ndata=(;x=g.ndata.x), edata=(;e=g.edata.e))
end

# message passing layer
struct M3MPLayer
    forward_conv
    drop1
    bn_n
    bn_e
    rev
    backward_conv
    #drop2
end

Flux.@functor M3MPLayer

function MyMEGNetConv(n; drop_p=0.1)
    phie = Chain([Flux.Dense(3*n=>2*n, relu),
		  Flux.Dropout(drop_p),
                  x -> Flux.normalise(x),
		  Flux.Dense(2*n=>n, relu)
		  ])
    phiv = Chain([Flux.Dense(2*n=>2*n, relu),
		  Flux.Dropout(drop_p),
                  x -> Flux.normalise(x),
		  Flux.Dense(2*n=>n, relu)
		  ])
    return MEGNetConv(phie, phiv)
end

function M3MPLayer(n::Int; drop_p::Float64=.1)
    return M3MPLayer(MyMEGNetConv(n), Flux.Dropout(drop_p), Flux.BatchNorm(n), Flux.BatchNorm(n), M3EdgeReverseLayer(), MyMEGNetConv(n))

end

function (l::M3MPLayer)(g, x, e)
    x,e = l.forward_conv(g, x, e)
    x = l.drop1(x)
    x = l.bn_n(x)
    e = l.bn_e(e)
    ng = l.rev(g)
    x,e = l.backward_conv(ng, x, e)
    return x, e
end

