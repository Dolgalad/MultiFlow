using Flux
using GraphNeuralNetworks
import Base: +,-,vcat
import Statistics: mean

function mean(g::GNNGraph...) 
    if !isempty(g[1].ndata) && !isempty(g[1].edata)
        return GNNGraph(g[1], ndata=(;x=mean([_g.x for _g in g])), edata=(;e=mean([_g.e for _g in g])))
    elseif !isempty(g[1].ndata)
        return GNNGraph(g[1], ndata=(;x=mean([_g.x for _g in g])))
    elseif !isempty(g[1].edata)
        return GNNGraph(g[1], ndata=(;e=mean([_g.e for _g in g])))
    else
        return GNNGraph(g[1])
    end
end

#function Base.vcat(g::GNNGraph...)
#    if !isempty(g[1].ndata) && !isempty(g[1].edata)
#        return GNNGraph(g[1], ndata=(;x=vcat([_g.x for _g in g]...)), edata=(;e=vcat([_g.e for _g in g]...)))
#    elseif !isempty(g[1].ndata)
#        return GNNGraph(g[1], ndata=(;x=vcat([_g.x for _g in g]...)))
#    elseif !isempty(g[1].edata)
#        return GNNGraph(g[1], edata=(;e=vcat([_g.e for _g in g]...)))
#    else
#        return GNNGraph(g[1])
#    end
#end
function Base.vcat(g1::GNNGraph, g2::GNNGraph)
    if !isempty(g1.ndata) && !isempty(g1.edata)
        return GNNGraph(g1, ndata=(;x=vcat(g1.x,g2.x)), edata=(;e=vcat(g1.e,g2.e)))
    elseif !isempty(g1.ndata)
        return GNNGraph(g1, ndata=(;x=vcat(g1.x,g2.x)))
    elseif !isempty(g1.edata)
        return GNNGraph(g1, edata=(;e=vcat(g1.e,g2.e)))
    else
        return GNNGraph(g1)
    end
end

#function +(g1::GNNGraph, g2::GNNGraph)
function +(g1::GNNGraph, g2::GNNGraph)
    if !isempty(g1.ndata) && !isempty(g1.edata)
        return GNNGraph(g1, ndata=(;x=g1.x .+ g2.x), edata=(;e=g1.e .+ g2.e))
    elseif !isempty(g1.ndata)
        return GNNGraph(g1, ndata=(;x=g1.x .+ g2.x))
    elseif !isempty(g1.edata)
        return GNNGraph(g1, ndata=(;e=g1.e .+ g2.e))
    else
        return GNNGraph(g1)
    end
end
function -(g1::GNNGraph, g2::GNNGraph)
    if !isempty(g1.ndata) && !isempty(g1.edata)
        return GNNGraph(g1, ndata=(;x=g1.x .- g2.x), edata=(;e=g1.e .- g2.e))
    elseif !isempty(g1.ndata)
        return GNNGraph(g1, ndata=(;x=g1.x .- g2.x))
    elseif !isempty(g1.edata)
        return GNNGraph(g1, ndata=(;e=g1.e .- g2.e))
    else
        return GNNGraph(g1)
    end
end

#-(g1::GNNGraph, g2::GNNGraph) = GNNGraph(g1, ndata=(;x=g1.x .- g2.x), edata=(;e=g1.e .- g2.e))


# message passing layer
struct VGNNNet
    aggr
    encoder
    decoder
    internal
end

Flux.@functor VGNNNet

function VGNNNet((x_dim,e_dim)::Tuple{Int64,Int64}, n::Int=1)
    decoder_e_dim, decoder_x_dim = 2*(2*x_dim+e_dim), 2*x_dim+e_dim
    if n==0
        decoder_e_dim, decoder_x_dim = 2*x_dim+e_dim, x_dim+e_dim
    end
    return VGNNNet(
                      vcat,
                      MEGNetConv(Dense(2*x_dim+e_dim => e_dim), Dense(x_dim+e_dim=>x_dim)),
                      MEGNetConv(Dense(decoder_e_dim => e_dim), Dense(decoder_x_dim=>x_dim)),
                      n>0 ? VGNNNet((x_dim,e_dim), n-1) : nothing
                  )
end

#function VGNNNet((in_dim,out_dim)::Pair, n::Int=1; layer_type=GCNConv)
#    decoder_x_dim = 2 .* in_dim
#    if n==0
#        decoder_x_dim = in_dim
#    end
#    return VGNNNet(
#                      layer_type(in_dim => in_dim),
#                      layer_type(decoder_x_dim=>in_dim),
#                      n>0 ? VGNNNet(in_dim => out_dim, n-1, layer_type=layer_type) : nothing
#                  )
#end

function dimsum(dims)
    if dims isa AbstractVector{Int}
        return sum(dims)
    else
        return tuple(collect((sum([dims[i][j] for i in 1:length(dims)]) for j in 1:length(dims[1])))...)
    end
end
function VGNNNet((in_dim,out_dim)::Pair, n::Int=1; aggr=vcat, layer_type=GCNConv, inner_dim=nothing)
    if isnothing(inner_dim)
        inner_dim = [out_dim for _ in 1:n+1]
    elseif !(inner_dim isa AbstractVector)
        inner_dim = [inner_dim for _ in 1:n+1]
    end
    if !(aggr isa AbstractVector)
        aggr = [aggr for _ in 1:n+1]
    end
    if !(layer_type isa AbstractVector)
        layer_type = [layer_type for _ in 1:n+1]
    end
    if n+1!=length(inner_dim)
        throw("Expected length(inner_dim) == $(n+1), got $(length(inner_dim))")
    end
    if aggr[1]==vcat && n>=1
        decoder_x_dim = dimsum(inner_dim[1:2])
        encoder_dim = inner_dim[1]
        inner_in_dim = inner_dim[1]
        inner_out_dim = inner_dim[2]
    else
        decoder_x_dim = inner_dim[1]
        encoder_dim = inner_dim[1]
        inner_in_dim = inner_dim[1]
        inner_out_dim = inner_dim[1]
    end
    #println("VGNNNet ($(in_dim) => $(out_dim)) $n")
    #println("\tencoder : $(in_dim) => $(inner_dim[1])")
    #println("\tdecoder : $(decoder_x_dim) => $(out_dim)")
    #println("\tinternal: $(inner_in_dim) => $(inner_out_dim)")

    return VGNNNet(
                      aggr[1],
                      layer_type[1](in_dim => encoder_dim),
                      layer_type[1](decoder_x_dim=>out_dim),
                      n>0 ? VGNNNet(inner_in_dim => inner_out_dim, n-1, aggr=aggr[2:end], layer_type=layer_type[2:end], inner_dim=n>=1 ? inner_dim[2:end] : nothing) : nothing
                  )
end


function (l::VGNNNet)(g::GNNGraph)
    g = l.encoder(g)

    if !isnothing(l.internal)
        g = reduce(l.aggr, [g, l.internal(g)])
        g = l.decoder(g)
    else
        g = l.decoder(g)
    end

    return g
end

function (l::VGNNNet)(g::GNNGraph, x, e=nothing)
    if !isnothing(e)
        x,e = l.encoder(g, x, e)
        g = GNNGraph(g, ndata=(;x=x), edata=(;e=e))
    else
        x = l.encoder(g, x)
        g = GNNGraph(g, ndata=(;x=x))
    end

    if !isnothing(l.internal)
        g = l.aggr(g, l.internal(g))
        g = l.decoder(g)
    else
        g = l.decoder(g)
    end

    return g
end


