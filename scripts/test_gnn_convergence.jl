using UMFSolver
using Flux
using GraphNeuralNetworks
using LinearAlgebra
using Plots

# load instance and convert to graph
inst = UMFData("instances/prc/small/flexE/1")
g = UMFSolver.to_gnngraph(inst)

# create GNN model
act = relu
edge_nn = Chain(Dense(9 => 9, act), Dense(9 => 3, sigmoid))
node_nn = Chain(Dense(6=>6,act), Dense(6 => 3, sigmoid))
gnn = MEGNetConv(edge_nn, node_nn)

N = 1000
tol = 1e-3
x = zeros(3, nv(g))
e = g.e

# store updated state norms
Xn, En = [], []
# store states
X, E = [], []
for i in 1:N
    global x, e, Xn, En, X, E
    x1,e1 = gnn(g, x, e)
    #x1,e1 = normalize(x1), normalize(e1)
    # node and edge state update norm
    nd, ed = norm(x1-x), norm(e1-e)
    println("i=$i, |x'-x|=$nd, |e'-e|=$ed")
    push!(Xn, nd); push!(En, ed)
    push!(X, x1); push!(E, e1)
    if nd<tol && ed<tol
        break
    end
    x, e = x1, e1
end

p = plot(Xn)
savefig(p, "state_update_norms_Xn.png")
p = plot(En)
savefig(p, "state_update_norms_En.png")

p = plot([[norm(r[:,i]) for r in X] for i in 1:nv(g)])
savefig(p, "state_update_norms_X.png")
p = plot([[norm(r[:,i]) for r in E] for i in 1:ne(g)])
savefig(p, "state_update_norms_E.png")

transpose(X[end])
