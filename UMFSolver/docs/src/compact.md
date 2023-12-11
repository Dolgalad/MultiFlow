# Compact Formulation

The UMFP can be formulated as an integer linear program with a polynomial number of variables and constraints.

For every arc ``a \in A`` and every demand ``k\in  K``, let ``x_a^k`` be a binary variable defined as:
```math
x_a^k = 
\left\{ \begin{array}{ll} 1 & \textrm{if demand ``k'' is routed through arc ``a'',}\\0 & \textrm{otherwise.}
\end{array}
\right. 
``` 

The UMFP can be formulated as follows:

```math
\begin{aligned}
    \min \sum_{a\in A}r_a \sum_{k\in K} b_k x_a^k\\
    \sum_{a \in \delta^{\rm out}(v)} x_a^k - \sum_{a \in \delta^{\rm in}(v)} x_a^k & = 
    \left\{
    \begin{array}{rl} 
    1 & \textrm{if } v=s_k,\\
    -1 &  \textrm{if } v=t_k,\\
    0 & \textrm{otherwise,}
    \end{array} 
    \right.
    & \forall v \in V, \forall k \in K,\\
    \sum_{k \in K} b_k x_a^k &\leq c_a& \forall a \in A,\\
    x_a^k&\in \{0,1\}&\forall k \in K, a \in A.
\end{aligned}
```

The objective function corresponds to the routing cost of a solution. The first set of constraints are the well-known flow conservation constraints ensuring, together with the inegrality constraints, that there is an ``s_kt_k``-path for each demand ``k \in K``. The inequalities are the arc capacity constraints, which impose that the sum of the bandwidths of the demands routed through an arc does not exceed its capacity.

## Solve configuration

Considering the compact formulation, one can choose to solve an instance to the linear relaxation, or to integrality; in addition, the solver must be chosen between `CPLEX` and `HiGHS`, and finally the name for the file where to store the results results must be chosen. Those are the fields of the configuration structure [`CompactConfigBase`](@ref).
The method of function [`solve`](@ref) to solve the compact formulation is `solve(::UMFData, ::CompactConfigBase)`. The results written in the output file are indicated in the function documentation.