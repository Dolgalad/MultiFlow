# Extended Formulation

The UMF problem admits an extended formulation, in which the flow conservation constraints are replaced by the constraints that require that the paths for each demand are in the convex hull of the feasible paths. The new variables are the coefficients of the convex combination, and the formulation can be expressed as follows.

```math
\begin{aligned}
	\min \sum_{k\in K} b_k \sum_{p \in P_k} \sum_{a \in p} r_a  \lambda_p\\
	\sum_{p \in P_k} \lambda_p & = 1 & \forall k \in K\\
	\sum_{k \in K} b_k \sum_{p \in P_k\mid a \in p} \lambda_p &\leq c_a& \forall a \in A \\
	\lambda_p&\in \{0,1\}&\forall k \in K, p \in P_k.
\end{aligned}
```

The objective function is the sum of the routing costs of the demands. The first set of inequalities are the *capacity constraints*, while the second set of constraints are the *convexity constraints*, which require one and only one path for routing each demand ``k`` from its source to its target.

This formulation is called *extended*, as opposed to the *compact* formulation, since it contains an exponential number of variables. It can be solved using a *column generation* algorithm.

## Column Generation Solver Configuration

The linear relaxation of this formulation can be solved with column generation, in which a (restricted) master and a pricing subproblems are alternatively and repeatdly solved, in order to converge to the optimal solution.
The restricted master problem has the same form as the extended formulation, although the sets of extreme points ``P_k`` are restricted to small subsets. 
The pricing problems reduce to a shortest path problem for each demand ``k \in K``.

Options for solving the extended formulation are as follow.
The master problem can be solved with a linear solver. The pricing problems can be solved with the Dikstra algorithm, or with different solvers.
Moreover, since the master problem may need a LP solver, this can be chosen between `CPLEX` and `HiGHS`. Finally, also the name for the file where to store the results must be selected.
Those are the fields of the configuration structure [`ColGenConfigBase`](@ref).
The method of function [`solve`](@ref) to solve the compact formulation is `solve(::UMFData, ::ColGenConfigBase)`. The results written in the output file are indicated in the function documentation.
