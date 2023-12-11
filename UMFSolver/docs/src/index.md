# UMF solver

This package is developed to solve instances of the Unsplittable Multicommodity Flow (UMF) problem.

## Problem description

The multi-commodity flow problem (MF) consists in routing a set of commodities, or demands, through a capacitated network, from their source to their target, so that the capacity limits on the network arcs are respected and the overall commodity transportation cost is minimized.
In the unsplittable version (UMF) a commodity can only be routed through a single path from its source to its target.

### Problem Definition

Let ``G=(V,A)`` be an oriented graph. With each arc ``a \in A`` are associated a capacity ``c_a \in \mathbb{Z}_+^*`` (in MBs per second) and a routing cost ``r_a \in \mathbb{Z}_+^*``. Let ``K`` be a set of demands. For ``k \in K``, let ``s_k \in V`` and ``t_k \in V`` denote the source and target nodes of the demand and ``b_k`` its bandwith. 
A solution to the *Unsplittable Multicommodity Flows Problem (UMFP)* consists of a ``|K|``-tuple of ``s_kt_k``-paths, one for every demand ``k \in K``,   such that the total bandwith of the demands being routed on each arc does not exceed the arc capacity. The cost of a solution is the sum of the cost of each path, the latter one being given by the bandwith of the demand routed on it times the sum of the routing costs over the arcs belonging to that path. The UMFP consists in determining a solution at minimum cost.

## Language options

The code is written in *Julia* and can be wrapped in *C++*. Specific instructions on how to work with the package with each of these languages are in the [Readme](../../README.md) file of the *julia* package, and in the [Readme](../../../cpp/README.md) file for the *c++* wrapper. See also the project [Readme](../../../README.md) file.

## Solvers

Solvers may be used to solve the compact formulation and also the master subproblems in the column generation framework. Currently, two options are considered: [CPLEX](https://www.ibm.com/it-it/analytics/cplex-optimizer), and [HiGHS](https://highs.dev/).


## Content

This documentation describes how the package works. It starts with explaining how the instance data are [read](readsolve.md), then it describes how to use the different solvers for a UMF instance, specifically to solve the [compact formulation](compact.md) and the [column generation](colgen.md) one.