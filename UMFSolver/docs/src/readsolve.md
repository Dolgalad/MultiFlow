# Instances

Instance data are available as folders with two *.csv* files: *link.csv*, and *service.csv*; these files contain respectively the data information on the network (lists of arc sources and destinations, costs, capacities) and on the demands (source and target nodes, bandwidths).


## Data Structures

The instance data are collected in a structure called [`UMFData`](@ref). All data characteristics (number of nodes, edges, demands, arc costs and capacities, demand sources and destination nodes) are accessible through suitable functions. See the documentation for further details.


## Instance reader

The function 

[`readinstance`](@ref)(path)

allows to read the data of the instance whose *.csv* files are contained in `path` folder.


# Solvers

The function [`solve`](@ref) has several methods, and can be used to solve an instance of `UMFData` type in different ways, depending on the solution method chosen by the user.

## Instance solver

The different options can be selected by the user simply via some keywords, which are passed to the function [`solveUMF`](@ref). This function solves the instance with the suitable method and writes the results on a *.txt* file.
See the function documentation and the [Readme](../../README.md) file for more information.

## Configurations

The possible configuration choices regard firstly the type of formulation that is needed: it is possible to solve the [compact formulation](compact.md) with a LP solver, or the [extended formulation](colgen.md) with a column generation algorithm. Furthermore, the solver can be chosen: it can be either `CPLEX`, or `HiGHS`.
Finally, in a column generation framework, both the master and pricing subproblems can be customized. To tacke all possible options, several configuration structures are defined, and the suitable methods of the function `solve` are called, depending on the selected configuration. More details are in the following dedicated sections.

