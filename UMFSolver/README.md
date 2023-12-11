
# Julia Execution Instructions

Once the `UMFSolver` package is installed, and added in *julia* (see instructions in the project [Readme](../README.md) file), its functions can be called to solve an instance of UMF problem with *julia*.

The different solver methods are described in the package [documentation](docs/src/index.md). Depending on some configurations, they solve the instance with its compact formulation, or with column generation. All of the methods write the results in a *.txt* file.
Instances are in the folder `multiflow/instances`.
The ouptut files can be stored in the folder `multiflow/output`.
The function:
`solveUMF`
can be used to read an instance, solve it, and write results on a *.txt* file.
It requires the following parameters:

- `path::String`: the path of the instance files;
- `form_type::String`: the chosen formulation: `"compactINT"` for the compact formulation, `"compactLR"`: for the linear relaxation of the compact formulation, or `"CG"` for column generation;
- `optimizer::String`: the chosen optimier: either `cplex`, or `highs`;
- `outputfile::String`: the name for the output file.

An example is as follows:

```julia
solveUMF("../instances/toytests/test1/", "compactLR", "highs", "../output/julia/test1.txt")
```

Optionally, two more parameters can be added, to specify the master and pricing configurations if the column generation method is selected:

- `mastertype::String`: `"linear"`, for a linear solver;
- `pricingtype::String`: `"dijkstra"`, to solve the shortest paths of the pricing with the *Dijkstra* algorithm;

If not specified, and if the column generation option is selected, the master is solved with a linear solver chosen with `optimizer`, and the pricing with the *Dijkstra* algorithm.

In the following example, the instance is solved with column generation, where the master and pricing are set to their default configurations.

```julia
solveUMF("../instances/toytests/test1/", "CG", "highs", 
"../output/julia/test1_CG.txt", "linear", "dijkstra")
```

More details about handling the possible options are specified in the package [documentation](docs/src/index.md).
