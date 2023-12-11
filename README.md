# An Unsplittable Multicommodity Flow Problem Solver

This project is intended to solve instances of the Unsplittable Multicomodity Flow (UMF) problem.

Algorithms to read and solve the instances are written in *julia* and can be called from a *julia* or a *C++* application.

In either case, the package *UMFSolver* is used and must be installed with the following steps.

## Installation


The package *UMFSolver* can be downloaded from the GitLab project.

In order to install and use it, please note that *julia* must be installed (see [Julia](https://julialang.org/downloads/)). Once the project has been cloned, move to the folder `multiflow/UMFSolver`, then run the following command:

```bash
$ julia --project=.
```


### CPLEX

*CPLEX* is by default used as one of the linear solvers in the project, but it is not strictly needed, since another linear solver ([HiGHS](https://highs.dev/)) is also accepted. Moreover, the minumum version requirement for *CPLEX* is *12.10* or *20.1*.
In order *not* to use the solver *CPLEX*, open the file *Project.toml*, contained in the folder `multiflow/UMFSolver`, and if *CPLEX* is present in the list of dependencies *\[deps\]*, delete that line. Alternatively, open the package mode typing: "]" and type:
```
(UMFSolver) pkg > rm CPLEX
```

This will allow to install and use the package *UMFSolver* without any reference to *CPLEX*.

### Instantiate

The required dependencies (including or not *CPLEX*) are installed from the package mode, (opened typing: "]") with the following command:

```
(UMFSolver) pkg > instantiate
```
After this instruction, the package is installed. The package mode can be closed with a backspace and the application can be closed. The package can then be used from a *julia* application, or in a *C++* script. Below are the details for both cases.

## Julia

In order to use the package from *Julia*, the project can be activated when opening the *julia REPL* in the folder `multiflow/UMFSolver`, by typing `julia --project=.` as written above.
From *Julia* the package can be added as usual by just typing:

```julia
using UMFSolver
```

Once the above instructions are successfully completed, the package can be used as reported in the package [Readme](UMFSolver/README.md) file.

The complete package documentation is available with the help ("?") command, or via the documentation at: [Documentation](UMFSolver/docs/src/index.md).


## C++ 

This project provides a *C++* interface to call the *julia* functions. This is done with the package *jluna*, whose installation instructions are available at: [Jluna installation](https://clemens-cords.com/jluna/installation.html). Please note that *cmake* (version 3.12 or higher) should be installed on the machine, and the *julia* version must be 1.7.0 or higher. The *C++* compilers *g++10*, *g++11* and *clang++-12* are fully supported. Further information is available at: [Jluna](https://github.com/Clemapfel/jluna).

Once *jluna* is installed, in order to run the solver in *C++*, move to the folder `multiflow/cpp`; then
open the file `CMakeLists.txt` and set the correct paths for the target link libraries for *julia*, and the include *jluna* and *julia* directories. Then compile the source code, for instance as follows:

```bash
$ makedir build
$ cd build
$ cmake ..
$ cmake --build .
```

This will create the cpp executable, `UMFSolve`, in the newly created folder `multiflow/cpp/build`.

Information on how to run the executable are contained in the *cpp* [Readme](cpp/README.md) file.
