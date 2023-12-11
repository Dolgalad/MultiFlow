using UMFSolver

instance_dir = "./instances/"

output_dir = "default_solve_outputs"

exclude = ["N1000", "N1200", "N1400", "N1600", "N1800", "N2000", "N800", "_2", "_3", "_4", "_5"]

for (root, dirs, files) in walkdir(instance_dir)
    for dir in dirs
        idir = joinpath(root, dir)
        if UMFSolver.is_instance_path(idir)
            instance_name = replace(replace(idir, instance_dir=>""), "/"=>"_")
            if any(contains(instance_name, e) for e in exclude)
                continue
            end
            # make output dir
            mkpath(joinpath(output_dir, instance_name))
            # load instance
            #inst = UMFData(idir)
            println("instance path ", idir, ", ", instance_name)
            solveUMF(idir*"/", "CG", "highs", joinpath(output_dir, instance_name, "output.json"))
        end
    end
end
