using JLD

"""
    UMFSolutionData

Container for storing solution of a UMFData instance
"""
struct UMFSolutionData
    x
    x_round
end

"""
    accepted_demands(sol::UMFSolutionData)

Count number of accepted demands
"""
function accepted_demands(sol::UMFSolutionData)
    c = 0
    for i in 1:size(sol.x, 2)
        if sum(sol.x[:,i])>0
            c+=1
        end
    end
    return c
end


function save(sol::UMFSolutionData, path::String; verbose::Bool=false)
    JLD.jldopen(path, "w") do file
        write(file, "x", sol.x)
    end
end

function load_solution(path::String; verbose::Bool=false)
    if isfile(path)
        if verbose
            println("Loading solution from '$path'")
        end

        c = JLD.jldopen(path, "r") do file
            read(file, "x")
        end
        return c
    end
    if verbose
        println("Error: could not find solution file '$path'")
    end
end


