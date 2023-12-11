using JSON

"""
    SolverStatistics

Container for storing solver run statistics
"""
struct SolverStatistics
    stats::Dict
end

function getindex(s::SolverStatistics, k::String)
    return s.stats[k]
end

function json_compatible_data(s::SolverStatistics)
    a = Dict()
    for (key, value) in s.stats
        if (typeof(value)==Int) || (typeof(value)==Float64) || (typeof(value)==String) || (typeof(value)==Float32)
            a[key] = value
        #else
        #    println("Cant json type ", typeof(value), ", k : ", key, ", ", (typeof(value)==Int) ,(typeof(value)==Float64),(typeof(value)==String), (typeof(value)==Real))
        end
    end
    a["val"] = s.stats["val"]
    return a
end

function save(s::SolverStatistics, filename::String)
    json_string = JSON.json(json_compatible_data(s), 4)
    open(filename, "w") do f
        write(f, json_string)
    end
end

function load_solverstats(filename::String)
    if isfile(filename)
        return SolverStatistics(JSON.parsefile(filename))
    end
end
