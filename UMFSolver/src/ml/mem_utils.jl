using CUDA
function get_memory_usage()
    pid = getpid()
    out = read(`pmap $pid`, String)
    parse(Int64, replace(split(out, " ")[end], "K\n"=>""))
end

function get_gpu_memory_usage()
    if !CUDA.functional()
        return 0
    end
    pid = getpid()
    out = read(`nvidia-smi`, String)
    lines = split(out, "\n")
    for l in lines
        if occursin("$pid", l)
            return parse(Int64, replace(split(l, " ")[end-1], "MiB"=>""))
        end
    end
    return 0
end
