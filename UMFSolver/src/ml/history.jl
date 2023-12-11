function update!(history::Dict, key::String, value::Number)
    if haskey(history, key)
        push!(history[key], value)
    else
        history[key] = [value]
    end
end

function update!(history::Dict, values::Dict; prefix="train")
    for k in keys(values)
        nk = prefix*"_"*k
        update!(history, nk, values[k])
    end
end

function update!(history::Dict, values::Vector; prefix="train")
    for k in keys(values[1])
        nk = prefix*"_"*k
        update!(history, nk, mean([em[k] for em in values]))
    end
end

function last_metrics(history; prefix="train")
    return (; (Symbol(replace(k, prefix*"_"=>"")) => history[k][end] for (k,v) in history if startswith(k, prefix*"_"))...)
end

function last_value(history, key)
    return history[key][end]
end

