using DataFrames

function cleanup_s(v)
    if v isa String
        return replace(v, raw"_"=>raw"\_")
    end
    if v isa Number
        return raw"$" * string(round(v, digits=3)) * raw"$"
    end
    return v
end
function dataframe_to_latex(df)
    cols = names(df)
    headerline = ""
    lt = raw"\begin{tabular}{"
    for (i,n) in enumerate(cols)
        lt = lt * "l"
        if i == 1
            headerline = headerline * raw"\textbf{" * n * "} & "
        else
            headerline = headerline * raw" & \textbf{" * n * "}"
        end
    end
    lt = lt * raw"}"*"\n"
    lt = lt * "    " * headerline * raw"\\\\" * "\n"
    for row in eachrow(df)
        vals =values(row)
        vals = map(cleanup_s, vals)
        lt = lt * "    " * join(vals, " & ") * raw"\\\\" * "\n"
    end

    lt = lt * raw"\end{tabular}" * "\n"

    return lt

end
