function tikz_node_line(n, posx, posy; n_indent=0)
    return join(["\t" for _ in 1:n_indent],"")*raw"\node[vertex] ("*string(n)*raw") at "*string((posx,posy))*raw" {};"
e
    return join(["\t" for _ in 1:n_indent],"")*raw"\node[vertex] ("*string(n)*raw") at "*string((posx,posy))*raw" {"*string(n)*"};"
end
function tikz_edge_line(e; n_indent=0)
    return join(["\t" for _ in 1:n_indent],"")*raw"\path ("*string(src(e))*raw") edge ("*string(dst(e))*raw");"
end
function indentation(n)
    return join(["\t" for _ in 1:n],"")
end
function graph_to_tikz(g, posx, posy; n_indent=0)
    node_lines, edge_lines = [], []
    for n in 1:nv(g)
        push!(node_lines, tikz_node_line(n, posx[n], posy[n], n_indent=n_indent))
    end
    push!(edge_lines, indentation(n_indent)*raw"\begin{scope}[on background layer]")
    for e in edges(g)
        push!(edge_lines, tikz_edge_line(e, n_indent=n_indent+1))
    end
    push!(edge_lines, indentation(n_indent)*raw"\end{scope}")

    return join(vcat(node_lines, edge_lines), "\n")
end

function graph_to_tex(g, posx, posy, filename)
    header_lines = [
                    raw"\documentclass[tikz=true]{standalone}",
                    raw"\usepackage{tikz}",
                    raw"\usepackage{xcolor}",
                    raw"\fboxsep=0.2pt",
                    raw"\usetikzlibrary{graphs,quotes,arrows}",
                    raw"\usetikzlibrary{automata,positioning,shapes.geometric}",
                    raw"\usetikzlibrary{backgrounds}",
                    raw"\begin{document}",
                    raw"\begin{tikzpicture}[", 
                    raw"vertex/.style={",
                    "draw,fill=white,circle,",
                    "minimum width=1cm",
                    "}]",
                    "\n",
                   ]
    footer = "\n" * raw"\end{tikzpicture}"*"\n"*raw"\end{document}"
    f = open(filename, "w")
    write(f, join(header_lines,"\n") * graph_to_tikz(g, posx, posy, n_indent=1) * footer)
    close(f)
end
