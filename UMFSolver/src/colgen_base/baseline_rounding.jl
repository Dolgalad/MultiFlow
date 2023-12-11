"""
    baselinerounding(
        ndemands::Int64,
        cols::Vector{Vector{Vector{Int64}}},#columns
        bws::Vector{<:Real}, 
        csts::Vector{<:Real}, 
        caps::Vector{Float64},
        pr::UMFShortestPathPricingData,
        config::DefaultDijkstraPricingConfig, 
        x_k_p::Vector{Vector{Float64}},
        bigms::Vector{<:Real} 
    )

Compute the integral rounding, given the fractional solution vector `x_k_p`.

# Arguments
- `ndemands`: number of demands
- `cols`: vector of extreme columns
- `bws`: vector of bandwidths
- `csts`: vector of costs on the arcs
- `caps`: vector of arc capacities
- `pr`: pricing problem
- `config`: pricing configuration
- `x_k_p`: fractional solution vector
- `bigms`: vector of costs of non-acceptance

Return:
- `x_k_p_best`: the vector of the found integer solution
- `y_k_best`: the vector of non accepted demands in the integral solution
- `sol_best`: the value of the integral solution found.

"""
function baselinerounding(
    ndemands::Int64,
    cols::Vector{Vector{Vector{Int64}}},#columns
    bws::Vector{<:Real}, #b_k
    csts::Vector{<:Real}, #r_a (without b_k)
    caps::Vector{Float64}, # capacities
    pr::UMFShortestPathPricingData, # pricing
    #config::DefaultDijkstraPricingConfig, # pricing configuration
    config::PrColGenconfig, # pricing configuration
    x_k_p::Vector{Vector{Float64}},
    bigms::Vector{<:Real}, # bigM: costs of non-acceptance
)
    avail_capacities::Vector{Float64} = copy(caps)
    #newcosts::Vector{Float64} = csts + duals # once for all!
    solvalue::Float64 = 0
    x_k_p_feas::Vector{Vector{Float64}} = [zeros(size(cols[k], 1)) for k = 1:ndemands]
    y_k_feas::Vector{Float64} = zeros(ndemands)
    sol_best::Float64 = 1e30
    x_k_p_best::Vector{Vector{Float64}} = copy(x_k_p_feas)
    y_k_best::Vector{Float64} = copy(y_k_feas)


    # maxl::Int64 = maximum(size(p,1) for p in cols)
    # lengths::Vector{Float64} = zeros(maxl)
    #newlengths::Vector{Float64} = zeros(maxl)
    newcosts::Vector{Float64} = copy(csts)
    capmin::Float64 = 0.0
    prob::Vector{Float64} = []
    sum::Float64 = 0.0
    selected_indices::Vector{Int64} = []
    perm_ks::Vector{Int64} = 1:ndemands
    #seed
    #Random.seed!(1234)
    for s = 1:10
        perm_ks = shuffle(1:ndemands)
        for k in perm_ks
            # fill!(lengths, 0.)
            # fill!(newlengths, 0.)
            # println("lengths = ", newlengths)
            # selected_paths::Vector{Vector{Int64}} = []
            empty!(selected_indices)
            # perm_lengths::Vector{Int64} = sortperm(newlengths[1:size(cols[k],1)])
            # println("lengths = ", newlengths)
            # println(perm_lengths)
            # println("sorted lengths = ", newlengths[perm_lengths])
            for p in eachindex(cols[k])
                #println("p = ", p,", ", cols[k])
                # debug
                #if isempty(cols[k][p])
                #    continue
                #end
                capmin = minimum(avail_capacities[a] for a in cols[k][p])
                if capmin >= bws[k] # path entirely fits in solution
                    # push!(selected_paths, cols[k][p])
                    push!(selected_indices, p)
                    sum += x_k_p[k][p]
                end
            end
            if isempty(selected_indices)
                for a in eachindex(newcosts)
                    if avail_capacities[a] < bws[k]
                        newcosts[a] = 2 * bigms[k]
                    end
                end
                set_demand!(pr, k)
                chgcost!(pr, newcosts)
                solve!(pr, config)
                if sol(pr) >= 2 * bigms[k]
                    y_k_feas[k] = 1
                    solvalue += bigms[k]
                else
                    solvalue += sol(pr) * bws[k]
                    push!(x_k_p_feas[k], 1.0)
                    for a in xopt(pr)
                        avail_capacities[a] -= bws[k]
                    end
                end
                newcosts .= csts
            else
                resize!(prob, size(selected_indices, 1))
                for p in eachindex(selected_indices)
                    if sum > 0
                        prob[p] = x_k_p[k][selected_indices[p]] / sum
                    else
                        prob[p] = 0.
                    end
                    if isnan(prob[p]) || isinf(prob[p])
                        println("prob[p], p=", p, ", isnan=", isnan(prob[p]), ", isinf=", isinf(prob[p]), ", sum=", sum)
                    end
                end
                w = Weights(prob)
                p = sample(selected_indices, w)
                x_k_p_feas[k][p] = 1
                for a in cols[k][p]
                    avail_capacities[a] -= bws[k]
                end
                solvalue += bws[k] * compute_length_k_p(cols, csts, k, p)
            end
        end
        if solvalue < sol_best
            x_k_p_best .= x_k_p_feas
            y_k_best .= y_k_feas
            sol_best = solvalue
        end
    end
    return x_k_p_best, y_k_best, sol_best
end
