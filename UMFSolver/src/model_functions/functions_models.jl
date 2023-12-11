
"""
    createdirectmodel(optimizer::String)

Initialize the model in direct mode with the chosen optimizer.

Return the initialized model if the optimizer is `cplex`, and `CPLEX` is installed, or `highs`; throw error otherwise.
If using `cplex`, allow to use only one thread.

"""
function createdirectmodel(optimizer::String)
    if optimizer == "highs"
        model = direct_model(HiGHS.Optimizer())
        return model
    elseif isinstalled("CPLEX") && optimizer == "cplex"
        model = direct_model(CPLEX.Optimizer())
        set_optimizer_attribute(model, "CPXPARAM_Threads", 1)
        return model
    else
        throw(ArgumentError("Not accepted optimizer."))
        return
    end
end
