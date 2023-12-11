using ChainRules, ChainRulesCore

function compute_loss(loss_f, x)
    if x isa GNNGraph
        return loss_f(x)
    else
        return mean(loss_f(g |> device) for g in loader)
    end
end

function train_model(model,
                     optimizer,
                     loss,
                     train_dataloader,
                     test_dataloader;
                     device=CUDA.functional() ? Flux.gpu : Flux.cpu,
                     epochs=100,
                     batchsize=1,
                     callbacks=[],
                     early_stopping=nothing,
		     )

    # model parameters
    ps = Flux.params(model)

    # training history, this object is passed as second argument to each callback
    history = Dict("train_loss"=>[],
                   "test_loss"=>[]
                  )

    
    println("starting training")

    state = Flux.setup(optimizer, model)

    for epoch in 1:epochs
        # epoch progress bar
        bar = ProgressBar(train_dataloader)
        set_description(bar, "Epoch $epoch")

        epoch_losses = []
        epoch_metrics = []

        for g in bar
            # send batch to device
            # debug
            g = g |> device

            # compute gradient
            #grad = gradient(() -> compute_loss(loss, g), ps)
            batch_loss, grad = Flux.withgradient(model) do m
                pred = sigmoid(vec(m(g)))
                labs = vec(g.targets[g.target_mask])
                ignore_derivatives() do
                    push!(epoch_metrics, metrics(pred, labs))
                end
                loss(pred, labs)
            end

            #grad = gradient(() -> loss(g), ps)
            #grad = gradient(() -> loss(pred, labs), ps)

            # check if NaN found in gradient
            #nan_check = [sum(isnan.(grad[p] |> cpu)) > 0 for p in ps if !isnothing(grad[p])]
            #nothing_check = [isnothing(grad[p] |> cpu) for p in ps]
            #if any(nothing_check)
            #    println("nothing found in gradient : ", nothing_check)
            #end
            #if any(nan_check)
            #    for (i,p) in enumerate(ps)
            #        if any(nan_check[i])
            #            println("NaN found in gradient of parameter $i")
            #        end
            #    end
            #    throw("NaN found")
            #end

            # update parameters
            #Flux.Optimise.update!(optimizer, ps, grad)
            Flux.update!(state, model, grad[1])

            #push!(epoch_losses, Flux.cpu(loss(g)))
            #push!(epoch_losses, Flux.cpu(loss(pred, labs)))
            push!(epoch_losses, batch_loss)

            #push!(epoch_metrics, metrics(g, model, device=device))
            #push!(epoch_metrics, metrics(pred, labs))

        end
        

        # update training and testing history
        update!(history, "train_loss", mean(vcat(epoch_losses...)))
        update!(history, epoch_metrics, prefix="train")
        
        test_loss = loss(test_dataloader, model)
        update!(history, "test_loss", test_loss)
        test_metrics = metrics_loader(test_dataloader, model, device=device)
        update!(history, test_metrics, prefix="test")


        # call callbacks
        for callback in callbacks
            callback(epoch, history)
        end
        # early stopping
        if !isnothing(early_stopping)
            if early_stopping(epoch, history)
                println("Early stopping")
                break
            end
        end
        # reclaim memory at end of batch
        GC.gc()
	if CUDA.functional()
	    CUDA.reclaim()
	end
    end
end 


