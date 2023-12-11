using Plots
using CSV
using DataFrames

data_dir = "training_data"

categories = ["flexE", "vlan", "mixed"]
img_size = (300, 300)

for cat in categories
    # loss
    train_df = CSV.read(joinpath(data_dir, cat, "train_loss.csv"), DataFrame)
    test_df = CSV.read(joinpath(data_dir, cat, "test_loss.csv"), DataFrame)
    println(train_df)
    p1= plot(train_df[!,"Step"], train_df[!,"Value"], label="train", size=img_size);
    plot!(test_df[!,"Step"], test_df[!,"Value"], label="test");
    xlabel!("Epoch")
    ylabel!("Loss")

    savefig(joinpath(data_dir, cat, "loss.png"))

    # precision
    train_df = CSV.read(joinpath(data_dir, cat, "train_precision.csv"), DataFrame)
    test_df = CSV.read(joinpath(data_dir, cat, "test_precision.csv"), DataFrame)
    println(train_df)
    p2 = plot(train_df[!,"Step"], train_df[!,"Value"], label="train", size=img_size);
    plot!(test_df[!,"Step"], test_df[!,"Value"], label="test");
    xlabel!("Epoch")
    ylabel!("Precision")

    savefig(joinpath(data_dir, cat, "precision.png"))


    # recall
    train_df = CSV.read(joinpath(data_dir, cat, "train_recall.csv"), DataFrame)
    test_df = CSV.read(joinpath(data_dir, cat, "test_recall.csv"), DataFrame)
    println(train_df)
    p3 =plot(train_df[!,"Step"], train_df[!,"Value"], label="train", size=img_size);
    plot!(test_df[!,"Step"], test_df[!,"Value"], label="test");
    xlabel!("Epoch")
    ylabel!("Recall")

    savefig(joinpath(data_dir, cat, "recall.png"))
    #
    plot(p1,p2,p3, layout=(1,3))
    savefig(joinpath(data_dir, cat, "metrics.png"))


end
