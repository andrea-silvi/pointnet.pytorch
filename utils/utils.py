import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def upload_args_from_json(file_path=os.path.join("parameters", "fixed_params.json")):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    setattr(args, "runNumber", 0)
    return args


def farthest_and_nearest_points(x, batch_points, n):
    # x should be (batch_size, num_points, 3)
    # batch_points should be (batch_size, 1, 3)
    batch_size = x.size(0)
    num_points = x.size(1)
    assert batch_points.size(0) == batch_size
    assert batch_points.size(2) == 3
    assert batch_points.size(1) == 1
    # batch_points: for each batch (point cloud) we have one point (x,y,z)
    distances = torch.sum((x - batch_points) ** 2, dim=-1)
    idx = distances.topk(k=num_points, dim=-1)[1]
    idx_farthest = idx[:, :n]
    idx_nearest = idx[:, n:]
    idx_base = torch.arange(0, batch_size, device="cuda").view(-1, 1) * num_points
    idx_farthest = (idx_farthest + idx_base).view(-1)
    idx_nearest = (idx_nearest + idx_base).view(-1)
    x = x.view(-1, 3)
    x_far = x[idx_farthest, :]
    x_near = x[idx_nearest, :]
    return x_far.view(batch_size, n, 3), idx_farthest, x_near.view(batch_size, -1, 3)


def plot_neptune_losses(files, log_scale=True):
    colors = ["blue", "g", "y", "r", "black", "fuchsia"]
    count_color = 0
    dict_files = dict()
    sns.set_style("darkgrid")
    min_loss = 100000000
    max_loss = 0
    for file in files:
        filename = file.split("\\")[-1]
        file_suffix = filename.split("___")[1]
        if dict_files.get(file_suffix, 0) == 0:
            color = colors[count_color]
            count_color += 1
            dict_files[file_suffix] = color
        else:
            color = dict_files[file_suffix]
        if filename.startswith("val"):
            linestyle = "--"
        else:
            linestyle = None
        losses = pd.read_csv(file, header=None)
        losses = losses[[0, 2]]
        losses.columns = ["epoch", "loss"]
        sns.lineplot(x="epoch", y="loss", data=losses, linestyle=linestyle, color=color, label=filename)
        min_loss = min(min(losses["loss"]), min_loss)
        max_loss = max(max(losses["loss"]), max_loss)
    plt.gca().set_xlabel("EPOCH", fontsize=14)
    plt.gca().set_ylabel("MEAN CHAMFER LOSS", fontsize=14)
    plt.gca().set_title("OnionNet: Training and Validation Losses")
    if log_scale:
        sample_count = np.around(np.logspace(math.log10(min_loss), math.log10(max_loss), 8), 4)
        plt.gca().set(yscale='log')
        plt.gca().set(yticks = sample_count, yticklabels = sample_count)
        plt.gca().set_ylabel("MEAN CHAMFER LOSS (log scale)", fontsize=14)

    plt.savefig("train_val_losses.png", bbox_inches="tight")

file = "C:\\Users\\vitto\\Downloads\\train_batch_seg_loss (1).csv"