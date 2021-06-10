import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
import utils.dataset as ds
import torch


# Insert the cloud path in order to print it
def printCloudFile(cloud_original, cloud_decoded, name):
    alpha = 0.5
    xyz = np.loadtxt(cloud_original)
    # print(xyz)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    ax.append(fig.add_subplot(111,projection='3d'))

    ax[-1].plot(cloud_decoded[:, 0], cloud_decoded[:, 1], cloud_decoded[:, 2], 'o', alpha=alpha)
    ax[-1].set_title("decoded cloud")
    plt.savefig(f"/content/pointnet.pytorch/{name}.png")


def printCloud(cloud_original, name, alpha=0.5, opt=None):

    xyz = cloud_original[0]
    print("sono qui")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    folder = "/content/pointnet.pytorch/images/" if opt is None else os.path.join(opt.outf, "images")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    plt.savefig(os.path.join(folder, f"{hash(str(opt))}_{name}.png"))


def printCloudM(cloud_original, cloud_decoded, name, alpha=0.5, opt=None):
    xyz = cloud_original[0]
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("original cloud")
    xyz = cloud_decoded[0]
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    ax.set_title("decoded cloud")
    folder = "/content/pointnet.pytorch/images/" if opt is None else os.path.join(opt.outf, "images")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    plt.savefig(os.path.join(folder, f"{hash(str(opt))}_{name}.png"))


def savePtsFile(type, category, opt, array):
    folder = os.path.join(opt.outf, "visualizations", opt.runNumber, category)
    try:
        os.makedirs(folder)
    except OSError:
        pass
    file = open(os.path.join(folder, f"{type}.pts"), 'w')
    np.savetxt(file, array)
    file.close()
def print_original_decoded_point_clouds(dataset, category, model, opt):
    point_cloud = dataset.get_point_cloud_by_category(category)
    model.eval()
    point_cloud_np = point_cloud.cuda()
    point_cloud_np = torch.unsqueeze(point_cloud_np, 0)
    decoded_point_cloud = model(point_cloud_np)
    original_pc_np = point_cloud_np.cpu().numpy()
    decoded_pc_np = decoded_point_cloud.cpu().data.numpy()
    #printCloudM(point_cloud_np, dec_val_stamp, name=category, opt=opt)
    original_pc_np = original_pc_np.reshape((1024, 3))
    decoded_pc_np = decoded_pc_np.reshape((1024, 3))
    savePtsFile("original", category, opt, original_pc_np)
    savePtsFile("decoded", category, opt, decoded_pc_np)
