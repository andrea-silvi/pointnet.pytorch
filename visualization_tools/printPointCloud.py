import numpy as np
import matplotlib.pyplot as plt
import os


# Insert the cloud path in order to print it
def printCloudFile(cloud, name):
    alpha = 0.5
    xyz = np.loadtxt(cloud)
    # print(xyz)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    plt.savefig(f"/content/pointnet.pytorch/{name}.png")


def printCloud(cloud, name, opt=None):
    alpha = 0.5
    xyz = cloud[0]
    # print("sono qui")
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', alpha=alpha)
    folder = "/content/pointnet.pytorch/images/" if opt is None else os.path.join(opt.outf, "images")
    try:
        os.makedirs("images")
    except OSError:
        pass
    plt.savefig(os.path.join(folder, f"{hash(str(opt))}_{name}.png"))
