import numpy as np
import matplotlib.pyplot as plt
import os


# Insert the cloud path in order to print it
def printCloudFile(cloud_original, cloud_decoded, name):
    alpha = 0.5
    #aaa
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
    alpha = 0.5
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
    alpha = 0.5
    xyz = cloud_original[0]
    print("sono qui")
    fig = plt.figure(figsize=(20, 15))
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