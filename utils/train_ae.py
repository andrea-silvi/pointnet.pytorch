from __future__ import print_function
import torch
from pointnet.pointnet_model import PointNet_AutoEncoder
from loss import PointLoss
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from utils.dataset import ShapeNetDataset
import torch.nn.functional as F
from tqdm import tqdm


# TODO - create a json file for setting all the arguments. Actually:
# TODO - create a json for the FINAL arguments (after the cross-validation, e.g.: {'batchSize': 32})
# TODO - and a json for the GRID SEARCH phase (e.g.: {'batchSize': [16, 32, 64], ...}
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)


# the following function doesn't make the training of the network!!
# It shows the interfaces with the classes necessary for the point cloud completion task
# N.B.: only with PointNetAE and PointLoss (the one used for evaluating the Chamfer distance)
# the gcnn interface is not implemented yet
def example_AE_and_chamfer_loss():
    """
    Instantiate a PointNetAutoEncoder
    Feed it with a synthetic point cloud
    Compute the encoded point cloud (output of encoder)
    Compute the decoded point cloud (output of decoder)
    Compute chamfer loss
    :return:
    """
    batch_size = 32
    input_points = 1024

    # Instantiate a fake batch of point clouds
    points = torch.rand(batch_size, input_points, 3)
    print("Input points: ", points.size())

    # Instantiate the AE
    pointnet_AE = PointNet_AutoEncoder(num_points=input_points)

    # Move everything (data + model) to GPU
    assert torch.cuda.device_count() > 0, "Fail: No GPU device detected"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points = points.to(device)
    pointnet_AE = pointnet_AE.to(device)

    # try AE forward
    decoded = pointnet_AE(points)
    print("Decoded output: ", decoded.size())

    # chamfer loss
    chamfer_loss = PointLoss()  #  instantiate the loss
    print("Input shape: ", points.size())
    print("Decoded shape: ", decoded.size())

    # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
    loss = chamfer_loss(decoded, points)
    print(loss)


# TODO - Implement training phase (you should also implement cross-validation for tuning the hyperparameters)
# TODO - You should also implement the visualization tools (visualization_tools package)