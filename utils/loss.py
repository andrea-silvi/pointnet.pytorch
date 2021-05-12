"""
## Loss Function - Chamfer
Given the Original Point Cloud (PC) X and the decoded output Y we need a way to penalize the predicted/decoded w.r.t. the original PC X.<br>
Standard MSE (Mean Squared Error) can't work cause both input and output are unordered sets of xyz-points.<br>
Think at PCs as vectors of xyz-points, the order of each of the points is absolutely meaningless.
Thus meaning that, since the lack of ordering for both input and decoded sets, you cannot simply compute the euclidean distance between the i-th element of the input tensor X and the i-th one of the decoded tensor Y.<br>
We'll use the Chamfer Distance as training loss to keep the generated/decoded points close to original/(not encoded) points.
The Chamfer Distance is not actually a distance but a pseudo-distance (dist(A,B) != dist(B,A)).<br>
Let's consider the two sets X and Y. For each point in X we'll compute the euclidean distance to the nearest in the Y set. The same in the opposite direction, given each point of Y we'll compute the distance to the nearest in the X set. By doing so we're not making any assumption on the ordering of the X and Y sets.<br>
The final loss will be the sum of two equally weighted contribution:
1.   AVG_Y2X - Average of all nearest-dist Y->X: for each point in Y consider the distance to the nearest in X
2.   AVG_X2Y - Average of all nearest-dist X->Y: same as 1. but opposite

Thus:

> CD_Loss = cd_weight * ( 0.5 * AVG_Y2X + 0.5 * AVG_X2Y )

Where 'cd_weight' is a loss weight term that you should cross-validate. Hint: try with cd_weight=100!
"""

import torch
import torch.nn as nn


# Simple Chamfer Distance / Loss implementation
# Implementation from https://github.com/zztianzz/PF-Net-Point-Fractal-Network/blob/master/utils.py
# CVPR2020 Paper PF-Net: Point Fractal Network for 3D Point Cloud Completion.

def array2samples_distance(array1, array2):
    """
    arguments:
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
    """
    num_point1, num_features1 = array1.shape
    num_point2, num_features2 = array2.shape
    expanded_array1 = array1.repeat(num_point2, 1)
    expanded_array2 = torch.reshape(
        torch.unsqueeze(array2, 1).repeat(1, num_point1, 1),
        (-1, num_features2))
    distances = (expanded_array1 - expanded_array2) * (expanded_array1 - expanded_array2)
    # distances = torch.sqrt(distances)
    distances = torch.sum(distances, dim=1)
    distances = torch.reshape(distances, (num_point2, num_point1))
    distances = torch.min(distances, dim=1)[0]
    distances = torch.mean(distances)
    return distances


def chamfer_distance_numpy(array1, array2, cd_weight):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (0.5 * av_dist1 + 0.5 * av_dist2) / batch_size
    return cd_weight * dist


class PointLoss(nn.Module):
    def __init__(self, cd_weight=100):
        super(PointLoss, self).__init__()
        self.cd_weight = cd_weight

    def forward(self, array1, array2):
        return chamfer_distance_numpy(array1, array2, self.cd_weight)

