#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #
    # batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.size_encoder)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.size_encoder, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # TODO - la rete riportata sopra rappresenta l'ENCODER
        # TODO - inserire i seguenti layer all'interno di una classe DECODER


    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = x.view(batch_size, self.args.size_encoder*2)
        return x

class Decoder(nn.Module):
    ''' Just a lightweight Fully Connected decoder:
    '''



    def __init__(self, args):
        super(Decoder, self).__init__()
        self.num_points = args.num_points
        self.linear1 = nn.Linear(args.size_encoder, 1280, bias=False)
        self.bn1 = nn.BatchNorm1d(1280)
        self.linear2 = nn.Linear(1280, 1536)
        self.bn2 = nn.BatchNorm1d(1536)
        self.linear3 = nn.Linear(1536, 1792)
        self.bn3 = nn.BatchNorm1d(1792)
        self.linear4 = nn.Linear(1792, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(p=args.dropout)
        self.linear5 = nn.Linear(2048, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.linear3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear4(x)), negative_slope=0.2)
        x = self.dp(x)
        x = self.th(self.linear5(x))
        x = x.view(batch_size, 3, self.num_points)
        return x

class DGCNN_AutoEncoder(nn.Module):
    '''
  Complete AutoEncoder Model:
  Given an input point cloud X:
      - Step 1: encode the point cloud X into a latent low-dimensional code
      - Step 2: Starting from the code geneate a representation Y as close as possible to the original input X


  '''

    def __init__(self, args):
        super(DGCNN_AutoEncoder, self).__init__()
        #print("PointNet AE Init - num_points (# generated): %d" % num_points)

        # Encoder Definition
        self.encoder = torch.nn.Sequential(
            DGCNN(args=args),
            nn.Linear(args.size_encoder, int(2*args.size_encoder/3)),
            nn.ReLU(),
            nn.Linear(int(2*args.size_encoder/3), args.size_encoder))


        # Decoder Definition
        self.decoder = Decoder(args=args)

    def forward(self, x):
        BS, N, dim = x.size()
        #print(x.size())
        assert dim == 3, f"Fail: expecting 3 (x-y-z) as last tensor dimension! Found {dim}"

        #  Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding
        code = self.encoder(x)  # [BS, 3, N] => [BS, size_encoder]

        # Decoding
        decoded = self.decoder(code)  #  [BS, 3, num_points]

        # Reshaping decoded output before returning..
        decoded = decoded.permute(0, 2, 1)  #  [BS, 3, num_points] => [BS, num_points, 3]

        return decoded

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=0.999, help="decay rate for second moment")
    parser.add_argument("--size_encoder", type=int, default=7, help="How long to wait after last time val loss improved.")
    parser.add_argument("--dropout", type=int, default=0, help="How long to wait after last time val loss improved.")
    opt = parser.parse_args()
    model = DGCNN(opt)
    model.forward(torch.rand((32, 3, 1024)))
