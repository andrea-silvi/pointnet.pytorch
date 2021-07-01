import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            raise NotImplementedError("Feature Transformer not implemented.")
            # self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # [batch_size, num_points, emb_size]
        x = torch.max(x, 2, keepdim=True)[0]  # [batch_size, num_points, emb_size] ==> [batch_size, emb_size]
        x = x.view(-1, 1024)
        if self.global_feat:
            # 'x' are the global features: embedding vector which can be used for Classification or other tasks on the whole shape
            # Obtained by performing maxpooling on per-point features (see row 35)
            # Shape is: [batch_size, emb_size]
            return x  # , trans, trans_feat
        else:
            # returning here the features of each point!
            # without maxpooling reduction
            # Shape is: [batch_size, num_points, emb_size]
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)  # , trans, trans_feat


class Decoder(nn.Module):
    ''' Just a lightweight Fully Connected decoder:
    '''

    def __init__(self, num_points=1024, size_encoder=1024, dropout=0):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(size_encoder, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        #self.dp = nn.Dropout(p=dropout)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = self.dp(x)
        x = self.th(self.fc5(x))
        x = x.view(batchsize, self.num_points, 3)
        return x

class PyramidDecoder(nn.Module):
    ''' Point pyramid decoder from PF_Net:
    '''

    def __init__(self, args):
        super(PyramidDecoder, self).__init__()
        self.num_points = args.num_points
        self.fc1 = nn.Linear(args.size_encoder, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 256 * self.num_points)
        self.fc2_1 = nn.Linear(512, 128 * 256)
        self.fc3_1 = nn.Linear(256, 128 * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        self.conv1_1 = torch.nn.Conv1d(self.num_points, self.num_points, 1)
        self.conv1_2 = torch.nn.Conv1d(self.num_points, 512, 1)
        self.conv1_3 = torch.nn.Conv1d(512, int((self.num_points * 3) / 256), 1)
        self.conv2_1 = torch.nn.Conv1d(256, 6, 1)

        #        self.bn1_ = nn.BatchNorm1d(512)
        #        self.bn2_ = nn.BatchNorm1d(256)

    def forward(self, x):
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 128, 3)  # 128x3 [center1 ,coarse sampling] final!

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 256, 128)
        pc2_xyz = self.conv2_1(pc2_feat)  # 256x128 -> 6x128 [center2, fine sampling]

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, self.num_points, 256)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))  # 1024x256 -> 1024x256
        pc3_feat = F.relu(self.conv1_2(pc3_feat))  # 1024x256 -> 512x256
        pc3_xyz = self.conv1_3(pc3_feat)  # 512x256 -> 12x256 complete

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)  # 128x1x3
        pc2_xyz = pc2_xyz.transpose(1, 2)  # 128x6
        pc2_xyz = pc2_xyz.reshape(-1, 128, 2, 3)  # 128x2x3
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 256, 3)  # 128x2x3 -> 256x3 final!

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)  # 256x1x3
        pc3_xyz = pc3_xyz.transpose(1, 2)  # 256x12
        pc3_xyz = pc3_xyz.reshape(-1, 256, int(self.num_points / 256), 3)  # 256x4x3
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.num_points, 3)  # 1024x3 final!

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,complete

class PointNet_AutoEncoder(nn.Module):
    '''
  Complete AutoEncoder Model:
  Given an input point cloud X:
      - Step 1: encode the point cloud X into a latent low-dimensional code
      - Step 2: Starting from the code geneate a representation Y as close as possible to the original input X

  Details:
  1. the 'code' size is hardocoded to 100 at line 45 - could be detrimental such a small code size
  1.1. If you change the code size you must modify accordingly also the decoder
  2. 'num_points' is the parameter controlling the number of points to be generated. In general we want to generate a number of points equal to the number of input points.
  '''

    def __init__(self, args, num_points=1024, size_encoder=1024, feature_transform=False, dropout=0):
        super(PointNet_AutoEncoder, self).__init__()
        #print("PointNet AE Init - num_points (# generated): %d" % num_points)

        #  Encoder Definition
        self.encoder = torch.nn.Sequential(
            PointNetfeat(global_feat=True, feature_transform=feature_transform),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, size_encoder))

        # Decoder Definition
        self.decoder = PyramidDecoder(args=args) if args.type_decoder == "pyramid" else \
            Decoder(num_points=num_points, size_encoder=size_encoder, dropout=dropout)

    def forward(self, x):
        BS, N, dim = x.size()
        #print(x.size())
        assert dim == 3, f"Fail: expecting 3 (x-y-z) as last tensor dimension! Found {dim}"

        #  Refactoring batch for 'PointNetfeat' processing
        x = x.permute(0, 2, 1)  # [BS, N, 3] => [BS, 3, N]

        # Encoding
        code = self.encoder(x)  # [BS, 3, N] => [BS, size_encoder]

        # Decoding
        decoded = self.decoder(code)  #either a [BS, N, 3] pointcloud or 3 X [BS, N, 3] pointclouds (based on decoder)

        return decoded


# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.utils.data
# from torch.autograd import Variable
# import numpy as np
# import torch.nn.functional as F
#
#
# class STN3d(nn.Module):
#     def __init__(self):
#         super(STN3d, self).__init__()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 9)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, 3, 3)
#         return x
#
#
# class STNkd(nn.Module):
#     def __init__(self, k=64):
#         super(STNkd, self).__init__()
#         self.conv1 = torch.nn.Conv1d(k, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#         self.relu = nn.ReLU()
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#         self.k = k
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
#         if x.is_cuda:
#             iden = iden.cuda()
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x
#
# class PointNetfeat(nn.Module):
#     def __init__(self, global_feat = True, feature_transform = False):
#         super(PointNetfeat, self).__init__()
#         self.stn = STN3d()
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)
#
#     def forward(self, x):
#         n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))
#
#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2,1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2,1)
#         else:
#             trans_feat = None
#
#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat
#
# class PointNetCls(nn.Module):
#     def __init__(self, k=2, feature_transform=False):
#         super(PointNetCls, self).__init__()
#         self.feature_transform = feature_transform
#         self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p=0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1), trans, trans_feat
#
#
# class PointNetDenseCls(nn.Module):
#     def __init__(self, k = 2, feature_transform=False):
#         super(PointNetDenseCls, self).__init__()
#         self.k = k
#         self.feature_transform=feature_transform
#         self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
#         self.conv1 = torch.nn.Conv1d(1088, 512, 1)
#         self.conv2 = torch.nn.Conv1d(512, 256, 1)
#         self.conv3 = torch.nn.Conv1d(256, 128, 1)
#         self.conv4 = torch.nn.Conv1d(128, self.k, 1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         n_pts = x.size()[2]
#         x, trans, trans_feat = self.feat(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x)
#         x = x.transpose(2,1).contiguous()
#         x = F.log_softmax(x.view(-1,self.k), dim=-1)
#         x = x.view(batchsize, n_pts, self.k)
#         return x, trans, trans_feat
#
# def feature_transform_regularizer(trans):
#     d = trans.size()[1]
#     batchsize = trans.size()[0]
#     I = torch.eye(d)[None, :, :]
#     if trans.is_cuda:
#         I = I.cuda()
#     loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
#     return loss
#
# if __name__ == '__main__':
    # sim_data = Variable(torch.rand(32,3,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))
    #
    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    #
    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())
    #
    # seg = PointNetDenseCls(k = 3)
    # out, _, _ = seg(sim_data)
    # print('seg', out.size())
