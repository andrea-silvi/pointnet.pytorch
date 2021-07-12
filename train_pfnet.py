from __future__ import print_function
import numpy as np
import torch

from point_completion.multitask_ext_code_model import OnionNet
from point_completion.naive_model import PointNet_NaiveCompletionNetwork
from utils.loss import PointLoss, PointLoss_test
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from utils.dataset_seg import ShapeNetPart
import gc
import csv
from utils.early_stopping import EarlyStopping
from utils.FPS import farthest_point_sample, index_points
from utils.utils import farthest_and_nearest_points
import sys, json
from utils.utils import upload_args_from_json
from point_completion.multitask_model import MultiTaskCompletionNet
from visualization_tools import printPointCloud
from visualization_tools.printPointCloud import *
import neptune.new as neptune

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils.FPS import farthest_point_sample, index_points

class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128), 2)
        x_256 = torch.squeeze(self.maxpool(x_256), 2)
        x_512 = torch.squeeze(self.maxpool(x_512), 2)
        x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class Latentfeature(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList(
            [Convlayer(point_scales=self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0]))
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1]))
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2]))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)
        #        latentfeature_64 = F.relu(self.bn1(self.conv1(latentfeature)))
        #        latentfeature = F.relu(self.bn2(self.conv2(latentfeature_64)))
        #        latentfeature = F.relu(self.bn3(self.conv3(latentfeature)))
        #        latentfeature = latentfeature + latentfeature_64
        #        latentfeature_256 = F.relu(self.bn4(self.conv4(latentfeature)))
        #        latentfeature = F.relu(self.bn5(self.conv5(latentfeature_256)))
        #        latentfeature = F.relu(self.bn6(self.conv6(latentfeature)))
        #        latentfeature = latentfeature + latentfeature_256
        #        latentfeature = F.relu(self.bn7(self.conv7(latentfeature)))
        #        latentfeature = F.relu(self.bn8(self.conv8(latentfeature)))
        #        latentfeature = self.maxpool(latentfeature)
        #        latentfeature = torch.squeeze(latentfeature,2)
        return latentfeature


class PointcloudCls(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, k=40):
        super(PointcloudCls, self).__init__()
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.latentfeature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class _netG(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super(_netG, self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales, each_scales_size, point_scales_list)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.fc1_1 = nn.Linear(1024, 128 * 512)
        self.fc2_1 = nn.Linear(512, 64 * 128)  # nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256, 64 * 3)

        #        self.bn1 = nn.BatchNorm1d(1024)
        #        self.bn2 = nn.BatchNorm1d(512)
        #        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
        #        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
        #        self.bn5 = nn.BatchNorm1d(64*128)
        #
        self.conv1_1 = torch.nn.Conv1d(512, 512, 1)  # torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
        self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
        self.conv2_1 = torch.nn.Conv1d(128, 6, 1)  # torch.nn.Conv1d(256,12,1) !

    #        self.bn1_ = nn.BatchNorm1d(512)
    #        self.bn2_ = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.latentfeature(x)
        x_1 = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x_1))  # 512
        x_3 = F.relu(self.fc3(x_2))  # 256

        pc1_feat = self.fc3_1(x_3)
        pc1_xyz = pc1_feat.reshape(-1, 64, 3)  # 64x3 center1

        pc2_feat = F.relu(self.fc2_1(x_2))
        pc2_feat = pc2_feat.reshape(-1, 128, 64)
        pc2_xyz = self.conv2_1(pc2_feat)  # 6x64 center2

        pc3_feat = F.relu(self.fc1_1(x_1))
        pc3_feat = pc3_feat.reshape(-1, 512, 128)
        pc3_feat = F.relu(self.conv1_1(pc3_feat))
        pc3_feat = F.relu(self.conv1_2(pc3_feat))
        pc3_xyz = self.conv1_3(pc3_feat)  # 12x128 fine

        pc1_xyz_expand = torch.unsqueeze(pc1_xyz, 2)
        pc2_xyz = pc2_xyz.transpose(1, 2)
        pc2_xyz = pc2_xyz.reshape(-1, 64, 2, 3)
        pc2_xyz = pc1_xyz_expand + pc2_xyz
        pc2_xyz = pc2_xyz.reshape(-1, 128, 3)

        pc2_xyz_expand = torch.unsqueeze(pc2_xyz, 2)
        pc3_xyz = pc3_xyz.transpose(1, 2)
        pc3_xyz = pc3_xyz.reshape(-1, 128, int(self.crop_point_num / 128), 3)
        pc3_xyz = pc2_xyz_expand + pc3_xyz
        pc3_xyz = pc3_xyz.reshape(-1, self.crop_point_num, 3)

        return pc1_xyz, pc2_xyz, pc3_xyz  # center1 ,center2 ,fine


class _netlocalD(nn.Module):
    def __init__(self, crop_point_num):
        super(_netlocalD, self).__init__()
        self.crop_point_num = crop_point_num
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool = torch.nn.MaxPool2d((self.crop_point_num, 1), 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(448, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x_64 = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x_64)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_64 = torch.squeeze(self.maxpool(x_64))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        Layers = [x_256, x_128, x_64]
        x = torch.cat(Layers, 1)
        x = F.relu(self.bn_1(self.fc1(x)))
        x = F.relu(self.bn_2(self.fc2(x)))
        x = F.relu(self.bn_3(self.fc3(x)))
        x = self.fc4(x)
        return x


def cropping(batch_point_cloud, batch_target=None, num_cropped_points=512, fixed_choice=None):
    # batch_point_cloud: (batch_size, num_points, 3)
    batch_size = batch_point_cloud.size(0)
    num_points = batch_point_cloud.size(1)
    k = num_points - num_cropped_points
    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0]),
              torch.Tensor([-1, 1, 0])] if fixed_choice is None else [fixed_choice]
    p_center = []
    for m in range(batch_size):
        index = random.sample(choice, 1)
        p_center.append(index[0])
    # it should have shape (batch_size, 3): one center for one point_cloud inside the batch
    batch_centers = torch.cat(p_center, dim=0).cuda()
    batch_centers = batch_centers.view(-1, 1, 3)
    incomplete_input, idx, cropped_input = farthest_and_nearest_points(batch_point_cloud, batch_centers, k)
    if batch_target is not None:
        batch_target = batch_target.view(-1, 1)
        batch_target = batch_target[idx, :]
        return incomplete_input, batch_target.view(-1, k, 1), cropped_input
    # cropped_input is our incomplete_input
    return incomplete_input, cropped_input
    #
    # for n in range(opt.pnum):
    #     distance_list.append(distance_squre(real_point[m, 0, n], p_center))
    # distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
    #
    # for sp in range(opt.crop_point_num):
    #     input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
    #     real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

    # k = num_points-num_cropped_points
    # idx = torch.randint(0, num_points, (batch_size,), device="cuda")
    # idx_base = torch.arange(0, batch_size, device="cuda").view(-1) * num_points
    # idx = (idx + idx_base).view(-1)
    # batch_points = batch_point_cloud.view(-1, 3)[idx, :].view(-1, 1, 3)
    # incomplete_input, idx, cropped_input = farthest_and_nearest_points(batch_point_cloud, batch_points, k)
    # if batch_target is not None:
    #     batch_target = batch_target.view(-1, 1)
    #     batch_target = batch_target[idx, :]
    #     return incomplete_input,  batch_target.view(-1, k, 1), cropped_input

    # real center is our cropped_input
    # return incomplete_input, cropped_input


def test_example(test_dataloader, model, n_classes, n_crop_points=512):
    # initialize lists to monitor test loss and accuracy
    test_loss_2048 = 0.0
    test_loss_512 = 0.0
    chamfer_loss = PointLoss_test()
    seg_test_loss = 0.0
    accuracy_test_loss = 0.0
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for data in test_dataloader:
            gc.collect()
            torch.cuda.empty_cache()

            points = data.cuda()
            incomplete_input_test, cropped_input_test = cropping(points, None)
            incomplete_input_test, cropped_input_test = incomplete_input_test.cuda(), cropped_input_test.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(incomplete_input_test)
            loss_cropped_pc = chamfer_loss(cropped_input_test, output)
            test_loss_512 += np.array(loss_cropped_pc) * points.size(0)
            output = torch.cat((output, incomplete_input_test), dim=1)
            loss_2048 = chamfer_loss(points, output)
            test_loss_2048 += np.array(loss_2048) * points.size(0)

        # calculate and print avg test loss
        test_loss_2048 = test_loss_2048 / len(test_dataloader.dataset)
        test_loss_512 = test_loss_512 / len(test_dataloader.dataset)
        print(f'Test Loss (overall pc: mean, gt->pred, pred->gt): {test_loss_2048}\n')

        return test_loss_2048, test_loss_512


def evaluate_loss_by_class(opt, autoencoder, run, n_classes):
    classes = ["airplane", "car", "chair", "lamp", "mug", "motorbike", "table"]
    novel_classes = []
    # n.b.: training_classes should be equal to classes.
    training_classes = opt.dict_category_offset if hasattr(opt, "dict_category_offset") else None

    novel_classes = ["bag", "cap", "earphone", "guitar", "knife", "laptop", "pistol", "rocket", "skateboard"]
    classes.extend(novel_classes)
    autoencoder.cuda()
    print("Start evaluation loss by class")
    for classs in classes:
        print(f"\t{classs}")
        test_dataset = ShapeNetPart(opt.dataset,
                                    class_choice=classs,
                                    split='test',
                                    segmentation=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=36,
            shuffle=True,
            num_workers=int(2))
        losss = test_example(test_dataloader, autoencoder, n_classes)
        run[f"loss/overall_pc/{classs}_cd_(gt->pred)"] = losss[0][1]
        run[f"loss/overall_pc/{classs}_cd_(pred->gt)"] = losss[0][2]
        run[f"loss/cropped_pc/{classs}_cd_(gt->pred)"] = losss[1][1]
        run[f"loss/cropped_pc/{classs}_cd_(pred->gt)"] = losss[1][2]




def train_pc(opt):
    neptune_info = json.loads(open(os.path.join("parameters", "neptune_params.json")).read())
    tag = "PF-Net"
    run = neptune.init(project=neptune_info['project'],
                       tags=[tag],
                       api_token=neptune_info['api_token'])
    run['params'] = vars(opt)
    random_seed = 43
    num_classes = None
    n_crop_points = 512
    torch.manual_seed(random_seed)

    training_dataset = ShapeNetPart(
        root="/content/drive/MyDrive/shapenetpart_hdf5_2048",
        class_choice=None,
        segmentation=False,
        split="trainval"
    )

    train_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=36,
        shuffle=True,
        num_workers=2)

    try:
        os.makedirs("cls")
    except OSError:
        pass
    num_classes = 0
    point_netG = _netG(3, 1, [1536, 768, 384], 512)
    point_netD = _netlocalD(512)

    optimizerG = optim.Adam(point_netG.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-5,
                           weight_decay=0)
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)
    optimizerD = optim.Adam(point_netD.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-5,
                            weight_decay=0)
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    point_netG.cuda()
    point_netD.cuda()
    real_label = 1
    fake_label = 0
    label = torch.FloatTensor(36)
    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_PointLoss = PointLoss().cuda()
    checkpoint_path = os.path.join("cls", f"checkpoint{0}.pt")
    gc.collect()
    torch.cuda.empty_cache()
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=checkpoint_path)
    #  instantiate the loss
    chamfer_loss = PointLoss()
    weight_sl = 0.6
    for epoch in range(130):
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        training_losses = []
        for i, data in enumerate(train_dataloader, 0):

            real_points = data
            real_points = real_points.cuda()
            incomplete_input1, cropped_input = cropping(real_points, None)
            incomplete_input1, cropped_input = incomplete_input1.cuda(), cropped_input.cuda()
            label.resize_([36, 1]).fill_(real_label)
            label = label.cuda()
            point_netG.train()
            point_netD.train()
            input_cropped2_idx = farthest_point_sample(incomplete_input1, 64, RAN=False)
            input_cropped2 = index_points(incomplete_input1, input_cropped2_idx)
            input_cropped3_idx = farthest_point_sample(incomplete_input1, 128, RAN=True)
            input_cropped3 = index_points(incomplete_input1, input_cropped3_idx)
            input_cropped1 = incomplete_input1.cuda()
            input_cropped2 = input_cropped2.cuda()
            input_cropped3 = input_cropped3.cuda()
            input_cropped = [input_cropped1, input_cropped2, input_cropped3]

            point_netD.zero_grad()
            output = point_netD(input_cropped)
            errD_real = criterion(output, label)
            errD_real.backward()
            fake_center1, fake_center2, fake = point_netG(input_cropped)
            label.data.fill_(fake_label)
            output = point_netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            point_netG.zero_grad()
            label.data.fill_(real_label)
            output = point_netD(fake)
            errG_D = criterion(output, label)
            coarse_sampling_idx = farthest_point_sample(cropped_input, 64, RAN=False)
            coarse_sampling = index_points(cropped_input, coarse_sampling_idx)
            coarse_sampling = coarse_sampling.cuda()
            fine_sampling_idx = farthest_point_sample(cropped_input, 128, RAN=True)
            fine_sampling = index_points(cropped_input, fine_sampling_idx)
            fine_sampling = fine_sampling.cuda()
            CD_loss = criterion_PointLoss(cropped_input, fake)
            loss = CD_loss \
                   + alpha1 * chamfer_loss(coarse_sampling, fake_center1) \
                   + alpha2 * chamfer_loss(fine_sampling, fake_center2)
            errG = errG_D
            errG.backward()
            optimizerG.step()


        print(f'\tepoch: {epoch}')


    torch.save(point_netG.state_dict(), checkpoint_path)

    evaluate_loss_by_class(point_netG, 0, num_classes)
    run.stop()
    return point_netG, 0


if __name__ == '__main__':
    # filename = "D:\\UNIVERSITA\\PRIMO ANNO\\SECONDO SEMESTRE\\Machine learning and Deep learning\\PROJECTS\\P1\\shapenetcorev2_hdf5_2048"
    # training_dataset = ShapeNetPart(
    #         root=filename,
    #         class_choice="None",
    #         segmentation=True,
    #         split="trainval"
    #     )
    # train_dataloader = torch.utils.data.DataLoader(
    #     training_dataset,
    #     batch_size=8,
    #     shuffle=True)
    opt = upload_args_from_json(os.path.join("parameters", "pc_fixed_params.json"))
    # for i, data in enumerate(train_dataloader, 0):
    #     val_points, target = data
    #     incomplete_input_val, target, cropped_input_val = cropping(val_points, target)
    #     inc_np = incomplete_input_val.cpu().detach().numpy()
    #     cropped = cropped.cpu().detach().numpy()
    #     savePtsFile(f"incomplete", "prova", opt, inc_np)
    #     savePtsFile(f"cropped", "prova", opt, cropped)
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_pc(opt)
