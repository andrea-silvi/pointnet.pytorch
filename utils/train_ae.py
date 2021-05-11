from __future__ import print_function
import torch
from utils.pointnet_model import PointNet_AutoEncoder
from loss import PointLoss
import argparse
import os
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset


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


def train_example(opt):
    dataset = ShapeNetDataset(
        root=opt.dataset,
        class_choice="Knife",
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        split='test',
        class_choice="Knife",
        npoints=opt.num_points,
        data_augmentation=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    # testdataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=opt.batchSize,
    #     shuffle=True,
    #     num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    autoencoder = PointNet_AutoEncoder(opt.num_points, opt.feature_transform)

    # TODO - import pointnet parameters (encoder network)
    if opt.model != '':
        autoencoder.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    autoencoder.cuda()

    num_batch = len(dataset) / opt.batchSize
    # TODO - modify number of epochs (from 5 to opt.nepoch)
    for epoch in range(5):
        scheduler.step()
        for i, points in enumerate(dataloader, 0):
            print(f"Points size: {points.size()}")
            points = points.transpose(2, 1)
            points = points.cuda()
            optimizer.zero_grad()
            autoencoder = autoencoder.train()
            decoded_points = autoencoder(points)
            print(f"Decoded points size: {decoded_points.size()}")
            decoded_points = decoded_points.cuda()
            chamfer_loss = PointLoss()  #  instantiate the loss
            # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
            loss = chamfer_loss(decoded_points, points)
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            print('[%d: %d/%d] train loss: %f' % (
            epoch, i, num_batch, loss.item()))

            # if i % 10 == 0:
            #     j, data = next(enumerate(testdataloader, 0))
            #     points, target = data
            #     target = target[:, 0]
            #     points = points.transpose(2, 1)
            #     points, target = points.cuda(), target.cuda()
            #     classifier = classifier.eval()
            #     pred, _, _ = classifier(points)
            #     loss = F.nll_loss(pred, target)
            #     pred_choice = pred.data.max(1)[1]
            #     correct = pred_choice.eq(target.data).cpu().sum()
            #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
            #     epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))

        torch.save(autoencoder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    # total_correct = 0
    # total_testset = 0
    # for i, data in tqdm(enumerate(testdataloader, 0)):
    #     points, target = data
    #     target = target[:, 0]
    #     points = points.transpose(2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     pred, _, _ = classifier(points)
    #     pred_choice = pred.data.max(1)[1]
    #     correct = pred_choice.eq(target.data).cpu().sum()
    #     total_correct += correct.item()
    #     total_testset += points.size()[0]
    #
    # print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__=='__main__':
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
    # parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    opt = parser.parse_args()
    print(opt)
    train_example(opt)

# TODO - Implement training phase (you should also implement cross-validation for tuning the hyperparameters)
# TODO - You should also implement the visualization tools (visualization_tools package)