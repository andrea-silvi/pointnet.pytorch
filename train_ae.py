from __future__ import print_function

import numpy as np
import torch
from pointnet.pointnet_model import PointNet_AutoEncoder
from utils.loss import PointLoss
import argparse
import os
import torch.optim as optim
import torch.utils.data
from utils.dataset import ShapeNetDataset
from visualization_tools import printPointCloud as ptPC
import gc
import csv
from utils.early_stopping import EarlyStopping
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


def print_loss_graph(training_history, val_history, opt):
    folder = os.path.join(opt.outf, "grid_search_results")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(os.path.join(folder, f'{hash(str(opt))}_losses.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows([training_history, val_history])
    # plt.plot(training_history, '-bx')
    # plt.plot(val_history, '-rx')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['Training', 'Validation'])
    # plt.title('Loss vs. No. of epochs')
    # plt.savefig('loss.png', bbox_inches='tight',)


def train_example(opt):
    random_seed = 43
    torch.manual_seed(random_seed)
    # writer = SummaryWriter('runs/train_ae_experiment_1')
    training_dataset = ShapeNetDataset(
        root=opt.dataset,
        class_choice=opt.train_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size)

    validation_dataset = ShapeNetDataset(
        root=opt.dataset,
        split='val',
        class_choice=opt.train_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        split='test',
        class_choice=opt.test_class_choice,
        npoints=opt.num_points,
        set_size=opt.set_size)

    train_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    val_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataloader = torch.utils.data.DataLoader(
         test_dataset,
         batch_size=opt.batchSize,
         shuffle=True,
         num_workers=int(opt.workers))

    print(f"Length training/validation/test datasets: {len(training_dataset)}, {len(validation_dataset)}, "
          f"{len(test_dataset)}")

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    autoencoder = PointNet_AutoEncoder(opt.num_points, opt.size_encoder)

    # TODO - import pointnet parameters (encoder network)
    if opt.model != '':
        autoencoder.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),
                           weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_stepSize, gamma=opt.scheduler_gamma)
    autoencoder.cuda()


    #num_batch = len(dataset) / opt.batchSize
    # TODO - modify number of epochs (from 5 to opt.nepoch)
    checkpoint_path = os.path.join(opt.outf, f"{hash(str(opt))}_checkpoint.pt")
    training_history = []
    val_history = []
    gc.collect()
    torch.cuda.empty_cache()
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=checkpoint_path)
    # flag_stampa = False
    n_epoch = opt.nepoch
    for epoch in range(n_epoch):
        scheduler.step()
        training_losses = []
        #running_loss = 0.0
        for i, points in enumerate(train_dataloader, 0):
            # print(f"Points size: {points.size()}")
            # points = points.transpose(2, 1)
            points = points.cuda()
            optimizer.zero_grad()
            autoencoder.train()
            decoded_points = autoencoder(points)
            #print(f"Decoded points size: {decoded_points.size()}")
            decoded_points = decoded_points.cuda()
            chamfer_loss = PointLoss()  #  instantiate the loss
            # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
            loss = chamfer_loss(decoded_points, points)
            if epoch==0 and i==0:
                print(f"LOSS: first epoch, first batch: \t {loss}")
            training_losses.append(loss.item())
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            #running_loss += loss.item()
            optimizer.step()
            #if i % 1000 == 999: #every 1000 mini batches
                #writer.add_scalar('training loss', running_loss/1000, epoch * len(train_dataloader))
        gc.collect()
        torch.cuda.empty_cache()

        # TODO - VALIDATION PHASE
        with torch.no_grad():
            val_losses = []
            for j, val_points in enumerate(val_dataloader, 0):
                autoencoder.eval()
                val_points = val_points.cuda()
                decoded_val_points = autoencoder(val_points)
                # if (flag_stampa is False) and (epoch == n_epoch-1):
                #     val_stamp = val_points[0,:,:].cpu().numpy()
                #     dec_val_stamp = decoded_points[0,:,:].cpu().numpy()
                #     #np.savetxt("validation_point", val_stamp, delimiter=" ")
                #     #np.savetxt("decoded_validation_point", dec_val_stamp, delimiter=" ")
                #     flag_stampa=True
                #     #print("sono qui")
                #     ptPC.printCloud(val_stamp, "original_validation_points")
                #     ptPC.printCloud(dec_val_stamp,"decoded_validation_points")




                decoded_val_points = decoded_val_points.cuda()
                chamfer_loss = PointLoss()  #  instantiate the loss
                val_loss = chamfer_loss(decoded_val_points, val_points)
                if j==0:
                    print(f"LOSS FIRST VALIDATION BATCH: {val_loss}")
                val_losses.append(val_loss.item())

            train_mean = np.average(training_losses)
            val_mean = np.average(val_losses)
            print(f'epoch: {epoch} , training loss: {train_mean}, validation loss: {val_mean}')

        early_stopping(val_mean, autoencoder)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        else:
            training_history.append(train_mean)
            val_history.append(val_mean)

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

        #Commented: early_stopping already saves the best model
        #torch.save(autoencoder.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
    autoencoder.load_state_dict(torch.load(checkpoint_path))

    # TODO PLOT LOSSES
    #print(training_history)
    #print(val_history)
    print_loss_graph(training_history, val_history, opt)
    return autoencoder, val_history
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
    parser.add_argument("--set_size", type=float, default=1, help="Subset size (between 0 and 1) of the training set. "
                                                                "Use it for fake test")
    parser.add_argument("--size_encoder", type=int, default=1024, help="Size latent code")
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='Number points from point cloud')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--train_class_choice', type=str, default=None, help="Training class")
    parser.add_argument('--test_class_choice', type=str, default=None, help="Test class")
    # parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="reduction factor of the learning rate")
    parser.add_argument("--scheduler_stepSize", type=int, default=20)
    parser.add_argument("--dropout", type=int, default=1, help="dropout percentage")
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--beta_1", type=float, default=0.9, help="decay rate for first moment")
    parser.add_argument("--beta_2", type=float, default=0.999, help="decay rate for second moment")
    parser.add_argument("--patience", type=int, default=7, help="How long to wait after last time val loss improved.")

    opt = parser.parse_args()
    print(opt)
    train_example(opt)

# TODO - Implement training phase (you should also implement cross-validation for tuning the hyperparameters)
# TODO - You should also implement the visualization tools (visualization_tools package)