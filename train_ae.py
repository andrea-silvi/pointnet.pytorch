from __future__ import print_function
import numpy as np
import torch
from pointnet.pointnet_model import PointNet_AutoEncoder
from pointnet.deeper_pointnet_model import PointNet_DeeperAutoEncoder
from utils.loss import PointLoss
import argparse
import os
import torch.optim as optim
import torch.utils.data
from utils.dataset import ShapeNetDataset
import gc
import csv
from utils.early_stopping import EarlyStopping
import sys, json

from visualization_tools import printPointCloud
from visualization_tools.printPointCloud import *

import neptune.new as neptune


# the following function doesn't make the training of the network!!
# It shows the interfaces with the classes necessary for the point cloud completion task
# N.B.: only with PointNetAE and PointLoss (the one used for evaluating the Chamfer distance)
# the gcnn interface is not implemented yet
# def example_AE_and_chamfer_loss():
#     """
#     Instantiate a PointNetAutoEncoder
#     Feed it with a synthetic point cloud
#     Compute the encoded point cloud (output of encoder)
#     Compute the decoded point cloud (output of decoder)
#     Compute chamfer loss
#     :return:
#     """
#     batch_size = 32
#     input_points = 1024
#
#     # Instantiate a fake batch of point clouds
#     points = torch.rand(batch_size, input_points, 3)
#     print("Input points: ", points.size())
#
#     # Instantiate the AE
#     pointnet_AE = PointNet_DeeperAutoEncoder(num_points=input_points)
#
#     # Move everything (data + model) to GPU
#     assert torch.cuda.device_count() > 0, "Fail: No GPU device detected"
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     points = points.to(device)
#     pointnet_AE = pointnet_AE.to(device)
#
#     # try AE forward
#     decoded = pointnet_AE(points)
#     print("Decoded output: ", decoded.size())
#
#     # chamfer loss
#     chamfer_loss = PointLoss()  #  instantiate the loss
#     print("Input shape: ", points.size())
#     print("Decoded shape: ", decoded.size())
#
#     # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
#     loss = chamfer_loss(decoded, points)
#     print(loss)
#

def print_loss_graph(training_history, val_history, opt):
    folder = os.path.join(opt.outf, "grid_search_results")
    try:
        os.makedirs(folder)
    except OSError:
        pass
    #with open(os.path.join(folder, f'{hash(str(opt))}_losses.csv'), 'w') as f:
    with open(os.path.join(folder, f'{opt.runNumber}_losses.csv'), 'w') as f:
        writer = csv.writer(f)
        if val_history == None:
            writer.writerow(training_history)
        else:
            writer.writerows([training_history, val_history])
    # plt.plot(training_history, '-bx')
    # plt.plot(val_history, '-rx')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['Training', 'Validation'])
    # plt.title('Loss vs. No. of epochs')
    # plt.savefig('loss.png', bbox_inches='tight',)


def upload_args_from_json(file_path=os.path.join("parameters", "fixed_params.json")):
    parser = argparse.ArgumentParser(description=f'Arguments from json')
    args = parser.parse_args()
    json_params = json.loads(open(file_path).read())
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    setattr(args, "runNumber", 0)
    return args


def test_example(opt, test_dataloader, model):
    # initialize lists to monitor test loss and accuracy
    chamfer_loss = PointLoss()
    test_loss = 0.0

    model.eval()  # prep model for evaluation

    for data in test_dataloader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = chamfer_loss(data, output)
        # update test loss
        test_loss += loss.item() * data.size(0)

    # calculate and print avg test loss
    test_loss = test_loss / len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    return test_loss


def train_example(opt):
    run = neptune.init(project='vittoriop.17/PointNet',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NzIxMmE4MC05OTBjLTRiMTMtODAzZi0yNzgzZTMwNjQ3OGUifQ==')
    run['params'] = vars(opt)
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

    final_training = opt.final_training
    if final_training:
        if opt.runNumber == 0:
            print("!!!!!!Final training starts!!!!!!")
        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            split='test',
            class_choice=opt.test_class_choice,
            npoints=opt.num_points,
            set_size=opt.set_size)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))
        training_dataset = torch.utils.data.ConcatDataset([training_dataset, validation_dataset])

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

    # print(f"Length training/validation/test datasets: {len(training_dataset)}, {len(validation_dataset)}, "
    #      f"{len(test_dataset)}")

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    autoencoder = PointNet_DeeperAutoEncoder(opt.num_points, opt.size_encoder, dropout=opt.dropout) \
                    if opt.architecture == "deep" else \
                    PointNet_AutoEncoder(opt.num_points, opt.size_encoder, dropout=opt.dropout)
    if opt.runNumber == 0 and opt.architecture == "deep":
        print("!!!!!!Training a deeper model!!!!!!")
    # TODO - import pointnet parameters (encoder network)
    if opt.model != '':
        autoencoder.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),
                           weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_stepSize, gamma=opt.scheduler_gamma)
    autoencoder.cuda()
    run["model"] = autoencoder
    # num_batch = len(dataset) / opt.batchSize
    # TODO - modify number of epochs (from 5 to opt.nepoch)
    #checkpoint_path = os.path.join(opt.outf, f"{hash(str(opt))}_checkpoint.pt")
    checkpoint_path = os.path.join(opt.outf, "checkpoint.pt")
    training_history = []
    val_history = []
    gc.collect()
    torch.cuda.empty_cache()
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=checkpoint_path)
    # flag_stampa = False
    n_epoch = opt.nepoch
    n_batches = np.floor(training_dataset.__len__() / opt.batchSize)
    for epoch in range(n_epoch):
        if epoch > 0:
            scheduler.step()
        training_losses = []
        # running_loss = 0.0
        for i, points in enumerate(train_dataloader, 0):
            # print(f"Points size: {points.size()}")
            # points = points.transpose(2, 1)
            points = points.cuda()
            optimizer.zero_grad()
            autoencoder.train()
            decoded_points = autoencoder(points)
            # print(f"Decoded points size: {decoded_points.size()}")
            decoded_points = decoded_points.cuda()
            chamfer_loss = PointLoss()  #  instantiate the loss
            # let's compute the chamfer distance between the two sets: 'points' and 'decoded'
            loss = chamfer_loss(decoded_points, points)
            # if epoch==0 and i==0:
            # print(f"LOSS: first epoch, first batch: \t {loss}")
            training_losses.append(loss.item())
            run["train/batch_loss"].log(loss.item())
            # if opt.feature_transform:
            #     loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            # running_loss += loss.item()
            optimizer.step()
            # if i % 1000 == 999: #every 1000 mini batches
            # writer.add_scalar('training loss', running_loss/1000, epoch * len(train_dataloader))
            # print_there(text=f"TRAINING: \t Epoch: {epoch}/{n_epoch},\t batch: {i}/{n_batches}")
        gc.collect()
        torch.cuda.empty_cache()
        train_mean = np.average(training_losses)
        run["train/epoch_loss"].log(train_mean)

        # TODO - VALIDATION PHASE
        if not final_training:
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
                    # if j==0:
                    # print(f"LOSS FIRST VALIDATION BATCH: {val_loss}")
                    val_losses.append(val_loss.item())
                    run["validation/batch_loss"].log(val_loss.item())
                val_mean = np.average(val_losses)
                run["validation/epoch_loss"].log(val_mean)

                print(f'\tepoch: {epoch}, training loss: {train_mean}, validation loss: {val_mean}')
        else:
            print(f'\tepoch: {epoch}, training loss: {train_mean}')

        if epoch >= 50:
            early_stopping(val_mean if not final_training else train_mean, autoencoder)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        training_history.append(train_mean)
        if not final_training:
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

        # Commented: early_stopping already saves the best model
    torch.save(autoencoder.state_dict(), checkpoint_path)
    autoencoder.load_state_dict(torch.load(checkpoint_path))
    printPointCloud.print_original_decoded_point_clouds(ShapeNetDataset(
            root=opt.dataset,
            split='test',
            class_choice=opt.test_class_choice,
            npoints=opt.num_points,
            set_size=opt.set_size), opt.test_class_choice, autoencoder, opt, run, train=True)

    # TODO PLOT LOSSES
    # print(training_history)
    # print(val_history)
    if not final_training:
        print_loss_graph(training_history, val_history, opt)
        run.stop()
        return autoencoder, val_history
    else:
        print_loss_graph(training_history, None, opt)
        test_loss = test_example(opt, test_dataloader, autoencoder)
        run.stop()
        return autoencoder, test_loss


def train_model_by_class(opt):
    worst_test_loss = 0
    worst_class = ""
    dataset = ShapeNetDataset(
        root=opt.dataset,
        class_choice=None,
        split='test',
        npoints=opt.num_points)
    classes = ["Airplane", "Car", "Chair", "Lamp", "Motorbike", "Mug", "Table"]
    base_folder = opt.outf
    for class_choice in classes:
        setattr(opt, "train_class_choice", class_choice)
        setattr(opt, "test_class_choice", class_choice)
        output_folder = os.path.join(base_folder, class_choice)
        setattr(opt, "outf", output_folder)
        # setattr(opt, "final_training", 0)
        # Implement for transfer learning
        # settattr(opt, "model", "path of trained general model")
        try:
            os.makedirs(output_folder)
        except Exception as e:
            print(e)
        print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
        model, test_loss = train_example(opt)

        if test_loss[-1] > worst_test_loss:
            print(f"--- Worst validation loss found! {test_loss[-1]} (previous one: {worst_test_loss})")
            worst_class = class_choice
            worst_test_loss = test_loss[-1]

        # Save final model + final test loss
        # torch.save(model.state_dict(), os.path.join(opt.outf, "final_checkpoint.pt"))
        # with open(os.path.join(opt.outf, "test_loss.csv"), 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(test_loss)
        #
        # print(opt)
        # model, val_loss = train_example(opt)
        # for class_choice_pc in classes:
        #     print_original_decoded_point_clouds(dataset, class_choice_pc, model, opt)
    print(f"Worst class: {worst_class},\t last loss: {worst_test_loss} (20th epoch)")


if __name__ == '__main__':
    # TODO - create a json file for setting all the arguments. Actually:
    # TODO - create a json for the FINAL arguments (after the cross-validation, e.g.: {'batchSize': 32})
    # TODO - and a json for the GRID SEARCH phase (e.g.: {'batchSize': [16, 32, 64], ...}
    parser = argparse.ArgumentParser()
    # parser.add_argument("--set_size", type=float, default=1, help="Subset size (between 0 and 1) of the training set. "
    #                                                               "Use it for fake test")
    # parser.add_argument("--size_encoder", type=int, default=1024, help="Size latent code")
    # parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    # parser.add_argument('--num_points', type=int, default=1024, help='Number points from point cloud')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    # parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    # parser.add_argument('--outf', type=str, default='cls', help='output folder')
    # parser.add_argument('--model', type=str, default='', help='model path')
    # parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    # parser.add_argument('--train_class_choice', type=str, default=None, help="Training class")
    # parser.add_argument('--test_class_choice', type=str, default=None, help="Test class")
    # # parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    # parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    # parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="reduction factor of the learning rate")
    # parser.add_argument("--scheduler_stepSize", type=int, default=20)
    # parser.add_argument("--dropout", type=int, default=1, help="dropout percentage")
    # parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    # parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
    # parser.add_argument("--beta_1", type=float, default=0.9, help="decay rate for first moment")
    # parser.add_argument("--beta_2", type=float, default=0.999, help="decay rate for second moment")
    # parser.add_argument("--patience", type=int, default=7, help="How long to wait after last time val loss improved.")
    # opt = parser.parse_args()
    # TODO - remove the following instruction (it overrides all the previous args)
    opt = upload_args_from_json()
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_example(opt)
    # train_model_by_class(opt)

# TODO - Implement training phase (you should also implement cross-validation for tuning the hyperparameters)
# TODO - You should also implement the visualization tools (visualization_tools package)
