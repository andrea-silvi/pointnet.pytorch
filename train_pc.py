from __future__ import print_function
import numpy as np
import torch
from point_completion.naive_model import PointNet_NaiveCompletionNetwork
from utils.loss import PointLoss
import argparse
import os
import torch.optim as optim
import torch.utils.data
from utils.dataset_seg import ShapeNetPart
import gc
import csv
from utils.early_stopping import EarlyStopping
from utils.FPS import farthest_point_sample, index_points
from utils.utils import  farthest_points
import sys, json
from utils.utils import upload_args_from_json

from visualization_tools import printPointCloud
from visualization_tools.printPointCloud import *
import neptune.new as neptune


def cropping(batch_point_cloud, num_cropped_points=512):
    # batch_point_cloud: (batch_size, num_points, 3)
    batch_size = batch_point_cloud.size(0)
    num_points = batch_point_cloud.size(1)
    idx = torch.randint(0, num_points, (batch_size,), device="cuda")
    idx_base = torch.arange(0, batch_size, device="cuda").view(-1) * num_points
    idx = (idx + idx_base).view(-1)
    batch_points = batch_point_cloud.view(-1, 3)[idx, :].view(-1, 1, 3)
    incomplete_input = farthest_points(batch_point_cloud, batch_points, num_points-num_cropped_points)
    return incomplete_input


def test_example(opt, test_dataloader, model):
    # initialize lists to monitor test loss and accuracy
    chamfer_loss = PointLoss()
    test_loss = 0.0

    model.eval()  # prep model for evaluation

    for data in test_dataloader:
        # forward pass: compute predicted outputs by passing inputs to the model
        data = data.cuda()
        incomplete_input_test = cropping(data)
        incomplete_input_test = incomplete_input_test.cuda()
        output = model(incomplete_input_test)
        output = output.cuda()
        # calculate the loss
        loss = chamfer_loss(data, output)
        # update test loss
        test_loss += loss.item() * data.size(0)

    # calculate and print avg test loss
    test_loss = test_loss / len(test_dataloader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    return test_loss


def evaluate_loss_by_class(opt, autoencoder, run):
    run["params"] = vars(opt)
    classes = ["Airplane", "Car", "Chair", "Lamp", "Mug", "Motorbike", "Table"] if opt.test_class_choice is None\
        else [opt.test_class_choice]
    autoencoder.cuda()
    print("Start evaluation loss by class")
    for classs in classes:
        print(f"\t{classs}")
        test_dataset = ShapeNetPart(opt.dataset,
                                    class_choice=classs,
                                    split='test',
                                    segmentation=opt.segmentation)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))
        run[f"loss/{classs}"] = test_example(opt, test_dataloader, autoencoder)
        print()
    # if opt.test_class_choice is None:
    #     evaluate_novel_categories(opt, autoencoder, run)


def train_pc(opt):
    neptune_info = json.loads(open(os.path.join("parameters", "neptune_params.json")).read())
    run = neptune.init(project=neptune_info['project'],
                       tags=[str(opt.train_class_choice), str(opt.size_encoder), "Naive point completion"],
                       api_token=neptune_info['api_token'])
    run['params'] = vars(opt)
    random_seed = 43
    torch.manual_seed(random_seed)

    training_dataset = ShapeNetPart(
        root=opt.dataset,
        class_choice=opt.train_class_choice,
        segmentation=opt.segmentation
    )

    validation_dataset = ShapeNetPart(
        root=opt.dataset,
        class_choice=opt.train_class_choice,
        segmentation=opt.segmentation,
        split="val"
    )

    final_training = opt.final_training
    if final_training:
        if opt.runNumber == 0:
            print("!!!!!!Final training starts!!!!!!")
        test_dataset = ShapeNetPart(
            root=opt.dataset,
            class_choice=opt.train_class_choice,
            segmentation=opt.segmentation,
            split="test"
        )
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

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    pc_architecture = PFNet_MultiTaskCompletionNet() \
        if opt.segmentation else PointNet_NaiveCompletionNetwork(num_points=opt.num_points, size_encoder=opt.size_encoder)

    optimizer = optim.Adam(pc_architecture.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2),
                           weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_stepSize, gamma=opt.scheduler_gamma)
    pc_architecture.cuda()
    run["model"] = pc_architecture
    checkpoint_path = os.path.join(opt.outf, f"checkpoint{opt.runNumber}.pt")
    training_history = []
    val_history = []
    gc.collect()
    torch.cuda.empty_cache()
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=checkpoint_path)
    #  instantiate the loss
    chamfer_loss = PointLoss()
    n_epoch = opt.nepoch
    for epoch in range(n_epoch):
        if epoch > 0:
            scheduler.step()
        training_losses = []
        for i, points in enumerate(train_dataloader, 0):
            points = points.cuda()
            optimizer.zero_grad()
            pc_architecture.train()
            incomplete_input = cropping(points)
            incomplete_input = incomplete_input.cuda()
            decoded_points = pc_architecture(incomplete_input)
            decoded_points = decoded_points.cuda()
            CD_loss = loss = chamfer_loss(points, decoded_points)
            training_losses.append(CD_loss.item())
            run["train/batch_loss"].log(CD_loss.item())
            loss.backward()
            optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        train_mean = np.average(training_losses)
        run["train/epoch_loss"].log(train_mean)
        # VALIDATION PHASE
        if not final_training:
            with torch.no_grad():
                val_losses = []
                for j, val_points in enumerate(val_dataloader, 0):
                    pc_architecture.eval()
                    val_points = val_points.cuda()
                    incomplete_input_val = cropping(val_points)
                    incomplete_input_val = incomplete_input_val.cuda()
                    decoded_val_points = pc_architecture(incomplete_input_val)
                    decoded_val_points = decoded_val_points.cuda()
                    val_loss = chamfer_loss(val_points, decoded_val_points)
                    val_losses.append(val_loss.item())
                    run["validation/batch_loss"].log(val_loss.item())
                val_mean = np.average(val_losses)
                run["validation/epoch_loss"].log(val_mean)
                print(f'\tepoch: {epoch}, training loss: {train_mean}, validation loss: {val_mean}')
        else:
            print(f'\tepoch: {epoch}, training loss: {train_mean}')
        if epoch >= 50:
            early_stopping(val_mean if not final_training else train_mean, pc_architecture)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        training_history.append(train_mean)
        if not final_training:
            val_history.append(val_mean)
    if opt.nepoch <= 50:
        torch.save(pc_architecture.state_dict(), checkpoint_path)
    pc_architecture.load_state_dict(torch.load(checkpoint_path))
    printPointCloud.print_original_incomplete_decoded_point_clouds(opt.test_class_choice, pc_architecture, opt, run)
    if not final_training:
        run.stop()
        return pc_architecture, val_history
    else:
        run["model_dictionary"].upload(checkpoint_path)
        evaluate_loss_by_class(opt, pc_architecture, run)
        test_loss = test_example(opt, test_dataloader, pc_architecture)
        run["test/loss"].log(test_loss)
        run.stop()
        return pc_architecture, 0


if __name__ == '__main__':
    opt = upload_args_from_json()
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_pc(opt)
