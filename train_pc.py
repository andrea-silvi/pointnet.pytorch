from __future__ import print_function
import numpy as np
import torch
from point_completion.naive_model import PointNet_NaiveCompletionNetwork
from utils.loss import PointLoss
import argparse
import os
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from utils.dataset_seg import ShapeNetPart
import gc
import csv
from utils.early_stopping import EarlyStopping
from utils.FPS import farthest_point_sample, index_points
from utils.utils import  farthest_points
import sys, json
from utils.utils import upload_args_from_json
from point_completion.mutlitask_model import PFNet_MultiTaskCompletionNet
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


def test_example(opt, test_dataloader, model, n_crop_points=512):
    # initialize lists to monitor test loss and accuracy
    chamfer_loss = PointLoss()
    test_loss = 0.0
    seg_test_loss = 0.0
    accuracy_test_loss = 0.0
    model.eval()  # prep model for evaluation

    for data in test_dataloader:
        if opt.segmentation:
            points, target = data
            points, target = points.cuda(), target.cuda()
        else:
            points = data.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        incomplete_input_test = cropping(points)
        incomplete_input_test = incomplete_input_test.cuda()
        if opt.segmentation:
            output_clouds, pred = model(incomplete_input_test)
            output, pred = output_clouds[2].cuda(), pred.cuda()
            pred = pred.view(-1, 50)
            target = target.view(-1, 1)[:, 0] - 1
            seg_loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            seg_test_loss += seg_loss * points.size(0)
            accuracy_test_loss += (correct.item() / float(points.size(0) * (opt.num_points - n_crop_points))) * points.size(0)
        else:
            output = model(incomplete_input_test)
        # calculate the loss
        loss = chamfer_loss(points, output)
        # update test loss
        test_loss += loss.item() * points.size(0)
    # calculate and print avg test loss
    test_loss = test_loss / len(test_dataloader.dataset)
    if opt.segmentation:
        accuracy_test_loss = accuracy_test_loss /  len(test_dataloader.dataset)
        seg_test_loss = seg_test_loss / len(test_dataloader.dataset)
        print(f"Test Accuracy: {accuracy_test_loss}\t Test neg log likelihood: {seg_test_loss}")
    print('Test Loss: {:.6f}\n'.format(test_loss))

    return test_loss, seg_test_loss, accuracy_test_loss if opt.segmentation else test_loss


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
        losss = test_example(opt, test_dataloader, autoencoder)
        if opt.segmentation:
            run[f"loss/chamfer_{classs}"] = losss[0]
            run[f"loss/nll_seg_{classs}"] = losss[1]
            run[f"loss/accuracy_seg_{classs}"] = losss[2]
        else:
            run[f"loss/chamfer_{classs}"] = losss

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
    num_classes = 50
    n_crop_points = 512
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
    pc_architecture = PFNet_MultiTaskCompletionNet(crop_point_num=n_crop_points) \
        if opt.segmentation else PointNet_NaiveCompletionNetwork(num_points=opt.num_points, size_encoder=opt.size_encoder)

    optimizer = optim.Adam(pc_architecture.parameters(), lr=opt.lr, betas=(opt.beta_1, opt.beta_2), eps=1e-5,
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
    num_batch = len(training_dataset) / opt.batchSize
    for epoch in range(n_epoch):
        # TODO - change weight segmentation loss
        if epoch > 0:
            scheduler.step()
        if epoch < 30:
            weight_sl = 0.1
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            weight_sl = 0.4
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            weight_sl = 0.8
            alpha1 = 0.1
            alpha2 = 0.2
        training_losses = []
        segmentation_losses = []
        for i, data in enumerate(train_dataloader, 0):
            if opt.segmentation:
                points, target = data
                points, target = points.cuda(), target.cuda()
            else:
                points = data
                points = points.cuda()
            optimizer.zero_grad()
            pc_architecture.train()
            incomplete_input = cropping(points)
            incomplete_input = incomplete_input.cuda()
            if opt.segmentation:
                decoded_points, pred = pc_architecture(incomplete_input)
                pred = pred.cuda()
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                seg_loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, seg_loss.item(), correct.item() / float(opt.batchSize * (opt.num_points - n_crop_points))))
                decoded_coarse = decoded_points[0].cuda()
                decoded_fine = decoded_points[1].cuda()
                decoded_input = decoded_points[2].cuda()

                coarse_sampling_idx = farthest_point_sample(points, 64, RAN=False)
                coarse_sampling = index_points(points, coarse_sampling_idx)
                coarse_sampling = coarse_sampling.cuda()
                fine_sampling_idx = farthest_point_sample(points, 128, RAN=True)
                fine_sampling = index_points(points, fine_sampling_idx)
                fine_sampling = fine_sampling.cuda()

                CD_loss = chamfer_loss(points, decoded_input)
                loss = chamfer_loss(points, decoded_input) \
                       + alpha1 * chamfer_loss(coarse_sampling, decoded_coarse) \
                       + alpha2 * chamfer_loss(fine_sampling, decoded_fine) \
                       + weight_sl*seg_loss
                run["train/batch_seg_loss"].log(seg_loss)
                segmentation_losses.append(seg_loss)
            else:
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
        if opt.segmentation:
            seg_train_mean = np.average(segmentation_losses)
            run["train/epoch_seg_loss"].log(seg_train_mean)
        # VALIDATION PHASE
        if not final_training:
            with torch.no_grad():
                val_losses = []
                val_seg_losses = []
                for j, data in enumerate(val_dataloader, 0):
                    if opt.segmentation:
                        val_points, target = data
                        val_points, target = val_points.cuda(), target.cuda()
                    else:
                        val_points = data
                        val_points = val_points.cuda()
                    pc_architecture.eval()
                    incomplete_input_val = cropping(val_points)
                    incomplete_input_val = incomplete_input_val.cuda()
                    if opt.segmentation:
                        decoded_point_clouds, seg_predictions = pc_architecture(incomplete_input_val)
                        pred = pred.cuda()
                        pred = pred.view(-1, num_classes)
                        target = target.view(-1, 1)[:, 0] - 1
                        val_seg_loss = F.nll_loss(pred, target)
                        pred_choice = pred.data.max(1)[1]
                        correct = pred_choice.eq(target.data).cpu().sum()
                        print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                            epoch, j, num_batch, val_seg_loss.item(), correct.item() / float(opt.batchSize * (opt.num_points - n_crop_points))))
                        decoded_val_points = decoded_point_clouds[2].cuda()
                        val_seg_losses.append(val_seg_loss)
                        run["validation/batch_seg_loss"].log(val_seg_loss)
                        val_seg_losses.append(val_seg_loss)
                    else:
                        decoded_val_points = pc_architecture(incomplete_input_val)
                        decoded_val_points = decoded_val_points.cuda()
                    val_loss = chamfer_loss(val_points, decoded_val_points)
                    val_losses.append(val_loss.item())
                    run["validation/batch_loss"].log(val_loss.item())
                val_mean = np.average(val_losses)
                run["validation/epoch_loss"].log(val_mean)
                if opt.segmentation:
                    val_seg_mean = np.average(val_seg_losses)
                    run["validation/epoch_seg_loss"].log(val_seg_mean)
                    print(f'Segmentation loss:\tepoch: {epoch}, training loss: {seg_train_mean}, validation loss: {seg_train_mean}')
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
        # test_loss = test_example(opt, test_dataloader, pc_architecture)
        # run["test/loss"].log(test_loss)
        run.stop()
        return pc_architecture, 0


if __name__ == '__main__':
    opt = upload_args_from_json()
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_pc(opt)
