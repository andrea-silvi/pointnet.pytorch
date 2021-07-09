from __future__ import print_function
import numpy as np
import torch
from point_completion.naive_model import PointNet_NaiveCompletionNetwork
from utils.loss import PointLoss, PointLoss_test
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
from utils.utils import  farthest_and_nearest_points
import sys, json
from utils.utils import upload_args_from_json
from point_completion.mutlitask_model import PFNet_MultiTaskCompletionNet
from visualization_tools import printPointCloud
from visualization_tools.printPointCloud import *
import neptune.new as neptune


def cropping(batch_point_cloud, batch_target=None, num_cropped_points=512):
    # batch_point_cloud: (batch_size, num_points, 3)
    batch_size = batch_point_cloud.size(0)
    num_points = batch_point_cloud.size(1)
    k = num_points-num_cropped_points
    idx = torch.randint(0, num_points, (batch_size,), device="cuda")
    idx_base = torch.arange(0, batch_size, device="cuda").view(-1) * num_points
    idx = (idx + idx_base).view(-1)
    batch_points = batch_point_cloud.view(-1, 3)[idx, :].view(-1, 1, 3)
    incomplete_input, idx, cropped_input = farthest_and_nearest_points(batch_point_cloud, batch_points, k)
    if batch_target is not None:
        batch_target = batch_target.view(-1, 1)
        batch_target = batch_target[idx, :]
        return incomplete_input,  batch_target.view(-1, k, 1), cropped_input
    return incomplete_input, cropped_input


def test_example(opt, test_dataloader, model, n_classes, n_crop_points=512):
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
            if opt.segmentation:
                points, target = data
                points, target = points.cuda(), target.cuda()
                incomplete_input_test, target, cropped_input_test = cropping(points, target)
                incomplete_input_test, target, cropped_input_test = incomplete_input_test.cuda(), target.cuda(), cropped_input_test.cuda()
            else:
                points = data.cuda()
                incomplete_input_test, cropped_input_test = cropping(points, None)
                incomplete_input_test, cropped_input_test = incomplete_input_test.cuda(), cropped_input_test.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model

            if opt.segmentation:
                output_clouds, pred = model(incomplete_input_test)
                output, pred = output_clouds[2].cuda(), pred.cuda()
                pred = pred.view(-1, n_classes)
                target = target.view(-1, 1)[:, 0]
                seg_loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1].cuda()
                correct = pred_choice.eq(target.data).sum()
                seg_test_loss += seg_loss * points.size(0)
                accuracy_test_loss += (correct.item() / float(points.size(0) * (opt.num_points - n_crop_points))) * points.size(0)
                loss_cropped_pc = chamfer_loss(cropped_input_test, output)
                test_loss_512 += np.array(loss_cropped_pc) * points.size(0)
                output = torch.cat((output, incomplete_input_test), dim=1)
            else:
                output = model(incomplete_input_test)
            loss_2048 = chamfer_loss(points, output)
            # update test loss
            test_loss_2048 += np.array(loss_2048) * points.size(0)

            # t = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # f = r - a  # free inside reserved
            # print(f"CUDA memory: total memory: {t}, reserved: {r}, allocated: {a}, free: {f}")
            # calculate the loss between the ORIGINAL CROPPED POINT CLOUD and the OUTPUT OF THE MODEL (the 512 points of the missing part)

        # calculate and print avg test loss
        test_loss_2048 = test_loss_2048 / len(test_dataloader.dataset)
        if opt.segmentation:
            test_loss_512 = test_loss_512 / len(test_dataloader.dataset)
            accuracy_test_loss = accuracy_test_loss /  len(test_dataloader.dataset)
            seg_test_loss = seg_test_loss / len(test_dataloader.dataset)
            print(f"Test Accuracy: {accuracy_test_loss}\t Test neg log likelihood: {seg_test_loss}")
        print(f'Test Loss (overall pc: mean, gt->pred, pred->gt): {test_loss_2048}\n')
        if opt.segmentation:
            return test_loss_2048, seg_test_loss, accuracy_test_loss, test_loss_512
        else:
            return test_loss_2048


def evaluate_loss_by_class(opt, autoencoder, run, n_classes):
    run["params"] = vars(opt)
    classes = ["airplane", "car", "chair", "lamp", "mug", "motorbike", "table"] if opt.test_class_choice is None\
        else [opt.test_class_choice]
    novel_classes = []
    if opt.test_class_choice is None:
        novel_classes = ["bag", "cap", "earphone", "guitar", "knife", "laptop", "pistol", "rocket", "skateboard"]
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
        losss = test_example(opt, test_dataloader, autoencoder, n_classes)
        if opt.segmentation:
            run[f"loss/overall_pc/{classs}_cd_mean"] = losss[0][0]
            run[f"loss/overall_pc/{classs}_cd_(gt->pred)"] = losss[0][1]
            run[f"loss/overall_pc/{classs}_cd_(pred->gt)"] = losss[0][2]
            run[f"loss/{classs}_nll_seg"] = losss[1]
            run[f"loss/{classs}_accuracy_seg"] = losss[2]
            run[f"loss/cropped_pc/{classs}_cd_mean"] = losss[3][0]
            run[f"loss/cropped_pc/{classs}_cd_(gt->pred)"] = losss[3][1]
            run[f"loss/cropped_pc/{classs}_cd_(pred->gt)"] = losss[3][2]
        else:
            run[f"loss/overall_pc/{classs}_cd_mean"] = losss[0]
            run[f"loss/overall_pc/{classs}_cd_(gt->pred)"] = losss[1]
            run[f"loss/overall_pc/{classs}_cd_(pred->gt)"] = losss[2]
        if classs in novel_classes:
            print_original_incomplete_decoded_point_clouds(classs, autoencoder, opt, run)


def train_pc(opt):
    neptune_info = json.loads(open(os.path.join("parameters", "neptune_params.json")).read())
    tag = "Multitask net" if opt.segmentation else "Naive net"
    run = neptune.init(project=neptune_info['project'],
                       tags=[str(opt.train_class_choice), tag],
                       api_token=neptune_info['api_token'])
    run['params'] = vars(opt)
    random_seed = 43
    num_classes = None
    n_crop_points = 512
    torch.manual_seed(random_seed)
    final_training = opt.final_training
    if final_training:
        if opt.runNumber == 0:
            print("!!!!!!Final training starts!!!!!!")
        training_dataset = ShapeNetPart(
            root=opt.dataset,
            class_choice=opt.train_class_choice,
            segmentation=opt.segmentation,
            split="trainval"
        )
    else:
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
        val_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    train_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    num_classes = training_dataset.seg_num_all if opt.segmentation else 0
    pc_architecture = PFNet_MultiTaskCompletionNet(num_classes=num_classes, crop_point_num=n_crop_points,\
                                                   pfnet_encoder=opt.pfnet_encoder)\
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
    weight_sl = 0.6
    for epoch in range(n_epoch):
        # TODO - change weight segmentation loss
        if epoch > 0:
            scheduler.step()
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
        segmentation_losses = []
        training_accuracies = []
        for i, data in enumerate(train_dataloader, 0):
            if opt.segmentation:
                points, target = data
                points, target = points.cuda(), target.cuda()
                incomplete_input, target, cropped_input = cropping(points, target)
                incomplete_input, target, cropped_input = incomplete_input.cuda(), target.cuda(), cropped_input.cuda()
            else:
                points = data
                points = points.cuda()
                incomplete_input, cropped_input = cropping(points, None)
                incomplete_input, cropped_input = incomplete_input.cuda(), cropped_input.cuda()
            optimizer.zero_grad()
            pc_architecture.train()
            if opt.segmentation:
                decoded_points, pred = pc_architecture(incomplete_input)
                pred = pred.cuda()
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0]
                seg_loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1].cuda()
                correct = pred_choice.eq(target.data).sum()
                accuracy = correct.item() / float(points.size(0) * (opt.num_points - n_crop_points))
                training_accuracies.append(accuracy)
                decoded_coarse = decoded_points[0].cuda()
                decoded_fine = decoded_points[1].cuda()
                decoded_input = decoded_points[2].cuda()

                coarse_sampling_idx = farthest_point_sample(cropped_input, 64, RAN=False)
                coarse_sampling = index_points(cropped_input, coarse_sampling_idx)
                coarse_sampling = coarse_sampling.cuda()
                fine_sampling_idx = farthest_point_sample(cropped_input, 128, RAN=True)
                fine_sampling = index_points(cropped_input, fine_sampling_idx)
                fine_sampling = fine_sampling.cuda()

                CD_loss = chamfer_loss(cropped_input, decoded_input)
                loss = CD_loss \
                       + alpha1 * chamfer_loss(coarse_sampling, decoded_coarse) \
                       + alpha2 * chamfer_loss(fine_sampling, decoded_fine) \
                       + weight_sl*seg_loss
                run["train/batch_seg_loss"].log(seg_loss)
                segmentation_losses.append(seg_loss.item())
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
            train_mean_accuracy = np.average(training_accuracies)
            run["train/epoch_seg_loss"].log(seg_train_mean)
            run["train/epoch_accuracy"].log(train_mean_accuracy)
        # VALIDATION PHASE
        if not final_training:
            with torch.no_grad():
                val_losses = []
                val_losses_cropped_pc = []
                val_seg_losses = []
                val_accuracies = []
                for j, data in enumerate(val_dataloader, 0):
                    if opt.segmentation:
                        val_points, target = data
                        val_points, target = val_points.cuda(), target.cuda()
                        incomplete_input_val, target, cropped_input_val = cropping(val_points, target)
                        incomplete_input_val, target, cropped_input_val = incomplete_input_val.cuda(), target.cuda(), cropped_input_val.cuda()
                    else:
                        val_points = data
                        val_points = val_points.cuda()
                        incomplete_input_val, cropped_input_val = cropping(val_points, None)
                        incomplete_input_val, cropped_input_val = incomplete_input_val.cuda(), cropped_input_val.cuda()
                    pc_architecture.eval()
                    if opt.segmentation:
                        decoded_point_clouds, pred = pc_architecture(incomplete_input_val)
                        pred = pred.cuda()
                        pred = pred.view(-1, num_classes)
                        target = target.view(-1, 1)[:, 0]
                        val_seg_loss = F.nll_loss(pred, target)
                        pred_choice = pred.data.max(1)[1].cuda()
                        correct = pred_choice.eq(target.data).sum()
                        accuracy = correct.item() / float(val_points.size(0) * (opt.num_points - n_crop_points))
                        val_accuracies.append(accuracy)
                        decoded_val_points = decoded_point_clouds[2].cuda()
                        val_seg_losses.append(val_seg_loss.item())
                        run["validation/batch_seg_loss"].log(val_seg_loss)
                        val_loss = chamfer_loss(cropped_input_val, decoded_val_points)
                        val_losses_cropped_pc.append(val_loss.item())
                        run["validation/batch_loss_cropped_pc"].log(val_loss.item())
                    else:
                        decoded_val_points = pc_architecture(incomplete_input_val)
                        decoded_val_points = decoded_val_points.cuda()
                        val_loss = chamfer_loss(val_points, decoded_val_points)
                    val_losses.append(val_loss.item())
                    run["validation/batch_loss"].log(val_loss.item())

                val_mean = np.average(val_losses)
                run["validation/epoch_loss"].log(val_mean)
                print(f"epoch: {epoch}")
                print(f'\tPOINT COMPLETION:\t training loss: {train_mean}, validation loss: {val_mean}')
                if opt.segmentation:
                    val_seg_mean = np.average(val_seg_losses)
                    run["validation/epoch_seg_loss"].log(val_seg_mean)
                    val_mean_accuracy = np.average(val_accuracies)
                    run["validation/epoch_accuracy"].log(val_mean_accuracy)
                    print(
                        f'\tSEGMENTATION:\t training accuracy/nnl: {train_mean_accuracy:.2f}/{seg_train_mean:.2f}, validation accuracy/nnl: {val_mean_accuracy:.2f}/{val_seg_mean:.2f}')

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
        evaluate_loss_by_class(opt, pc_architecture, run, num_classes)
        run.stop()
        return pc_architecture, 0


if __name__ == '__main__':
    opt = upload_args_from_json(os.path.join("parameters", "pc_fixed_params.json"))
    print(f"\n\n------------------------------------------------------------------\nParameters: {opt}\n")
    train_pc(opt)
