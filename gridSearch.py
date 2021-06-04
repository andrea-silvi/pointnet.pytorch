from sklearn.model_selection import ParameterGrid
import subprocess
import json
import os
from random import uniform
import argparse
from train_ae import train_example
from utils.dataset import ShapeNetDataset
from visualization_tools import printPointCloud as ptPC


def fake_test(set_size=0.2):
    json_params = json.loads(open("gridParameters.json").read())
    for setup in ParameterGrid(json_params):
        param_sets = []
        for i in setup:
            if i in ['nepoch', 'train_class_choice', 'test_class_choice']:
                continue
            command = "--" + i
            value = str(setup[f"{i}"])
            param_sets.append(command)
            param_sets.append(value)
        param_sets.append("--set_size")
        param_sets.append(str(set_size))
        param_sets.append("--nepoch")
        param_sets.append(str(10))
        subprocess.run(["python", "train_ae.py"] + param_sets)


def optimize_lr():
    json_params = json.loads(open(os.path.join("parameters", "lr_params.json")).read())
    lr_boundaries = json_params.pop("lr")
    upper_lr = lr_boundaries[1]
    lower_lr = lr_boundaries[0]
    parser = argparse.ArgumentParser(description='Preliminary grid search (for learning rate)')
    args = parser.parse_args()

    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    validation_dataset = ShapeNetDataset(
        root=args.dataset,
        split='val',
        class_choice="Airplane",
        npoints=1024)
    n_point_clouds = validation_dataset.__len__()
    image_index = int(uniform(0, n_point_clouds - 1))
    point_cloud = validation_dataset.__getitem__(image_index)
    # try 20 different learning rate
    for count in range(15):
        setattr(args, "lr", 10 ** uniform(lower_lr, upper_lr))
        model = train_example(args)

        model.eval()
        point_cloud = point_cloud.cuda()
        decoded_point_cloud = model(point_cloud)

        point_cloud = point_cloud[0, :, :].cpu().numpy()
        dec_val_stamp = decoded_point_cloud[0, :, :].cpu().numpy()
        ptPC.printCloud(point_cloud, "original_validation_points")
        ptPC.printCloud(dec_val_stamp, "decoded_validation_points")


if __name__ == '__main__':
    optimize_lr()
    # json_params = json.loads(open("gridParameters.json").read())
    # setup = json_params['fixed_params']
    # param_sets = []
    # for r_param in json_params['random_params']:
    #     (low, high) = json_params['random_params'][r_param]
    #     setup[r_param] = int(uniform(low, high)) if r_param == 'size_encoder' else \
    #         (uniform(low, high) if r_param == 'scheduler_gamma' else 10 ** uniform(low, high))
    # for opt in setup:
    #     command = "--"+ opt
    #     value = str(setup[f"{opt}"])
    #     param_sets.append(command)
    #     param_sets.append(value)
    # subprocess.run(["python", "train_ae.py"]+param_sets)
