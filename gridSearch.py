import json
import os
from random import uniform
import argparse
from train_ae import train_example
from utils.dataset import ShapeNetDataset
from visualization_tools import printPointCloud as ptPC
import torch
import sys


# def fake_test(set_size=0.2):
#     json_params = json.loads(open("parameters/gridParameters.json").read())
#     for setup in ParameterGrid(json_params):
#         param_sets = []
#         for i in setup:
#             if i in ['nepoch', 'train_class_choice', 'test_class_choice']:
#                 continue
#             command = "--" + i
#             value = str(setup[f"{i}"])
#             param_sets.append(command)
#             param_sets.append(value)
#         param_sets.append("--set_size")
#         param_sets.append(str(set_size))
#         param_sets.append("--nepoch")
#         param_sets.append(str(10))
#         subprocess.run(["python", "train_ae.py"] + param_sets)
#

def optimize_lr():
    json_params = json.loads(open(os.path.join("parameters", "lr_params.json")).read())
    lr_boundaries = json_params.pop("lr")
    upper_lr = lr_boundaries[1]
    lower_lr = lr_boundaries[0]
    parser = argparse.ArgumentParser(description='Preliminary grid search (for learning rate)')
    args = parser.parse_args()
    dict_params = {}
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    val_dataset = ShapeNetDataset(
        root=args.dataset,
        split='val',
        class_choice="Airplane",
        npoints=1024)
    n_point_clouds = val_dataset.__len__()
    image_index = int(uniform(0, n_point_clouds - 1))
    point_cloud = val_dataset.__getitem__(image_index)
    # try 20 different learning rate
    for count in range(10):
        setattr(args, "lr", uniform(lower_lr, upper_lr))
        print(args)
        model = train_example(args)

        model.eval()
        point_cloud_np = point_cloud.cuda()
        point_cloud_np = torch.unsqueeze(point_cloud_np, 0)
        decoded_point_cloud = model(point_cloud_np)

        point_cloud_np = point_cloud_np.cpu().numpy()
        dec_val_stamp = decoded_point_cloud.cpu().data.numpy()
        ptPC.printCloud(point_cloud_np, dec_val_stamp, "scifi", args)
        # ptPC.printCloud(point_cloud_np, "original_validation_points", args)
        # ptPC.printCloud(dec_val_stamp, "decoded_validation_points", args)
        dict_params[hash(str(args))] = str(args)
    folder = args.outf
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(os.path.join(folder, f'hash_params.json'), 'w') as f:
        json.dump(dict_params, f)


def optimize_params(filepath=os.path.join("parameters", "lr_params.json"), default_params=None):
    """
    :param filepath: string: json file path (contains ALL the hyperparameters, also those fixed: see
        lr_params.json for reference).
        N.B.: it should not contain the hyperparameters passed through default_params
    :param default_params: DICTIONARY: {hyperparam1: value, hyperparam2: value, ...}
        hyperparam1 and hyperparam2 should be not present inside the json file!!
        Use this variable in order to pass the learning rate found at the first phase of
        the random search.
        TIP: use the dictionary 'best_hyperparams' returned by this function
    :return:
    """
    json_params = json.loads(open(filepath).read())
    parser = argparse.ArgumentParser(description=f'Random search')
    hyperparam_boundaries = {}
    upper_boundary = {}
    lower_boundary = {}
    dict_params = {}
    best_val_loss = sys.float_info.max
    best_hyperparams = {}  # contains the best hyperparameters (only those randomly generated) {hyperparam: value, ...}
    current_hyperparams = {}  # contains the best hyperparameters (only those randomly generated)
    hyperparams = []
    for hyperparam, value in json_params.items():
        if isinstance(value, list):
            hyperparams.append(hyperparam)
            try:
                # the json file should contain HYPERPARAM as key and [low_boundary, high_boundary] as VALUE!!!!
                hyperparam_boundaries[hyperparam] = json_params[hyperparam]
                upper_boundary[hyperparam] = hyperparam_boundaries[hyperparam][1]
                lower_boundary[hyperparam] = hyperparam_boundaries[hyperparam][0]
            except Exception as e:
                print(e)
    [json_params.pop(hyperparam) for hyperparam in hyperparams]
    args = parser.parse_args()
    # Add the default parameters to the parameters downloaded from the json
    if default_params is not None:
        for def_param, def_value in default_params.items():
            json_params[def_param] = def_value
    for option, option_value in json_params.items():
        if option_value == 'None':
            option_value = None
        setattr(args, option, option_value)
    val_dataset = ShapeNetDataset(
        root=args.dataset,
        split='val',
        class_choice="Airplane",
        npoints=1024)
    n_point_clouds = val_dataset.__len__()
    image_index = int(uniform(0, n_point_clouds - 1))
    point_cloud = val_dataset.__getitem__(image_index)
    # try 3 random values for each hyperparameter
    for count in range(10):
        for hyperparam in hyperparams:
            value = 10 ** uniform(lower_boundary[hyperparam], upper_boundary[hyperparam])
            setattr(args, hyperparam, value)
            current_hyperparams[hyperparam] = value
        print(f"\n\n------------------------------------------------------------------\nParameters: {args}\n")
        # val_losses is the list of losses obtained during validation
        model, val_losses = train_example(args)
        if val_losses[-1] < best_val_loss:
            print(f"--- Best validation loss found! {val_losses[-1]} (previous one: {best_val_loss}), corresponding to"
                  f"hyperparameters {current_hyperparams.items()}")
            best_val_loss = val_losses[-1]
            best_hyperparams = current_hyperparams
        model.eval()
        point_cloud_np = point_cloud.cuda()
        point_cloud_np = torch.unsqueeze(point_cloud_np, 0)
        decoded_point_cloud = model(point_cloud_np)

        point_cloud_np = point_cloud_np.cpu().numpy()
        dec_val_stamp = decoded_point_cloud.cpu().data.numpy()
        ptPC.printCloudM(point_cloud_np, dec_val_stamp, "", opt=args)
        #ptPC.printCloud(point_cloud_np, "original_validation_points", args)
        #ptPC.printCloud(dec_val_stamp, "decoded_validation_points", args)
        dict_params[hash(str(args))] = str(args)
    folder = args.outf
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(os.path.join(folder, f'hash_params.json'), 'w') as f:
        json.dump(dict_params, f)
    return best_hyperparams




if __name__ == '__main__':
    best_lr = optimize_params()
    #print(f"\t\t\t-------BEST LEARNING RATE: {best_lr['lr']}\t\t\t")
    # print(f"BEST LEARNING RATE: {0.00020589232338423906}")
    # best_params = optimize_params(os.path.join("parameters", "others_params.json"), ["weight_decay"], best_lr)
    # print(f"-------BEST HYPERPARAMS: {best_params}")
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
