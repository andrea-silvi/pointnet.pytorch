import json
import os
from random import uniform
import argparse
from train_ae import train_example, train_model_by_class
from utils.dataset import ShapeNetDataset
from visualization_tools import printPointCloud as ptPC
import torch
import sys
from sklearn.model_selection import ParameterGrid


def optimize_params(filepath=os.path.join("parameters", "params.json")):
    """
    :param filepath: string: json file path (contains ALL the hyperparameters, also those fixed: see
        lr_params.json for reference).
        N.B.: it should not contain the hyperparameters passed through default_params
    :return:
    """
    json_params = json.loads(open(filepath).read())
    parser = argparse.ArgumentParser(description=f'Model validation')
    args = parser.parse_args()
    best_val_loss = sys.float_info.max
    dict_params = {}
    best_hyperparams = {}  # contains the best hyperparameters (only those randomly generated) {hyperparam: value, ...}
    current_hyperparams = {}  # contains the best hyperparameters (only those randomly generated)
    param_grid = {}
    for hyperparam, value in json_params.items():
        # check if 'value' is a list
        if isinstance(value, list):
            param_grid[hyperparam] = value
        else:
            if value == 'None':
                value = None
            setattr(args, hyperparam, value)

    test_dataset = ShapeNetDataset(
        root=args.dataset,
        split='test',
        class_choice="Airplane",
        npoints=1024)

    for current_param_grid in ParameterGrid(param_grid):
        for hyperparam, value in current_param_grid.items():
            setattr(args, hyperparam, value)
            current_hyperparams[hyperparam] = value
        print(f"\n\n------------------------------------------------------------------\nParameters: {args}\n")
        # val_losses is the list of losses obtained during validation
        model, val_losses = train_example(args)
        if val_losses[-1] < best_val_loss:
            print(f"--- Best validation loss found! {val_losses[-1]} (previous one: {best_val_loss}), corresponding to "
                  f"hyperparameters {current_hyperparams.items()}")
            best_val_loss = val_losses[-1]
            best_hyperparams = current_hyperparams
        ptPC.print_original_decoded_point_clouds(test_dataset, "Airplane", model, args)
        dict_params[hash(str(args))] = str(args)
    folder = args.outf
    try:
        os.makedirs(folder)
    except OSError:
        pass
    with open(os.path.join(folder, f'hash_params.json'), 'w') as f:
        json.dump(dict_params, f)
    return best_hyperparams


def wrapper_train_by_class(json_path=os.path.join("parameters", "fixed_params.json")):
    json_params = json.loads(open(json_path).read())
    parser = argparse.ArgumentParser(description=f'Training models with different classes')
    args = parser.parse_args()
    for hyperparam, value in json_params.items():
        if value == 'None':
            value = None
        setattr(args, hyperparam, value)
    train_model_by_class(args)


if __name__ == '__main__':
    # best_params = optimize_params()
    # print(f"Best parameters: \t{best_params}\n")
    wrapper_train_by_class()

