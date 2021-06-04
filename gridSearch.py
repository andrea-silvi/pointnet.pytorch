from sklearn.model_selection import ParameterGrid
import subprocess
import json
from random import uniform

if __name__=='__main__':
    json_params = json.loads(open("gridParameters.json").read())
    setup = json_params['fixed_params']
    param_sets = []
    for r_param in json_params['random_params']:
        (low, high) = json_params['random_params'][r_param]
        setup[r_param] = int(uniform(low, high)) if r_param == 'size_encoder' else \
            (uniform(low, high) if r_param == 'scheduler_gamma' else 10 ** uniform(low, high))
    for opt in setup:
        command = "--"+ opt
        value = str(setup[f"{opt}"])
        param_sets.append(command)
        param_sets.append(value)
    subprocess.run(["python", "train_ae.py"]+param_sets)


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
