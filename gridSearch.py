from sklearn.model_selection import ParameterGrid
import subprocess
import json


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

fake_test()

# json_params = json.loads(open("gridParameters.json").read())
# for setup in ParameterGrid(json_params):
#     param_sets = []
#     for i in setup:
#         command = "--"+ i
#         value = str(setup[f"{i}"])
#         param_sets.append(command)
#         param_sets.append(value)
#     subprocess.run(["python", "train_ae.py" ]+param_sets)
#




