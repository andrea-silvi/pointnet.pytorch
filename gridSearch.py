from sklearn.model_selection import ParameterGrid
import subprocess
import json

json_params = json.loads(open("gridParameters.json").read())
for setup in ParameterGrid(json_params):
    param_sets = []
    for i in setup:
        command = "--"+ i
        value = str(setup[f"{i}"])
        param_sets.append(command)
        param_sets.append(value)
    subprocess.run(["python", "train_ae.py" ]+param_sets)
