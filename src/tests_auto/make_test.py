
import sys
sys.path.append("./")
import os
import json
import _pickle as cPickle
import numpy as np
import errno

from .automated_test import Brownian_V_separation

from ..models.build_model import build_model


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def load_parameters(filename):
    with open(filename, "r") as f:
        traj = json.load(f)
    return traj


if __name__ == "__main__":
    param_file = sys.argv[1]
    parameters = load_parameters(param_file)
    # print(sys.argv)
    if "sumatra_label" in parameters:
        parameters["data_folder"] = os.path.join(parameters["data_folder"],
                                                 parameters["sumatra_label"])
        parameters.pop("sumatra_label")
    else:
        print("no extra label")

    with open(os.path.join(parameters["data_folder"], "params.json"), "w") as f:
        s = json.dumps(parameters)
        f.write(s)

    parameters["data_folder"] = os.path.join(parameters["data_folder"], "")
    parameters["filename"] = param_file

    os.makedirs(parameters["data_folder"], exists=True)

    model = build_model(n_states=parameters["n_states"], n_cat=parameters["n_cat"],
                        n_layers=parameters["n_layers"],
                        inputsize=parameters["n_dim"],
                        hidden=parameters["hidden"], simple=parameters["simple"],
                        segmentation=parameters["segmentation"])

    model.load_weights(parameters["weights"])

    if parameters["type_test"] == "BV":
        r = Brownian_V_separation(model,
                                  ndim=parameters["inputsize"] - 3, batch_size=10,
                                  noise_level=parameters["noise_level"])

    with open(os.path.join(parameters["data_folder"], parameters["type_test"] + ".pick"), "w"):
        cPickle.write(r)
