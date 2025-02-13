#!/usr/anaconda3/envs/lfc-dmpcrl/bin/python
from datetime import *
import argparse
import json

from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl import UpdateStrategy

from lfc_train import train

# To run code from command-line using json-file, use the following command: python lfc_train_cmd.py --config_file training_config.json
if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description="Train LFC agents")

    # Add arguments
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file")
    # note: output-dir not necessary, as it is hardcoded in the function

    # parse arguments
    args = parser.parse_args()

    # load config file
    with open(args.config_file, "r") as file:
        param_sets = json.load(file)

        for param_set in param_sets:
            # Convert the "learning_rate" JSON object into an ExponentialScheduler
            if "learning_rate" in param_set:
                lr = param_set["learning_rate"]
                if lr["type"] == "ExponentialScheduler":
                    param_set["learning_rate"] = ExponentialScheduler(**lr["args"])

            # Convert the "epsilon" JSON object into an ExponentialScheduler
            if "epsilon" in param_set:
                eps = param_set["epsilon"]
                if eps["type"] == "ExponentialScheduler":
                    param_set["epsilon"] = ExponentialScheduler(**eps["args"])

            # Convert the "experience" JSON object into an ExperienceReplay
            if "experience" in param_set:
                exp = param_set["experience"]
                if exp["type"] == "ExperienceReplay":
                    param_set["experience"] = ExperienceReplay(**exp["args"])

            if "update_strategy" in param_set:
                us = param_set["update_strategy"]
                if us["type"] == "UpdateStrategy":
                    param_set["update_strategy"] = UpdateStrategy(**us["args"])

            param_set["cmd_flag"] = True # to toggle different plotting styles
            train(**param_set)
            print("Training complete")
        print("All training complete")
