from ddpg_agent import train_ddpg
import argparse
import json

# ddpg from command-line
if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(description="Train LFC agents for DDPG")

    # Add arguments
    parser.add_argument("--config_file", type=str, required=True, help="Path to config file")
    # note: output-dir not necessary, as it is hardcoded in the function

    # parse arguments
    args = parser.parse_args()

    # load config file
    with open(args.config_file, "r") as file:
        param_sets = json.load(file)

        for param_set in param_sets:
            # Per set of params, call the DDPG training
            train_ddpg(**param_set)
            print("Training complete")
        print("All training complete")