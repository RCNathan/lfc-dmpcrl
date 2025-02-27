import os
import numpy as np
import pickle 
import time

# modules for DDPG training using stable baselines3
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import VecNormalize
from ddpg_agent import make_env # func to create env (has venv wrapper though!)


"""File for processing the data obtained by training DDPG models on the LFC problem.
The data is saved in the form of .pkl files and .zip files. The .pkl files contain the data from the MonitorEpisodes
class, which is used to monitor the episodes during training and evaluation. The .zip files contain the model parameters
The model can be run on a test-dataset to evaluate and compare performance with dmpcrl and sc-mpc"""

#############################################################################################
# update 23-1: save-names changed for consistency to (all in ddpg\):
# lfc_[...]_eval.pkl        contains the MonitorEpisodes data from eval, can be plotted with vis_large_eps.py
# lfc_[...]_train.pkl       contains the MonitorEpisodes data from train, can be plotted with vis_large_eps.py
# lfc_[...]_model.zip       contains the model params (weights)
# lfc_[...]_env.pkl         contains the vectorized env used for model.get_env() and model.predict etc.
# additionally; best models are being saved in a folder:
# \best_model\lfc_[...]\best_model.zip

# From the sb3 docs:
# ''parameters'' refer to neural network parameters (also called “weights”). This is a dictionary mapping variable name to a PyTorch tensor.
# ''data'' refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space. These depend on the algorithm used. 
# This is a dictionary mapping classes variable names to their values.
# zip-archived JSON dump
#############################################################################################

def DDPG_evaluate(
    model_path: str,
    vec_norm_path: str,
    numEpisodes: int,
    numSteps: int = 1000,
    save_name_info: str = None, # scenario, which model is used etc
    bestModelFlag: bool = False, # if model comes from \ddpg\best_model\
    scenario: int = None, # which scenario is being evaluated
):
    """
    Evaluate a trained DDPG-agent on the LFC environment. 

    ----------------
    Inputs:
    model_path: str
        Path to the saved model weights.
    vec_norm_path: str
        Path to the saved VecNormalize statistics.

    Outputs:
    data stored in a .pkl-file containing X, U, R, Pl, Pl_noises
    """

    if scenario not in {0, 1, 2}:
        raise ValueError("Please provide a scenario from {0, 1, 2}")
    
    # create the environment using the same AugmentedObservationWrapper as used during training
    # normalization statistics need to be applied to rewrap the env
    env = make_env(isEval=True, scenario=scenario).unwrapped # remove last wrapper (the VecNormalize)
    if bestModelFlag:
         venv = VecNormalize(env, training=False)
    else:
        venv = VecNormalize.load(vec_norm_path, env) # vecnormalize statistics from save
        venv.training = False # turn off training

    # load model with weights
    model = DDPG.load(model_path, env=venv)

    # test if model and env work
    obs = venv.reset() # gets the initial observation (with x0)
    # note: env.reset() yields x0 as expected; [0.1, ..., 0, 0.1, ...] (54,)
    # but the venv.reset() already is normalized so these are very different w/ shape still (54,)!
    # to retrieve the correct/actual states later, remember, we have to:
    #   env = venv.envs[0].env.env.env 
    #   X = np.asarray(env.ep_observations) (before end of episode) or env.observations (after episode)
    # Then you'll see that X[0] actually yields the [0.1, ...] (12, 1) !

    start_time = time.time()
    for _ in range(numEpisodes * numSteps):
        action, _ = model.predict(obs)
        obs, _, _, _ = venv.step(action)
    end_time = time.time()
    print("(DDPG eval) Time elapsed:", end_time - start_time)

    # retrieve from the venv by pealing away wrappers;
    env = venv.envs[0].env.env.env
    X = np.asarray(env.observations)
    U = np.asarray(env.actions)
    R = np.asarray(env.rewards)
    Pl = np.asarray(env.unwrapped.loads)
    Pl_noise = np.asarray(env.unwrapped.load_noises)

    # check shape of x, and if load etc and resetting works as expected
    print("Shape of x: ", X.shape)
    print(f"check env-reset: \n {X[0, 0, :, :]}, \n {X[1, 0, :, :]}")
    print("Shape of load: ", Pl.shape)
    print(f"check load-reset: \n {Pl[0, :, :]}, \n {Pl[1000, :, :]}")

    # make sure dir exists, save plot and close after
    saveloc = r'evaluate_data'
    os.makedirs(saveloc, exist_ok=True)
    savename = f"ddpg_{numEpisodes}eps_{save_name_info}_scenario{scenario}"
    file_path = os.path.join(saveloc, savename)
    
    with open(
            file_path + '.pkl',
            "wb",  # w: write mode, creates new or truncates existing. b: binary mode
        ) as file:
            pickle.dump(
                {
                    "X": X,
                    "U": U,
                    "R": R,
                    "Pl": Pl,
                    "Pl_noise": Pl_noise,
                    'elapsed_time': end_time - start_time,
                    "scenario": scenario,
                },
                file,
            )

    print("Evaluation succesful, file saved as", file_path)

# data from eval env during training
# with open('ddpg/lfc_ddpg4_eval.pkl', 'rb') as f:
#     data = pickle.load(f)
    # data: .keys() = X, U, R, Pl, Pl_noise
    # X.shape = (1000, 1001, 12, 1)

# evaluate the DDPG trained agent (i.e create new data) for scenario 1 & 2 (same)
DDPG_evaluate(
    model_path=r"ddpg\lfc_ddpg5_model", # r"ddpg\best_model\lfc_ddpg4\best_model"
    vec_norm_path='ddpg\lfc_ddpg5_env.pkl', # 'ddpg\lfc_ddpg4_env.pkl'
    numEpisodes=20,
    numSteps=1000,
    save_name_info="ddpg5_scenario1and2_newenv", # stuff like ddpg4 etc 
    bestModelFlag=False,
    scenario=0,
) # scenario 1 & 2 both have noise; scenario 0 does not, so needs separate evaluation.


# TODO: # by changing the env!!
# evaluate the DDPG trained agent (i.e create new data) for scenario 0 (no noise)
# DDPG_evaluate(
#     model_path="ddpg/lfc_ddpg4_model",
#     vec_norm_path='ddpg/lfc_ddpg4_env.pkl',
#     numEpisodes=20,
#     numSteps=1000,
#     save_name_info="ddpg4_scenario0", # stuff like scenario, ddpg4 etc 
# ) # scenario 1 & 2 both have noise; scenario 0 does not, so needs separate evaluation.