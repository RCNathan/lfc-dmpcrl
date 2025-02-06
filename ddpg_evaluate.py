import os
import numpy as np
import pickle 
import zipfile

from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit, TransformReward
from mpcrl.wrappers.envs import MonitorEpisodes
from operator import neg

# modules for DDPG training using stable baselines3
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

from lfc_env import LtiSystem  # environment providing the 'true' dynamics in lfc
from lfc_model import Model  # linear model used for learning in dmpcrl
from vis_large_eps import vis_large_eps


"""File for processing the data obtained by training DDPG models on the LFC problem.
The data is saved in the form of .pkl files and .zip files. The .pkl files contain the data from the MonitorEpisodes
class, which is used to monitor the episodes during training and evaluation. The .zip files contain the model parameters
The model can be run on a test-dataset to evaluate and compare performance with dmpcrl and sc-mpc"""

# A total of four files will be saved when running ddpg_agent.py or the ddpg_cmd.py: [tweak the names pls]
# [...]env_eval[...].pkl        contains the MonitorEpisodes data from eval
# [...]env_train[...].pkl       contains the MonitorEpisodes data from train
# [...].zip                     contains the model params (weights)
# [...]_env.pkl                 contains the vectorized env used for model.get_env() and model.predict etc.

# update 23-1: save-names change for consistency to:
# lfc_[...]_eval.pkl        contains the MonitorEpisodes data from eval
# lfc_[...]_train.pkl       contains the MonitorEpisodes data from train
# lfc_[...]_model.zip       contains the model params (weights)
# lfc_[...]_env.pkl         contains the vectorized env used for model.get_env() and model.predict etc.
# additionally; best models are being saved in a folder:
# \best_model\lfc_[...]\best_model.zip

# what is in the ddpg's zips or pkl-files?
# ddpg_env_evalddpg4.pkl
# ddpg_lfc_ddpg4.zip

with open('ddpg/ddpg_env_evalddpg4.pkl', 'rb') as f:
    data = pickle.load(f)
    # data: .keys() = X, U, R, Pl, Pl_noise
    # X.shape = (1000, 1001, 12, 1)

# From the sb3 docs:
# ''parameters'' refer to neural network parameters (also called “weights”). This is a dictionary mapping variable name to a PyTorch tensor.
# ''data'' refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space. These depend on the algorithm used. 
# This is a dictionary mapping classes variable names to their values.
# zip-archived JSON dump

model = DDPG.load("ddpg/ddpg_lfc_ddpg4")
venv = model.get_env()  # venv.envs[0].env.env.env is LtiSystem
obs = venv.reset()

    

print("test")