import os
import numpy as np
import pickle 

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

# from lfc_env import LtiSystem  # environment providing the 'true' dynamics in lfc
# from lfc_model import Model  # linear model used for learning in dmpcrl
from vis_large_eps import vis_large_eps

model = DDPG.load("ddpg/ddpg_lfc_repeatabilityCheck") # slide 58 in ppt - to run for 10k instead using config.
buffer_size = model.buffer_size
batch_size = model.batch_size
learning_rate = model.learning_rate
policy_kwargs = model.policy_kwargs
gamma = model.gamma
seed = model.seed

# print all variables: name, value of the ones defined above
print("buffer_size: ", buffer_size) # 1e6
print("batch_size: ", batch_size) # 256
print("learning_rate: ", learning_rate) # 1e-6
print("policy_kwargs: ", policy_kwargs) # 'net_arch': [256, 256], 'optimizer_kwargs': {'weight_decay': 1e-06}, 'n_critics': 1
print("gamma: ", gamma) # 0.999
print("seed: ", seed) # 1

vis_large_eps("ddpg/ddpg_env_trainrepeatabilityCheck") # yep this is the one -> slide 58 in ppt -> these params into config-file @ 10k eps.
# vis_large_eps("ddpg/ddpg_env_evalrepeatabilityCheck") # check eval?

# vis_large_eps("ddpg/ddpg_env_trainddpg4") # yes, these are the correct hyperparameters, i.e same result as slide 61 for example.