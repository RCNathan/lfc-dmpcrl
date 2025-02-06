import os
import numpy as np
import pickle
import time 

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

# from lfc_ddpg_env import ddpgEnv


class AugmentedObservationWrapper(ObservationWrapper):
    """An environment for training a DDPG agent on the LFC problem.
    The state is augmented with the previous state `x(k-1)` and the load along the horizon `Pl(k), ..., Pl(k+N)`.
    """

    def __init__(self, env):
        super().__init__(env)

        lfc_model = Model()
        ubnd = lfc_model.ubnd
        n = lfc_model.n
        nxl = lfc_model.nx_l
        nx = n * nxl

        # Augment the observation_space with the previous state and the load
        # [x(k), x(k-1), load(k), ... load(k+N)] => [(12,), (12,), (3, N)] => (12 + 12 + 3N,)
        loadShape = (
            self.unwrapped.loads_over_horizon.flatten().shape
        )  # from (n, N) to (n*N,)
        self.observation_space = spaces.Box(
            np.concatenate(
                [
                    np.full(nx, -np.inf),
                    np.full(nx, -np.inf),
                    np.full(loadShape, -np.inf),
                ]
            ),
            np.concatenate(
                [np.full(nx, np.inf), np.full(nx, np.inf), np.full(loadShape, np.inf)]
            ),
            dtype=np.float64,  # float64 recommended by SB3
        )
        env.action_space = spaces.Box(
            low=-ubnd, high=ubnd, shape=(n,), dtype=np.float32
        )  # float32 recommended by SB3

    def observation(self, observation):
        observation = observation.flatten()
        # augment the state from the env to include the previous state and the load
        state = self.unwrapped.x.flatten() # to check if it's identical to state or not..
        if not np.allclose(state, observation):
            raise ValueError(
                "The state from the environment is not the same as the observation."
            )
        prev_state = self.unwrapped.last_xkm1.flatten()
        loads = self.unwrapped.loads_over_horizon.flatten()
        augmented_observation = np.concatenate([observation, prev_state, loads])
        return augmented_observation


def make_env(
    steps_per_episode: int=1000,  
    prediction_horizon: int=10,
    flipFlag: bool=True,
    isEval: bool=False,
) -> VecNormalize:
    
    # Create the environment
    lfc_model = Model() # state space model
    env = MonitorEpisodes(
        TimeLimit(
            LtiSystem(
                model=lfc_model, predicton_horizon=prediction_horizon
            ), 
            max_episode_steps=int(steps_per_episode),
        )
    )

    # Add SB3 wrappers
    env = AugmentedObservationWrapper(env) # to augment the observation
    if flipFlag: 
        env = TransformReward(env, neg) # to transform the cost into reward (env returns cost, but RL algorithms expects reward)
    # check_env(env, warn=True) # to check if the environment is compatible with SB3 ## TURN OFF after checking; it steps through env w/o resetting

    env = Monitor(env) # it is bugging me to do it.
    venv = DummyVecEnv([lambda: env]) # allows multithreading, necessary to vectorize env
    if isEval:
        venv = VecNormalize(venv, training=False) # TODO: check: Should norm_obs, norm_rewards be false?
    else:
        venv = VecNormalize(venv, training=True) # moving average and normalization 
    
    return venv

def train_ddpg(
        steps_per_episode: int=1000, 
        num_episodes: int=10, 
        prediction_horizon: int=10,
        numEvals: int=10,
        savename_info: str="",
        learning_rate: int=1e-3,
        weight_decay: int=1e-5,
        train_freq: tuple=(5, "step"),
        buffer_size: int=1e5,
        batch_size:int=256,
        net_arch: list=[256, 256],
        gamma: float=0.99,
        seed: int=1,
        flipFlag: bool=True,
        makePlots: bool=False,
        ):
    # create the environment for training
    venv = make_env(steps_per_episode=steps_per_episode, prediction_horizon=prediction_horizon, flipFlag=flipFlag, isEval=False)

    # Create callbacks
    # Create your own by inherting from BaseCallback, or use the built-in ones:
    # EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
    # - EvalCallback: Evaluate periodically and save the best model
    # - CheckpointCallback: Save the model at regular intervals
    # - StopTrainingOnRewardThreshold: Stop training when the mean reward exceeds a certain threshold
    # ProgressCallback: Report training progress -> indicator: progress, time, estimated remaining time --> integrated in learn() method

    # # Simulation parameters
    totalSteps = steps_per_episode * num_episodes
    # numEvals = 10 # how many evaluations to perform during training

    # create the path for saving the callbacks (best model), env (w/ normalizations), model (i.e params/weights) and the envs.env... (i.e MonitorEpisodes)
    savename = "lfc_" + savename_info
    saveloc = os.path.join("ddpg", savename) # makes ddpg/ddpg_lfc_ddpg4 for example
    saveloc_best = os.path.join("ddpg", "best_model")
    saveloc_best = os.path.join(saveloc_best, savename) # makes ddpg/best_model/ddpg_lfc_ddpg4.zip for example 

    # make sure the dirs exist (ugly I know)
    os.makedirs("ddpg", exist_ok=True)
    os.makedirs(os.path.join("ddpg", "best_model"), exist_ok=True)

    # create the eval callback
    eval_env = make_env(steps_per_episode=steps_per_episode, prediction_horizon=prediction_horizon, flipFlag=flipFlag, isEval=True)
    cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=int(totalSteps / numEvals),
        best_model_save_path=saveloc_best,  # saves after new best model is found by evaluating on eval_env.
    )

    # Define the model to train
    lfc_model = Model()
    na = venv.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(na), np.full(na, lfc_model.ubnd), dt=lfc_model.ts_env
    )
    model = DDPG(
        policy="MlpPolicy",
        env=venv,
        action_noise=action_noise,
        verbose=1,
        learning_rate=learning_rate,
        train_freq=train_freq,  # update the model every train_freq steps, alternatively pass tuple; (5, "step"), or (2, "episode")
        buffer_size=int(buffer_size),
        batch_size=batch_size,
        learning_starts=batch_size,
        # tau=1e-3, # the soft update coefficient ("Polyak update", between 0 and 1)
        gamma=gamma,
        # train_freq=(96, "step"),
        # gradient_steps=-1, # How many gradient steps to do after each rollout, -1 means to do as many gradient steps as steps done in the environment
        policy_kwargs={
            "net_arch": net_arch, # [256, 256], -- supplying only one list means actor/critic share archetecture, else: pass dict: e.g. net_arch = dict(pi=[64,64], qf=[400,300])
            "optimizer_kwargs": {
                "weight_decay": weight_decay,
            },  # l2_regularization=1e-5,
        },
        # verbose=verbose,
        seed=seed,
        # device=device,
    )
    # train
    start_time = time.time()
    model.learn(total_timesteps=totalSteps, log_interval=1, progress_bar=True, callback=cb)
    end_time = time.time()
    print("Time elapsed:", end_time - start_time)
    
    # save the trained model and the training env (with normalizations)
    env = model.get_env()
    model.save(saveloc + "_model") # save the model; all parameters (or weights)
    env.save(saveloc + "_env.pkl") # save the env with normalizations (to use with get_env() later for evaluation)
    print("Model saved as", savename)

    # return as data the `MonitorEpisodes` from the training and evaluation envs - ugly,
    # but they must be digged out from the `VecNormalize` wrapper
    if flipFlag:
        train_env, eval_env =  venv.envs[0].env.env.env, eval_env.envs[0].env.env.env # with flipreward
    else:
        train_env, eval_env =  venv.envs[0].env.env, eval_env.envs[0].env.env # without flipreward
    for env_type, env in reversed([( "train", train_env), ("eval", eval_env)]):
        X = np.asarray(env.observations)
        U = np.asarray(env.actions)
        R = np.asarray(env.rewards)
        Pl = np.asarray(env.unwrapped.loads)
        Pl_noise = np.asarray(env.unwrapped.load_noises)
        # TODO TD errors => no.

        print("Shape:",X.shape)
        # save the MonitorEpisodes data
        # savename = f"ddpg_env_{env_type}" + savename_info
        # saveloc = os.path.join("ddpg", savename)
        
        with open(
            f"{saveloc}_{env_type}.pkl", # use the saveloc defined earlier for consistency; saveloc = ddpg/lfc_ddpg4 for example; then add _train, _eval
            "wb",
        ) as file:
            pickle.dump(
                {
                    "X": X,
                    "U": U,
                    "R": R,
                    "Pl": Pl,
                    "Pl_noise": Pl_noise,
                    'elapsed_time': end_time - start_time,
                },
                file,
            )
    if makePlots:
        print("Plotting for", saveloc)
        vis_large_eps(saveloc)

# call the training function
# Simulation parameters
steps_per_episode = 1000 # total timesteps for simulation, should be identical to lfc-dmpcrl case
num_episodes = 20 # how many episodes to train for
numEvals = 2 # how many evaluations to perform during training (default 10)

if __name__ == "__main__":
    print("Executing from __main__")
    train_ddpg(
        steps_per_episode=steps_per_episode, 
        num_episodes=num_episodes, 
        numEvals=numEvals, 
        learning_rate=1e-6, # from baseline; 1e-6 seems best
        weight_decay=1e-5,
        train_freq=(5, "step"),
        buffer_size=int(1e6),
        batch_size=256,
        gamma=0.999,
        seed=1,
        net_arch = [256, 256],
        flipFlag=True,
        makePlots=True,
        savename_info="test",
    )
# vis_large_eps(r"ddpg\ddpg_env_trainchangelr",)




# venv = model.get_env()  # venv.envs[0].env.env.env is LtiSystem
# model = DDPG.load("ddpg_lfc_agent")
# obs = venv.reset()
# # timestep = 0 # breakpoint: timestep % 100 == 0 or timestep == 1002
# # while env.unwrapped.step_counter < 1010: # env.unwrapped.step_counter instead?
# dones = False
# while dones == False:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = venv.step(action)
#     # print(obs)
#     # timestep += 1


# you can get information from the env, i.e state, action etc using:
# env.envs[0].env.env.env.last_action, similarly x, last_x, last_xkm1, load, load_noise
# -> self.last_x == self.x

# print("ddpg_lfc_agent is a model trained on 100k total steps, 17-12 17h40")