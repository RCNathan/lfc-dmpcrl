import logging
import pickle
from copy import deepcopy
from datetime import *

import casadi as cs
import numpy as np
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from dmpcrl.core.admm import AdmmCoordinator
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import EpsilonGreedyExploration, StepWiseExploration
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.optim import GradientDescent
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from lfc_agent import LfcLstdQLearningAgentCoordinator # TODO: integrate lfc_agent
from lfc_env import LtiSystem  # change environment in lfc_env.py (= ground truth)
from lfc_learnable_mpc import (  # change learnable mpc, start with centralized
    CentralizedMpc,
    LearnableMpc,
    LocalMpc,
)
from lfc_model import Model  # change model in model.py (own repo)
from lfc_visualization import visualize

save_data = True
make_plots = True

centralized_flag = True
learning_flag = False

prediction_horizon = 10 # higher seems better but takes significantly longer/more compute time & resources | not the issue at hand.
admm_iters = 50
rho = 0.5
model = Model()  # model class defines dynamic model
G = AdmmCoordinator.g_map(model.adj)  # network topology G

# centralised mpc and params
centralized_mpc = CentralizedMpc(model, prediction_horizon)  # for comparison/debugging
centralized_learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=centralized_mpc.parameters[name])
        for name, val in centralized_mpc.learnable_pars_init.items()
    )
)

# distributed agents
distributed_mpcs: list[LocalMpc] = [
    LocalMpc(
        model=model,
        prediction_horizon=prediction_horizon,
        num_neighbours=len(G[i]) - 1,
        my_index=G[i].index(i),
        rho=rho,
    )
    for i in range(Model.n)
]
distributed_learnable_parameters: list[LearnableParametersDict] = [
    LearnableParametersDict[cs.SX](
        (
            LearnableParameter(
                name, val.shape, val, sym=distributed_mpcs[i].parameters[name]
            )
            for name, val in distributed_mpcs[i].learnable_pars_init.items()
        )
    )
    for i in range(Model.n)
]
distributed_fixed_parameters: list = [
    distributed_mpcs[i].fixed_pars_init for i in range(Model.n)
]

# learning arguments
update_strategy = 2 # Frequency to update the mpc parameters with. Updates every `n` env's steps
if learning_flag:
    optimizer = GradientDescent(        
        learning_rate=ExponentialScheduler(5e-5, factor=0.9996)    
    )
    base_exp = EpsilonGreedyExploration( # TODO: SAM: to clarify type (completely random/perturbation)
        epsilon=ExponentialScheduler(0.7, factor=0.9), # (value, decay-rate: 1 = no decay)
        strength=0.1 * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
        seed=1,
    )
    experience = ExperienceReplay(
        maxlen=100, sample_size=15, include_latest=10, seed=1 # smooths learning
    )  
else: # NO LEARNING
    optimizer = GradientDescent(learning_rate=0) # learning-rate 0: alpha = 0: no updating theta's.
    base_exp = EpsilonGreedyExploration(epsilon = 0, strength = 0, seed=1,) # 0 exploration adds no perturbation
    experience = ExperienceReplay(maxlen=100, sample_size=15, include_latest=10, seed=1) 

agents = [
    RecordUpdates(
        LstdQLearningAgent(
            mpc=distributed_mpcs[i],
            update_strategy=deepcopy(update_strategy),
            discount_factor=LearnableMpc.discount_factor,
            optimizer=deepcopy(optimizer),
            learnable_parameters=distributed_learnable_parameters[i],
            fixed_parameters=distributed_fixed_parameters[i],
            exploration=StepWiseExploration(
                base_exploration=deepcopy(base_exp),
                step_size=admm_iters,
                stepwise_decay=False,
            ),
            experience=deepcopy(experience),
            hessian_type="none",
            record_td_errors=True,
            name=f"agent_{i}",
        )
    )
    for i in range(Model.n)
]

env = MonitorEpisodes(TimeLimit(LtiSystem(model=model), max_episode_steps=int(2e2))) # Lti system:
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LfcLstdQLearningAgentCoordinator(
            agents=agents,
            N=prediction_horizon,
            nx=4,
            nu=1,
            adj=model.adj,
            rho=rho,
            admm_iters=admm_iters,
            consensus_iters=100,
            centralized_mpc=centralized_mpc,
            centralized_learnable_parameters=centralized_learnable_pars,
            centralized_fixed_parameters=centralized_mpc.fixed_pars_init, # TODO: implement delta PL
            centralized_exploration=deepcopy(base_exp),
            centralized_experience=deepcopy(experience),
            centralized_update_strategy=deepcopy(update_strategy),
            centralized_optimizer=deepcopy(optimizer),
            centralized_discount_factor=LearnableMpc.discount_factor,
            hessian_type="none",
            record_td_errors=True,
            centralized_flag=centralized_flag,
            centralized_debug=False,
            name="coordinator",
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 100},
)

agent.train(env=env, episodes=1, seed=1, raises=False)

# extract data
# from agent
TD = (
    agent.td_errors if centralized_flag else agent.agents[0].td_errors
)  # all smaller agents have global TD error
param_dict = {}
if centralized_flag:
    for name, val in agent.updates_history.items():
        param_dict[name] = np.asarray(val)
else:
    for i in range(Model.n):
        for name, val in agent.agents[i].updates_history.items():
            param_dict[f"{name}_{i}"] = np.asarray(val)
X = np.asarray(env.observations)
U = np.asarray(env.actions)
R = np.asarray(env.rewards)

if save_data:
    if centralized_flag:
        pklname = 'cent'
    else:
        pklname = 'decent'
    if learning_flag == False:
        pklname = pklname + '_no_learning'
    with open(
        f"{pklname}.pkl",
        "wb", # w: write mode, creates new or truncates existing. b: binary mode
    ) as file:
        pickle.dump({"TD": TD, "param_dict": param_dict, "X": X, "U": U, "R": R}, file)
    print("Training succesful, file saved as", pklname)

    if make_plots:
        visualize(pklname)
