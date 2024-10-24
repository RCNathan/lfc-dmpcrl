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
from mpcrl import UpdateStrategy 

from lfc_agent import LfcLstdQLearningAgentCoordinator 
from lfc_env import LtiSystem  # change environment in lfc_env.py (= ground truth)
from lfc_learnable_mpc import (  # change learnable mpc, start with centralized
    CentralizedMpc,
    LearnableMpc,
    LocalMpc,
)
from lfc_model import Model  # change model in model.py (own repo)
from lfc_visualization import visualize
from vis_large_eps import vis_large_eps

def train(
        centralized_flag: bool,   # centralized (True) vs distributed (False)
        learning_flag: bool,      # learn vs no learn
        numEpisodes: int,         # number of episodes | x0, load etc reset on episode start
        numSteps: int,            # number of steps per episode | steps*ts = time
        admm_iters = 50,          # number of ADMM iterations
        rho = 0.5,                # for ADMM
        update_strategy: int | UpdateStrategy = 2,                # int or UpdateStrategy obj
        learning_rate = ExponentialScheduler(1e-10, factor=0.99), # alpha: ExponentialScheduler(1e-10, factor=1)
        epsilon = ExponentialScheduler(0.9, factor=0.99),         # exploration probability: ExponentialScheduler(0.9, factor=0.99) 
        eps_strength: int = 0.1,                                  # exploration strength
        experience = ExperienceReplay(maxlen=100, sample_size=20, include_latest=10, seed=1), # experience replay
        prediction_horizon = 10,                                  # MPC prediction horizon N
        ) -> None:
        
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
            global_index=i, # for the initial values of learnable params, taken from list in model
            G = G, # also for getting correct initial values
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
    # update_strategy = 2 # Frequency to update the mpc parameters with. Updates every `n` env's steps
    # update_strategy = UpdateStrategy(1, skip_first=0, hook="on_episode_end")
    if learning_flag:
        optimizer = GradientDescent(        
            # learning_rate=ExponentialScheduler(1e-10, factor=1)    
            learning_rate=learning_rate
        )
        base_exp = EpsilonGreedyExploration( # TODO: SAM: to clarify type (completely random OR perturbation on chosen input)
            # epsilon=ExponentialScheduler(0.9, factor=0.99), # (probability, decay-rate: 1 = no decay)
            epsilon=epsilon,
            # strength= 0.1 * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
            strength=eps_strength * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
            seed=1,
        )
        experience = ExperienceReplay(
            maxlen=100, sample_size=20, include_latest=10, seed=1 # smooths learning
        )  
    else: # NO LEARNING
        optimizer = GradientDescent(learning_rate=0) # learning-rate 0: alpha = 0: no updating theta's.
        base_exp = EpsilonGreedyExploration(epsilon = 0, strength = 0, seed=1,) # 0 exploration adds no perturbation
        experience = ExperienceReplay(maxlen=10, sample_size=2, include_latest=1, seed=1) 

    agents = [
        RecordUpdates(
            LstdQLearningAgent(
                mpc=distributed_mpcs[i],
                update_strategy=deepcopy(update_strategy),
                discount_factor=LearnableMpc.discount_factor,
                optimizer=deepcopy(optimizer),
                learnable_parameters=distributed_learnable_parameters[i],
                fixed_parameters=distributed_fixed_parameters[i], # Pl, y, z
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

    env = MonitorEpisodes(TimeLimit(LtiSystem(model=model), max_episode_steps=int(numSteps))) # Lti system:
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
                centralized_fixed_parameters=centralized_mpc.fixed_pars_init, # fixed: Pl
                centralized_exploration=deepcopy(base_exp),
                centralized_experience=deepcopy(experience),
                centralized_update_strategy=deepcopy(update_strategy),
                centralized_optimizer=deepcopy(optimizer),
                centralized_discount_factor=LearnableMpc.discount_factor,
                hessian_type="none",
                record_td_errors=True,
                centralized_flag=centralized_flag,
                centralized_debug=True,
                name="coordinator",
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
    )
    agent.train(env=env, episodes=numEpisodes, seed=1, raises=False)

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
    Pl = np.asarray(env.unwrapped.loads) # WARNING: use env.unwrapped.loads or env.get_wrapper_attr('loads') in new v1.0 (if ever updated)
    Pl_noise = np.asarray(env.unwrapped.load_noises)
    learning_params = {
        'update_strategy': update_strategy,
        'optimizer': optimizer,
        'epsilon': base_exp.epsilon_scheduler,
        'eps_strength': base_exp.strength_scheduler,
        'experience': {'max_len' : experience.maxlen, 'sample_size': experience.sample_size, "incl_latest" : experience.include_latest},
    }
    infeasibles = agent.unwrapped.numInfeasibles
    print("MPC failures at following eps, timesteps:", infeasibles)
    print("Total infeasible steps:")
    for key in infeasibles:
        print(key, np.asarray(infeasibles[key]).shape)

    if save_data: 
        if centralized_flag:
            pklname = 'cent'
        else:
            pklname = 'distr'
        if learning_flag == False:
            pklname = pklname + '_no_learning'
        pklname = pklname + '_' + str(numEpisodes) + 'ep' + '_scenario_0'
        with open(
            f"{pklname}.pkl",

            "wb", # w: write mode, creates new or truncates existing. b: binary mode
        ) as file:
            pickle.dump({"TD": TD, "param_dict": param_dict, "X": X, "U": U, "R": R, "Pl": Pl, "Pl_noise": Pl_noise, 
                         "learning_params": learning_params, "infeasibles": infeasibles}, file)
        print("Training succesful, file saved as", pklname)

        if make_plots:
            if numEpisodes > 0: # for larger amounts of eps -> 0 as this became standard.
                vis_large_eps(pklname)
            else:
                visualize(pklname) # outdated atm

model = Model()  # model class defines dynamic model
G = AdmmCoordinator.g_map(model.adj)  # network topology G
save_data = True
make_plots = True

### Sanity Checks ###

# Noise seed affects results a lot; weird.
# centralized, no learning: check if no noise on load, but WITH noise on A,B,F leads to same result every time (if not: problem.)
# train(centralized_flag=True, learning_flag=False, numEpisodes=5, numSteps=300, prediction_horizon=15) 
# 1) no stochasticities (none on A,B,F, load): different costs :/
# cost for noise on A,B,F only?
# 2) for noise on A,B,F -> cent_no_learning_5ep_sanityCheck: different costs. Similar? yes.
# 3) for noise on load


# added terminal penalty to cost in learnable mpc.
# train(centralized_flag=True, learning_flag=False, numEpisodes=1, numSteps=500, prediction_horizon=15)


# distributed, no learning: check if same results as centralized -> should be.
# admm_iters = 1000 # tune this and rho if the MPCs are correct but different results.
# rho = 0.1

# After adding GRC, everything breaks. ehm. 
# With perfect knowledge (no noise on A,B,F, load), x0 = zeros, and only small step of 0.01, infeasibilities all over the place...
# train(centralized_flag=True, learning_flag=False, numEpisodes=1, numSteps=500, prediction_horizon=15) 
# increasing prediction horizon fixed this! in literature: N = 12, 13 and even 15 (Venkat) -> slacking the GRC is necessary!
# filename cent_no_learning_5ep_grc

# fixed with learning?
# train(centralized_flag=True, learning_flag=True, numEpisodes=5, numSteps=200, prediction_horizon=15)
# no xD

# now that GRC gets slacked, horizon can go back to 10

# investigating why stochasticity-free leads to different results still
# [5.47128098e-05] run 1
# [5.47128098e-05] run 2
# [5.47128098e-05 2.07741712e-05 7.27197147e-06] run 3, 3 eps -> clearly its with the reset that introduces differences.
# train(centralized_flag=True, learning_flag=False, numEpisodes=3, numSteps=10, prediction_horizon=10)

### SCENARIO 0: no stochasticities ###

# cent no learn, filename = cent_no_learning_5ep_scenario_0
# train(centralized_flag=True, learning_flag=False, numEpisodes=1, numSteps=500, prediction_horizon=10)

# cent learning 
# train(centralized_flag=True, learning_flag=True, numEpisodes=5, numSteps=500, prediction_horizon=10, 
#       update_strategy=10, 
#       learning_rate=ExponentialScheduler(1e-12, factor=0.9999), 
#       epsilon=ExponentialScheduler(0.9, factor=0.99),
#       eps_strength=2000, # values depend on setup, might need large values!
#       experience=ExperienceReplay(maxlen=100, sample_size=20, include_latest=10, seed=1))


# distr working pls
train(centralized_flag=False, learning_flag=False, numEpisodes=2, numSteps=20, prediction_horizon=10,
      admm_iters=1000, rho=0.5)
# filename = cent_no_learning_1ep_scenario_0, return[460.72410733]