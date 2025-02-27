#!/usr/anaconda3/envs/lfc-dmpcrl/bin/python
import logging
import pickle
from copy import deepcopy
from datetime import *
import os
import time

import casadi as cs
import numpy as np

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
from mpcrl.core.update import UpdateStrategy 

from lfc_agent import LfcLstdQLearningAgentCoordinator
from lfc_env import LtiSystem 
from lfc_learnable_mpc import ( 
    CentralizedMpc,
    LearnableMpc,
    LocalMpc,
)
from lfc_model import Model 
from solve_time_wrapper import SolverTimeRecorder


def dmpcrl_evaluate(
        filename: str,
        numEpisodes: int,
        numSteps: int = 1000,
        save_name_info: str = None,  # scenario, which model is used etc
        best_ep: int = None, # if a specific episode is to be evaluated instead of the last one
        log_freqs: int = 100, # frequency of logging evaluation progress 
        scenario: int = None, # which scenario is being evaluated
):
    """
    Evaluate a trained (d)mpcrl-agent on the LFC environment. 

    ----------------
    Inputs:
    filename: str
        Path to the saved file which has the trained parameters.
    etc: str
        stuff
    
    Outputs:
    data stored in a .pkl-file containing X, U, R, Pl, Pl_noises
    """
    
    if scenario not in {0, 1, 2}:
        raise ValueError("Please provide a scenario from {0, 1, 2}")

    with open(
        filename + '.pkl',
        "rb"
    ) as file:
        data = pickle.load(file)

    # some constants
    model = Model(scenario=scenario) # load the model
    n = model.n
    G = AdmmCoordinator.g_map(model.adj)  # network topology G

    # get other constants from the data
    centralized_flag = data['cent_flag']
    prediction_horizon = 10 # I'm not currently saving this!
    admm_iters = 50 # idem.
    rho = 0.5 # also not saving
    consensus_iters = 100 # idem dito
    solver = "qpoases" # also not saving, but doesn't really matter!
    seed = 1

    learning_params = data['learning_params']
    update_strategy = learning_params['update_strategy']
    # update_strategy = learning_params.get('update_strategy', UpdateStrategy(frequency=10, skip_first=100))
    # optimizer = learning_params.get('optimizer', GradientDescent(learning_rate=0))
    # epsilon = learning_params.get('epsilon', 0)
    # eps_strength = learning_params.get('eps_strength', 0)
    # experience = learning_params.get('experience', ExperienceReplay(maxlen=10, sample_size=2, include_latest=1, seed=1))
    # base_exp = EpsilonGreedyExploration(epsilon=epsilon, strength=0, seed=seed)

    # for evaluating: no learning needs to happen!
    update_strategy = UpdateStrategy(frequency=10, skip_first=100)
    optimizer = GradientDescent(learning_rate=0)
    base_exp = EpsilonGreedyExploration(epsilon=0, strength=0, seed=seed) 
    experience = ExperienceReplay(maxlen=10, sample_size=2, include_latest=1, seed=seed)


    param_dict = data.get('param_dict', None) # param are stored for every update-step
    if param_dict == None:
        raise ValueError("No param_dict in the data file")

    # store the (final) parameters in a dict to be pushed to the agent.evaluate() later
    params = {}
    for key in param_dict.keys():
        if best_ep != None:
            params[key] = param_dict[key][int(best_ep*numSteps/update_strategy.frequency)]
        else:
            params[key] = param_dict[key][-1]

    # centralised mpc and params
    centralized_mpc = SolverTimeRecorder(
        CentralizedMpc(
            model, prediction_horizon, solver=solver
        )
    ) 

    # inject the learned values of parameters
    for key in centralized_mpc.learnable_pars_init.keys():
        centralized_mpc.learnable_pars_init[key] = params[key]
    # for indexing with i: use [...]_init.update() - see learnable_mpc.py
    # for dict: simply set the value

    centralized_learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(
                name, val.shape, val, sym=centralized_mpc.parameters[name]
            )
            for name, val in centralized_mpc.learnable_pars_init.items()
        )
    )


    # distributed agents
    distributed_mpcs: list[LocalMpc] = [
        SolverTimeRecorder(
            LocalMpc(
                model=model,
                prediction_horizon=prediction_horizon,
                num_neighbours=len(G[i]) - 1,
                my_index=G[i].index(i),  # index where agent i is located inside of G[i]
                global_index=i,  # for the initial values of learnable params, taken from list in model
                G=G,  # also for getting correct initial values
                rho=rho,
                solver=solver,
            )
        )
        for i in range(n)
    ]

    # inject the learned values of parameters
    for i in range(n):
        for key in distributed_mpcs[i].learnable_pars_init.keys():
            distributed_mpcs[i].learnable_pars_init[key] = params[f"{key}_{i}"]


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
        distributed_mpcs[i].fixed_pars_init for i in range(n)
    ]

    # make the agent(s) - even though no learning is happening, still need to provide learning params
    agents = [
        RecordUpdates(
            LstdQLearningAgent(
                mpc=distributed_mpcs[i],
                update_strategy=deepcopy(update_strategy),
                discount_factor=LearnableMpc.discount_factor,
                optimizer=deepcopy(optimizer),
                learnable_parameters=distributed_learnable_parameters[i],
                fixed_parameters=distributed_fixed_parameters[i],  # Pl, y, z
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

    env = MonitorEpisodes(
        TimeLimit(LtiSystem(model=model, predicton_horizon=prediction_horizon), max_episode_steps=int(numSteps))
    )  # Lti system:
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
                consensus_iters=consensus_iters,
                centralized_mpc=centralized_mpc,
                centralized_learnable_parameters=centralized_learnable_pars,
                centralized_fixed_parameters=centralized_mpc.fixed_pars_init,  # fixed: Pl
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
        log_frequencies={"on_timestep_end": log_freqs},
    )
    start_time = time.time()
    # evaluate agent (NOT TRAIN!)
    agent.evaluate(env=env, episodes=numEpisodes, seed=seed, raises=False)
    end_time = time.time()
    print("Time elapsed:", end_time - start_time)


    # retrieve from env
    X = np.asarray(env.observations)
    U = np.asarray(env.actions)
    R = np.asarray(env.rewards)
    Pl = np.asarray(env.unwrapped.loads)
    Pl_noise = np.asarray(env.unwrapped.load_noises)

    infeasibles = agent.unwrapped.numInfeasibles
    print("MPC failures at following eps, timesteps:", infeasibles)
    print("Total infeasible steps:")
    for key in infeasibles:
        print(key, np.asarray(infeasibles[key]).shape)

    if centralized_flag:
        solver_times = np.asarray(centralized_mpc.solver_time)
        print(f"Total mpc solve time {np.sum(solver_times)} s")
    else:
        solver_times = np.asarray([distributed_mpcs[i].solver_time for i in range(Model.n)])
        print(f"Total mpc solve time {np.sum(np.max(solver_times, axis=0))} s")


    # make sure dir exists, save plot and close after
    saveloc = r'evaluate_data'
    os.makedirs(saveloc, exist_ok=True)
    savename = f"dmpcrl_{numEpisodes}eps_{save_name_info}_scenario{scenario}"
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
                "infeasibles": infeasibles,
                "cent_flag": centralized_flag,
                "elapsed_time": end_time - start_time,
                "solver_times": solver_times,
                "scenario": scenario,
            },
            file,
        )
    print("Evaluation succesful, file saved as", savename)



# let's see what is in the saved files first (data\pkls\) for dmpcrl
# filename = r"data\pkls\tcl63_cent_100ep_scenario_2" # centralized
# filename = r"data\pkls\periodic\tdl67\periodic_ep120" # distributed periodic
# filename = r"data\pkls\tdl67_distr_50ep_scenario_2" # distributed

# dmpcrl_evaluate(
#     filename=r"data\pkls\tcl63_cent_100ep_scenario_2",
#     numEpisodes=20,
#     numSteps=1000,
#     save_name_info = "tcl63_scenario2",
#     # best_ep=20,
#     )

dmpcrl_evaluate(
    filename=r"data\pkls\periodic\tdl67\periodic_ep120",
    numEpisodes=20,
    numSteps=1000,
    save_name_info = "tdl67",
    log_freqs=1,
    )

print("debug")