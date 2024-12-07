#!/usr/anaconda3/envs/lfc-dmpcrl/bin/python
import logging
import pickle
from copy import deepcopy
from datetime import *
import os
import argparse
import json

import casadi as cs
import numpy as np

# from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
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
from masterplot import large_plot


def train(
    centralized_flag: bool,  # centralized (True) vs distributed (False)
    learning_flag: bool,  # learn vs no learn
    numEpisodes: int,  # number of episodes | x0, load etc reset on episode start
    numSteps: int = 1000,  # number of steps per episode | steps*ts = time
    admm_iters=50,  # number of ADMM iterations
    rho=0.5,  # for ADMM
    consensus_iters: int = 100,
    update_strategy: int | UpdateStrategy = 2,  # int or UpdateStrategy obj
    learning_rate=ExponentialScheduler(
        1e-10, factor=0.99
    ),  # alpha: ExponentialScheduler(1e-10, factor=1)
    epsilon=ExponentialScheduler(
        0.9, factor=0.99
    ),  # exploration probability: ExponentialScheduler(0.9, factor=0.99)
    eps_strength: int = 0.1,  # exploration strength
    experience=ExperienceReplay(
        maxlen=100, sample_size=20, include_latest=10, seed=1
    ),  # experience replay
    prediction_horizon: int = 10,  # MPC prediction horizon N
    centralized_debug=False,  # debug flag for centralized mpc
    save_name_info: str =None, # optional name to provide more info for saves (plots etc)
    solver: str = "qpoases",  # solver (qpoases or ipopt)
) -> None:
    
    # High-level stuff
    model = Model()  # model class defines dynamic model
    G = AdmmCoordinator.g_map(model.adj)  # network topology G
    save_data = True
    make_plots = True

    if save_name_info == None:
        raise NameError ('Please provide some info (e.g Test1) to differentiate simulations, \
                         by providing save_name_info')
    if type(save_name_info) != str:
        raise TypeError('Please provide a descriptive string (str) for save_name_info')

    # centralised mpc and params
    centralized_mpc = CentralizedMpc(
        model, prediction_horizon, solver=solver,
    )  # for comparison/debugging
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
        base_exp = EpsilonGreedyExploration(  # TODO: SAM: to clarify type (completely random OR perturbation on chosen input)
            # epsilon=ExponentialScheduler(0.9, factor=0.99), # (probability, decay-rate: 1 = no decay)
            epsilon=epsilon,
            # strength= 0.1 * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
            strength=eps_strength * (model.u_bnd_l[1, 0] - model.u_bnd_l[0, 0]),
            seed=1,
        )
        experience = ExperienceReplay(
            maxlen=100, sample_size=20, include_latest=10, seed=1  # smooths learning
        )
    else:  # NO LEARNING
        optimizer = GradientDescent(
            learning_rate=0
        )  # learning-rate 0: alpha = 0: no updating theta's.
        base_exp = EpsilonGreedyExploration(
            epsilon=0,
            strength=0,
            seed=1,
        )  # 0 exploration adds no perturbation
        experience = ExperienceReplay(
            maxlen=10, sample_size=2, include_latest=1, seed=1
        )

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
        TimeLimit(LtiSystem(model=model), max_episode_steps=int(numSteps))
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
                centralized_debug=centralized_debug,
                name="coordinator",
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 100},
    )
    if learning_flag:
        agent.train(env=env, episodes=numEpisodes, seed=1, raises=False)
    else:
        agent.train(env=env, episodes=numEpisodes, seed=1, raises=False)
        # agent.evaluate(env=env, episodes=numEpisodes, seed=1, raises=False) # bugged atm

    # extract data
    # from agent
    TD = (
        # agent.td_errors if centralized_flag else agent.agents[0].td_errors
        agent.td_errors if centralized_flag else [agent.agents[i].td_errors for i in range(Model.n)]
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
    Pl = np.asarray(
        env.unwrapped.loads
    )  # WARNING: use env.unwrapped.loads or env.get_wrapper_attr('loads') in new v1.0 (if ever updated)
    Pl_noise = np.asarray(env.unwrapped.load_noises)
    learning_params = {
        "update_strategy": update_strategy,
        "optimizer": optimizer,
        "epsilon": base_exp.epsilon_scheduler,
        "eps_strength": base_exp.strength_scheduler,
        "experience": {
            "max_len": experience.maxlen,
            "sample_size": experience.sample_size,
            "incl_latest": experience.include_latest,
        },
    }
    infeasibles = agent.unwrapped.numInfeasibles
    print("MPC failures at following eps, timesteps:", infeasibles)
    print("Total infeasible steps:")
    for key in infeasibles:
        print(key, np.asarray(infeasibles[key]).shape)

    if save_data:
        if centralized_flag:
            pklname = "cent"
        else:
            pklname = "distr"
        if learning_flag == False:
            pklname = pklname + "_no_learning"
        pklname = pklname + "_" + str(numEpisodes) + "ep" + "_scenario_1"

        # make sure dir exists, save plot and close after
        # saveloc = r'data\pkls'
        saveloc = os.path.join('data', 'pkls') # to ensure cross-platform compatibility
        os.makedirs(saveloc, exist_ok=True)
        savename = save_name_info + "_" + pklname
        file_path = os.path.join(saveloc, savename) # again, to ensure cross-platform compatibility

        with open(
            # f'{saveloc}\{savename}.pkl',
            file_path + '.pkl',
            "wb",  # w: write mode, creates new or truncates existing. b: binary mode
        ) as file:
            pickle.dump(
                {
                    "TD": TD,
                    "param_dict": param_dict,
                    "X": X,
                    "U": U,
                    "R": R,
                    "Pl": Pl,
                    "Pl_noise": Pl_noise,
                    "learning_params": learning_params,
                    "infeasibles": infeasibles,
                    "cent_flag": centralized_flag,
                },
                file,
            )
        print("Training succesful, file saved as", savename)

        if make_plots:
            large_plot(
                # f'{saveloc}\{savename}',
                file_path, 
                optional_name=save_name_info 
                )

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

            train(**param_set)
            print("Training complete")
        print("All training complete")
