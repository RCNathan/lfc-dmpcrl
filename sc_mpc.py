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

from gymnasium.wrappers import TimeLimit
from mpcrl.optim import GradientDescent
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes
from dmpcrl.utils.solver_options import SolverOptions

from lfc_stochastic_agent import LfcScMPCAgent
from lfc_env import LtiSystem  # change environment in lfc_env.py (= ground truth)
from lfc_model import Model  # change model in model.py (own repo)
from block_diag import block_diag
from vis_large_eps import vis_large_eps

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from csnlp.wrappers.mpc.scenario_based_mpc import ScenarioBasedMpc, _n
from mpcrl.util.seeding import RngType

class sampleBasedMpc(ScenarioBasedMpc):
    """Sample-based MPC for scenario-based optimization problems for the LFC case.
    Note that 'scenario' defines the level of stochasticities in the system, as defined for dmpcrl, 
    and has nothing to do with the scenarios in scenario-based mpc."""
    
    def __init__(
        self, 
        model,
        n_scenarios, 
        scenario: int = 1 | 2, # scenario 1 or 2; defines the level of stochasticities
        solver: str = "qpoases", # qpoases or ipopt
        prediction_horizon: int = 10, # Np
        control_horizon: int = 10, # Nc
        input_spacing = 1, 
        shooting = "multi"
    ) -> None:
        
        # get vars from model
        self.n = model.n
        self.ts = model.ts
        self.nx_l, self.nu_l = model.nx_l, model.nu_l
        self.nx, self.nu = model.n * model.nx_l, model.n * model.nu_l
        self.x_bnd_l, self.u_bnd_l = model.x_bnd_l, model.u_bnd_l
        self.x_bnd = np.tile(model.x_bnd_l, model.n)
        self.u_bnd = np.tile(model.u_bnd_l, model.n)
        self.w_l = np.array(
            [[1e3, 1e1, 1e1, 1e1]]  # TODO: change
        )  # penalty weight for slack variables!
        self.w = np.tile(self.w_l, (1, self.n))
        self.w_grc_l = np.array([1e1])
        self.w_grc = np.tile(self.w_grc_l, (1, self.n))
        self.adj = model.adj
        self.GRC_l = model.GRC_l
        self.GRC = np.tile(model.GRC_l, model.n).reshape(-1, 1)

        # get the actual A, B, and F from the model (same as used in env), to be perturbed for scenario 2.
        A, B, F = model.A, model.B, model.F
        noise_A, noise_B, noise_F = model.noise_A, model.noise_B, model.noise_F  # for perturbing in scenario 2

        # define the Qx, Qu and Qf for the objective function
        Qx = np.array([[1e2, 0, 0, 0], [0, 1e0, 0, 0], [0, 0, 1e1, 0], [0, 0, 0, 2e1]])  # quadratic cost on states (local)
        Qx = block_diag(Qx, n=self.n) # from local Q to centralized Q (4,4) -> (12,12)
        Qu = np.array([0.5]) # 'quadaratic' cost on acyions (local)
        Qu = np.eye(self.nu) * Qu # from local Q to centralized (1,1) -> (3,3)
        Qf = np.array([[1e2, 0, 0, 0], [0, 1e0, 0, 0], [0, 0, 1e1, 0], [0, 0, 0, 2e1]])  # quadratic terminal penalty
        Qf = block_diag(Qf, n=self.n) # from local Q to centralized Q (4,4) -> (12,12)

        # create nlp object; generic, to build our mpc problem.
        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        super().__init__(nlp, n_scenarios, prediction_horizon, control_horizon, input_spacing, shooting)
        N = prediction_horizon
        gamma = 1 # discount factor, same value as lfc_learnable_mpc.py

        # variables (state, action, slack) | optimized over in mpc
        _, xs, _ = self.state("x", self.nx) # x: single state of single scenario, xs: list of states
        u, _ = self.action(
            "u",
            self.nu,
            lb=self.u_bnd[0].reshape(-1, 1),
            ub=self.u_bnd[1].reshape(-1, 1),
        )
        s = [
            self.variable(_n("s", i), (self.nx, N), lb=0)[0] for i in range(n_scenarios)
        ]  # slacks on states for each scenario | _n is a helper function to name variables
        s_grc = [
            self.variable(_n("s_grc", i), (self.nu, N), lb=0)[0] for i in range(n_scenarios)
        ]  # same but for grc

        # Fixed parameters: load
        Pl = self.parameter("Pl", (3, N))  # creates parameter obj for load
        self.fixed_pars_init = {
            "Pl": np.zeros((3, N))
        }  # value to be changed in agent; 'lfc_stochastic_agent' for example.

        # noise e is sampled from the same distribution as for the dmpcrl, for scenarios 1 and 2
        offset = 0.05
        e = [
            0.05 * (
                np.random.uniform(0, 2, (self.nu, N + 1)) - 1 + offset
            ) for _ in range(n_scenarios)
        ] # shape of e; list [n_scenarios][nx, N+1]; consistent with xs and s

        # add dynamics manually due dynamic load over horizon - for Ns scenarios
        ## scenario 1 ##
        if scenario == 1:
            for n in range(n_scenarios):
                for k in range(N):
                    self.constraint(
                        f"dynam_{n}_{k}",
                        A @ xs[n][:, [k]] + B @ u[:, [k]] + F @ (Pl[:, [k]] + e[n][:, [k]]),
                        "==",
                        xs[n][:, [k + 1]],
                        # A @ x[:, [k]] + B @ u[:, [k]] + F @ Pl[:, [k]] + b, 
                        # "==",
                        # x[:, [k + 1]],
                    ) # setting Pl[:, [0]] is identical to previous implementation where load is not tracked over N
        ## scenario 2 ##
        elif scenario == 2:
            # we need to perturb A, B, and F for scenario 2; constant over horizon k in Np
            # perturbation is sampled from same distribution as the noise applied to initialization of A,B,F in model for learnable mpc
            A_perturbed = [
                A + noise_A * np.block([
                    [2*np.random.random((4,4))-1, np.zeros((4,8))],
                    [np.zeros((4,4)), 2*np.random.random((4,4))-1, np.zeros((4,4))],
                    [np.zeros((4,8)), 2*np.random.random((4,4))-1]                    
                ])
                for _ in range(n_scenarios)
            ] # same convention as xs, s, e | 2*[0, 1)-1 = [-1, 1) | noise only on block_diags, similar to dmpcrl
            B_perturbed = [
                B + noise_B * np.block([
                    [2 * np.random.random((4, 1)) - 1, np.zeros((4, 2))],
                    [np.zeros((4, 1)), 2 * np.random.random((4, 1)) - 1, np.zeros((4, 1))],
                    [np.zeros((4, 2)), 2 * np.random.random((4, 1)) - 1]
                ])
                for _ in range(n_scenarios)
            ]
            F_perturbed = [
                F + noise_F * np.block([
                    [2 * np.random.random((4, 1)) - 1, np.zeros((4, 2))],
                    [np.zeros((4, 1)), 2 * np.random.random((4, 1)) - 1, np.zeros((4, 1))],
                    [np.zeros((4, 2)), 2 * np.random.random((4, 1)) - 1]
                ])
                for _ in range(n_scenarios)
            ]
            # dynamics altered for the perturbed matrices
            for n in range(n_scenarios):
                for k in range(N):
                    self.constraint(
                        f"dynam_{n}_{k}",
                        A_perturbed[n] @ xs[n][:, [k]] + B_perturbed[n] @ u[:, [k]] + F_perturbed[n] @ (Pl[:, [k]] + e[n][:, [k]]),
                        "==",
                        xs[n][:, [k + 1]],
                        # A @ x[:, [k]] + B @ u[:, [k]] + F @ Pl[:, [k]] + b, 
                        # "==",
                        # x[:, [k + 1]],
                    ) # setting Pl[:, [0]] is identical to previous implementation where load is not tracked over N
        else:
            raise ValueError("Scenario must be 1 or 2.")
        
        # other constraints | states and GRC | for Ns scenarios
        for n in range(n_scenarios):
            self.constraint(_n("x_lb", n), self.x_bnd[0].reshape(-1, 1) - s[n], "<=", xs[n][:, 1:])
            self.constraint(_n("x_ub", n), xs[n][:, 1:], "<=", self.x_bnd[1].reshape(-1, 1) + s[n])
            self.constraint(
                _n("GRC_lb", n),
                -self.GRC - s_grc[n],
                "<=",
                1 / self.ts * (xs[n][[1, 5, 9], 1:] - xs[n][[1, 5, 9], :-1]),
                soft=False,
            )  # generation-rate constraint
            self.constraint(
                _n("GRC_ub", n),
                1 / self.ts * (xs[n][[1, 5, 9], 1:] - xs[n][[1, 5, 9], :-1]),
                "<=",
                self.GRC + s_grc[n],
                soft=False,
            )  # generation-rate constraint
        # note: _n() is func for naming variables

        # objective | x.shape = (nx, N+1), u.shape = (nu, N)    |   sum1 is row-sum, sum2 is col-sum
        xs_stacked = [xs[n].reshape((-1, 1)) for _ in range(n_scenarios)] # shape: [x0', x1', .. xN']'
        u_stacked = u.reshape((-1, 1))
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        
        # objective to be minimized
        objective = 0
        for n in range(n_scenarios):
            objective += xs_stacked[n][:-self.nx].T @ block_diag(Qx, n=N) @ xs_stacked[n][:-self.nx]  # x'Qx
            # -> note: the n=N in block_diag is the number of blocks of Qx (size (12,12))
            objective += xs_stacked[n][-self.nx:].T @ Qf @ xs_stacked[n][-self.nx:]  # x(N)' Q_f x(N), terminal cost
            objective += cs.sum2(self.w @ s[n])  # punishes slack variables
            objective += + cs.sum2(self.w_grc @ s_grc[n])  # punishes slacks on grc
        self.minimize(
            1/n_scenarios * objective # 1/ns * sum[x'Qx + slacks] over all scenarios ns
            + u_stacked.T @ block_diag(Qu, n=N) @ u_stacked 
        )
        # solver = "qpoases"  # qpoases or ipopt
        solver = solver # qpoases or ipopt
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)


### Inspiration from lfc_train.py: ###
# centralised mpc and params
# centralized_mpc = CentralizedMpc(
#     model, prediction_horizon, solver=solver
# )  # for comparison/debugging
# -> learnable params not necessary

# initialize the mpc
model = Model()
prediction_horizon = 10
numSteps = 1000
numEpisodes = 1
centralized_scmpc = sampleBasedMpc(
    model=model, 
    scenario=1, 
    n_scenarios=5,
    solver="qpoases", 
    prediction_horizon=prediction_horizon, 
    control_horizon=prediction_horizon, 
    input_spacing=1, 
    shooting="multi"
)

# make the env
env = MonitorEpisodes(
        TimeLimit(LtiSystem(model=model, predicton_horizon=prediction_horizon), max_episode_steps=int(numSteps))
    )  # Lti system:

# initialize the agent
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LfcScMPCAgent(
            mpc=centralized_scmpc,
            learnable_parameters=None,
            fixed_parameters=centralized_scmpc.fixed_pars_init,  # fixed: Pl
            update_strategy=2,
            discount_factor=1,
            optimizer = GradientDescent(
            learning_rate=0
            )
            # centralized fixed params?
            # other args?
        ),
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 100},
)

# call evaluate(), since no learning is required
seed = 1
agent.evaluate(env=env, episodes=numEpisodes, seed=seed, raises=False)

# save all data from env; X, U, R, ... - different scenarios?
# env.observations


# in lfc_train: 
        # LfcLstdQLearningAgentCoordinator(
        #     agents=agents, # distributed mpcs
        #     ...,
        #     consensus_iters=consensus_iters,
        #     centralized_mpc=centralized_mpc,
        # ) -> we need to define a new agent, no learning required.