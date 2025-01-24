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
from mpcrl.core.update import UpdateStrategy 

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



from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp import multistart as ms
from csnlp.wrappers import Mpc
from csnlp.wrappers.mpc.scenario_based_mpc import ScenarioBasedMpc, _n
from mpcrl.util.seeding import RngType

class sampleBasedMpc(ScenarioBasedMpc):
    """Sample-based MPC for scenario-based optimization problems for the LFC case."""
    
    def __init__(
        self, 
        n_scenarios, 
        prediction_horizon, 
        control_horizon = None, 
        input_spacing = 1, 
        shooting = "multi"
    ) -> None:
        
        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        super().__init__(nlp, n_scenarios, prediction_horizon, control_horizon, input_spacing, shooting)
        # Mpc.__init__(self, nlp, prediction_horizon)

    # from the scenario_based_mpc.py; 
    # def __init__(
    #     self,
    #     nlp: Nlp[SymType],
    #     n_scenarios: int,
    #     prediction_horizon: int,
    #     control_horizon: Optional[int] = None,
    #     input_spacing: int = 1,
    #     shooting: Literal["single", "multi"] = "multi",
    # ) 

testClass = sampleBasedMpc(n_scenarios=10, prediction_horizon=10, control_horizon=10, input_spacing=1, shooting="multi")

# in lfc_train:
# centralised mpc and params
# centralized_mpc = CentralizedMpc(
#     model, prediction_horizon, solver=solver
# )  # for comparison/debugging
# -> learnable params not necessary


