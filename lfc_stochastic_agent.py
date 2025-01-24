from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from mpcrl.agents.lstd_q_learning import LstdQLearningAgent
from mpcrl.agents.common.agent import Agent
from csnlp import Solution
from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
from numpy import ndarray
import numpy.typing as npt
from typing import Any, Callable, Literal, Optional, TypeVar, Union
from collections.abc import Collection, Iterable, Iterator
import pickle
import casadi as cs
from casadi import DM

SymType = TypeVar("SymType", cs.SX, cs.MX)

from lfc_env import LtiSystem
from plot_running import plotRunning
from plot_dual_vars import plotDualVars

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# the agent for the scenario-based MPC
class LfcScMPCAgent(LstdQLearningAgent):
    """Agent for Scenario-based MPC - for the LFC problem. This agent hence handles load changes."""

    def update_load_info(self, env: Env[ObsType, ActType]) -> None:
        # Update load at end of timestep | only centralized is implemented
        self.fixed_parameters["Pl"] = env.unwrapped.loads_over_horizon
    
    # For updating load values at the start of the episode
    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        # update load info for MPCs
        self.update_load_info(env)
        return super().on_episode_start(env, episode, state)

    def on_timestep_end(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        # update load information for MPCs
        self.update_load_info(env)
        return super().on_timestep_end(env, episode, timestep)