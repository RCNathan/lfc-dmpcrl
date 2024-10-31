from casadi.casadi import DM
from csnlp import Solution
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from gymnasium import Env
from gymnasium.spaces import Box
from numpy import ndarray
import pickle

from lfc_env import LtiSystem
from plot_running import plotRunning

from typing import Any, Callable, Literal, Optional, TypeVar, Union

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""

    saveRunningData = (
        True  # toggle whether during runtime, every x steps, data gets saved.
    )
    plotRunningFlag = True  # toggle whether plotted immediately as well. note: not sure if this causes multiple terminals

    def on_timestep_end(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        if not self.saveRunningData or self.centralized_flag:
            return super().on_timestep_end(env, episode, timestep)

        if timestep % 10 == 0:
            X = [env.observations, env.ep_observations]
            U = [env.actions, env.ep_actions]
            R = [env.rewards, env.ep_rewards]
            TD = self.agents[0].agent.td_errors
            info = {
                "admm_iters": self.admm_coordinator.iters,
                "consensus_iters": self.consensus_coordinator.iters,
            }
            with open(
                f"running_pkls\ep{episode}timestep{timestep}.pkl",
                "wb",
            ) as file:
                pickle.dump(
                    {
                        "TD": TD,
                        "X": X,
                        "U": U,
                        "R": R,
                        "info": info,
                    },
                    file,
                )
            print(f"Episode {episode}, timestep {timestep}")
            if self.plotRunningFlag:
                plotRunning(f"ep{episode}timestep{timestep}")
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        if self.centralized_flag:
            self.fixed_parameters["Pl"] = env.unwrapped.load
        else:
            for i in range(len(self.agents)):
                self.agents[i].fixed_parameters["Pl"] = env.unwrapped.load[i]
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: LtiSystem, episode: int, timestep: int) -> None:
        if self.centralized_flag:
            self.fixed_parameters["Pl"] = env.unwrapped.load
        else:
            for i in range(len(self.agents)):
                self.agents[i].fixed_parameters["Pl"] = env.unwrapped.load[i]
        return super().on_env_step(env, episode, timestep)

    # numInfeasibles
    numInfeasibles = {}  # make empty dict

    def on_mpc_failure(
        self, episode: int, timestep: int | None, status: str, raises: bool
    ) -> None:
        if episode in self.numInfeasibles:
            self.numInfeasibles[episode].append(timestep)
        else:
            self.numInfeasibles[episode] = [timestep]
        return super().on_mpc_failure(episode, timestep, status, raises)

    def distributed_state_value(
        self, state: ndarray, deterministic=False, action_space: Box | None = None
    ) -> tuple[DM, list[Solution]]:
        return super().distributed_state_value(state, deterministic, action_space)
