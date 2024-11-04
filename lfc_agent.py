from csnlp import Solution
from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
from gymnasium import Env
from gymnasium.spaces import Box
from numpy import ndarray
import pickle
import numpy as np
import casadi as cs

from lfc_env import LtiSystem
from plot_running import plotRunning
from plot_dual_vars import plotDualVars

from typing import Any, Callable, Literal, Optional, TypeVar, Union

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""

    saveRunningData = True  # toggle whether during runtime, every x steps, data gets saved.
    plotRunningFlag = False  # toggle whether plotted immediately as well. note: not sure if this causes multiple terminals
    plotDualVarsFlag = True # toggle whether dual vars are being plotted at every timestep.

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
        # env.reset()
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
        self,
        state: np.ndarray,
        deterministic=False,
        action_space: Box | None = None,
    ) -> tuple[cs.DM, list[Solution]]:
        """Computes the distributed state value function using ADMM.

        Parameters
        ----------
        state : cs.DM
            The centralized state for which to compute the value function.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`."""
        (
            local_actions,
            local_sols,
            info_dict,
        ) = self.admm_coordinator.solve_admm(
            state, deterministic=deterministic, action_space=action_space
        )
        if self.plotDualVarsFlag:
            # Save a pkl (to make the plot func)
            # with open("dualvarstest.pkl", "wb",) as file:
            #         pickle.dump(
            #             {"info_dict": info_dict,}, file,
            #         )
            plotDualVars(info_dict)
        return cs.DM(local_actions), local_sols

    def distributed_action_value(
        self, state: np.ndarray, action: cs.DM
    ) -> list[Solution]:
        """Computes the distributed action value function using ADMM.

        Parameters
        ----------
        state : cs.DM
            The centralized state for which to compute the value function.
        action : cs.DM
            The centralized action for which to compute the value function.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`."""
        (_, local_sols, info_dict) = self.admm_coordinator.solve_admm(
            state, action=action, deterministic=True
        )
        if self.plotDualVarsFlag:
            plotDualVars(info_dict)
        return local_sols
