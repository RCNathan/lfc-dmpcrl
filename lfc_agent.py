from dmpcrl.agents.lstd_ql_coordinator import LstdQLearningAgentCoordinator
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
import os
import time

SymType = TypeVar("SymType", cs.SX, cs.MX)

from lfc_env import LtiSystem
from plot_running import plotRunning
from plot_dual_vars import plotDualVars

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LfcLstdQLearningAgentCoordinator(LstdQLearningAgentCoordinator):
    """A coordinator for LSTD-Q learning agents - for the Lfc problem. This agent hence handles load changes."""

    saveRunningData = False  # toggle whether during runtime, every x steps, data gets saved.
    plotRunningFlag = False  # toggle whether plotted immediately as well. 
    plotDualVarsFlag = False  # toggle whether dual vars are being plotted at every timestep. 
    small_start_time = time.time() # used for periodical saves

    cent_debug_info_dict = {
        "state": [],
        "action_opt": [],
        "cent_sol": [],
    }  # make empty dict to store centralized solution

    def update_load_info(self, env: Env[ObsType, ActType]) -> None:
        # Update load at end of timestep
        if self.centralized_flag:
            self.fixed_parameters["Pl"] = env.unwrapped.loads_over_horizon
        else:
            for i in range(len(self.agents)):
                self.agents[i].fixed_parameters["Pl"] = env.unwrapped.loads_over_horizon[[i], :] # notation necessary else it'd be (10,)

    def on_timestep_end(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        # update load information for MPCs
        self.update_load_info(env)

        if (
            not self.saveRunningData or self.centralized_flag
        ):  # if centralized, don't save running data
            return super().on_timestep_end(env, episode, timestep)

        if self.centralized_debug == False:
            print(
                "Warning: centralized_debug is False, so no centralized solution is being saved."
            )

        if timestep % 10 == 0:
            X = [env.observations, env.ep_observations]
            U = [env.actions, env.ep_actions]
            R = [env.rewards, env.ep_rewards]
            TD = [self.agents[i].agent.td_errors for i in range(self.n)]
            debug_flag = self.centralized_debug
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
                        "debug_flag": debug_flag,
                        "cent_debug_info_dict": self.cent_debug_info_dict,
                    },
                    file,
                )
            print(
                f"Episode {episode}, timestep {timestep}, ADMM iters {self.admm_coordinator.iters}, GAC iters {self.consensus_coordinator.iters}"
            )
            if self.plotRunningFlag:
                plotRunning(f"ep{episode}timestep{timestep}")

        if self.plotDualVarsFlag:
            plotDualVars(r"dual_vars\dist_av.pkl", r"dual_vars\centdebug_av.pkl")
        return super().on_timestep_end(env, episode, timestep)

    # For updating load values 
    def on_episode_start(self, env: LtiSystem, episode: int, state) -> None:
        # update load info for MPCs
        self.update_load_info(env)
        return super().on_episode_start(env, episode, state)

    # # For updating load values-> moved to on_timestep_end since it causes spikes in TD when s+ and s dont align (V(s+), Q(s,a))
    # def on_env_step(self, env: LtiSystem, episode: int, timestep: int) -> None:
    #     # update load info for MPCs
    #     # self.update_load_info(env)
    #     return super().on_env_step(env, episode, timestep)
    # ---> from testing: doesn't matter if here or on_timestep_end, same returns.

    def on_episode_end(self, env, episode, rewards):
        # save periodically, note episode starts at 0
        if env.unwrapped.save_periodically != False:
            # print("testing for periodically saving")
            if (episode + 1) % env.unwrapped.save_periodically == 0:
                print(f"Saving after episode {episode + 1}!")
                # extract from env
                X = np.asarray(env.observations)
                U = np.asarray(env.actions)
                R = np.asarray(env.rewards)
                Pl = np.asarray(env.unwrapped.loads)
                Pl_noise = np.asarray(env.unwrapped.load_noises)
                # extract from 'self' (agent or coordinator)
                TD = (
                    # agent.td_errors if centralized_flag else agent.agents[0].td_errors
                    self.td_errors if self.centralized_flag else [self.agents[i].td_errors for i in range(self.n)]
                )  # all smaller agents have global TD error
                param_dict = {}
                if self.centralized_flag:
                    param_loc = self
                    for name, val in self.updates_history.items():
                        param_dict[name] = np.asarray(val)
                else:
                    param_loc = self.agents[0]
                    for i in range(self.n):
                        for name, val in self.agents[i].updates_history.items():
                            param_dict[f"{name}_{i}"] = np.asarray(val)
                learning_params = {
                    "update_strategy": param_loc.update_strategy,
                    "optimizer": param_loc.optimizer,
                    "exploration": param_loc.exploration,
                    "experience": {
                        "max_len": param_loc.experience.maxlen,
                        "sample_size": param_loc.experience.sample_size,
                        "incl_latest": param_loc.experience.include_latest,
                    },
                }
                infeasibles = self.numInfeasibles
                # save in pkl
                saveloc = os.path.join( "data", "pkls", "periodic")
                os.makedirs(saveloc, exist_ok=True)
                with open(
                    os.path.join(saveloc, f"periodic_ep{episode + 1}.pkl"),
                    "wb",
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
                            "cent_flag": self.centralized_flag,
                            'elapsed_time': time.time() - self.small_start_time 
                        },
                        file,
                    )
        return super().on_episode_end(env, episode, rewards)

    # To store infeasible timesteps when MPC fails
    numInfeasibles = {}  # make empty dict to store infeasible timesteps

    def on_mpc_failure(
        self, episode: int, timestep: int | None, status: str, raises: bool
    ) -> None:
        if episode in self.numInfeasibles:
            self.numInfeasibles[episode].append(timestep)
        else:
            self.numInfeasibles[episode] = [timestep]
        return super().on_mpc_failure(episode, timestep, status, raises)

    # For plotting dual vars and comparing them to centralized solution
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
            if self.centralized_debug == False:
                print(
                    "Warning: centralized_debug is False, so no centralized solution is being saved."
                )
            # Save a pkl (to make the plot func) - add to existing info_dict: local_actions, local_sols
            info_dict["local_actions"] = local_actions
            # -> == info_dict['u_iters'][-1, :, 0], i.e. final iteration of first timestep
            info_dict["local_sols"] = np.array([local_sols[i].f for i in range(self.n)])
            info_dict["local_dual_vals"] = [
                {key: val.toarray() for key, val in local_sols[i].dual_vals.items()}
                for i in range(self.n)
            ]
            with open(
                "dual_vars\dist_sv.pkl",
                "wb",
            ) as file:
                pickle.dump(
                    {
                        "info_dict": info_dict,
                    },
                    file,
                )
            # plotDualVars(info_dict)
        return cs.DM(local_actions), local_sols

    # For plotting dual vars and comparing them to centralized solution
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
            performed. By default, `deterministic=False`.

        Returns
        -------
        list[Solution]
            A list of solutions for each agent, containing the local solutions and
            the information dictionary.
            The info_dict contains the following keys: augmented_x_iters, u_iters, y_iters, z_iters
        """
        (_, local_sols, info_dict) = self.admm_coordinator.solve_admm(
            state, action=action, deterministic=True
        )
        if self.plotDualVarsFlag:
            # Save a pkl (to make the plot func)
            info_dict["local_actions"] = action  # action is the centralized action
            info_dict["local_sols"] = np.array([local_sols[i].f for i in range(self.n)])
            info_dict["local_dual_vals"] = [
                {key: val.toarray() for key, val in local_sols[i].dual_vals.items()}
                for i in range(self.n)
            ]
            with open(
                "dual_vars\dist_av.pkl",
                "wb",
            ) as file:
                pickle.dump(
                    {
                        "info_dict": info_dict,
                    },
                    file,
                )
        return local_sols

    # For plotting dual vars and comparing them to centralized solution
    def state_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        deterministic: bool = False,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        action_space: Optional[Box] = None,
        **kwargs,
    ) -> tuple[cs.DM, Solution[SymType]]:
        # Get the optimal action and the solution from the parent class
        action_opt, sol = super().state_value(
            state, deterministic, vals0, action_space, **kwargs
        )

        if self.plotRunningFlag:
            self.cent_debug_info_dict["state"].append(state)
            self.cent_debug_info_dict["action_opt"].append(action_opt.toarray())
            self.cent_debug_info_dict["cent_sol"].append(sol.f)

        # Save it in a pkl
        if self.plotDualVarsFlag:
            # Save a pkl (to make the plot func)
            # make a dictionary called info_dict which saves the state, action, dual vars, and the centralized solution
            info_dict = {
                "state": state,
                "action_opt": action_opt.toarray(),
                "cent_sol": sol.f,
                # "dual_vars": sol.dual_vars['lam_g_dyn'].reshape((-1, self.N))
                "dual_vals": {key: val.toarray() for key, val in sol.dual_vals.items()},
            }
            with open(
                "dual_vars\centdebug_sv.pkl",
                "wb",
            ) as file:
                pickle.dump(
                    {
                        "info_dict": info_dict,
                    },
                    file,
                )
        # return the optimal action and the solution
        return action_opt, sol

    # For plotting dual vars and comparing them to centralized solution
    def action_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        action: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs,
    ) -> Solution[SymType]:
        # Get the solution from the parent class
        sol = super().action_value(state, action, vals0, **kwargs)

        # if self.plotRunningFlag: # TODO: check if this is correct
        #     self.cent_debug_info_dict["state"].append(state)
        #     self.cent_debug_info_dict["action_opt"].append(action.toarray())
        #     self.cent_debug_info_dict["cent_sol"].append(sol.f)

        # Save it in a pkl
        if self.plotDualVarsFlag:
            # Save a pkl (to make the plot func)
            # make a dictionary called info_dict which saves the state, action, dual vars, and the centralized solution
            info_dict = {
                "state": state,
                "action_opt": action.toarray(), # named action_opt to be the same as in state_value, though not technically correct.
                "cent_sol": sol.f,
                "dual_vals": {key: val.toarray() for key, val in sol.dual_vals.items()},
            }
            with open(
                "dual_vars\centdebug_av.pkl",
                "wb",
            ) as file:
                pickle.dump(
                    {
                        "info_dict": info_dict,
                    },
                    file,
                )
        # return the solution
        return sol
