from typing import Any
from dmpcrl.core.admm import AdmmCoordinator

import casadi as cs
import numpy as np
from csnlp import Solution
from gymnasium.spaces import Box
from mpcrl import Agent

from dmpcrl.mpc.mpc_admm import MpcAdmm

# code from the init:
# # create auxillary vars for ADMM procedure
# self.y = [
#     np.zeros((nx_l * len(self.G[i]), N + 1)) for i in range(self.n)
# ]  # multipliers


class Lfc_AdmmCoordinator(AdmmCoordinator):
    def __init__(
        self,
        agents: list[Agent],
        Adj: np.ndarray,
        N: int,
        nx_l: int,
        nu_l: int,
        rho: float,
        iters: int = 50,
    ) -> None:
        super().__init__(agents, Adj, N, nx_l, nu_l, rho, iters)

    def solve_admm(
        self,
        state: np.ndarray,
        action: cs.DM | None = None,
        deterministic: bool = True,
        action_space: Box | None = None,
    ) -> tuple[np.ndarray, list[Solution], dict[str, Any]]:
        """Solve the mpc problem for the network of agents using ADMM. If an
        action provided, the first action is constrained to be the provided action.

        Parameters
        ----------
        state: np.ndarray
            Global state of the network.
        action: np.ndarray | None = None
            Global action of the network. If None, the action is solved for.
        deterministic: bool = True
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed.
        action_space: Optional[Box]
            Only applicable if action=None. The action space of the environment. If provided, the action is clipped to
            the action space.

        Returns
        -------
        tuple[np.ndarray, list[Solution], dict[str, Any]
            A tuple containing the local actions, local solutions, and an info dictionary
        """
        u_iters = np.empty(
            (self.iters, self.n, self.nu_l, self.N)
        )  # store actions over iterations

        loc_actions = np.empty((self.n, self.nu_l))
        local_sols: list[Solution] = [None] * len(self.agents)
        x_l = np.split(state, self.n)  # split global state and action into local states
        u_l: list[np.ndarray] = np.split(action, self.n) if action is not None else []

        self.y_iter_data = np.zeros(
            (self.iters, self.n, self.nx_l * len(self.G[i]), self.N + 1)
        )  # for visualizing convergence of dual vars

        for iter in range(self.iters):
            # x update: solve local minimisations
            for i in range(len(self.agents)):
                # set parameters in augmented lagrangian
                self.agents[i].fixed_parameters["y"] = self.y[i]
                # G[i] is the agent indices relevant to agent i. Reshape stacks them in the local augmented state
                self.agents[i].fixed_parameters["z"] = self.z[self.G[i], :].reshape(
                    -1, self.N + 1
                )

                if action is None:
                    loc_actions[i], local_sols[i] = self.agents[i].state_value(
                        x_l[i], deterministic=deterministic, action_space=action_space
                    )
                else:
                    local_sols[i] = self.agents[i].action_value(x_l[i], u_l[i])
                if not local_sols[i].success:
                    # not raising an error on MPC failures
                    u_iters[iter, i] = np.nan
                    self.agents[i].on_mpc_failure(
                        episode=0, status=local_sols[i].status, raises=False, timestep=0
                    )
                else:
                    u_iters[iter, i] = local_sols[i].vals["u"]

                # construct solution to augmented state from local state and coupled states
                self.augmented_x[i] = cs.vertcat(
                    local_sols[i].vals["x_c"][: self.nx_l * self.G[i].index(i), :],
                    local_sols[i].vals["x"],
                    local_sols[i].vals["x_c"][self.nx_l * self.G[i].index(i) :, :],
                )

            # z update: an averaging of all agents' optinions on each z
            for i in range(self.n):
                self.z[i] = np.mean(
                    np.stack(
                        [
                            self.augmented_x[j][
                                self.nx_l
                                * self.G[j].index(i) : self.nx_l
                                * (self.G[j].index(i) + 1),
                                :,
                            ]
                            for j in self.G[i]
                        ]
                    ),
                    axis=0,
                )

            # y update: increment by the residual
            for i in range(self.n):
                self.y[i] = self.y[i] + self.rho * (
                    self.augmented_x[i] - self.z[self.G[i], :].reshape(-1, self.N + 1)
                )

                # save resulting y's in list
                self.y_iter_data[iter, i, :, :] = self.y[i]

        return (
            loc_actions,
            local_sols,
            {"u_iters": u_iters},
        )  # return actions and solutions from last ADMM iter
