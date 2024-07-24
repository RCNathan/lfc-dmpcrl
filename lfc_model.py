from typing import ClassVar

import casadi as cs
import numpy as np



class Model():
    """Class to store model information for the system."""

    n: ClassVar[int] = 3  # number of agents
    nx_l: ClassVar[int] = 4  # local state dimension
    nu_l: ClassVar[int] = 1  # local control dimension

    # constants (to be moved later maybe? Also: differences between areas)
    # taken from Yan et al. three area network (for now, area 1)
    Tg = 0.10 # governor time constant
    Tt = 0.40 # turbine time constant
    H = 0.0833 # synchronous machine inertia
    D = 0.0015 # damping coefficient
    R = 0.33 # speed drop
    Tij = 0.015 # T12

    # note: unchanged as of 24-7
    x_bnd_l: ClassVar[np.ndarray] = np.array(
        [[0, -1], [1, 1]]
    )  # local state bounds x_bnd[0] <= x <= x_bnd[1] 
    u_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-1], [1]]
    )  # local control bounds u_bnd[0] <= u <= u_bnd[1]

    # adj is atm set up for 3 agents, so I guess I keep it at 3 agents for now.
    adj: ClassVar[np.ndarray] = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32 #  [1, 0, 1]: connected to 1st and 3rd.
    )  # adjacency matrix of coupling in network

    # TRUE/REAL (unknown) dynamics
    # Note: this means dynamics are the same for each area (right now) 
    A_l: ClassVar[np.ndarray] = np.array(
        [[-D/(2*H), 1/(2*H), 0, -1/(2*H)], 
         [0, -1/Tt, 1/Tt, 0], 
         [-1/(R*Tg), 0, -1/Tg, 0], 
         [np.inf, 0, 0, 0]]
    )  # local state-space matrix A
    A_l[3,0] = 2 * np.pi * n * Tij # to be replaced by actual different Tij's
    B_l: ClassVar[np.ndarray] = np.array(
        [[0], [0], [1/Tg], [0]]
    )  # local state-space matrix B
    A_c_l: ClassVar[np.ndarray] = np.array(
        np.zeros((4,4))
    )  # local coupling matrix A_c
    A_c_l[3,0] = -2 * np.pi * n * Tij # to be replaced with sum T_ij

    # starting point (inaccurate guess) for learning (excluding learnable params)
    A_l_innacurate: ClassVar[np.ndarray] = np.asarray(
        [[1, 0.25], [0, 1]]
    )  # inaccurate local state-space matrix A
    B_l_innacurate: ClassVar[np.ndarray] = np.asarray(
        [[0.0312], [0.25]]
    )  # inaccurate local state-space matrix B
    A_c_l_innacurate: ClassVar[np.ndarray] = np.array(
        [[0, 0], [0, 0]]
    )  # inaccurate local coupling matrix A_c

    def __init__(self):
        """Initializes the model."""
        self.A, self.B = self.centralized_dynamics_from_local(
            [self.A_l] * self.n,
            [self.B_l] * self.n,
            [[self.A_c_l for _ in range(np.sum(self.adj[i]))] for i in range(self.n)],
        )

    def centralized_dynamics_from_local(
        self,
        A_list: list[np.ndarray | cs.SX],
        B_list: list[np.ndarray | cs.SX],
        A_c_list: list[list[np.ndarray | cs.SX]],
    ) -> tuple[np.ndarray | cs.SX, np.ndarray | cs.SX]:
        """Creates centralized representation from a list of local dynamics matrices.

        Parameters
        ----------
        A_list : list[np.ndarray | cs.SX]
            List of local state-space matrices A.
        B_list : list[np.ndarray | cs.SX]
            List of local state-space matrices B.
        A_c_list : list[list[np.ndarray | cs.SX]]
            List of local coupling matrices A_c. A_c[i][j] is coupling
            effect of agent j on agent i.

        Returns
        -------
        tuple[np.ndarray | cs.SX, np.ndarray | cs.SX]
            Global state-space matrices A and B.
        """
        if any(len(A_c_list[i]) != np.sum(self.adj[i]) for i in range(self.n)):
            raise ValueError(
                "A_c_list must have the same length as the number of neighbors."
            )

        if isinstance(A_list[0], np.ndarray):
            row_func = lambda x: np.hstack(x)
            col_func = lambda x: np.vstack(x)
            zero_func = np.zeros
        else:
            row_func = lambda x: cs.horzcat(*x)
            col_func = lambda x: cs.vertcat(*x)
            zero_func = cs.SX.zeros
        A = col_func(  # global state-space matrix A
            [
                row_func(
                    [
                        (
                            A_list[i]
                            if i == j
                            else (
                                A_c_list[i].pop(0)
                                if self.adj[i, j] == 1
                                else zero_func((self.nx_l, self.nx_l))
                            )
                        )
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        B = col_func(
            [
                row_func(
                    [
                        B_list[i] if i == j else zero_func((self.nx_l, self.nu_l))
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        return A, B


m = Model()
print("Local A matrix for one agent/area: \n", m.A_l)
print("Local B matrix for one agent/area:", m.B_l)
print("Local A_{ij} matrix for one agent/area: \n", m.A_c_l)


print("dot is for debugging :)")