from typing import ClassVar

import casadi as cs
import numpy as np
from dmpcrl.utils.discretisation import zero_order_hold


class Model:
    """Class to store model information for the system."""

    print("Model instance created")
    n: ClassVar[int] = 3  # number of agents
    nx_l: ClassVar[int] = 4  # local state dimension
    nu_l: ClassVar[int] = 1  # local control dimension

    ts = 0.1  # sampling time for ZOH discretization (not sure about value)

    # Constants taken from Yan et al.'s three area network
    # Area 1
    Tg1 = 0.10  # governor time constant
    Tt1 = 0.40  # turbine time constant
    H1 = 0.0833  # synchronous machine inertia
    D1 = 0.0015  # damping coefficient
    R1 = 0.33  # speed drop
    # Area 2
    Tg2 = 0.12  # governor time constant
    Tt2 = 0.38  # turbine time constant
    H2 = 0.1000  # synchronous machine inertia
    D2 = 0.0020  # damping coefficient
    R2 = 0.28  # speed drop
    # Area 3
    Tg3 = 0.08  # governor time constant
    Tt3 = 0.35  # turbine time constant
    H3 = 0.0750  # synchronous machine inertia
    D3 = 0.0010  # damping coefficient
    R3 = 0.40  # speed drop
    # interconnection between area's (note T12 = T21)
    T12 = 0.015
    T13 = 0.02
    T23 = 0.01

    # note: changed dimensions only (physical constraints?)
    x_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-1, -1, -1, -1], [1, 1, 1, 1]]
    )  # local state bounds x_bnd[0] <= x <= x_bnd[1]
    u_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-1], [1]]
    )  # local control bounds u_bnd[0] <= u <= u_bnd[1]

    # Yan et al.'s three-area network is fully connected
    adj: ClassVar[np.ndarray] = np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=np.int32,  #  [1, 0, 1]: connected to 1st and 3rd.
    )  # adjacency matrix of coupling in network

    # TRUE/REAL (unknown) dynamics (from Liao et al.)
    # Area 1
    A_l_1: ClassVar[np.ndarray] = np.array(
        [
            [-D1 / (2 * H1), 1 / (2 * H1), 0, -1 / (2 * H1)],
            [0, -1 / Tt1, 1 / Tt1, 0],
            [-1 / (R1 * Tg1), 0, -1 / Tg1, 0],
            [np.inf, 0, 0, 0],
        ]
    )  # local state-space matrix A
    A_l_1[3, 0] = 2 * np.pi * (T12 + T13)
    B_l_1: ClassVar[np.ndarray] = np.array(
        [[0], [0], [1 / Tg1], [0]]
    )  # local state-space matrix B
    F_l_1: ClassVar[np.ndarray] = np.array(
        [[-1 / (2 * H1)], [0], [0], [0]]
    )  # local state-space matrix F

    # Area 2
    A_l_2: ClassVar[np.ndarray] = np.array(
        [
            [-D2 / (2 * H2), 1 / (2 * H2), 0, -1 / (2 * H2)],
            [0, -1 / Tt2, 1 / Tt2, 0],
            [-1 / (R2 * Tg2), 0, -1 / Tg2, 0],
            [np.inf, 0, 0, 0],
        ]
    )  # local state-space matrix A
    A_l_2[3, 0] = 2 * np.pi * (T12 + T23)
    B_l_2: ClassVar[np.ndarray] = np.array(
        [[0], [0], [1 / Tg2], [0]]
    )  # local state-space matrix B
    F_l_2: ClassVar[np.ndarray] = np.array(
        [[-1 / (2 * H2)], [0], [0], [0]]
    )  # local state-space matrix F

    # Area 3
    A_l_3: ClassVar[np.ndarray] = np.array(
        [
            [-D3 / (2 * H3), 1 / (2 * H3), 0, -1 / (2 * H3)],
            [0, -1 / Tt3, 1 / Tt3, 0],
            [-1 / (R3 * Tg3), 0, -1 / Tg3, 0],
            [np.inf, 0, 0, 0],
        ]
    )  # local state-space matrix A
    A_l_3[3, 0] = 2 * np.pi * (T13 + T23)  #
    B_l_3: ClassVar[np.ndarray] = np.array(
        [[0], [0], [1 / Tg3], [0]]
    )  # local state-space matrix B
    F_l_3: ClassVar[np.ndarray] = np.array(
        [[-1 / (2 * H3)], [0], [0], [0]]
    )  # local state-space matrix F

    # Coupling Matrices
    A12: ClassVar[np.ndarray] = np.array(
        np.zeros((4, 4))
    )  # local coupling matrix A_12 = A_21
    A12[3, 0] = -2 * np.pi * T12
    A13: ClassVar[np.ndarray] = np.array(
        np.zeros((4, 4))
    )  # local coupling matrix A_13 = A_31
    A13[3, 0] = -2 * np.pi * T13
    A23: ClassVar[np.ndarray] = np.array(
        np.zeros((4, 4))
    )  # local coupling matrix A_23 = A_32
    A23[3, 0] = -2 * np.pi * T23

    # these can be used in formulation of A_c_l
    # A21 = A12
    # A31 = A13
    # A32 = A23

    # combine into one matrix (for ease of change later)
    A_c_l = np.array(
        [
            [np.zeros((4, 4)), A12, A13],
            [A12, np.zeros((4, 4)), A23],
            [A13, A23, np.zeros((4, 4))],
        ]
    )  # zeros are placeholders/not used

    # starting point (inaccurate guess) for learning (excluding learnable params)
    np.random.seed(420)  # set seed for consistency/repeatability
    A_l_innacurate: ClassVar[np.ndarray] = A_l_1 + 0.1 * np.random.random(
        (4, 4)
    )  # inaccurate local state-space matrix A
    B_l_innacurate: ClassVar[np.ndarray] = B_l_1 + 0.1 * np.random.random(
        (4, 1)
    )  # inaccurate local state-space matrix B
    A_c_l_innacurate: ClassVar[np.ndarray] = 0.1 * np.random.random(
        (4, 4)
    )  # inaccurate local coupling matrix A_c
    F_l_innacurate: ClassVar[np.ndarray] = F_l_1 + 0.1 * np.random.random(
        (4, 1)
    )  # inaccurate local state-space matrix F TODO: maybe change name to inaccurate (optional, also for A, B, Ac, note: needs to be the same in lfc_learnable)

    # old:
    # # A_l_innacurate: ClassVar[np.ndarray] = np.asarray(
    #     [[1, 0.25], [0, 1]]
    # )  # inaccurate local state-space matrix A
    # B_l_innacurate: ClassVar[np.ndarray] = np.asarray(
    #     [[0.0312], [0.25]]
    # )  # inaccurate local state-space matrix B
    # A_c_l_innacurate: ClassVar[np.ndarray] = np.array(
    #     [[0, 0], [0, 0]]
    # )  # inaccurate local coupling matrix A_c

    # testing/debugging:
    # nx_l = 2
    # A_l_1 = np.reshape(np.arange(4),(2,2))
    # A_l_2 = np.reshape(np.arange(4,8),(2,2))
    # A_l_3 = np.reshape(np.arange(8,12),(2,2))
    # A_c_l = np.reshape(np.arange(20, 20+36), (3,3,2,2)) # list of lists; 3x3 with matrices Aij which are 2x2 for the example
    # B_l_1 = np.array([[0],[1]])
    # B_l_2 = np.array([[0],[2]])
    # B_l_3 = np.array([[0],[3]])

    def __init__(self):
        """Initializes the model."""
        self.A, self.B, self.F = self.centralized_dynamics_from_local(
            [self.A_l_1, self.A_l_2, self.A_l_3],
            [self.B_l_1, self.B_l_2, self.B_l_3],
            # [[self.A_c_l for _ in range(np.sum(self.adj[i]))] for i in range(self.n)], # original from model.py
            self.A_c_l,  # n by n matrix with coupling matrices (which are nx_l by nx_l) # works for model but not with learnable_mpc...
            [self.F_l_1, self.F_l_2, self.F_l_3],
            self.ts,
        )

    def centralized_dynamics_from_local(
        self,
        A_list: list[np.ndarray | cs.SX],
        B_list: list[np.ndarray | cs.SX],
        A_c_list: list[list[np.ndarray | cs.SX]],
        F_list: list[np.ndarray | cs.SX],
        ts: float,
    ) -> tuple[np.ndarray | cs.SX, np.ndarray | cs.SX, np.ndarray | cs.SX]:
        """Creates centralized representation from a list of local dynamics matrices.

        Parameters
        ----------
        A_list : list[np.ndarray | cs.SX]
            List of local state-space matrices A.
        B_list : list[np.ndarray | cs.SX]
            List of local state-space matrices B.
        A_c_list : list[list[np.ndarray | cs.SX]]
            List of local coupling matrices A_c: [[-, A12, A13], [...], [...]]
        F_list: list[npndarray | cs.SX]
            List of local state-space matrices F.
        ts: float
            sampling time for ZOH discretization.

        Returns
        -------
        tuple[np.ndarray | cs.SX, np.ndarray | cs.SX, np.ndarray | cs.SX]
            Global state-space matrices A and B and matrix F for load disturbance delta P.
        """
        # if any(len(A_c_list[i]) != np.sum(self.adj[i]) for i in range(self.n)):
        #     raise ValueError(
        #         "A_c_list must have the same length as the number of neighbors."
        #     )
        if isinstance(A_list[0], np.ndarray):
            row_func = lambda x: np.hstack(x)
            col_func = lambda x: np.vstack(x)
            zero_func = np.zeros
        else:  # for learnable params in lfc_learnable_mpc
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
                                A_c_list[i][j]
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
                        (
                            B_list[i] if i == j else zero_func((self.nx_l, self.nu_l))
                        )  # dim nx x nu; 4x1
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        F = col_func(
            [
                row_func(
                    [  # remember: centralized dynamics:
                        # [x1_dot] = [F1 0] [dP1]
                        # [x2_dot]   [0 F2] [dP2]
                        (
                            F_list[i] if i == j else zero_func((self.nx_l, self.nu_l))
                        )  # dim nx x nu; 4x1
                        for j in range(self.n)
                    ]
                )
                for i in range(self.n)
            ]
        )
        # return A, B, F
        # Discretization using ZOH
        B_comb = row_func((B, F))
        A_d, B_d_comb = zero_order_hold(A, B_comb, ts)
        B_d = B_d_comb[:, : self.n]
        F_d = B_d_comb[:, self.n :]
        return A_d, B_d, F_d


m = Model()
print("\nLocal A matrix for one agent/area: \n", m.A_l_1)
print("After discretization: \n", m.A[0:4, 0:4])
print("Local B matrix for one agent/area: \n", m.B_l_1)
print("Local A_{ij} matrix for one agent/area: \n", m.A_c_l[0][1])
print("dot is for debugging :)")
