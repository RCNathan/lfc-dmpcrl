from typing import ClassVar

import casadi as cs
import numpy as np
from lfc_discretization import lfc_forward_euler, lfc_zero_order_hold
from controllability import ctrb


class Model:
    """Class to store model information for the system."""

    print("Model instance created")
    n: ClassVar[int] = 3  # number of agents
    nx_l: ClassVar[int] = 4  # local state dimension
    nu_l: ClassVar[int] = 1  # local control dimension

    ts = 0.01  # sampling time for ZOH discretization/ Forward Euler | 0.1 gives problems, 0.01 seems fine
    ts_env = (
        0.1 * ts
    )  # sampling time for env, which will approx the real system better.

    # noise on matrices for inaccurate guess used by learnable MPC
    noise_A = 1e-1  # default 1e0
    noise_B = 1e-2  # default 1e-1
    noise_F = 1e-2  # default 1e-1
    # noise_A, noise_B, noise_F = 0, 0, 0  # Perfect knowledge of system matrices
    ubnd = 3e-1  # 2e-1 in Zhao et al., 3e-1 in Venkat et al., 0.25 in Mohamed et al., 3e-1 in Ma et al.
    # GRC_l = 0.00017 # p.u/s in Yan et al.
    # GRC_l = 0.0017  # p.u/s in Ma et al., Zhao et al., Liao et al. <- most stable..
    GRC_l = 0.1

    # note: changed dimensions only (physical constraints?)
    x_bnd_l: ClassVar[np.ndarray] = np.array(
        # [[-0.2, -1e3, -1e3, -1e3], [0.2, 1e3, 1e3, 1e3]]
        # [[-0.2, -0.3, -0.5, -0.1], [0.2, 0.3, 0.5, 0.1]]
        [[-0.2, -1, -1, -0.1], [0.2, 1, 1, 0.1]]  # with GRC now
    )  # local state bounds x_bnd[0] <= x <= x_bnd[1]
    u_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-ubnd], [ubnd]]  # Yan: GRC: |u| <= 2e-4   =/=  input constraint!
    )  # local control bounds u_bnd[0] <= u <= u_bnd[1]

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

    # Discretize the dynamics using Forward Euler. This way, the centralized, distributed and inaccurate guesses are the same.
    A_l_1d, B_l_1d, F_l_1d = lfc_forward_euler(A_l_1, B_l_1, F_l_1, ts)
    A_l_2d, B_l_2d, F_l_2d = lfc_forward_euler(A_l_2, B_l_2, F_l_2, ts)
    A_l_3d, B_l_3d, F_l_3d = lfc_forward_euler(A_l_3, B_l_3, F_l_3, ts)
    A12d, A13d, A23d = ts * A12, ts * A13, ts * A23

    # env has different ts_env
    A_l_1_env, B_l_1_env, F_l_1_env = lfc_forward_euler(A_l_1, B_l_1, F_l_1, ts_env)
    A_l_2_env, B_l_2_env, F_l_2_env = lfc_forward_euler(A_l_2, B_l_2, F_l_2, ts_env)
    A_l_3_env, B_l_3_env, F_l_3_env = lfc_forward_euler(A_l_3, B_l_3, F_l_3, ts_env)
    A12_env, A13_env, A23_env = ts_env * A12, ts_env * A13, ts_env * A23

    # Construct coupling matrix after discretization: combine into one matrix (for ease of change later)
    A_c_ld = np.array(
        [
            [np.zeros((4, 4)), A12d, A13d],
            [A12d, np.zeros((4, 4)), A23d],
            [A13d, A23d, np.zeros((4, 4))],
        ]
    )  # and, below, for the env (different ts_env)
    A_c_l_env = np.array(
        [
            [np.zeros((4, 4)), A12_env, A13_env],
            [A12_env, np.zeros((4, 4)), A23_env],
            [A13_env, A23_env, np.zeros((4, 4))],
        ]
    )  # zeros are placeholders/not used

    # adding NOISE | starting point (inaccurate guess) for learning (excluding learnable params)
    np.random.seed(
        420
    )  # set seed for consistency/repeatability | 2*rand-1 returns uniform distribution in [-1, 1)
    A_l_inac: ClassVar[np.ndarray[np.ndarray]] = noise_A * (
        2 * np.random.random((3, 4, 4)) - 1
    ) + np.array(
        [A_l_1d, A_l_2d, A_l_3d]
    )  # inaccurate local state-space matrix A
    B_l_inac: ClassVar[np.ndarray[np.ndarray]] = noise_B * (
        2 * np.random.random((3, 4, 1)) - 1
    ) + np.array(
        [B_l_1d, B_l_2d, B_l_3d]
    )  # inaccurate local state-space matrix B
    F_l_inac: ClassVar[np.ndarray[np.ndarray]] = noise_F * (
        2 * np.random.random((3, 4, 1)) - 1
    ) + np.array(
        [F_l_1d, F_l_2d, F_l_3d]
    )  # inaccurate local state-space matrix F
    A_c_l_inac: ClassVar[np.ndarray[np.ndarray[np.ndarray]]] = 0 * np.random.random(
        (3, 3, 4, 4)
    ) + (
        A_c_ld
    )  # inaccurate local coupling matrix A_c

    def __init__(self):
        """Initializes the model."""
        self.A, self.B, self.F = self.centralized_dynamics_from_local(
            [self.A_l_1d, self.A_l_2d, self.A_l_3d],
            [self.B_l_1d, self.B_l_2d, self.B_l_3d],
            self.A_c_ld,  # n by n matrix with coupling matrices (which are nx_l by nx_l)
            [self.F_l_1d, self.F_l_2d, self.F_l_3d],
            self.ts,
        )
        self.A_env, self.B_env, self.F_env = self.centralized_dynamics_from_local(
            [self.A_l_1_env, self.A_l_2_env, self.A_l_3_env],
            [self.B_l_1_env, self.B_l_2_env, self.B_l_3_env],
            self.A_c_l_env,
            [self.F_l_1_env, self.F_l_2_env, self.F_l_3_env],
            self.ts_env,  # different sampling time for the env
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
            sampling time for ZOH discretization. [NOT USED]

        Returns
        -------
        tuple[np.ndarray | cs.SX, np.ndarray | cs.SX, np.ndarray | cs.SX]
            Global state-space matrices A and B and matrix F for load disturbance delta P.
        """
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
        return A, B, F


# m = Model()
# # print("\nLocal A matrix for one agent/area: \n", m.A_l_1)
# print("Sampling time {} s".format(m.ts))

# # controllability of (discretized) A,B:
# K, rank = ctrb(m.A, m.B)
# print("Rank of controllability matrix is", rank)
# if rank != m.A.shape[0]:
#     print("Rank is smaller than dimension, meaning system (A,B) has uncontrollable modes")
# # eigvals, eigvec = np.linalg.eig(m.A)

# print("Debug")
