from typing import ClassVar

import casadi as cs
import numpy as np
from dmpcrl.utils.discretisation import zero_order_hold, forward_euler
from lfc_discretization import lfc_forward_euler, lfc_zero_order_hold
from controllability import ctrb


class Model:
    """Class to store model information for the system."""

    print("Model instance created")
    discretizationFlag: ClassVar[str] = "FE" # change discretization. Options: 'ZOH' or 'FE' 
    n: ClassVar[int] = 3  # number of agents
    nx_l: ClassVar[int] = 4  # local state dimension
    nu_l: ClassVar[int] = 1  # local control dimension

    ts = 0.01  # sampling time for ZOH discretization/ Forward Euler (not sure about value)

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
        # [[-0.2, -1e3, -1e3, -1e3], [0.2, 1e3, 1e3, 1e3]]
        [[-0.2, -1, -1, -0.2], [0.2, 1, 1, 0.2]]
    )  # local state bounds x_bnd[0] <= x <= x_bnd[1]
    u_bnd_l: ClassVar[np.ndarray] = np.array(
        [[-1e-1], [1e-1]] # Yan: GRC: |u| <= 2e-4
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

    # starting point (inaccurate guess) for learning (excluding learnable params)
    np.random.seed(420)  # set seed for consistency/repeatability
    A_l_inac: ClassVar[np.ndarray[np.ndarray]] = 0 * np.random.random((3, 4, 4)) + np.array(
        [A_l_1, A_l_2, A_l_3]
    )  # inaccurate local state-space matrix A
    B_l_inac: ClassVar[np.ndarray[np.ndarray]] = 0 * np.random.random((3, 4, 1)) + np.array(
        [B_l_1, B_l_2, B_l_3]
    )  # inaccurate local state-space matrix B
    F_l_inac: ClassVar[np.ndarray[np.ndarray]] = 0 * np.random.random((3, 4, 1)) + np.array(
        [F_l_1, F_l_2, F_l_3]
    ) # inaccurate local state-space matrix F 


    # Coupling matrix: after discretizatoin: combine into one matrix (for ease of change later)
    A_c_l = np.array(
        [
            [np.zeros((4, 4)), A12, A13],
            [A12, np.zeros((4, 4)), A23],
            [A13, A23, np.zeros((4, 4))],
        ]
    )  # zeros are placeholders/not used
    A_c_l_inac: ClassVar[np.ndarray[np.ndarray[np.ndarray]]] = 0 * np.random.random((3, 3, 4, 4)) + (
        A_c_l
    )  # inaccurate local coupling matrix A_c

    def __init__(self):
        """Initializes the model."""
        self.A, self.B, self.F = self.centralized_dynamics_from_local(
            [self.A_l_1, self.A_l_2, self.A_l_3],
            [self.B_l_1, self.B_l_2, self.B_l_3],
            self.A_c_l,  # n by n matrix with coupling matrices (which are nx_l by nx_l)
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
        #  Toggle between ZOH or FE discretization
        if self.discretizationFlag == 'ZOH':
            # using Zero-Order Hold | expm(ts*M) of augmented matrix M = [A, I; 0, 0] actually is [expm(A*ts), int_0^ts(expm(A*ts)); 0, I]!
            Ad, Bd, Fd = lfc_zero_order_hold(A, B, F, ts)
            print("Using Zero-Order Hold discretization")
        elif self.discretizationFlag == 'FE':
            # using forward Euler | centralized: x+ = (I + ts*A)x + (ts*B)u | local:  xi+ = (I + ts*Ai)xi + (ts*Bi)ui + (ts*Aij)xj
            Ad, Bd, Fd = lfc_forward_euler(A, B, F, ts)
            print("Using Forward Euler discretization")
        else:
            raise Exception("No valid option for discretization given. Choose between 'ZOH' or 'FE' for Zero-Order Hold or Forward Euler, respectively.")
        return Ad, Bd, Fd

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