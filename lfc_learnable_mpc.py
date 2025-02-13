import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from dmpcrl.mpc.mpc_admm import MpcAdmm
from dmpcrl.utils.solver_options import SolverOptions

from lfc_model import Model
from block_diag import block_diag
from csnlp.wrappers import Wrapper

class SolverTimeRecorder(Wrapper):
    """A wrapper class of that records the time taken by the solver."""

    def __init__(self, nlp: Nlp) -> None:
        super().__init__(nlp)
        self.solver_time: list[float] = []

    def solve(self, *args, **kwds):
        sol = self.nlp.solve(*args, **kwds)
        self.solver_time.append(sol.stats["t_wall_total"])
        return sol


class LearnableMpc(Mpc[cs.SX]):
    """Abstract class for learnable MPC controllers. Implemented by centralized and distributed child classes"""

    discount_factor = 1  # was 0.9

    def __init__(self, model: Model) -> None:
        """Initializes the learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model of the system.
        prediction_horizon : int
            The prediction horizon."""
        self.n = model.n
        self.ts = model.ts
        self.nx_l, self.nu_l = model.nx_l, model.nu_l
        self.nx, self.nu = model.n * model.nx_l, model.n * model.nu_l
        self.x_bnd_l, self.u_bnd_l = model.x_bnd_l, model.u_bnd_l
        self.x_bnd = np.tile(model.x_bnd_l, model.n)
        self.u_bnd = np.tile(model.u_bnd_l, model.n)
        self.w_l = np.array(
            [[1e3, 1e1, 1e1, 1e1]]  # TODO: change
        )  # penalty weight for slack variables!
        self.w = np.tile(self.w_l, (1, self.n))
        self.w_grc_l = np.array([1e1])
        self.w_grc = np.tile(self.w_grc_l, (1, self.n))
        self.adj = model.adj
        self.GRC_l = model.GRC_l
        self.GRC = np.tile(model.GRC_l, model.n).reshape(-1, 1)

        # standard learnable parameters dictionary for local agent - initial values
        self.learnable_pars_init_local = {
            "V0": np.zeros((1, 1)),
            "x_lb": np.reshape(
                # [-0.2, -0.3, -0.1, -0.1], (-1, 1)
                [0, 0, 0, 0],
                (-1, 1),
            ),  # how does this compare with ub/lb in model? -> this is learned. TODO: check below statement
            # I think makes more sense to put at 0 initially? automatica: [0, -1]' + x_lb - sigma <= x <= [1, 1]' + x_ub + sigma
            "x_ub": np.reshape(
                # [0.2, 0.3, 0.1, 0.1], (-1, 1)),
                [0, 0, 0, 0],
                (-1, 1),
            ),
            "b": np.zeros(self.nx_l),
            "f": np.zeros(self.nx_l + self.nu_l),
            "Qx": np.array(
                [[1e2, 0, 0, 0], [0, 1e0, 0, 0], [0, 0, 1e1, 0], [0, 0, 0, 2e1]]
            ),  # quadratic cost on states (local)
            "Qu": np.array([0.5]),
            "Qf": np.array(
                [[1e2, 0, 0, 0], [0, 1e0, 0, 0], [0, 0, 1e1, 0], [0, 0, 0, 2e1]]
            ),  # quadratic terminal penalty
        }


class CentralizedMpc(LearnableMpc):
    """A centralised learnable MPC controller."""

    def __init__(self, model: Model, prediction_horizon: int, solver: str = "qpoases",) -> None:
        """Initializes the centralized learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model of the system.
        prediction_horizon : int
            The prediction horizon."""
        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        Mpc.__init__(self, nlp, prediction_horizon)
        LearnableMpc.__init__(self, model)

        # renaming them for ease of use
        N = prediction_horizon
        gamma = self.discount_factor

        # create MPC parameters - parameters are learned but not optimized over during MPC
        # dynamics parameters
        A_list = [
            self.parameter(f"A_{i}", (self.nx_l, self.nx_l)) for i in range(self.n)
        ]
        B_list = [
            self.parameter(f"B_{i}", (self.nx_l, self.nu_l)) for i in range(self.n)
        ]
        F_list = [
            self.parameter(f"F_{i}", (self.nx_l, self.nu_l)) for i in range(self.n)
        ]
        # if no coupling between i and j, A_c_list[i, j] = None, otherwise we add a parameterized matrix
        A_c_list = [
            [
                (
                    self.parameter(f"A_c_{i}_{j}", (self.nx_l, self.nx_l))
                    if self.adj[i, j]
                    else cs.SX.zeros((self.nx_l, self.nx_l))
                )
                for j in range(self.n)
            ]
            for i in range(self.n)
        ]
        b_list = [self.parameter(f"b_{i}", (self.nx_l, 1)) for i in range(self.n)]

        # cost parameters
        V0_list = [self.parameter(f"V0_{i}", (1,)) for i in range(self.n)]
        f_list = [
            self.parameter(f"f_{i}", (self.nx_l + self.nu_l, 1)) for i in range(self.n)
        ]
        Qx_list = [
            self.parameter(f"Qx_{i}", (self.nx_l, self.nx_l)) for i in range(self.n)
        ]
        Qu_list = [self.parameter(f"Qu_{i}", (1,)) for i in range(self.n)]
        Qf_list = [
            self.parameter(f"Qf_{i}", (self.nx_l, self.nx_l)) for i in range(self.n)
        ]

        # constraints parameters
        x_lb_list = [self.parameter(f"x_lb_{i}", (self.nx_l,)) for i in range(self.n)]
        x_ub_list = [self.parameter(f"x_ub_{i}", (self.nx_l,)) for i in range(self.n)]

        # initial values for learnable parameters
        A_l_inac, B_l_inac, A_c_l_inac, F_l_inac = (
            model.A_l_inac,
            model.B_l_inac,
            model.A_c_l_inac,
            model.F_l_inac,
        )

        # using .update: sets the initialized theta's to some values, for all learnable params with a name in learnable_pars_init (defined in LearnableMPC)
        self.learnable_pars_init = {
            f"{name}_{i}": val
            for name, val in self.learnable_pars_init_local.items()
            for i in range(self.n)
        }
        self.learnable_pars_init.update(
            {f"A_{i}": A_l_inac[i] for i in range(self.n)}
        )  # different inaccurate guesses now
        self.learnable_pars_init.update({f"B_{i}": B_l_inac[i] for i in range(self.n)})
        self.learnable_pars_init.update(
            {
                f"A_c_{i}_{j}": A_c_l_inac[i][j]
                for i in range(self.n)
                for j in range(self.n)
                if self.adj[i, j]
            }
        )
        self.learnable_pars_init.update({f"F_{i}": F_l_inac[i] for i in range(self.n)})

        # concat some params for use in cost and constraint expressions
        V0 = cs.vcat(V0_list)
        x_lb = cs.vcat(x_lb_list)
        x_ub = cs.vcat(x_ub_list)
        b = cs.vcat(b_list)
        f = cs.vcat(f_list)
        Qu = block_diag(*Qu_list)
        Qx = block_diag(*Qx_list)
        Qf = block_diag(*Qf_list)
        # -> uses custom block_diag func and *unpacking operator; == block_diag(Qx_list[0], Qx_list[1], Qx_list[2])

        # get centralized symbolic dynamics
        A, B, F = model.centralized_dynamics_from_local(
            A_list, B_list, A_c_list, F_list, self.ts
        )  # A_c_list has zero's in formulation too.

        # variables (state, action, slack) | optimized over in mpc
        x, _ = self.state("x", self.nx)
        u, _ = self.action(
            "u",
            self.nu,
            lb=self.u_bnd[0].reshape(-1, 1),
            ub=self.u_bnd[1].reshape(-1, 1),
        )
        # d = self.disturbance("d", size=self.nx) # TODO: implement load this way?
        s, _, _ = self.variable("s", (self.nx, N), lb=0)
        s_grc, _, _ = self.variable("s_grc", (3, N), lb=0)

        # Fixed parameters: load
        Pl = self.parameter("Pl", (3, N))  # creates parameter obj for load
        self.fixed_pars_init = {
            "Pl": np.zeros((3, N))
        }  # !!! initial value of 0.0 will be changed by agent on episode start, and then every env step (see lfc_agent.py) !!!

        # add dynamics manually due dynamic load over horizon
        for k in range(N):
            self.constraint(
                f"dynam_{k}",
                A @ x[:, [k]] + B @ u[:, [k]] + F @ Pl[:, [k]] + b, 
                "==",
                x[:, [k + 1]],
            ) # setting Pl[:, [0]] is identical to previous implementation where load is not tracked over N


        # other constraints
        self.constraint("x_lb", self.x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", self.x_bnd[1].reshape(-1, 1) + x_ub + s)
        self.constraint(
            "GRC_lb",
            -self.GRC - s_grc,
            "<=",
            1 / self.ts * (x[[1, 5, 9], 1:] - x[[1, 5, 9], :-1]),
            soft=False,
        )  # generation-rate constraint
        self.constraint(
            "GRC_ub",
            1 / self.ts * (x[[1, 5, 9], 1:] - x[[1, 5, 9], :-1]),
            "<=",
            self.GRC + s_grc,
            soft=False,
        )  # generation-rate constraint

        # objective | x.shape = (nx, N+1), u.shape = (nu, N)    |   sum1 is row-sum, sum2 is col-sum
        x_stacked = x.reshape((-1, 1)) # shape: [x0', x1', .. xN']'
        u_stacked = u.reshape((-1, 1))
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            cs.sum1(V0)
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))  # f'([x, u]')
            # + x_stacked.T @ block_diag(Qx, n=N+1) @ x_stacked
            + x_stacked[:-self.nx].T @ block_diag(Qx, n=N) @ x_stacked[:-self.nx] # x' Q_x x, stage cost
            + x_stacked[-self.nx:].T @ Qf @ x_stacked[-self.nx:] # x(N)' Q_f x(N), terminal cost
            + u_stacked.T @ block_diag(Qu, n=N) @ u_stacked 
            + cs.sum2(self.w @ s)  # punishes slack variables
            + cs.sum2(self.w_grc @ s_grc)  # punishes slacks on grc
            # + cs.sum1(cs.sum2(x**2))
        )

        # solver
        # solver = "qpoases"  # qpoases or ipopt
        solver = solver # qpoases or ipopt
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)


class LocalMpc(MpcAdmm, LearnableMpc):
    """Local learnable MPC."""

    def __init__(
        self,
        model: Model,
        prediction_horizon: int,
        num_neighbours: int,
        my_index: int,
        global_index: int,
        G: list[list],
        rho: float = 0.5,
        solver: str = "qpoases",
    ) -> None:
        """Initializes the local learnable MPC controller.

        Parameters
        ----------
        model : Model
            The model object containing system information.
        prediction_horizon : int
            The prediction horizon.
        num_neighbours : int
            The number of neighbours for the agent.
        my_index : int
            The index of the agent within its local augmented state.
        rho : float, optional
            The ADMM penalty parameter, by default 0.5.
        """
        N = prediction_horizon
        gamma = self.discount_factor
        self.rho = rho

        nlp = Nlp[cs.SX]()  # optimization problem object for MPC
        LearnableMpc.__init__(self, model)
        MpcAdmm.__init__(self, nlp=nlp, prediction_horizon=prediction_horizon)

        # MPC parameters
        V0 = self.parameter("V0", (1,))
        x_lb = self.parameter("x_lb", (self.nx_l,))
        x_ub = self.parameter("x_ub", (self.nx_l,))
        b = self.parameter("b", (self.nx_l, 1))
        f = self.parameter("f", (self.nx_l + self.nu_l, 1))
        A = self.parameter("A", (self.nx_l, self.nx_l))
        B = self.parameter("B", (self.nx_l, self.nu_l))
        F = self.parameter("F", (self.nx_l, self.nu_l))
        A_c_list = [
            self.parameter(f"A_c_{j}", (self.nx_l, self.nx_l))
            for j in G[global_index]
            if j != global_index
        ]
        Qx = self.parameter("Qx", (self.nx_l, self.nx_l))
        Qf = self.parameter("Qf", (self.nx_l, self.nx_l))
        Qu = self.parameter("Qu", (1,))

        # dictionary containing initial values for local learnable parameters
        self.learnable_pars_init = (
            self.learnable_pars_init_local.copy()
        )  # copies from the LearnableMPC; V0, b, f, etc.
        self.learnable_pars_init["A"] = model.A_l_inac[global_index]
        self.learnable_pars_init["B"] = model.B_l_inac[global_index]
        self.learnable_pars_init["F"] = model.F_l_inac[global_index]
        self.learnable_pars_init.update(
            # gets Ac[i][j] from model, when i=/=j, (i,j) in adj(i,j), and where i is local MPCs own global index.
            {
                f"A_c_{j}": model.A_c_l_inac[global_index][j]
                for j in G[global_index]
                if j != global_index
            }  # note: name corresponds to neighbours numbers, i.e for agent 1 (i = 0), only Ac1, Ac2 exist
        )

        # Fixed parameters: load
        Pl = self.parameter("Pl", (1, N))
        self.fixed_pars_init.update({ 
            "Pl": np.zeros((1, N))
            })  # value changed in lfc_agent

        # variables (state+coupling, action, slack)
        x, x_c = self.augmented_state(num_neighbours, my_index, self.nx_l)
        u, _ = self.action(
            "u",
            self.nu_l,
            lb=self.u_bnd_l[0][0],
            ub=self.u_bnd_l[1][0],
        )
        s, _, _ = self.variable("s", (self.nx_l, N), lb=0)
        s_grc, _, _ = self.variable("s_grc", (1, N), lb=0)

        x_c_list = cs.vertsplit(
            x_c, np.arange(0, self.nx_l * num_neighbours + 1, self.nx_l)
        )  # store the bits of x that are couplings in a list for ease of access

        # dynamics - added manually due to coupling
        # coup = [cs.SX.zeros(self.nx_l, 1) for _ in range(N)]
        for k in range(N):  # N: horizon
            coup = cs.SX.zeros(self.nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A @ x[:, [k]] + B @ u[:, [k]] + F @ Pl[:, [k]] + coup + b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints
        self.constraint(f"x_lb", self.x_bnd_l[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", self.x_bnd_l[1] + x_ub + s)
        self.constraint(
            "GRC_lb",
            -self.GRC_l - s_grc,
            "<=",
            1 / self.ts * (x[1, 1:] - x[1, :-1]),
            soft=False,
        )  # grc constraint
        self.constraint(
            "GRC_ub",
            1 / self.ts * (x[1, 1:] - x[1, :-1]),
            "<=",
            self.GRC_l + s_grc,
            soft=False,
        )  # grc constraint

        # objective
        x_stacked = x.reshape((-1, 1)) # shape: check that [x0', x1', .. xN']'
        u_stacked = u.reshape((-1, 1))
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            # + x_stacked.T @ block_diag(Qx, n=N+1) @ x_stacked
            + x_stacked[:-self.nx_l].T @ block_diag(Qx, n=N) @ x_stacked[:-self.nx_l] # x' Q_x x, stage cost
            + x_stacked[-self.nx_l:].T @ Qf @ x_stacked[-self.nx_l:] # x(N)' Q_f x(N), terminal cost
            + u_stacked.T @ block_diag(Qu, n=N) @ u_stacked 
            + cs.sum2(self.w_l @ s)  # punishes slack variables
            + cs.sum2(self.w_grc_l @ s_grc)  # punishes slacks on grc
            # + cs.sum1(cs.sum2(x**2))
        )

        # solver
        # solver = "qpoases"
        solver = solver # qpoases or ipopt
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)