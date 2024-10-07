import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from dmpcrl.mpc.mpc_admm import MpcAdmm
from dmpcrl.utils.solver_options import SolverOptions

from lfc_model import Model
from lfc_discretization import lfc_forward_euler_cs, lfc_zero_order_hold
from block_diag import block_diag


class LearnableMpc(Mpc[cs.SX]):
    """Abstract class for learnable MPC controllers. Implemented by centralized and distributed child classes"""

    discount_factor = 0.9

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
        self.x_bnd  = np.tile(
            model.x_bnd_l, model.n)
        self.u_bnd = np.tile(
            model.u_bnd_l, model.n
        )
        self.w_l = np.array(
            [[1e5, 1e1, 1e1, 1e1]]  # TODO: change
        )  # penalty weight for slack variables!
        self.w = np.tile(self.w_l, (1, self.n))
        self.adj = model.adj

        # standard learnable parameters dictionary for local agent - initial values
        self.learnable_pars_init_local = {
            "V0": np.zeros((1, 1)),
            "x_lb": np.reshape(
                [-0.2, -1, -1, -0.2], (-1, 1)
            ),  # how does this compare with ub/lb in model? -> this is learned.
            "x_ub": np.reshape([0.2, 1, 1, 0.2], (-1, 1)),
            "b": np.zeros(self.nx_l),
            "f": np.zeros(self.nx_l + self.nu_l),
            "Qx": np.array(
            [[1e2, 0, 0, 0], 
             [0, 1e0, 0, 0],
             [0, 0, 1e1, 0],
             [0, 0, 0, 1e-1]]), # quadratic cost on states (local)
            "Qu": np.array([0.5]),
        }


class CentralizedMpc(LearnableMpc):
    """A centralised learnable MPC controller."""

    def __init__(self, model: Model, prediction_horizon: int) -> None:
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
        b_list = [
            self.parameter(f"b_{i}", (self.nx_l, 1)) for i in range(self.n)
        ] 

        # cost parameters
        V0_list = [
            self.parameter(f"V0_{i}", (1,)) for i in range(self.n)
        ]  
        f_list = [
            self.parameter(f"f_{i}", (self.nx_l + self.nu_l, 1)) for i in range(self.n)
        ] 
        Qx_list = [
            self.parameter(f"Qx_{i}", (self.nx_l, self.nx_l)) for i in range(self.n)
        ]
        Qu_list= [
            self.parameter(f"Qu_{i}", (1,)) for i in range(self.n)
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
        self.learnable_pars_init.update(
            {f"B_{i}": B_l_inac[i] for i in range(self.n)})
        self.learnable_pars_init.update(
            {
                f"A_c_{i}_{j}": A_c_l_inac[i][j] 
                for i in range(self.n)
                for j in range(self.n)
                if self.adj[i, j] 
            }
        )
        self.learnable_pars_init.update(
            {f"F_{i}": F_l_inac[i] for i in range(self.n)})

        # concat some params for use in cost and constraint expressions
        V0 = cs.vcat(V0_list)
        x_lb = cs.vcat(x_lb_list)
        x_ub = cs.vcat(x_ub_list)
        b = cs.vcat(b_list)
        f = cs.vcat(f_list)
        Qu = cs.vcat(Qu_list)
        Qx = block_diag(*Qx_list) 
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
        s, _, _ = self.variable("s", (self.nx, N), lb=0)

        # Fixed parameters: load
        Pl = self.parameter("Pl", (3, 1))  # creates parameter obj for load
        self.fixed_pars_init = {
            "Pl": np.zeros((3, 1))
        }  # !!! initial value of 0.0 will be changed by agent on episode start, and then every env step (see lfc_agent.py) !!!

        # dynamics
        self.set_dynamics(
            # lambda x, u: A @ x + B @ u + b, n_in=2, n_out=1
            lambda x, u: A @ x + B @ u + F @ Pl + b, n_in=2, n_out=1
        )  # TODO: b is removed, maybe return later

        # other constraints
        self.constraint("x_lb", self.x_bnd[0].reshape(-1, 1) + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", self.x_bnd[1].reshape(-1, 1) + x_ub + s)

        # objective | x.shape = (nx, N+1), u.shape = (nu, N)    |   sum1 is row-sum, sum2 is col-sum
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize( 
            cs.sum1(V0) 
            # + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u)) # f'([x, u]')
            + cs.sum2(Qu.T @ u**2) # u'Q_u u
            + cs.sum2(
                cs.sum1(
                    x.T @ Qx @ x # x' Q_x x 
                )
            ) 
        )
        
        # solver
        solver = "qpoases" # qpoases or ipopt
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
        rho: float = 0.5,
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
        A_c_list = [
            self.parameter(f"A_c_{i}", (self.nx_l, self.nx_l))
            for i in range(num_neighbours)
        ]
        Qx = self.parameter("Qx", (self.nx_l, self.nx_l)) # TODO: this is placeholder, needs to be changed in stage cost
        Qu = self.parameter("Qu", (1,))

        # dictionary containing initial values for local learnable parameters
        self.learnable_pars_init = self.learnable_pars_init_local.copy()
        self.learnable_pars_init["A"] = model.A_l_inac[my_index] # TODO: this is placeholder, to be changed when tackling distributed
        self.learnable_pars_init["B"] = model.B_l_inac[my_index]
        self.learnable_pars_init.update(
            {f"A_c_{i}": model.A_c_l_inac[1][0] for i in range(num_neighbours)} # TODO: this is placeholder, to be changed when tackling distributed
        )
        self.learnable_pars_init

        # variables (state+coupling, action, slack)
        x, x_c = self.augmented_state(num_neighbours, my_index, self.nx_l)
        u, _ = self.action(
            "u",
            self.nu_l,
            lb=self.u_bnd_l[0][0],
            ub=self.u_bnd_l[1][0],
        )
        s, _, _ = self.variable("s", (self.nx_l, N), lb=0)

        x_c_list = cs.vertsplit(
            x_c, np.arange(0, self.nx_l * num_neighbours + 1, self.nx_l)
        )  # store the bits of x that are couplings in a list for ease of access

        # dynamics - added manually due to coupling
        for k in range(N):
            coup = cs.SX.zeros(self.nx_l, 1)
            for i in range(num_neighbours):  # get coupling expression
                coup += A_c_list[i] @ x_c_list[i][:, [k]]
            self.constraint(
                f"dynam_{k}",
                A @ x[:, [k]] + B @ u[:, [k]] + coup + b,
                "==",
                x[:, [k + 1]],
            )

        # other constraints
        self.constraint(f"x_lb", self.x_bnd_l[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint(f"x_ub", x[:, 1:], "<=", self.x_bnd_l[1] + x_ub + s)

        # objective
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.set_local_cost(
            V0
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + self.w_l @ s)
            )
        )

        # solver
        solver = "ipopt"
        opts = SolverOptions.get_solver_options(solver)
        self.init_solver(opts, solver=solver)


# model = Model()
# centralized_mpc = CentralizedMpc(model, prediction_horizon=10) # for comparison/debugging
# print("Debug point.")
