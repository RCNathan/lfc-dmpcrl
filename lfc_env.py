from typing import Any

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from lfc_model import Model
from block_diag import block_diag


class LtiSystem(
    gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]
):  # underlying system/simulation/ground truth
    """A discrete time network of LTI systems."""

    noise_bnd = np.array(
        [[-1e-1], [0]]
    )  # uniform noise bounds for process noise on local systems

    def __init__(self, model: Model, predicton_horizon: int) -> None:
        """Initializes the environment.

        Parameters
        ----------
        model : Model
            The model of the system.
        predicton_horizon : integer, horizon for prediction in MPC
        Needs to be passed to allow loads over prediction_horizon."""
        super().__init__()
        self.A, self.B, self.F = model.A_env, model.B_env, model.F_env
        self.n = model.n
        self.nx = model.n * model.nx_l
        self.nx_l = model.nx_l
        self.x_bnd = np.tile(model.x_bnd_l, self.n)
        self.ts_env = model.ts_env  # different sampling time for env
        self.ts = model.ts  # needed to decide how many times to run loop
        self.w = np.tile(
            [[1e3, 1e1, 1e1, 1e1]], (1, self.n)
        )  # penalty weight for bound violations
        self.w_grc = np.tile(
            [0, 1e1, 0, 0], (1, self.n)
        )  # weight on slacks for grc violation
        # note: with the way it's set up with dist_cost, I pass entire state (12) or local state (4),
        # making this the easiest way to implement
        self.grc = model.GRC_l
        self.N = predicton_horizon # to have loads over horizon for MPC's

        # Initialize step_counter, load and load_noise
        self.load = np.array([0.0, 0.0, 0.0]).reshape(self.n, -1)
        self.load_noise = self.load  # initialize also at 0
        self.step_counter = 0
        np.random.seed(1)

        # Saving data on load and noise for plotting
        self.loads: list = []
        self.load_noises: list = []
        self.loads_over_horizon = np.zeros((self.n, self.N))

        # information available in env, to be used in DDPG's augmented state/observation
        self.last_action = np.zeros((self.n, 1))
        self.last_x = np.zeros((self.nx, 1)) # store last state
        self.last_xkm1 = np.zeros((self.nx, 1)) # store previous state x(k-1)

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the environment. Gets called at the start of each episode.

        Parameters
        ----------
        seed : int, optional
            The seed for the random number generator.
        options : dict[str, Any], optional
            The options for the reset.

        Returns
        -------
        tuple[npt.NDArray[np.floating], dict[str, Any]]
            The initial state and an info dictionary."""
        super().reset(seed=seed, options=options)
        if options is not None and "x0" in options:
            self.x = options["x0"]
        else:  # Remember: n:num_agents(=3), nx_l:local_state_dim(=4), nx:n*nx_l(=12) -> reshaping is transposing
            self.x = np.hstack(
                [
                    [0.1, 0, 0, 0],  # x0 for agent 1
                    [0.1, 0, 0, 0],  # x0 for agent 2
                    [0.1, 0, 0, 0],  # x0 for agent 3
                ]
            ).reshape(self.nx, 1)

        #  Fixed parameters: time, load, load-noise
        self.step_counter = 0
        self.load = np.array([0.0, 0.0, 0.0]).reshape(
            self.n, -1
        )  
        self.load_noise = np.array([0.0, 0.0, 0.0]).reshape(
            self.n, -1
        )  
        self.loads_over_horizon = np.zeros((self.n, self.N))

        # reset previous action and next state as well
        self.last_action = np.zeros((self.n, 1))
        self.last_x = np.zeros((self.nx, 1))
        self.last_xkm1 = np.zeros((self.nx, 1))

        return self.x, {}

    def get_stage_cost(
        self,
        state: np.ndarray,
        statekp1: np.ndarray,
        action: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        w: np.ndarray,
        w_grc: np.ndarray,
        Qs: np.ndarray,
        Qa: np.ndarray,
    ) -> float:
        """Returns the stage cost of the system for a given state and action.

        Parameters
        ----------
        state : np.ndarray
            The state of the system. Shape (nx, 1).
        state+ : np.ndarray
            The state of the system at the next time-step, to calculate GRC. Shape (nx, 1)
        action : np.ndarray
            The action of the system. Shape (nu,1).
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
        w : np.ndarray
            The penalty weight for bound violations.
        w_grc: np.ndarray
            The penalty weight on GRC violations
        Qs: np.ndarray
            Matrix defining quadratic cost on states (LQR-like)
        Qa: np.ndarray
            Matrix defining quadratic cost on actions (LQR-like)

        Returns
        -------
        float
            The stage cost."""
        # replaced by ||s||^2_{Q_s} + ||a||^2_{Q_a} + w*(constraint violations)
        return float(
            state.T.squeeze() @ Qs @ state.squeeze()  # s'Qs: quadratic
            + Qa * np.square(action).sum()  # 0.5*a^2
            # necessary to punish constraint violation
            + w @ np.maximum(0, lb[:, np.newaxis] - state)  # = 0 if x > x_lower
            + w @ np.maximum(0, state - ub[:, np.newaxis])  # = 0 if x < x_upper
            + w_grc
            @ np.maximum(
                0, statekp1 - state - self.grc
            )  # = 0 if x_dot > -grc    or  -grc < x_dot
            + w_grc
            @ np.maximum(
                0, -(statekp1 - state) - self.grc
            )  # = 0 if x_dot < grc, nonzero if x_dot > grc
        )

    def get_dist_stage_cost(  # distributed
        self,
        state: np.ndarray,
        statekp1: np.ndarray,
        action: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        w: np.ndarray,
        w_grc: np.ndarray,
        Qs_l: np.ndarray,
        Qa_l: np.ndarray,
    ) -> list[float]:
        """Returns the distributed costs of the system for a given centralized state and action.

        Parameters
        ----------
        state : np.ndarray
            The centralized state of the system.
        state+ : np.ndarray
            The centralized state of the system at the next time-step.
        action : np.ndarray
            The centralized action of the system.
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
        w : np.ndarray
            The penalty weight for bound violations.
        w_grc: np.ndarray
            The penalty weight for GRC violations.
        Qs_l: np.ndarray
            LOCAL Matrix defining quadratic cost on states (LQR-like)
        Qa_l: np.ndarray
            LOCAL Matrix defining quadratic cost on actions (LQR-like)

        Returns
        -------
        list[float]
            The distributed costs."""
        x_l, x_lp1, u_l, lb_l, ub_l, w_l, w_l_grc = (
            np.split(state, self.n),
            np.split(statekp1, self.n),
            np.split(action, self.n),
            np.split(lb, self.n),
            np.split(ub, self.n),
            np.split(w, self.n, axis=1),
            np.split(w_grc, self.n, axis=1),
        )  # break into local pieces
        return [
            self.get_stage_cost(
                x_l[i],
                x_lp1[i],
                u_l[i],
                lb_l[i],
                ub_l[i],
                w_l[i],
                w_l_grc[i],
                Qs_l,
                Qa_l,
            )
            for i in range(self.n)
        ]

    def return_load(self, step_counter: int) -> np.ndarray:
        """Return loads for given timestep."""
        
        #  step function for load | time = step_counter*ts
        sim_time = step_counter * self.ts
        # with ts = 0.01
        c1, c2 = 0.1, -0.1 # 0.1 -0.1 interesting result fr, touching constraint at f1, f3, tie1, tie3
        t1, t2, t3 = 1, 2, 3

        load = np.zeros((self.n, 1))
        if sim_time < t1:
            load = np.array([0.0, 0.0, 0.0]).reshape(self.n, -1)
        elif sim_time < t2:  # from t = 10 - 20
            load = np.array([c1, 0.0, 0.0]).reshape(self.n, -1)
        elif sim_time < t3:  # from t = 20 - 30
            load = np.array([c1, 0.0, c2]).reshape(self.n, -1)
        elif sim_time < 40:
            load = np.array([c1, 0.0, c2]).reshape(self.n, -1)
        else:
            load = np.array([c1, 0.0, c2]).reshape(self.n, -1)
        return load
        

    def step(
        self, action: cs.DM
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Takes a step in the environment.

        Parameters
        ----------
        action : cs.DM
            The action to take.

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            The new state, the reward, truncated flag, terminated flag, and an info dictionary.
        """
        # Define load for current step and also over control horizon
        self.load = self.return_load(self.step_counter)
        for k in range(self.N):
            self.loads_over_horizon[:, k] = self.return_load(self.step_counter + k).squeeze()

        # noise on load | += self.F @ noise_on_load | noise is uniform and bounded (rn 0.01)
        offset = 0.05
        self.load_noise = 0.05 * (
            np.random.uniform(0, 2, (3, 1)) - 1 + offset
        )  # (low, high, size) -> in [-1, 1) -- what about [-1 + a, 1 + a)

        # self.load = np.zeros((3,1)) # to toggle load on/off
        # self.load_noise = np.zeros((3, 1))

        if type(action) == cs.DM:
            action = action.full()  # convert action from casadi DM to numpy array
        else:
            action = action.reshape((-1, 1)) # reshape (3,) -> (3,1) for matrix multiplication

        # x_new = self.A @ self.x + self.B @ action
        x = self.x
        for _ in range(int(self.ts / self.ts_env)):
            x_new = self.A @ x + self.B @ action + self.F @ self.load
            x_new += self.F @ self.load_noise
            x = x_new

        # Defines the quadratic cost on states
        Qs_l = np.array(
            [[1e2, 0, 0, 0], [0, 1e0, 0, 0], [0, 0, 1e1, 0], [0, 0, 0, 2e1]]
        )
        Qs = block_diag(Qs_l, n=self.n)
        Qa = 0.5

        r = self.get_stage_cost(
            self.x,
            x_new,
            action,
            lb=self.x_bnd[0],
            ub=self.x_bnd[1],
            w=self.w,
            w_grc=self.w_grc,
            Qs=Qs,
            Qa=Qa,
        )
        r_dist = self.get_dist_stage_cost(
            self.x,
            x_new,
            action,
            lb=self.x_bnd[0],
            ub=self.x_bnd[1],
            w=self.w,
            w_grc=self.w_grc,
            Qs_l=Qs_l,
            Qa_l=Qa,
        )

        # store to have available in env 
        self.last_xkm1 = self.x # store state as previous state 
        self.last_x = x_new # store next state 
        self.last_action = action # store last action

        # update and store for next step
        self.x = x_new
        self.step_counter += 1

        # save data for plots
        self.loads.append(self.load)
        self.load_noises.append(self.load_noise)

        return x_new, r, False, False, {"r_dist": r_dist}
