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

    def __init__(self, model: Model) -> None:
        """Initializes the environment.

        Parameters
        ----------
        model : Model
            The model of the system."""
        super().__init__()
        self.A, self.B, self.F = model.A, model.B, model.F
        self.n = model.n
        self.nx = model.n * model.nx_l
        self.nx_l = model.nx_l
        self.x_bnd = np.tile(model.x_bnd_l, self.n)
        self.ts = model.ts
        self.w = np.tile(
            [[5e2, 1e1, 1e1, 1e1]], (1, self.n)
        )  # penalty weight for bound violations

        # Initialize step_counter, load and load_noise
        self.load = np.array([0.0, 0.0, 0.0]).reshape(
            self.n, -1
        )
        self.load_noise = self.load # initialize also at 0 
        self.step_counter = 0 
        np.random.seed(1)

        # Saving data on load and noise for plotting 
        self.loads: list = []
        self.load_noises: list = []

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
            self.x = np.hstack([[0.1, 0, 0, 0], # x0 for agent 1
                                [0.1, 0, 0, 0], # x0 for agent 2
                                [0.1, 0, 0, 0], # x0 for agent 3
                                ]).reshape(self.nx, 1)

        #  Fixed parameters: time, load, load-noise
        self.step_counter = 0
        self.load = np.array([0.0, 0.0, 0.0]).reshape(
            self.n, -1
        ) # TODO: decide: does this continue past episodes? or reset each time? <- probably reset
        self.load_noise = np.array([0.0, 0.0, 0.0]).reshape(
            self.n, -1
        ) # TODO: decide: does this continue past episodes? or reset each time? <- probably continue

        return self.x, {}

    def get_stage_cost(
        self,
        state: np.ndarray,
        action: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        w: np.ndarray,
        Qs: np.ndarray,
        Qa: np.ndarray,
    ) -> float:
        """Returns the stage cost of the system for a given state and action.

        Parameters
        ----------
        state : np.ndarray
            The state of the system. Shape (nx, 1).
        action : np.ndarray
            The action of the system. Shape (nu,1).
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
        w : np.ndarray
            The penalty weight for bound violations.
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
            state.T.squeeze() @ Qs @ state.squeeze() # s'Qs: quadratic
            + Qa * np.square(action).sum() # 0.5*a^2
            # necessary to punish constraint violation
            + w @ np.maximum(0, lb[:, np.newaxis] - state) # = 0 if x > x_lower
            + w @ np.maximum(0, state - ub[:, np.newaxis]) # = 0 if x < x_upper
        )

    def get_dist_stage_cost(  # distributed
        self,
        state: np.ndarray,
        action: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        w: np.ndarray,
        Qs_l: np.ndarray,
        Qa_l: np.ndarray,
    ) -> list[float]:
        """Returns the distributed costs of the system for a given centralized state and action.

        Parameters
        ----------
        state : np.ndarray
            The centralized state of the system.
        action : np.ndarray
            The centralized action of the system.
        lb : np.ndarray
            The lower bounds of the states.
        ub : np.ndarray
            The upper bounds of the states.
        w : np.ndarray
            The penalty weight for bound violations.
        Qs_l: np.ndarray
            LOCAL Matrix defining quadratic cost on states (LQR-like)
        Qa_l: np.ndarray
            LOCAL Matrix defining quadratic cost on actions (LQR-like)

        Returns
        -------
        list[float]
            The distributed costs."""
        x_l, u_l, lb_l, ub_l, w_l = (
            np.split(state, self.n),
            np.split(action, self.n),
            np.split(lb, self.n),
            np.split(ub, self.n),
            np.split(w, self.n, axis=1),
        )  # break into local pieces
        return [
            self.get_stage_cost(x_l[i], u_l[i], lb_l[i], ub_l[i], w_l[i], Qs_l, Qa_l)
            for i in range(self.n)
        ]

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
        #  step function for load | time = step_counter*ts
        sim_time = self.step_counter*self.ts
        if(sim_time <= 10):
            self.load = np.array(
                [0.0, 0.0, 0.0]    
            ).reshape(self.n, -1)
        elif(sim_time <= 20):   # from t = 10 - 20
             self.load = 5*np.array(
                [0.01, 0.0, 0.01]    
            ).reshape(self.n, -1)
        elif(sim_time <= 30):   # from t = 20 - 30
             self.load = 5*np.array(
                [0.01, -0.01, 0.0]    
            ).reshape(self.n, -1)
        elif(sim_time <= 40):
             self.load = 5*np.array(
                [0.01, -0.01, 0.0]    
            ).reshape(self.n, -1)
        else:
            self.load = 5*np.array(
                [0.01, -0.01, -0.01]    
            ).reshape(self.n, -1)
        
        # noise on load | += self.F @ noise_on_load | noise is uniform and bounded (rn 0.01)
        self.load_noise = (0.01*(np.random.uniform(0, 2, (3,1)) -1)) # (low, high, size) -> in [-1, 1)
        
        # self.load = np.zeros((3,1)) # to toggle load on/off
        self.load_noise =  np.zeros((3,1)) 

        action = action.full()  # convert action from casadi DM to numpy array

        # x_new = self.A @ self.x + self.B @ action  
        x_new = self.A @ self.x + self.B @ action  + self.F @ self.load
        x_new += self.F @ self.load_noise 
        
        # Defines the quadratic cost on states
        Qs_l = np.array(
            [[1e4, 0, 0, 0], 
             [0, 1e0, 0, 0],
             [0, 0, 1e1, 0],
             [0, 0, 0, 2e1]])
        Qs = block_diag(Qs_l, n=self.n)
        Qa = 0.5


        r = self.get_stage_cost(
            self.x, action, lb=self.x_bnd[0], ub=self.x_bnd[1], w=self.w, Qs=Qs, Qa=Qa
        )
        r_dist = self.get_dist_stage_cost( # TODO: change for distributed setting
            self.x, action, lb=self.x_bnd[0], ub=self.x_bnd[1], w=self.w, Qs_l=Qs_l, Qa_l=Qa
        )
        self.x = x_new

        self.step_counter += 1

        # save data for plots
        self.loads.append(self.load)
        self.load_noises.append(self.load_noise)

        return x_new, r, False, False, {"r_dist": r_dist}