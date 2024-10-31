import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model


"""Makes plots to visualize TD-error, rewards, states and inputs"""

# Change filename below -> update: gets passed into visualize()
filename = "cent_no_learning_1ep_scenario_0"  # centralized, return [460.55373678]
with open(
    filename + ".pkl",
    "rb",
) as file:
    data_cent = pickle.load(file)

filename = "distr_no_learning_1ep_scenario_0"  # distributed, return [459.15050864]
with open(
    filename + ".pkl",
    "rb",
) as file:
    data_dist = pickle.load(file)

# cent
x = data_cent.get("X")
x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
u = data_cent.get("U")
u = u.reshape(u.shape[0], u.shape[1], -1)  # (4, 200, 3)     | (eps, steps, inputs)
R = data_cent.get("R")  # shape = (4, 200)                        | (eps, steps)
TD = np.asarray(data_cent.get("TD")).reshape(
    1, -1
)  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)

# dist
x2 = data_dist.get("X")
x2 = x2.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
u2 = data_dist.get("U")
u2 = u2.reshape(u.shape[0], u.shape[1], -1)  # (4, 200, 3)     | (eps, steps, inputs)
R2 = data_dist.get("R")  # shape = (4, 200)                        | (eps, steps)
TD2 = np.asarray(data_dist.get("TD")).reshape(
    1, -1
)  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)

x_len = x.shape[1]
t = np.linspace(0, Model.ts * (x_len - 1), x_len)  # time = sampling_time * num_samples
_, axs = plt.subplots(5, 3, constrained_layout=True)
for j in range(Model.n):
    # states
    for i in range(Model.nx_l):
        axs[i, j].plot(t, x[0, :, 4 * j + i], label="cent")  # cent
        axs[i, j].plot(t, x2[0, :, 4 * j + i], label="dist", linestyle="--")  # dist
    # inputs
    axs[4, j].plot(t[:-1], u[0, :, j], label="cent")
    axs[4, j].plot(t[:-1], u2[0, :, j], label="dist", linestyle="--")
axs[0, 0].legend()

_, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(t[:-1], TD[0, :], label="cent")
axs[0].plot(t[:-1], TD2[0, :], label="dist", linestyle="--")
axs[0].set_title("TD error over time")
axs[1].plot(t[:-1], TD[0, :] - TD2[0, :], label="difference")
axs[1].set_title("TD error difference between distributed and centralized")
axs[0].legend()
axs[1].legend()

plt.show()
