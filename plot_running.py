import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model


def plotRunning(running_filename):
    """Makes plots to visualize states during running. Note: manually change the centralized counterpart."""

    filename = "cent_no_learning_1ep_scenario_0"  # centralized, return [460.55373678]
    filename = "cent_no_learning_1ep_scenario_0.1"  # TODO: use centralized_debug's validation sol, but for now, manual :(
    with open(
        filename + ".pkl",
        "rb",
    ) as file:
        data_cent = pickle.load(file)

    # filename = 'running_pkls\ep0timestep60'
    with open(
        f"running_pkls\{running_filename}" + ".pkl",
        "rb",
    ) as file:
        data_dist = pickle.load(file)

    # cent
    x = data_cent.get("X")
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    u = data_cent.get("U")
    u = u.reshape(u.shape[0], u.shape[1], -1)  # (4, 201, 3)     | (eps, steps, inputs)
    R = data_cent.get("R")  # shape = (4, 200)                        | (eps, steps)
    TD = np.asarray(data_cent.get("TD")).reshape(
        1, -1
    )  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)

    # dist
    x2 = data_dist.get("X")
    u2 = data_dist.get("U")
    R2 = data_dist.get("R")
    TD2 = np.asarray(data_dist.get("TD")).reshape(1, -1)
    if str(x2[0]) == "deque([])":  # first ep uncompleted
        x2 = np.array(x2[1]).reshape((1, np.array(x2[1]).shape[0], -1))
        u2 = np.array(u2[1]).reshape(1, -1, 3)
        R2 = np.array(R2[1])
    else:
        x2 = np.array(x2[0])
        u2 = np.array(u2[0])
        R2 = np.array(R2[0])

    x_len = x2.shape[1]
    t = np.linspace(
        0, Model.ts * (x_len - 1), x_len
    )  # time = sampling_time * num_samples
    _, axs = plt.subplots(5, 4, constrained_layout=True, figsize=(15, 7.5))
    for j in range(Model.n):
        # states
        for i in range(Model.nx_l):
            axs[i, j].plot(t, x[0, :x_len, 4 * j + i], label="cent")  # cent
            axs[i, j].plot(t, x2[0, :, 4 * j + i], label="dist", linestyle="--")  # dist
        # inputs
        axs[4, j].plot(t[:-1], u[0, : x_len - 1, j], label="cent")
        axs[4, j].plot(t[:-1], u2[0, :, j], label="dist", linestyle="--")  # dist
    axs[0, 0].legend()
    # TD
    # after changing to use agent.evaluate() for non-learning; TD is non-existent in that case
    if TD.shape[1] != 0 and TD2.shape[1] != 0:
        axs[0, 3].plot(t[:-1], TD[0, : x_len - 1], label="cent")
        axs[0, 3].plot(t[:-1], TD2[0, :], label="dist", linestyle="--")
        axs[0, 3].set_title("TD error over time")
        axs[1, 3].plot(t[:-1], TD[0, : x_len - 1] - TD2[0, :], label="difference")
        axs[1, 3].set_title("TD error difference between distributed and centralized")
        axs[0, 3].legend()
        axs[1, 3].legend()

    # plt.get_current_fig_manager().full_screen_toggle()
    # plt.show()
    info = data_dist.get("info")
    admm, gac = info["admm_iters"], info["consensus_iters"]
    saveloc = r"running_pkls\figs"
    savename = r"\admm" + f"{admm}_gac{gac}_{running_filename}"
    plt.savefig(
        saveloc + ".png",
        bbox_inches="tight",
    )  # save so that it can continue running
    print(f"File {savename} saved in {saveloc}")
    # plt.close() # figures are retained until explicitly closed and may consume too much memory TODO: check out
    # print(f"\nADMM iterations used: {admm}, Consensus iterations used:{gac}")


# filename = 'ep0timestep10'
# filename = 'ep0timestep60'
# filename = 'ep0timestep500'
filename = 'ep0timestep10'
plotRunning(filename)
