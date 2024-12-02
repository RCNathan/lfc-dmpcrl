import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model
import os


def large_plot(file: str, optional_name=None) -> None:
    """Makes plots to visualize TD-error, rewards, states and inputs.
    This is a crude implementation to save everything in one image on a (7,4) grid."""

    # Change filename below -> update: gets passed into visualize()
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)

    # Note that .squeeze() will get rid of the (1,) in the first index for case 1 episode, making it incompatible
    param_dict = data.get(
        "param_dict"
    )  # params V0_0, ... all have len = steps   ---> shape is affected by update_strategy, i.e: if 10, only saves once every 10 steps
    x = data.get(
        "X"
    )  # shape = (4, 201, 12, 1)                 | (eps, steps, states, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    u = data.get(
        "U"
    )  # shape = (4, 201, 3, 1)                  | (eps, steps, inputs, 1)
    u = u.reshape(u.shape[0], u.shape[1], -1)  # (4, 201, 3)     | (eps, steps, inputs)
    R = data.get("R")  # shape = (4, 200)                        | (eps, steps)


    # TD now saved as agents[i].td_errors instead of agents[0], to compare to centralized. <- THIS IS WRONG!!
    # To account for that, we sum over all agents <- no. using the GAC td[0] = td[1] = td[2] = td_cent!!
    TD = np.asarray(data.get("TD")) 
    centralized_flag = data.get("cent_flag")
    if centralized_flag == False: # i.e: distributed, so TD has shape (n, eps*steps)
        # TD = np.sum(TD, axis=0) # sum over agents to shape (eps*steps) # no!
        TD =  TD[0, :] # sum over agents to shape (eps*steps)

    # bit trickier, TD, Pl and Pl_noise are reshaped later if numEps > 1
    TD = TD.reshape(
        1, -1
    )  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)
    Pl = (
        np.asarray(data.get("Pl")).squeeze().reshape(1, -1, 3)
    )  #           | (1, eps*steps, 3)
    Pl_noise = (
        np.asarray(data.get("Pl_noise")).squeeze().reshape(1, -1, 3)
    )  #           | (1, eps*steps, 3)

    learningFlag = True
    if (param_dict["A_0"][0] == param_dict["A_0"][-1]).all():
        learningFlag = False

    m = Model()
    numAgents = m.n
    u_bnd = m.u_bnd_l
    x_bnd = m.x_bnd_l.T
    x_len = (
        x.shape[1] if len(x.shape) == 3 else x.shape[0]
    )  # takes length of x   | if/else: shape of x depends on numEpisodes
    t = np.linspace(0, m.ts * (x_len - 1), x_len)  # time = sampling_time * num_samples
    numEpisodes = x.shape[0] if len(x.shape) == 3 else 1

    # get max and min of states
    xmax = np.max(x, axis=0)
    xmin = np.min(x, axis=0)
    umax = np.max(u, axis=0)
    umin = np.min(u, axis=0)

    grc = m.GRC_l
    if numEpisodes != 1:
            Pl = Pl.reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)
            Pl_noise = Pl_noise.reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)
    
    if numEpisodes != 1:
        TD = TD.reshape((numEpisodes, -1))  # newShape = (numEps, steps)
    
    # get max and min of TD, R
    TDmax = np.max(TD, axis=0)
    TDmin = np.min(TD, axis=0)
    Rmax = np.max(R, axis=0)
    Rmin = np.min(R, axis=0)
    # make cumulative sum, get max and min
    Rcumsum = np.cumsum(R, axis=1)
    Rcmax = np.max(Rcumsum, axis=0)
    Rcmin = np.min(Rcumsum, axis=0)

    TDlong = TD.reshape(
        -1,
    )  # reverting back to (1, eps*steps)
    tlong = m.ts * np.linspace(0, len(TDlong) - 1, len(TDlong))

    # Masterplot
    fig, axs = plt.subplots(
        7,
        4,
        constrained_layout=True,
        sharex=True,
        figsize=(18, 9.5),
    )  # figsize: (width, height)
    for j in range(numAgents):
        ### plot states of all agents
        axs[0, j].plot(t, xmax[:, 4 * j], linestyle="--", label="upper bound")
        axs[0, j].plot(t, xmin[:, 4 * j], linestyle="--", label="lower bound")
        axs[0, j].plot(t, x[0, :, 4 * j], color="green", label="first")
        axs[0, j].plot(t, x[-1, :, 4 * j], color="black", label="last")
        axs[1, j].plot(t, xmax[:, 4 * j + 1], linestyle="--", label="upper bound")
        axs[1, j].plot(t, xmin[:, 4 * j + 1], linestyle="--", label="lower bound")
        axs[1, j].plot(t, x[0, :, 4 * j + 1], color="green", label="first")
        axs[1, j].plot(t, x[-1, :, 4 * j + 1], color="black", label="last")
        axs[2, j].plot(t, xmax[:, 4 * j + 2], linestyle="--", label="upper bound")
        axs[2, j].plot(t, xmin[:, 4 * j + 2], linestyle="--", label="lower bound")
        axs[2, j].plot(t, x[0, :, 4 * j + 2], color="green", label="first")
        axs[2, j].plot(t, x[-1, :, 4 * j + 2], color="black", label="last")
        axs[3, j].plot(t, xmax[:, 4 * j + 3], linestyle="--", label="upper bound")
        axs[3, j].plot(t, xmin[:, 4 * j + 3], linestyle="--", label="lower bound")
        axs[3, j].plot(t, x[0, :, 4 * j + 3], color="green", label="first")
        axs[3, j].plot(t, x[-1, :, 4 * j + 3], color="black", label="last")
        axs[4, j].plot(t[:-1], umax[:, j], linestyle="--", label="upper bound")
        axs[4, j].plot(t[:-1], umin[:, j], linestyle="--", label="lower bound")
        axs[4, j].plot(t[:-1], u[0, :, j], color="green", label="first")
        axs[4, j].plot(t[:-1], u[-1, :, j], color="black", label="last")

        ### GRC plot | x: (eps, steps, states)
        axs[5, j].plot(t[:-1], 1 / m.ts*(x[0, 1:, 4 * j + 1] - x[0, :-1, 4 * j + 1]), color="green")
        axs[5, j].plot(t[:-1], 1 / m.ts*(x[-1, 1:, 4 * j + 1] - x[-1, :-1, 4 * j + 1]), color="black")
        axs[5, j].hlines(
            [-grc, grc], 0, t[-2], color="r", linestyle="--", label="GRC"
        )  # hlines(y_values, xmin, xmax)

        ### Plot loads and loads + load-noises
        axs[6, j].plot(t[:-1], Pl[0, :, j] + Pl_noise[0, :, j])
        axs[6, j].plot(t[:-1], Pl[-1, :, j] + Pl_noise[-1, :, j])
        axs[6, j].plot(t[:-1], Pl[0, :, j], linestyle="--", color="black")
        axs[6, j].plot(t[:-1], Pl[-1, :, j], linestyle="--", color="black", label='nominal load') # redundancy to check whether load stays same every ep


        # only needs to be plotted once for each agent
        axs[0, j].hlines(
            x_bnd[0, :], 0, t[-1], linestyles="--", color="r"
        )  # hlines(y_values, xmin, xmax)
        axs[0, j].set_title(
            "Agent {}".format(j + 1)
        )  # sets agent-title on every top plot only
        axs[1, j].hlines(
            x_bnd[1, :], 0, t[-1], linestyles="--", color="r"
        )  # hlines(y_values, xmin, xmax)
        axs[2, j].hlines(
            x_bnd[2, :], 0, t[-1], linestyles="--", color="r"
        )  # hlines(y_values, xmin, xmax)
        axs[3, j].hlines(
            x_bnd[3, :], 0, t[-1], linestyles="--", color="r"
        )  # hlines(y_values, xmin, xmax)
        axs[4, j].hlines(
            u_bnd, 0, t[-2], linestyles="--", color="r"
        )  # hlines(y_values, xmin, xmax)
        axs[4, j].set_xlabel(r"time $t$")  # x-labels (only bottom row)
        axs[0, j].set_ylim([-0.25, 0.25])

    ### TD, R, cumR for single eps (first, last, min, max)
    if TD.shape[1] != 0:
        axs[0, 3].plot(t[:-1], TDmax, linestyle="--", label="max")
        axs[0, 3].plot(t[:-1], TDmin, linestyle="--", label="min")
        axs[0, 3].plot(t[:-1], TD[0, :], color="green", label="first")
        axs[0, 3].plot(t[:-1], TD[-1, :], color="black", label="last")
    axs[1, 3].plot(t[:-1], Rmax, linestyle="--", label="max")
    axs[1, 3].plot(t[:-1], Rmin, linestyle="--", label="min")
    axs[1, 3].plot(t[:-1], R[0, :], color="green", label="first")
    axs[1, 3].plot(t[:-1], R[-1, :], color="black", label="last")
    axs[2, 3].plot(t[:-1], Rcumsum[0, :], color="green", label="first")
    axs[2, 3].plot(t[:-1], Rcumsum[-1, :], color="black", label="last")
    axs[2, 3].plot(t[:-1], Rcmax, linestyle="--", label="max")
    axs[2, 3].plot(t[:-1], Rcmin, linestyle="--", label="min")

    # only set once
    axs[0, 3].set_title("TD error")
    axs[0, 3].set_xlabel(r"time $t$")
    axs[0, 3].legend(loc="upper right")
    axs[1, 3].set_title("Costs (R)")  # for different episodes
    axs[1, 3].set_xlabel(r"time $t$")
    axs[1, 3].legend()
    axs[2, 3].set_title("Cumulative cost")
    axs[2, 3].set_xlabel(r"time $t$")
    axs[2, 3].legend()

    # only set once | y-axis labels (states, input)
    axs[0, 0].set_ylabel(r"$\Delta f_i$")
    axs[1, 0].set_ylabel(r"$\Delta P_{m,i}$")
    axs[2, 0].set_ylabel(r"$\Delta P_{g,i}$")
    axs[3, 0].set_ylabel(r"$\Delta P_{tie,i}$")
    axs[4, 0].set_ylabel(r"$u$")
    axs[5, 0].set_ylabel(r"$\Delta P_{m,i}(k+1)$ - $\Delta P_{m,i}(k)$")
    axs[6, 0].set_ylabel(r"Load $\Delta P_l$")
    axs[3, 2].legend(loc="lower right")
    axs[5, 2].legend(loc="lower right")
    axs[6, 2].legend(loc="lower right")


    ### Plot TD error continously, (avg) rewards & TD per episode and evolution of learnable params over time
    if TD.shape[1] != 0:
        axs[3, 3].plot(tlong, TDlong)
        axs[3, 3].set_title("Continuous TD error (all eps)")
        axs[3, 3].set_xlabel(r"time $t$")
    axs[4, 3].plot(
        np.linspace(1, numEpisodes, numEpisodes), Rcumsum[:, -1], linestyle="--"
    )
    axs[4, 3].scatter(
        np.linspace(1, numEpisodes, numEpisodes),
        Rcumsum[:, -1],
        label="cumulative cost",
    )
    if TD.shape[1] != 0:
        axs[5, 3].plot(
            np.linspace(1, numEpisodes, numEpisodes), np.sum(np.abs(TD), axis=1), linestyle="--"
        )
        axs[5, 3].scatter(
            np.linspace(1, numEpisodes, numEpisodes),
            np.sum(TD, axis=1),
            label="sum of TD error",
        )
    axs[4, 3].set_title("Cost per episode")
    axs[4, 3].set_xlabel("Episodes")
    axs[4, 3].set_ylim(bottom=0, top=1.1 * np.max(Rcumsum[:, -1]))
    axs[4, 3].margins(y=0.5)
    axs[5, 3].set_title("TD per episode")
    axs[5, 3].set_xlabel("Episodes")

    # make sure dir exists, save plot and close after
    saveloc = r'data\plots'
    os.makedirs(saveloc, exist_ok=True)
    savename = 'centralized_' if centralized_flag else 'distributed_'
    savename += 'learning' if learningFlag else 'no_learning'

    if optional_name != None:
        savename = optional_name + '_' + savename 

    plt.savefig(
        f'{saveloc}\{savename}.png',
        bbox_inches='tight'
    )
    print(f"Figure saved as {saveloc}\{savename}.png")
    plt.close()


# filename= 'cent_no_learning_1ep_scenario_1' # [960.91]
# large_plot(filename, 'test')