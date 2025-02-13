import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model


def vis_large_eps(file: str) -> None:
    """Makes plots to visualize TD-error, rewards, states and inputs"""

    # Change filename below -> update: gets passed into visualize()
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)

    # Note that .squeeze() will get rid of the (1,) in the first index for case 1 episode, making it incompatible
    param_dict = data.get(
        "param_dict", None
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
    TD = np.asarray(data.get("TD", None)) # returns None if TD does not exist
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

    if isinstance(data, dict):
        # Find and print all the keys where the value is a list
        list_keys = [key for key, value in data.items()]
        # param_keys = [key for key, value in data.get('param_dict').items()]

        print("Keys in data:", list_keys)
        # print("Keys inside param_dict:", param_keys)
    else:
        print("The loaded data is not a dictionary.")
    learningFlag = True
    if param_dict != None:
        if (param_dict["A_0"][0] == param_dict["A_0"][-1]).all():
            print("\nNo learning of A_0")
            learningFlag = False

    m = Model()
    numAgents = m.n
    stateDim = m.nx_l
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

    # plot states of all agents
    _, axs = plt.subplots(
        5,
        3,
        constrained_layout=True,
        sharex=True,
        figsize=(8, 7.5),
    )  # figsize: (width, height)
    for j in range(numAgents):
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

        # shaded area between min and max
        for i in range(m.nx_l):
            axs[i, j].fill_between(t, xmax[:, 4 * j + i], xmin[:, 4 * j + i], color="gray", alpha=0.5, hatch="//")
        axs[4, j].fill_between(t[:-1], umax[:, j], umin[:, j], color="gray", alpha=0.5)
        # Show legend for each plot
        # axs1[0, j].legend()
        # axs1[1, j].legend()
        # axs1[2, j].legend()
        # axs1[3, j].legend()
        # axs1[4, j].legend()

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

    # only set once | y-axis labels (states, input)
    axs[0, 0].set_ylabel(r"$\Delta f_i$")
    axs[1, 0].set_ylabel(r"$\Delta P_{m,i}$")
    axs[2, 0].set_ylabel(r"$\Delta P_{g,i}$")
    axs[3, 0].set_ylabel(r"$\Delta P_{tie,i}$")
    axs[4, 0].set_ylabel(r"$u$")
    axs[3, 2].legend(loc="lower right")

    wm = (
        plt.get_current_fig_manager()
    )  # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    wm.window.move(-10, 0)

    if numEpisodes != 1 and TD.all() != None:
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

    # plot TD error, reward and cumulative reward
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True, figsize=(3, 5))
    # using agent.evaluate() to speed up in lfc_train.py, yields no TD data if not learning, so we dont plot for no learning.
    if TD.shape[1] != 0 and TD.all() != None:
        axs[0].plot(t[:-1], TDmax, linestyle="--", label="max")
        axs[0].plot(t[:-1], TDmin, linestyle="--", label="min")
        axs[0].plot(t[:-1], TD[0, :], color="green", label="first")
        axs[0].plot(t[:-1], TD[-1, :], color="black", label="last")
    axs[1].plot(t[:-1], Rmax, linestyle="--", label="max")
    axs[1].plot(t[:-1], Rmin, linestyle="--", label="min")
    axs[1].plot(t[:-1], R[0, :], color="green", label="first")
    axs[1].plot(t[:-1], R[-1, :], color="black", label="last")
    axs[2].plot(t[:-1], Rcumsum[0, :], color="green", label="first")
    axs[2].plot(t[:-1], Rcumsum[-1, :], color="black", label="last")
    axs[2].plot(t[:-1], Rcmax, linestyle="--", label="max")
    axs[2].plot(t[:-1], Rcmin, linestyle="--", label="min")

    # only set once
    axs[0].set_title("TD error")
    axs[0].set_xlabel(r"time $t$")
    axs[0].legend(loc="upper right")
    axs[1].set_title("Costs (R)")  # for different episodes
    axs[1].set_xlabel(r"time $t$")
    axs[1].legend()
    axs[2].set_title("Cumulative cost")
    axs[2].set_xlabel(r"time $t$")
    axs[2].legend()

    wm = (
        plt.get_current_fig_manager()
    )  # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    # print(wm.window.geometry()) # (x,y,dx,dy)
    # figx, figy, figdx, figdy = wm.window.geometry().getRect()
    # wm.window.setGeometry(1000, 0, figdx, figdy)
    wm.window.move(800, 0)

    # Plot TD error continously, (avg) rewards & TD per episode and evolution of learnable params over time
    TDlong = TD.reshape(
        -1,
    )  # reverting back to (1, eps*steps)
    tlong = m.ts * np.linspace(0, len(TDlong) - 1, len(TDlong))
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=False, figsize=(3, 5))
    if TD.shape[1] != 0:
        axs[0].plot(tlong, TDlong)
        axs[0].set_title("Continuous TD error (all eps)")
        axs[0].set_xlabel(r"time $t$")
    axs[1].plot(
        np.linspace(1, numEpisodes, numEpisodes), Rcumsum[:, -1], linestyle="--"
    )
    axs[1].scatter(
        np.linspace(1, numEpisodes, numEpisodes),
        Rcumsum[:, -1],
        label="cumulative cost",
    )
    if TD.shape[1] != 0 and TD.all() != None:
        axs[2].plot(
            np.linspace(1, numEpisodes, numEpisodes), np.sum(np.abs(TD), axis=1), linestyle="--"
        )
        axs[2].scatter(
            np.linspace(1, numEpisodes, numEpisodes),
            np.sum(np.abs(TD), axis=1),
            label="sum of TD error",
        )
    # axs[1].set_ylim(bottom=0)
    axs[1].set_title("Cost per episode")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylim(bottom=0, top=1.1 * np.max(Rcumsum[:, -1]))
    axs[1].margins(y=0.5)
    axs[2].set_title("TD per episode")
    axs[2].set_xlabel("Episodes")
    # axs[2].set_ylim(bottom=0, top=1.1 * np.max(np.sum(TD, axis=1)))

    wm = plt.get_current_fig_manager()
    # wm.window.move(1500,0)
    wm.window.move(1100, 0)

    if numEpisodes != 1:
        Pl = Pl[:, :, :].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)
        Pl_noise = Pl_noise[:,:,:].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)

    # Plot loads and loads + noise
    _, axs = plt.subplots(
        1, 3, constrained_layout=True, sharey=True, figsize=(3, 1.9)
    )  # figsize: (width, height)
    for j in range(numAgents):
        for n in range(min(numEpisodes, 10)): # max 10 eps
            axs[j].plot(t[:-1], Pl[n, :, j] + Pl_noise[n, :, j])  # , label='last ep.'
            axs[j].plot(t[:-1], Pl[n, :, j], linestyle="--", color="black")

        # only set once for every agent
        axs[j].set_title(
            "Agent {}".format(j + 1)
        )  # sets agent-title on every top plot only
        axs[j].set_xlabel(r"time $t$")
        # axs[j].legend()
    # axs[2].legend(loc='upper right')

    # only set once
    axs[0].set_ylabel(r"Load $\Delta P_l$")

    wm = (
        plt.get_current_fig_manager()
    )  # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    wm.window.move(1100, 560)

    # GRC plot | x: (eps, steps, states)
    grc = m.GRC_l
    fig, axs = plt.subplots(
        1, 3, constrained_layout=True, figsize=(3, 1.9), sharey=True
    )
    for n in range(m.n):
        axs[n].plot(
            t[:-1],
            1 / m.ts * (x[0, 1:, 4 * n + 1] - x[0, :-1, 4 * n + 1]),
            color="green",
        )
        axs[n].plot(
            t[:-1],
            1 / m.ts * (x[-1, 1:, 4 * n + 1] - x[-1, :-1, 4 * n + 1]),
            color="black",
        )
        axs[n].hlines(
            [-grc, grc], 0, t[-2], color="r", linestyle="--", label="GRC"
        )  # hlines(y_values, xmin, xmax)
        # axs[n].set_ylim([-1.1*grc, 1.1*grc])
        axs[n].set_title(f"Agent {n+1}")
    axs[0].set_ylabel(r"$\Delta P_{m,i}(k+1)$ - $\Delta P_{m,i}(k)$")
    axs[2].legend(loc="lower right")
    # fig.suptitle("$\Delta P_{m,i}(k+1)$ - $\Delta P_{m,i}(k)$")
    wm = plt.get_current_fig_manager()
    wm.window.move(800, 560)

    # Plot evolution of learnable parameters over time
    if TD.shape[1] != 0 and learningFlag and TD.all() != None:
        # plot for a lot (or all) of learnable params (debug-purposes)
        _, axs = plt.subplots(7, 6, constrained_layout=True, figsize=(7.5, 7))

        i = 1
        for item in param_dict.keys():
            # print(param_dict[f'{item}'])
            plt.subplot(7, 6, i)
            plotdata = param_dict[f"{item}"].squeeze()
            if len(plotdata.shape) > 2:
                plt.plot(plotdata[:, 0, 0])
            elif len(plotdata.shape) == 2:
                plt.plot(plotdata[:, 0])
            else:
                plt.plot(plotdata)
            plt.title(str(item))
            i += 1

        # removes tick-marks (for visibility)
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        wm = plt.get_current_fig_manager()
        wm.window.move(0, 0)

    # constraint_violations
    plt.figure(constrained_layout=True, figsize=(3, 1.9))
    x_bnd = np.vstack([x_bnd, x_bnd, x_bnd])
    # sum of steps in an episode that the constraints are violated for x by comparing to model bounds
    violations_upper = x >= x_bnd[:, 1] # True if violated
    violations_lower = x <= x_bnd[:, 0] # True if violated
    violations = violations_upper | violations_lower # if either is true
    violations_per_ep = np.sum(violations, axis=(1,2)) # per episode; sum over steps and states
    plt.plot(np.linspace(1, numEpisodes, numEpisodes), violations_per_ep, label='violations')
    plt.title('Constraint violations per episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of violations')
    wm = plt.get_current_fig_manager()
    wm.window.move(1100, 560)

    print("returns", Rcumsum[:, -1])
    infeasibles = data.get("infeasibles", None)
    if infeasibles != None:
        # print("MPC failures at following eps, timesteps:", infeasibles)
        print("Total infeasible steps:")
        for key in infeasibles:
            print(key, np.asarray(infeasibles[key]).shape)
    if data.get('learning_params', None) != None:
        print("learning rate", data['learning_params']['optimizer'].lr_scheduler) 
    plt.show()
    

# filename = r'data\pkls\tcl48_cent_50ep_scenario_2'
# vis_large_eps(filename)