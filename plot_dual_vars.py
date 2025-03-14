import numpy as np
import matplotlib.pyplot as plt
from lfc_model import Model
import pickle


def plotDualVars(dist, debug):
    """
    Plots evolution of augmented state minus z for all agents,
    within one timestep, where 'admm_iters' iterations of ADMM are executed.
    Additionally, plots the sum of the differences over all timesteps and the dual variables.
    Compares the distributed solution to the centralized solution from the debug_dict.
    """

    with open(
        dist,
        "rb",
    ) as file:
        data = pickle.load(file)
    info_dict = data["info_dict"]
    with open(
        debug,
        "rb",
    ) as file:
        datac = pickle.load(file)
    debug_dict = datac["info_dict"]

    # get from model
    n = Model.n
    nx_l = Model.nx_l
    nx = nx_l * n
    N = info_dict['u_iters'].shape[-1] # control horizon

    # shapes:
    # u_iters: (self.iters, self.n, self.nu_l, self.N)
    # y_iters: [(self.iters, self.nx_l * len(self.G[i]), self.N + 1) for i in range(self.n)]
    # z_iters: (self.iters, self.n, self.nx_l, self.N + 1)
    # augmented_x_iters: [(self.iters, self.nx_l * len(self.G[i]), self.N + 1) for i in range(self.n)]

    # distributed solution
    x_aug_iters = info_dict[
        "augmented_x_iters"
    ]  # list of (iters, 12, N+1) for n agents
    y_iters = info_dict["y_iters"]  # list of (iters, 12, N+1) for n agents
    iters = x_aug_iters[0].shape[0]
    # horizon = x_aug_iters[0].shape[2]
    # numAgents = len(x_aug_iters)
    u_iters = info_dict["u_iters"]  # (iters, 3, 1, N+1)
    u_iters = u_iters.reshape((iters, 3, -1))
    z_iters = info_dict["z_iters"]  # (iters, 3, 4, N+1)
    z_iters = z_iters.reshape((iters, 12, -1))
    f_iters = info_dict["f_iters"]  # (iters, 3)
    local_actions = info_dict[
        "local_actions"
    ]  # == u_iters[-1, :, 0], i.e the last iteration of the first timestep
    local_sols = info_dict["local_sols"]

    # centralized solution
    x_opt = debug_dict["state"]
    u_opt = debug_dict["action_opt"]
    f_opt = debug_dict["cent_sol"]

    # Bad coding practice, but: since fully interconnected, we can skip aligning the augmented x's with the z: its always (12,) and sorted
    # if that ever changes: use adj or G to align the augmented x's with the z

    it = np.arange(1, iters + 1)
    _, axs = plt.subplots(3, 3, constrained_layout=True)
    for j in range(n):
        dif = np.sum(
            np.abs(x_aug_iters[j] - z_iters), axis=1
        )  # takes abs value between x-z and sums over all states -> shape (iters, timesteps)
        vars = np.sum(y_iters[j], axis=(1, 2))
        for timestep in range(N + 1):
            if timestep == 0 or timestep == 1 or timestep == N:
                axs[j, 0].plot(it, dif[:, timestep], label=f"timestep k={timestep}")
            else:
                axs[j, 0].plot(it, dif[:, timestep])
        axs[j, 0].set_title(r"|$\tilde{x}$-$\tilde{z}$|" + f" for Agent {j+1}")
        axs[j, 0].set_xlabel("ADMM iters")

        axs[0, 1].plot(it, vars - vars[-1], label=f"Agent {j+1}")
        axs[0, 1].set_title(r"Dual vars $y$-$y_{\text{final}}$")

        # plot augmented x and [... optimal x from debug]
        axs[0, 2].plot(
            it, np.sum(x_aug_iters[j][:, :, 0], axis=1), label=f"Agent {j+1}"
        )  # for each agent, sums the states of the first timestep in the horizon to compare it to the sum of the optimal states

    # optimal x from debug
    axs[0, 2].plot(
        it, np.sum(x_opt) * np.ones(it.shape), linestyle=":", color="r", label="Optimal"
    )
    # plot optimal u from debug
    axs[1, 2].plot(it, np.sum(u_iters[:, :, 0], axis=1), label=f"Agents'")  # summed
    axs[1, 2].plot(
        it, np.sum(u_opt) * np.ones(it.shape), linestyle=":", color="r", label="Optimal"
    )  # summed
    axs[2, 2].plot(it, np.sum(f_iters, axis=1), label="Agents'")  # summed
    axs[2, 2].plot(
        it, f_opt * np.ones(it.shape), linestyle=":", color="r", label="Optimal"
    )

    axs[0, 1].legend()
    axs[0, 1].set_xlabel("ADMM iters")
    axs[0, 2].legend()
    axs[0, 2].set_xlabel("ADMM iters")
    axs[0, 2].set_title(r"$\tilde{x}_i$ compared to optimal $x^*$")
    axs[1, 2].legend()
    axs[1, 2].set_title(r"$\tilde{u}$ compared to optimal $u^*$")
    axs[1, 2].set_xlabel("ADMM iters")
    axs[2, 2].legend()
    axs[2, 2].set_title(r"$\tilde{f}$ compared to optimal $f^*$")
    axs[2, 2].set_xlabel("ADMM iters")
    axs[2, 0].legend()

    # dual vars lambda for dynamics
    dist_lambda_g = np.array(
        [
            [info_dict["local_dual_vals"][i][f"lam_g_dynam_{k}"] for k in range(N)]
            for i in range(n)
        ] # shape (n, N, nx, 1)
    ).transpose(0, 2, 1, 3) # shape (n, nx, N, 1)
    dist_lambda_g = dist_lambda_g.reshape((-1, 10), order='C') # shape (n*nx, N)
    dist_lambda_g = dist_lambda_g.reshape((-1,1), order='F') # shape (n*nx*N,1)
    # cent_lambda_g = debug_dict["dual_vals"]["lam_g_dyn"] # prior to change in learnable_mpc to have dynamics as constraints
    cent_lambda_g = np.asarray(
        [
            debug_dict["dual_vals"][f"lam_g_dynam_{k}"] 
            for k in range(N)
        ]
    ).reshape((-1,1)) # to match distributed after change to dynamics in learnable_mpc

    axs[1, 1].scatter(np.arange(120), np.abs(dist_lambda_g - cent_lambda_g))
    axs[1, 1].hlines(1e-7, 0, 120, color='r', label='1e-7') # y, xmin, xmax
    axs[1, 1].set_title(
        r"Dual vars-error: |$\lambda_{\text{g\_dyn, dist}}$ - $\lambda_{\text{g\_dyn, cent}}|$"
    )
    axs[1, 1].set_xlabel("Scatterplot of final iteration")
    axs[1, 1].set_yscale('log')
    axs[1, 1].legend()

    # for i in range(10):
    #     small_lamb_dist = info_dict["local_dual_vals"][0][f"lam_g_dynam_{i}"]
    #     small_lamb_cent = debug_dict["dual_vals"]["lam_g_dyn"][4*i:4*(i+1)]
    #     print(np.hstack([small_lamb_cent, small_lamb_dist]))
    print("dynam_dual_error", np.linalg.norm(cent_lambda_g - dist_lambda_g))

    # Maximize the figure window
    manager = plt.get_current_fig_manager()
    try:
        manager.window.showMaximized()  # For most backends (Qt5, TkAgg)
    except AttributeError:
        manager.full_screen_toggle()   # Alternative for some environments
    plt.show()


# dist_dict = r"dual_vars\dist_av.pkl"
# debug_dict = r"dual_vars\centdebug_av.pkl"
# plotDualVars(dist_dict, debug_dict)
