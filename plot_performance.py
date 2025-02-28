import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lfc_model import Model


# file for comparing (box/whisker-plots) the performance of n-different evaluate-pkl

# i.e: def plot_performance(file_paths: List[str]) -> None:
# and then have n subplots based on how many paths are given

def plot_performance(
        file_paths: list[str],
        names: list[str],
        title_info: str = "",
        colors: list[str] = None, # optional: set colors of boxplots
        showfliers: bool = True, # toggle visibility of outliers
        logscale: bool = False, # toggle logscale on y-axis
) -> None: 
    """
    File for plotting box/whisker plots of performance for arbitrary amount of files. \\
    Data gets stored into structures: `x.shape = [file][eps][steps][states]`, i.e augmenting with a file dimension.

    stuff..
    """
    n = len(file_paths)
    # initialize data structures
    x,u,R = [],[],[]

    # for i in range(n): open and save data
    for i in range(n):
        with open(file_paths[i] + '.pkl', 'rb') as f:
            data = pickle.load(f)
            tempX = data.get("X")
            x.append(data["X"].reshape(tempX.shape[0], tempX.shape[1], -1))
            tempU = data.get("U")
            u.append(data["U"].reshape(tempU.shape[0], tempU.shape[1], -1))
            R.append(data.get("R"))


    # sanity check: to compare, all data should have identical shape.
    assert n == len(names), "Number of names does not match number of file paths"
    refshape = x[0].shape
    for i in range(n):
        assert x[i].shape == refshape, "Shapes of data do not match"

    # some sanity checks; first: avg
    numEps = x[0].shape[0]
    for i in range(n):
        print(f"Mean cost per episode for {names[i]}: {np.mean(np.sum(R[i], axis=1))}")
        print(f"with standard deviation: {np.std(np.sum(R[i], axis=1))}")

    # plot data - box/whisker plot for cost R
    R_tot = np.sum(np.asarray(R), axis=2)
    if colors == None:
        # mpl.style.use('seaborn-v0_8-dark-palette') # ggplot, fivethirtyeight, seaborn-v0_8, seaborn-v0_8-dark-palette
        plt.figure(figsize=(5, 4))
        plt.boxplot(R_tot.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4))
        bplot = ax.boxplot(R_tot.T, patch_artist=True, tick_labels=names, showfliers=showfliers, 
                           medianprops={'color': 'white', "linewidth": 1.5},
                           boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                           whiskerprops={'color': 'grey', "linewidth": 1},
                           capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.yscale("log") if logscale else None
    plt.ylabel("Cost per episode")
    plt.title(f"Performance comparison: average cost per episode | {title_info}")
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(-10, 0)


    # constraint_violations
    x_bnd_l = Model.x_bnd_l
    n = Model.n
    x_bnd = np.tile(x_bnd_l, Model.n).T
    # sum of steps in an episode that the constraints are violated for x by comparing to Model bounds
    violations_upper = x >= x_bnd[:, 1] # True if violated
    violations_lower = x <= x_bnd[:, 0] # True if violated
    violations = violations_upper | violations_lower # if either is true
    violations_per_ep = np.sum(violations, axis=(2,3)) # per episode; sum over steps and states -> new shape [file][eps]

    # plot amount of violations:
    if colors == None:
        plt.figure(figsize=(5, 4))
        plt.boxplot(violations_per_ep.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4))
        bplot = ax.boxplot(violations_per_ep.T, patch_artist=True, tick_labels=names, showfliers=showfliers,
                            medianprops={'color': 'white', "linewidth": 1.5},
                            boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                            whiskerprops={'color': 'grey', "linewidth": 1},
                            capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.yscale("log") if logscale else None
    plt.ylabel("Number of constraint violations per episode")
    plt.title(f"Performance comparison: constraint violations | {title_info}")
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(500, 0)


    # magnitude of violations
    x_up = np.maximum(x - x_bnd[:, 1], 0)
    x_down = np.maximum(x_bnd[:, 0] - x, 0)
    violations_magnitude = np.sum(x_up + x_down, axis=(2,3))

    # plot magnitude of violations:
    if colors == None:
        plt.figure(figsize=(5, 4))
        plt.boxplot(violations_magnitude.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4))
        bplot = ax.boxplot(violations_magnitude.T, patch_artist=True, tick_labels=names, showfliers=showfliers,
                            medianprops={'color': 'white', "linewidth": 1.5},
                            boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                            whiskerprops={'color': 'grey', "linewidth": 1},
                            capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.yscale("log") if logscale else None
    plt.ylabel("Magnitude of constraint violations per episode")
    plt.title(f"Performance comparison: constraint violations magnitude | {title_info}")
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(1000, 0)
    
    
    # magnitude of GRC violations (no amount, just magnitude)
    grc = Model.GRC_l
    ts = Model.ts
    x = np.asarray(x)
    # grc: |Pm(k+1) - Pm(k)|/ts <= grc  --> Pm is 2nd entry for all agents: so [1,5,9]
    x_grc = 1/ts * x[:, :, 1:, [1,5,9]] - x[:, :, :-1, [1,5,9]]
    grc_up = np.maximum(x_grc - grc, 0)
    grc_down = np.maximum(grc - x_grc, 0)
    grc_violations_magnitude = np.sum(grc_up + grc_down, axis=(2,3))
    
    # plot magnitude of GRC violations:
    if colors == None:
        # plt.figure(figsize=(4, 3), constrained_layout=True,)
        plt.figure(figsize=(5, 4))
        plt.boxplot(grc_violations_magnitude.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4))
        bplot = ax.boxplot(grc_violations_magnitude.T, patch_artist=True, tick_labels=names, showfliers=showfliers,
                            medianprops={'color': 'white', "linewidth": 1.5},
                            boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                            whiskerprops={'color': 'grey', "linewidth": 1},
                            capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    plt.yscale("log") if logscale else None
    plt.ylabel("Magnitude of GRC violations per episode")
    plt.title(f"Performance comparison: GRC violations magnitude | {title_info}")
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(500, 300)
    
    plt.show()

# colors: https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def (scroll down to the bottom) - default is X11/CSS4, other colors use pre-fix xkcd:
# plot_performance(
#     file_paths=[
#         r"evaluate_data\dmpcrl_20eps_tcl63_scenario2",
#         r"evaluate_data\dmpcrl_20eps_tdl67_scenario2",  # change for the dmpcrl once done!!    
#         r"scmpc\ipopt_scmpc_20ep_scenario_2_ns_10",
#         r"evaluate_data\ddpg_20eps_ddpg5_scenario1and2_newenv",
#     ],
#     names=[
#         "MPC-RL", # centralized !!
#         "DMPC-RL", 
#         "Sc-MPC",
#         "DDPG",
#     ],
#     colors=[
#         "xkcd:aquamarine",
#         "xkcd:azure",
#         "xkcd:blue",
#         "xkcd:darkblue",
#         # "xkcd:purple",
#     ],
#     # showfliers=False,
#     logscale=True,
#     title_info = "Scenario 2"
# )
# Done: plot violations; amount and magnitude (and separate for GRC) - based on Model bounds. (see also vis_large_eps for how-to)
# TODO: figsize, consider constrained_layout=True, etc.







# test: compare whether last or ep20 is better for tcl63 (evaluated on 10 eps)
# plot_performance(
#    file_paths=[
#         r"evaluate_data\dmpcrl_10eps_tcl63_scenario2",
#         r"evaluate_data\dmpcrl_10eps_tcl63_scenario2_bestep20"
#     ],
#     names=[
#         "tcl63_last",
#         "tcl63_ep20",
#     ],
#     title_info = "Scenario 2"
# )

# test: compare whether last or ep20 is better for ddpg6
# plot_performance(
#    file_paths=[
#         r"evaluate_data\ddpg_20eps_ddpg6_scenario0_last_scenario0",
#         r"evaluate_data\ddpg_20eps_ddpg6_scenario0_best_scenario0",
#         r"evaluate_data\ddpg_20eps_ddpg6_scenario0_bestv2_scenario0", # bestModelFlag = False -> best eval result
#     ],
#     names=[
#         "ddpg6_last",
#         "ddpg6_best",
#         "ddpg6_bestv2",
#     ],
#     title_info = "Scenario 0"
# )

# sc-mpc: scenario 0 so these are identical of course..
plot_performance(
   file_paths=[
        r"scmpc\ipopt_scmpc_2ep_scenario_0_ns_10",
        r"scmpc\ipopt_scmpc_2ep_scenario_0_ns_5",
        r"data\pkls\scmpc_test_cent_no_learning_2ep_scenario_0"
    ],
    names=[
        "scmpc10",
        "scmpc5",
        "mpcrl"
    ],
    title_info = "Scenario 0"
)

