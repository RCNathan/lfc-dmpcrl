import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lfc_model import Model
from matplotlib.ticker import LogLocator, LogFormatterSciNotation


# file for comparing (box/whisker-plots) the performance of n-different evaluate-pkl

# i.e: def plot_performance(file_paths: List[str]) -> None:
# and then have n subplots based on how many paths are given

def print_performance_vals(
        file_paths: list[str],
        names: list[str],
) -> None: 
    """
    Prints values of performance metrics for each file in file_paths.
    """
    n = len(file_paths)
    # initialize data structures
    x,u,R,solve_times = [],[],[],[]

    # for i in range(n): open and save data
    for i in range(n):
        with open(file_paths[i] + '.pkl', 'rb') as f:
            data = pickle.load(f)
            tempX = data.get("X")
            x.append(data["X"].reshape(tempX.shape[0], tempX.shape[1], -1))
            tempU = data.get("U")
            u.append(data["U"].reshape(tempU.shape[0], tempU.shape[1], -1))
            R.append(data.get("R"))

    # constraint_violations
    x_bnd_l = Model.x_bnd_l
    n = Model.n
    x_bnd = np.tile(x_bnd_l, Model.n).T

    # magnitude of violations
    x_up = np.maximum(x - x_bnd[:, 1], 0)
    x_down = np.maximum(x_bnd[:, 0] - x, 0)
    violations_magnitude = np.sum(x_up + x_down, axis=(2,3))

    for i in range(len(file_paths)):
        print(f"{names[i]}: Cost: {np.mean(np.sum(R[i], axis=1))} +- {np.std(np.sum(R[i], axis=1))} --- Violation magnitudes: {np.mean(violations_magnitude[i])} +- {np.std(violations_magnitude[i])}")

def print_solver_times(
        file_paths: list[str],
        names: list[str],
) -> None:
    
    """
    Prints values of performance metrics for each file in file_paths.
    """
    n = len(file_paths)
    # initialize data structures
    solve_times, mean_solve_times, x_shape = [],[],None

    # for i in range(n): open and save data
    for i in range(n):
        with open(file_paths[i] + '.pkl', 'rb') as f:
            data = pickle.load(f)
            solver_time = data.get('solver_times', None) # mpcrl, dmpcrl
            if solver_time is None:
                solver_time = data.get('elapsed_time', None) # ddpg
                if type(solver_time) == float: # ddpg
                    x_shape = data['X'].shape
                if solver_time is None:
                    solver_time = data['meta_data'].get("solver_time", None) # scmpc
                    if solver_time is None:
                        raise ValueError("No solver times found") # or older version
        solve_times.append(solver_time)
    
    for i in range(n):
        if type(solve_times[i]) == float: # ddpg
            mean_solve_time = solve_times[i] / (x_shape[0]*(x_shape[1]-1))
        elif len(solve_times[i].shape) == 1: # mpcrl and sc-mpc
            mean_solve_time = np.mean(solve_times[i])
        elif len(solve_times[i].shape) == 2: # dmpcrl
            mean_solve_time = np.sum(np.max(solve_times[2], axis=0))
        else:
            raise ValueError("Unknown shape of solver times")
        mean_solve_times.append(mean_solve_time)
        print(f"{names[i]}: Mean solve time per step: {mean_solve_time:.4f} s")

def plot_performance( 
        file_paths: list[str],
        names: list[str],
        title_info: str = "",
        colors: list[str] = None, # optional: set colors of boxplots
        showfliers: bool = True, # toggle visibility of outliers
        logscale: bool = False, # toggle logscale on y-axis
        plotGRC_violations: bool = False, # toggle plotting of GRC violations
        y_lim_tuple: tuple = None, # set y-limits for cost
        y_lim_constraints: tuple = None, # set y-limits for constraint violations
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

    # constraint_violations
    x_bnd_l = Model.x_bnd_l
    n = Model.n
    x_bnd = np.tile(x_bnd_l, Model.n).T

    # magnitude of violations
    x_up = np.maximum(x - x_bnd[:, 1], 0)
    x_down = np.maximum(x_bnd[:, 0] - x, 0)
    violations_magnitude = np.sum(x_up + x_down, axis=(2,3))

    # magnitude of GRC violations (no amount, just magnitude)
    grc = Model.GRC_l
    ts = Model.ts
    x = np.asarray(x)
    # grc: |Pm(k+1) - Pm(k)|/ts <= grc  --> Pm is 2nd entry for all agents: so [1,5,9]
    x_grc = 1/ts * x[:, :, 1:, [1,5,9]] - x[:, :, :-1, [1,5,9]]
    grc_up = np.maximum(x_grc - grc, 0)
    grc_down = np.maximum(grc - x_grc, 0)
    grc_violations_magnitude = np.sum(grc_up + grc_down, axis=(2,3))

    # plot data - box/whisker plot for cost R
    R_tot = np.sum(np.asarray(R), axis=2)
    if colors == None:
        # mpl.style.use('seaborn-v0_8-dark-palette') # ggplot, fivethirtyeight, seaborn-v0_8, seaborn-v0_8-dark-palette
        plt.figure(figsize=(5, 4))
        plt.boxplot(R_tot.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        bplot = ax.boxplot(R_tot.T, patch_artist=True, tick_labels=names, showfliers=showfliers, 
                           medianprops={'color': 'white', "linewidth": 1.5},
                           boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                           whiskerprops={'color': 'grey', "linewidth": 1},
                           capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_ylabel(r"$J_\text{eval}$")
    ax.set_title(f"Cost per episode | {title_info}")
    if logscale:
        ax.set_yscale("log")  # Set log scale on y-axis
        # ax.set_ylabel("Cost per episode (log)")
    if y_lim_tuple != None:
        ax.set_ylim(y_lim_tuple)  # 9*10**2, 2*10**4
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(-10, 0)

    # plot magnitude of violations:
    if colors == None:
        plt.figure(figsize=(5, 4))
        plt.boxplot(violations_magnitude.T, labels=names, showfliers=showfliers)
    else:
        _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        bplot = ax.boxplot(violations_magnitude.T, patch_artist=True, tick_labels=names, showfliers=showfliers,
                            medianprops={'color': 'white', "linewidth": 1.5},
                            boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                            whiskerprops={'color': 'grey', "linewidth": 1},
                            capprops={'color': 'black', "linewidth": 1},)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    ax.set_ylabel(r"$\eta$")
    ax.set_title(f"Constraint violations magnitude | {title_info}")
    if logscale:
        ax.set_yscale("log")  # Set log scale on y-axis
        # ax.set_ylabel("Magnitude of constraint violations per episode (log)")
    if y_lim_constraints != None:
        ax.set_ylim(y_lim_constraints)  # 9*10**2, 2*10**4
    wm = plt.get_current_fig_manager() # move figure over
    wm.window.move(500, 0) 
    
    # plot magnitude of GRC violations:
    if plotGRC_violations:
        if colors == None:
            # plt.figure(figsize=(4, 3), constrained_layout=True,)
            plt.figure(figsize=(5, 4))
            plt.boxplot(grc_violations_magnitude.T, labels=names, showfliers=showfliers)
        else:
            _, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
            bplot = ax.boxplot(grc_violations_magnitude.T, patch_artist=True, tick_labels=names, showfliers=showfliers,
                                medianprops={'color': 'white', "linewidth": 1.5},
                                boxprops={'facecolor': 'C0', "edgecolor": "white", "linewidth": 0.5},
                                whiskerprops={'color': 'grey', "linewidth": 1},
                                capprops={'color': 'black', "linewidth": 1},)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
        plt.yscale("log") if logscale else None
        if logscale:
            plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))  # Major ticks at powers of 10
            plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))  # Minor ticks in between
            # Set a formatter to show labels
            plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation())  # Uses scientific notation (10^x format)
        plt.ylabel("Magnitude of GRC violations per episode")
        plt.title(f"Performance comparison: GRC violations magnitude | {title_info}")
        wm = plt.get_current_fig_manager() # move figure over
        wm.window.move(1000, 0)
    
    plt.show()

# Names of methods (and colors) are constant
names=[
        "MPC-RL", # centralized !!
        "DMPC-RL", 
        "Sc-MPC",
        "DDPG",
]
# colors: https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def (scroll down to the bottom) - default is X11/CSS4, other colors use pre-fix xkcd:
colors=[
        "xkcd:aquamarine",
        "xkcd:azure",
        "xkcd:blue",
        "xkcd:darkblue",
] 

### scenario 0  ### -> Only 1 episode needed; not included as box/whiskers: each episode is identical (no uncertainties)
files_scenario_0 = [
    r"evaluate_data\dmpcrl_2eps_tcl0_scenario0",
    r"evaluate_data\dmpcrl_2eps_tdl0_scenario0",
    r"scmpc\ipopt_scmpc_2ep_scenario_0_ns_10",        
    r"evaluate_data\ddpg_2eps_ddpg6_scenario0_bestv2_scenario0",
]
# print_solver_times(
#     file_paths=files_scenario_0,
#     names=names,
# )
# print_performance_vals(
#     file_paths=files_scenario_0,
#     names=names,
# )
# plot_performance(
#     file_paths=files_scenario_0,
#     names=names,
#     colors=colors,
#     # showfliers=False,
#     # logscale=True,
#     title_info = "Scenario 0"
# )

### scenario 1 ###
files_scenario_1=[
    r"evaluate_data\dmpcrl_20eps_tcl13_scenario1",
    r"evaluate_data\dmpcrl_20eps_tdl19_scenario1",
    r"scmpc\ipopt_scmpc_20ep_scenario_1_ns_10",
    r"evaluate_data\ddpg_20eps_ddpg5_scenario1and2_newenv", # did not have solver-time oops. 
    # r"evaluate_data\ddpg_20eps_ddpg5_sc_1_and_2_scenario2", # <- somehow they are not the same, just use this one for solve time only.
]
# print_solver_times(
#     file_paths=files_scenario_1,
#     names=names,
# )
# print_performance_vals(
#     file_paths=files_scenario_1,
#     names=names,
# )
# plot_performance(
#     file_paths=files_scenario_1,
#     names=names,
#     colors=colors,
#     # showfliers=False,
#     logscale=True,
#     y_lim_tuple = (9*10**2, 1.1*10**4), # for the log scaling (manually skip one outlier)
#     # y_lim_tuple = (0, 10**4), # for regular scale (manually skip one outlier)
#     y_lim_constraints = (-0.2, 5.2), # for regular scale (manually skip one outlier)
#     title_info = "Scenario 1"
# )


### scenario 2 ###
files_scenario_2 = [
    r"evaluate_data\dmpcrl_20eps_tcl63_scenario2",
    r"evaluate_data\dmpcrl_20eps_tdl67_scenario2",  # change for the dmpcrl once done!!    
    r"scmpc\ipopt_scmpc_20ep_scenario_2_ns_10",
    # r"evaluate_data\ddpg_20eps_ddpg5_scenario1and2_newenv",
    r"evaluate_data\ddpg_20eps_ddpg5_sc_1_and_2_scenario2", # <- somehow they are not the same, just use this one for solve time only.
]
# print_solver_times(
#     file_paths=files_scenario_2,
#     names=names,
# )
# print_performance_vals(
#     file_paths=files_scenario_2,
#     names=names,
# )
plot_performance(
    file_paths=files_scenario_2,
    names=names,
    colors=colors,
    # showfliers=False,
    logscale=True,
    y_lim_tuple = (1*10**3,  9*10**4), # for the log scaling (manually skip one outlier)
    y_lim_constraints=(4*10**-2, 10**2),
    title_info = "Scenario 2"
)
# # TODO: wait for dmpcrl scenario 2 results? :/





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

# # test: see whether sc-mpc: scenario 0 depends on ns (no) and whether it is identical than mpcrl (it is)
# plot_performance(
#    file_paths=[
#         r"scmpc\ipopt_scmpc_2ep_scenario_0_ns_10",
#         r"scmpc\ipopt_scmpc_2ep_scenario_0_ns_5",
#         r"data\pkls\scmpc_test_cent_no_learning_2ep_scenario_0"
#     ],
#     names=[
#         "scmpc10",
#         "scmpc5",
#         "mpcrl"
#     ],
#     title_info = "Scenario 0"
# )
