import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend like Agg
import matplotlib.pyplot as plt
import pickle
from typing import Tuple
from lfc_model import Model
from vis_large_eps import vis_large_eps
# from plot_performance import plot_performance
# from visualize_report import visualize as visualize_report

def return_load(step_counter: int) -> np.ndarray:
    """Return loads for given timestep."""
    #  step function for load | time = step_counter*ts
    ts = 0.01
    sim_time = step_counter * ts
    # with ts = 0.01
    c1, c2 = 0.1, -0.1 # 0.1 -0.1 interesting result fr, touching constraint at f1, f3, tie1, tie3
    t1, t2, t3 = 1, 2, 3

    load = np.zeros((n, 1))
    if sim_time < t1:
        load = np.array([0.0, 0.0, 0.0]).reshape(n, -1)
    elif sim_time < t2:  # from t = 10 - 20
        load = np.array([c1, 0.0, 0.0]).reshape(n, -1)
    elif sim_time < t3:  # from t = 20 - 30
        load = np.array([c1, 0.0, c2]).reshape(n, -1)
    elif sim_time < 40:
        load = np.array([c1, 0.0, c2]).reshape(n, -1)
    else:
        load = np.array([c1, 0.0, c2]).reshape(n, -1)
    return load

def plot_load_disturbance():    
    n = 3
    t = 0.01* np.arange(0, 1000)
    loads = np.hstack([return_load(i) for i in range(1000)])

    print("This file is used to plot the load disturbance, for the final report.")
    plt.figure(figsize=(5,3),constrained_layout=True)
    for i in range(n):    
        plt.plot(t, loads[i, :], label=f"Area {i+1}", linestyle='--',)
    plt.legend(loc='upper right')
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Nominal load $\Delta \hat{P}_{\text{L},i}$")
    plt.title("Nominal load disturbances for each area")
    # have tick marks that stand out at t = 1, 2, while leaving the tick marks 0 through 10 as is
    plt.xticks(np.arange(0, 11, 1), labels=[f"{i}" for i in range(0, 11)])
    plt.show()

def return_param_dict(file):
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)
    
    param_dict = data["param_dict"]
    return param_dict

def analyze_param_differences(param_dict):  
    """
    Computes the maximum absolute difference between the first and last iteration 
    for each parameter in param_dict and returns a sorted list of parameters.

    Args:
        param_dict (dict): Dictionary where keys are parameter names and values are 
                           numpy arrays of shape (n_iters, dim) or (n_iters, dim1, dim2).

    Returns:
        list: Sorted list of tuples, where each tuple contains:
              (parameter_name, {"max": max_difference, "argmax": index_of_max_difference})
              The list is sorted in descending order of max_difference.
    
    Example:
        param_dict = {
            "A_0": np.random.rand(10, 4, 4),
            "B_1": np.random.rand(10, 4)
        }
        sorted_params = analyze_param_differences(param_dict)

        # Output (example):
        # Max differences per parameter (sorted by value):
        # Parameter: A_0, Max: 9.123456e-01, Argmax: (2, 3)
        # Parameter: B_1, Max: 5.234567e-01, Argmax: (1,)
    """  
    differences_dict = {}
    for param, data in param_dict.items():
        # Compute the difference between the last and first iteration
        if len(data.shape) == 3:
            difference = data[-1, :, :] - data[0, :, :]
        elif len(data.shape) == 2:
            difference = data[-1, :] - data[0, :]
        else:
            raise ValueError(f"Unexpected shape {data.shape} for parameter {param}")

        # Take the absolute value before finding the max difference
        abs_difference = np.abs(difference)
        ind = np.unravel_index(np.argmax(abs_difference), abs_difference.shape)
        max_diff = abs_difference[ind]

        # Store results in dictionary
        differences_dict[param] = {"max": max_diff, "argmax": ind}

    # Sort the dictionary by max difference values in descending order
    sorted_params = sorted(differences_dict.items(), key=lambda x: x[1]["max"], reverse=True)

    # Print results in sorted order
    print("\nMax absolute differences per parameter (sorted by value):")
    for param, values in sorted_params:
        print(f"Parameter: {param}, Max: {values['max']:.4e}, Argmax: {values['argmax']}")

    return sorted_params

def analyze_relative_param_differences(param_dict):
    """
    Computes the maximum absolute and relative difference between the first and last iteration 
    for each parameter in param_dict and returns a sorted list of parameters.

    Args:
        param_dict (dict): Dictionary where keys are parameter names and values are 
                           numpy arrays of shape (n_iters, dim) or (n_iters, dim1, dim2).

    Returns:
        list: Sorted list of tuples, where each tuple contains:
              (parameter_name, {
                  "max_abs": max_absolute_difference, 
                  "argmax": index_of_max_absolute_difference, 
                  "relative_change": relative_difference
              })
              The list is sorted in descending order of relative_change.
    
    Example:
        param_dict = {
            "A_0": np.random.rand(10, 4, 4) * 0.1,  # Example with small values
            "B_1": np.random.rand(10, 4) * 10       # Example with large values
        }
        sorted_params = analyze_param_differences(param_dict)

        # Output (example):
        # Max relative differences per parameter (sorted by value):
        # Parameter: A_0, Max Abs: 5.432100e-02, Argmax: (2, 3), Relative Change: 2.543210e+00
        # Parameter: B_1, Max Abs: 3.210000e+00, Argmax: (1,), Relative Change: 1.234567e-01
    """
    differences_dict = {}
    for param, data in param_dict.items():
        # Compute the difference between the last and first iteration
        if len(data.shape) == 3:
            difference = data[-1, :, :] - data[0, :, :]
            initial_value = data[0, :, :]
            last_value = data[-1, :, :]
        elif len(data.shape) == 2:
            difference = data[-1, :] - data[0, :]
            initial_value = data[0, :]
            last_value = data[-1, :]
        else:
            raise ValueError(f"Unexpected shape {data.shape} for parameter {param}")

        # Take absolute difference
        abs_difference = np.abs(difference)

        # Find max absolute change and its index
        ind = np.unravel_index(np.argmax(abs_difference), abs_difference.shape)
        max_abs_diff = abs_difference[ind]

        # Determine the reference value for relative comparison
        ref_value = np.abs(initial_value[ind]) if initial_value[ind] != 0 else np.abs(last_value[ind])

        # Compute relative difference safely
        relative_change = max_abs_diff / ref_value if ref_value != 0 else 0

        # Store results in dictionary
        differences_dict[param] = {
            "max_abs": max_abs_diff,
            "argmax": ind,
            "relative_change": relative_change
        }

    # Sort the dictionary by relative change in descending order
    sorted_params = sorted(differences_dict.items(), key=lambda x: x[1]["relative_change"], reverse=True)

    # Print results in sorted order using scientific notation
    print("\nMax relative differences per parameter (sorted by value):")
    for param, values in sorted_params:
        print(f"Parameter: {param}, Max Abs: {values['max_abs']:.4e}, Argmax: {values['argmax']}, Relative Change: {values['relative_change']:.3e}")

    return sorted_params

def plot_4learnables(sorted_params, param_dict, top_n=4, manual_vars=None, analyze_method="relative"):
    """
    Plots the trajectories of the top 4 learnable parameters over time.

    Args:
        sorted_params (list): Sorted list of tuples from analyze_param_differences(), 
                              each containing (param_name, {"max": value, "argmax": index}).
        param_dict (dict): Original dictionary containing parameter data.
        top_n (int, optional): Number of top parameters to plot. Defaults to 4.
        manual_vars (list, optional): List of exactly 4 parameter names to plot instead of top-n sorted ones.
    """
    # Select the top `top_n` parameters unless manual variables are provided
    if manual_vars:
        selected_params = manual_vars
    else:
        selected_params = [param[0] for param in sorted_params[:top_n]]

    # Ensure exactly 4 parameters are selected
    if len(selected_params) != 4:
        raise ValueError(f"Expected 4 parameters to plot, but got {len(selected_params)}.")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes = axes.flatten()  # Flatten to iterate easily

    for ax, param in zip(axes, selected_params):
        data = param_dict[param]
        max_index = sorted_params[0][1]["argmax"]  # Use the top parameter's max index as an example

        # Extract the correct trajectory based on the shape of the parameter
        if len(data.shape) == 3:  # Shape (n_iters, dim1, dim2)
            index = sorted_params[[p[0] for p in sorted_params].index(param)][1]["argmax"]
            trajectory = data[:, index[0], index[1]]  # Extract time series at max difference index
        elif len(data.shape) == 2:  # Shape (n_iters, dim)
            index = sorted_params[[p[0] for p in sorted_params].index(param)][1]["argmax"]
            trajectory = data[:, index[0]]  # Extract time series at max difference index
        else:
            raise ValueError(f"Unexpected shape {data.shape} for parameter {param}")

        # Plot the trajectory
        ax.plot(trajectory, label=f"{param} @ {index}", linewidth=2)
        ax.set_title(param)
        # ax.legend()
        ax.grid(True)

    # Set common x-label only for the bottom plots
    for i in {2,3}:
        axes[i].set_xlabel("Learning iteration")

    # Display the figure
    # plt.suptitle(f"Using the max {analyze_method} difference")
    plt.suptitle(f"Evolution of the top {top_n} learnable parameters by max {analyze_method} difference")
    plt.show()

def vis_training_eps(
    file: str, 
    view_partly: Tuple = None, 
    show_agent: int = 0, # 0, 1, 2
    grid_on: bool = False,
) -> None:
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
    x = data.get("X")  # shape = (4, 201, 12, 1)                 | (eps, steps, states, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    u = data.get("U")  # shape = (4, 201, 3, 1)                  | (eps, steps, inputs, 1)
    u = u.reshape(u.shape[0], u.shape[1], -1)  # (4, 201, 3)     | (eps, steps, inputs)
    R = data.get("R")  # shape = (4, 200)                        | (eps, steps)
    TD = np.asarray(data.get("TD", None)) # returns None if TD does not exist
    
    centralized_flag = data.get("cent_flag")
    if centralized_flag == False and (TD != None).all(): # i.e: distributed, so TD has shape (n, eps*steps)
        # TD = np.sum(TD, axis=0) # sum over agents to shape (eps*steps) # no!
        TD =  TD[0, :] # sum over agents to shape (eps*steps)

    # bit trickier, TD, Pl and Pl_noise are reshaped later if numEps > 1
    TD = TD.reshape(1, -1)  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)
    Pl = (np.asarray(data.get("Pl")).squeeze().reshape(1, -1, 3)) # | (1, eps*steps, 3)
    Pl_noise = (np.asarray(data.get("Pl_noise")).squeeze().reshape(1, -1, 3))  # | (1, eps*steps, 3)

    numAgents = Model.n
    u_bnd = Model.u_bnd_l
    x_bnd = Model.x_bnd_l.T
    x_len = (
        x.shape[1] if len(x.shape) == 3 else x.shape[0]
    )  # takes length of x   | if/else: shape of x depends on numEpisodes
    t = np.linspace(0, Model.ts * (x_len - 1), x_len)  # time = sampling_time * num_samples
    numEpisodes = x.shape[0] if len(x.shape) == 3 else 1

    if numEpisodes != 1:
        Pl = Pl[:, :, :].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)
        Pl_noise = Pl_noise[:,:,:].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)

    if numEpisodes != 1 and TD.all() != None:
        TD = TD.reshape((numEpisodes, -1))  # newShape = (numEps, steps)

    # Optional: view only part of the data
    if view_partly != None:
        beginEp, endEp = view_partly[0], view_partly[1]
        x = x[beginEp:endEp, :, :]
        u = u[beginEp:endEp, :, :]
        R = R[beginEp:endEp, :]
        TD = TD[beginEp:endEp, :]
        Pl = Pl[beginEp:endEp, :, :]
        Pl_noise = Pl_noise[beginEp:endEp, :, :]
        numEpisodes = endEp-beginEp

    # get max and min of states
    xmax = np.max(x, axis=0)
    xmin = np.min(x, axis=0)
    umax = np.max(u, axis=0)
    umin = np.min(u, axis=0)

    # plot states of all agents
    _, axs = plt.subplots(
        5,
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(10, 7.5),
    )  # figsize: (width, height)
    for i in range(Model.nx_l):
        # plot states
        axs[i].plot(t, xmax[:, 4 * show_agent + i], color="gray", linestyle="--") # label="upper bound"
        axs[i].plot(t, xmin[:, 4 * show_agent + i], color="gray", linestyle="--") #label="lower bound"
        axs[i].plot(t, x[0, :, 4 * show_agent + i], color="xkcd:purple", label="First Episode")
        axs[i].plot(t, x[-1, :, 4 * show_agent + i], color="xkcd:red", label="Last Episode")
        axs[i].hlines(x_bnd[i, :], 0, t[-1], linestyles="--", color="black", label="Constraint")  # hlines(y_values, xmin, xmax)
        # shaded area between min and max
        axs[i].fill_between(t, xmax[:, 4 * show_agent + i], xmin[:, 4 * show_agent + i], color="gray", alpha=0.5, hatch="//", label="Envelope")

    # plot inputs
    axs[4].plot(t[:-1], umax[:, show_agent], color="gray", linestyle="--")
    axs[4].plot(t[:-1], umin[:, show_agent], color="gray", linestyle="--")
    axs[4].plot(t[:-1], u[0, :, show_agent], color="xkcd:purple")
    axs[4].plot(t[:-1], u[-1, :, show_agent], color="xkcd:red")
    axs[4].hlines(u_bnd, 0, t[-2], linestyles="--", color="black")  # hlines(y_values, xmin, xmax)
    axs[4].fill_between(t[:-1], umax[:, show_agent], umin[:, show_agent], color="gray", alpha=0.5, hatch="//")

    # only needs to be set once for each agent
    axs[0].set_title("Evolution of trajectories for agent {}".format(show_agent + 1))  # sets agent-title on every top plot only
    axs[4].set_xlabel(r"Time $t$ [s]")  # x-labels (only bottom row)
    axs[0].set_ylim([-0.25, 0.25])
    axs[1].set_ylim([-1.25, 1.25])
    axs[2].set_ylim([-1.25, 1.25])
    axs[3].set_ylim([-0.125, 0.125])

    # only set once | y-axis labels (states, input)
    axs[0].set_ylabel(r"$\Delta f_i$")
    axs[1].set_ylabel(r"$\Delta P_{\mathrm{m},i}$")
    axs[2].set_ylabel(r"$\Delta P_{\mathrm{g},i}$")
    axs[3].set_ylabel(r"$\Delta P_{\mathrm{tie},i}$")
    axs[4].set_ylabel(r"$u$")
    axs[0].legend(loc="lower right")

    for i in range(5):
        axs[i].grid(grid_on)

    wm = (
        plt.get_current_fig_manager()
    )  # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    wm.window.move(-10, 0)
    plt.show()

def vis_evaluate_eps(
    file: str, 
    view_partly: Tuple = None, 
    show_agent: int = 0, # 0, 1, 2
    grid_on: bool = False,
    color: str = "xkcd:red",
) -> None:
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
    x = data.get("X")  # shape = (4, 201, 12, 1)                 | (eps, steps, states, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    u = data.get("U")  # shape = (4, 201, 3, 1)                  | (eps, steps, inputs, 1)
    u = u.reshape(u.shape[0], u.shape[1], -1)  # (4, 201, 3)     | (eps, steps, inputs)
    R = data.get("R")  # shape = (4, 200)                        | (eps, steps)
    TD = np.asarray(data.get("TD", None)) # returns None if TD does not exist
    
    centralized_flag = data.get("cent_flag")
    if centralized_flag == False and (TD != None).all(): # i.e: distributed, so TD has shape (n, eps*steps)
        # TD = np.sum(TD, axis=0) # sum over agents to shape (eps*steps) # no!
        TD =  TD[0, :] # sum over agents to shape (eps*steps)

    # bit trickier, TD, Pl and Pl_noise are reshaped later if numEps > 1
    TD = TD.reshape(1, -1)  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)
    Pl = (np.asarray(data.get("Pl")).squeeze().reshape(1, -1, 3)) # | (1, eps*steps, 3)
    Pl_noise = (np.asarray(data.get("Pl_noise")).squeeze().reshape(1, -1, 3))  # | (1, eps*steps, 3)

    numAgents = Model.n
    u_bnd = Model.u_bnd_l
    x_bnd = Model.x_bnd_l.T
    x_len = (
        x.shape[1] if len(x.shape) == 3 else x.shape[0]
    )  # takes length of x   | if/else: shape of x depends on numEpisodes
    t = np.linspace(0, Model.ts * (x_len - 1), x_len)  # time = sampling_time * num_samples
    numEpisodes = x.shape[0] if len(x.shape) == 3 else 1

    if numEpisodes != 1:
        Pl = Pl[:, :, :].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)
        Pl_noise = Pl_noise[:,:,:].reshape((numEpisodes, -1, 3))  # newShape = (numEps, steps)

    if numEpisodes != 1 and TD.all() != None:
        TD = TD.reshape((numEpisodes, -1))  # newShape = (numEps, steps)

    # Optional: view only part of the data
    if view_partly != None:
        beginEp, endEp = view_partly[0], view_partly[1]
        x = x[beginEp:endEp, :, :]
        u = u[beginEp:endEp, :, :]
        R = R[beginEp:endEp, :]
        TD = TD[beginEp:endEp, :]
        Pl = Pl[beginEp:endEp, :, :]
        Pl_noise = Pl_noise[beginEp:endEp, :, :]
        numEpisodes = endEp-beginEp

    # get index of best episode
    best_index = np.argmin(np.cumsum(R, axis=1)[:, -1])

    # get max and min of states
    xmax = np.max(x, axis=0)
    xmin = np.min(x, axis=0)
    umax = np.max(u, axis=0)
    umin = np.min(u, axis=0)

    # plot states of all agents
    _, axs = plt.subplots(
        5,
        1,
        constrained_layout=True,
        sharex=True,
        figsize=(10, 7.5),
    )  # figsize: (width, height)
    for i in range(Model.nx_l):
        # plot states
        axs[i].plot(t, xmax[:, 4 * show_agent + i], color="gray", linestyle="--") # label="upper bound"
        axs[i].plot(t, xmin[:, 4 * show_agent + i], color="gray", linestyle="--") #label="lower bound"
        axs[i].plot(t, x[best_index, :, 4 * show_agent + i], color=color, label="Best Episode") # best
        # axs[i].plot(t, x[0, :, 4 * show_agent + i], color="xkcd:purple", label="First Episode")
        # axs[i].plot(t, x[-1, :, 4 * show_agent + i], color="xkcd:red", label="Last Episode")
        axs[i].hlines(x_bnd[i, :], 0, t[-1], linestyles="--", color="black", label="Constraint")  # hlines(y_values, xmin, xmax)
        # shaded area between min and max
        axs[i].fill_between(t, xmax[:, 4 * show_agent + i], xmin[:, 4 * show_agent + i], color="gray", alpha=0.5, hatch="//", label="Envelope")

    # plot inputs
    axs[4].plot(t[:-1], umax[:, show_agent], color="gray", linestyle="--")
    axs[4].plot(t[:-1], umin[:, show_agent], color="gray", linestyle="--")
    axs[4].plot(t[:-1], u[best_index, :, show_agent], color=color, label="best episode") # best
    # axs[4].plot(t[:-1], u[0, :, show_agent], color="xkcd:purple")
    # axs[4].plot(t[:-1], u[-1, :, show_agent], color="xkcd:red")
    axs[4].hlines(u_bnd, 0, t[-2], linestyles="--", color="black")  # hlines(y_values, xmin, xmax)
    axs[4].fill_between(t[:-1], umax[:, show_agent], umin[:, show_agent], color="gray", alpha=0.5, hatch="//")

    # only needs to be set once for each agent
    axs[0].set_title("Evolution of trajectories for agent {}".format(show_agent + 1))  # sets agent-title on every top plot only
    axs[4].set_xlabel(r"Time $t$ [s]")  # x-labels (only bottom row)
    axs[0].set_ylim([-0.25, 0.25])
    axs[1].set_ylim([-1.25, 1.25])
    axs[2].set_ylim([-1.25, 1.25])
    axs[3].set_ylim([-0.125, 0.125])

    # only set once | y-axis labels (states, input)
    axs[0].set_ylabel(r"$\Delta f_i$")
    axs[1].set_ylabel(r"$\Delta P_{\mathrm{m},i}$")
    axs[2].set_ylabel(r"$\Delta P_{\mathrm{g},i}$")
    axs[3].set_ylabel(r"$\Delta P_{\mathrm{tie},i}$")
    axs[4].set_ylabel(r"$u$")
    axs[0].legend(loc="lower right")

    for i in range(5):
        axs[i].grid(grid_on)

    wm = (
        plt.get_current_fig_manager()
    )  # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    wm.window.move(-10, 0)
    plt.show()

def plot_costs(
    file: str, 
    view_partly: Tuple = None, 
    color: str = "xkcd:blue",
    grid_on: bool = False,
    is_ddpg: bool = False,
    every_n: int = 10, # for ddpg: plot every 10th datapoint instead of all of them
) -> None:

    # Change filename below -> update: gets passed into visualize()
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)

    # Note that .squeeze() will get rid of the (1,) in the first index for case 1 episode, making it incompatible
    x = data.get("X")  # shape = (4, 201, 12, 1)                 | (eps, steps, states, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    R = data.get("R")  # shape = (4, 200)                        | (eps, steps)
    TD = np.asarray(data.get("TD", None)) # returns None if TD does not exist
    
    centralized_flag = data.get("cent_flag")
    if centralized_flag == False and (TD != None).all(): # i.e: distributed, so TD has shape (n, eps*steps)
        # TD = np.sum(TD, axis=0) # sum over agents to shape (eps*steps) # no!
        TD =  TD[0, :] # sum over agents to shape (eps*steps)

    # bit trickier, TD, Pl and Pl_noise are reshaped later if numEps > 1
    TD = TD.reshape(1, -1)  # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)
    x_bnd = Model.x_bnd_l.T
    numEpisodes = x.shape[0] if len(x.shape) == 3 else 1

    if numEpisodes != 1 and TD.all() != None:
        TD = TD.reshape((numEpisodes, -1))  # newShape = (numEps, steps)

    # Optional: view only part of the data
    if view_partly != None:
        beginEp, endEp = view_partly[0], view_partly[1]
        x = x[beginEp:endEp, :, :]
        R = R[beginEp:endEp, :]
        TD = TD[beginEp:endEp, :]
        numEpisodes = endEp-beginEp

    # make cumulative sum, get max and min
    Rcumsum = np.cumsum(R, axis=1)

    # magnitude of violations
    x_bnd_l = Model.x_bnd_l
    x_bnd = np.tile(x_bnd_l, Model.n).T
    x_up = np.maximum(x - x_bnd[:, 1], 0)
    x_down = np.maximum(x_bnd[:, 0] - x, 0)
    violations_magnitude = np.sum(x_up + x_down, axis=(1,2))

    # Plot TD error continously, (avg) rewards & TD per episode and evolution of learnable params over time
    if is_ddpg:
        num = 2
    else:
        num = 3
    _, axs = plt.subplots(
        num, 1, 
        constrained_layout=True, 
        sharex=True, 
        figsize=(8, 6), # figsize=(3, 5)  
    )
    if not is_ddpg:
        axs[0].plot(np.linspace(1, numEpisodes, numEpisodes), Rcumsum[:, -1], linestyle="--", color=color)
        axs[0].scatter(
            np.linspace(1, numEpisodes, numEpisodes),
            Rcumsum[:, -1],
            label="cumulative cost",
            color=color
        )
        axs[0].set_title("Total cost per episode")
        axs[0].set_xlabel("Episodes")
        axs[1].plot(np.linspace(1, numEpisodes, numEpisodes), np.sum(np.abs(TD), axis=1), linestyle="--", color=color)
        axs[1].scatter(
            np.linspace(1, numEpisodes, numEpisodes),
            np.sum(np.abs(TD), axis=1),
            label="sum of TD error",
            color=color
        )
        axs[1].set_title(r"Temporal difference error $\delta_t$ per episode")
        axs[1].set_xlabel("Episodes")
        axs[2].scatter(
            np.linspace(1, numEpisodes, numEpisodes),
            violations_magnitude,
            color=color
        )
        axs[2].plot(np.linspace(1, numEpisodes, numEpisodes), violations_magnitude, linestyle="--", color=color)
        axs[2].set_xlabel("Episodes")
        axs[2].set_title("Magnitude of constraint violations per episode")
        axs[2].set_xticks(range(0, numEpisodes+1, 5))
    if is_ddpg:
        axs[0].plot(np.linspace(1, numEpisodes, numEpisodes)[::every_n], Rcumsum[:, -1][::every_n], linestyle="--", color=color)
        axs[0].scatter(
            np.linspace(1, numEpisodes, numEpisodes)[::every_n],
            Rcumsum[:, -1][::every_n],
            label="cumulative cost",
            color=color
        )
        axs[0].set_title("Total cost per episode")
        axs[0].set_xlabel("Episodes")
        axs[1].scatter(
            np.linspace(1, numEpisodes, numEpisodes)[::every_n],
            violations_magnitude[::every_n],
            color=color
        )
        axs[1].plot(np.linspace(1, numEpisodes, numEpisodes)[::every_n], violations_magnitude[::every_n], linestyle="--", color=color)
        # axs[1].plot(violations_magnitude[::10], linestyle="--", color=color)
        axs[1].set_xlabel("Episodes")
        axs[1].set_title("Magnitude of constraint violations per episode")
        axs[1].set_xticks(range(0, numEpisodes+1, 5))
    # axs[1].set_ylim(bottom=0)
    # axs[0].set_ylim(bottom=0, top=1.1 * np.max(Rcumsum[:, -1]))
    # axs[0].margins(y=0.5)
    for i in range(num):
        axs[i].grid(grid_on)

    wm = plt.get_current_fig_manager()
    # wm.window.move(1500,0)
    wm.window.move(-10, 0)
    plt.show()


######################################################################################################

# easily toggle on/off different plots..
# plot_load_disturbance()

# scenario 0
filename = r"data\pkls\sc0_cent_20ep_scenario_0", # mpcrl training on scenario 0
filename = r"data\pkls\sc0_distr_20ep_scenario_0",  # dmpcrl training on scenario 0

# scenario 1
filename= r"data from server\batch 3\pkls\tcl13_cent_20ep_scenario_1" # tcl13 for mpcrl [13, 14, 15]
filename= r"data from server\batch 3\pkls\tdl23_distr_20ep_scenario_1" # tdl16 for dmpcrl [16, 19, 23?] - 

# scenario 2
filename = r"data\pkls\tcl63_cent_100ep_scenario_2" # 63 = 48 = 48rp..
filename = r"data\pkls\tcl48_cent_50ep_scenario_2" # 48


# learning plots
manual_vars = ["V0_0", "V0_2", "x_lb_0", "x_ub_2"]
param_dict = return_param_dict(filename)
sorted_params = analyze_relative_param_differences(param_dict) 
sorted_params = analyze_param_differences(param_dict) # or analyze_param_differences for absolute differences
# plot_4learnables(
#     sorted_params, 
#     param_dict, 
#     # manual_vars=manual_vars, 
#     analyze_method="relative"
# )

# vis_training_eps(filename) # view_partly=[0,20]
# plot_costs(filename)
# vis_large_eps(filename)

# ddpg scenario 1/2
filename = r"ddpg\lfc_ddpg5_eval"
filename = r"ddpg\lfc_ddpg5_train"
# plot_costs(filename, is_ddpg=True)
# vis_training_eps(filename, show_agent=2)

# evaluation plots
filename = r"scmpc\ipopt_scmpc_20ep_scenario_2_ns_5"
# filename = r"scmpc\ipopt_scmpc_20ep_scenario_2_ns_10"
# plot_costs(filename, is_ddpg=True, every_n=1)
# vis_evaluate_eps(filename, show_agent=0)
# visualize_report(filename, color="xkcd:red") # my favorites: xkcd red and xkcd blue

# vis_evaluate_eps(filename)

# maybe include in report?
filename = r"data\pkls\sc0_cent_20ep_scenario_0" # mpcrl training on scenario 0
filename = r"data\pkls\sc0_distr_20ep_scenario_0"  # dmpcrl training on scenario 0
vis_training_eps(filename, show_agent=2)