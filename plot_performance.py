import pickle
import numpy as np
import matplotlib.pyplot as plt
# file for comparing (box/whisker-plots) the performance of n-different evaluate-pkl

# i.e: def plot_performance(file_paths: List[str]) -> None:
# and then have n subplots based on how many paths are given

def plot_performance(
        file_paths: list[str],
        names: list[str],
        title_info: str = "",
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
    plt.boxplot(R_tot.T, labels=names)
    plt.title(f"Performance comparison: average cost per episode | {title_info}")
    plt.show()

    # TODO: plot violations; amount and magnitude (and separate for GRC) - based on model bounds. (see also vis_large_eps for how-to)

# test: compare whether last or ep20 is better for tcl63 (evaluate on 10 eps)
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

plot_performance(
    file_paths=[
        r"evaluate_data\dmpcrl_20eps_tcl63_scenario2",
        r"evaluate_data\ddpg_20eps_ddpg5_scenario1and2_newenv",
        r"scmpc\ipopt_scmpc_20ep_scenario_2_ns_10"
    ],
    names=[
        "mpcrl", # centralized !!
        "ddpg",
        "scmpc"
    ],
    title_info = "Scenario 2"
)