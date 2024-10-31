import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model


def plot_learnables(file: str) -> None:
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)

    param_dict = data.get("param_dict")
    # # print(param_dict.keys())
    # gridnames = np.array([])
    # for key in param_dict.keys():
    #     gridnames = np.append(gridnames, key)
    # print(gridnames.reshape((6,6)))

    # plot for a lot (or all) of learnable params (debug-purposes)
    _, axs = plt.subplots(6, 6, constrained_layout=True, figsize=(7.5, 7))

    i = 1
    for item in param_dict.keys():
        # print(param_dict[f'{item}'])
        plt.subplot(6, 6, i)
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

    # plot for a lot (or all) of learnable params (debug-purposes)
    _, axs = plt.subplots(6, 6, constrained_layout=True, figsize=(7.5, 7))

    i = 1
    for item in param_dict.keys():
        # print(param_dict[f'{item}'])
        plt.subplot(6, 6, i)
        plotdata = param_dict[f"{item}"].squeeze()
        if len(plotdata.shape) > 2:
            plt.plot(plotdata[:, :, 0])
            plt.plot(plotdata[:, :, 1])
            plt.plot(plotdata[:, :, 2])
            plt.plot(plotdata[:, :, 3])
        else:
            plt.plot(plotdata)
        plt.title(str(item))
        i += 1

    # removes tick-marks (for visibility)
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    wm = plt.get_current_fig_manager()
    wm.window.move(750, 0)

    plt.show()


# changed Qs,Qx and other stuff -> to get TD error down for numerical stability
filename = "cent_5ep"  # this one shows really promising results!
filename = "cent_50epTEST"  # holy shit this shit is amazing!
# filename = 'cent_5epTEST'
# filename = 'cent_5epTEST3'
# filename = 'cent_5epTEST4'
filename = "cent_20epTEST4"  # incredible. good stuffs!

# playing with w vs Q
filename = "cent_20epTEST5"
filename = "cent_50epTEST5"
filename = "cent_5epTEST5"
filename = "cent_20epTEST5"
plot_learnables(filename)
