import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model


def plot_grc(file: str) -> None:
    filename = file + ".pkl"
    with open(
        filename,
        "rb",
    ) as file:
        data = pickle.load(file)

    param_dict = data.get("param_dict")
    x = data.get("X")
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (4, 201, 12)    | (eps, steps, states)
    m = Model()
    grc = m.GRC_l
    ts = m.ts

    x_len = (
        x.shape[1] if len(x.shape) == 3 else x.shape[0]
    )  # takes length of x   | if/else: shape of x depends on numEpisodes
    t = np.linspace(0, m.ts * (x_len - 1), x_len)  # time = sampling_time * num_samples

    _, axs = plt.subplots(1, 3)
    for n in range(m.n):
        axs[n].plot(t[:-1], 1 / ts * (x[-1, 1:, 4 * n + 1] - x[-1, :-1, 4 * n + 1]))
        axs[n].hlines(
            [-grc, grc], 0, t[-2], color="r", linestyle="--"
        )  # hlines(y_values, xmin, xmax)
        axs[n].set_ylim([-1.1 * grc, 1.1 * grc])

    plt.show()


filename = "cent_no_learning_1epdist_time"
plot_grc(filename)
