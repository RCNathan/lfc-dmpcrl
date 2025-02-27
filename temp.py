import numpy as np
import matplotlib.pyplot as plt

def return_load(step_counter: int) -> np.ndarray:
    """Return loads for given timestep."""
    #  step function for load | time = step_counter*ts
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

ts = 0.01
n = 3
t = 0.01* np.arange(0, 1000)
loads = np.hstack([return_load(i) for i in range(1000)])

plt.figure(figsize=(5,3),constrained_layout=True)
for i in range(n):    
    plt.plot(t, loads[i, :], label=f"Area {i+1}", linestyle='--',)
plt.legend(loc='upper right')
plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Load $\Delta \hat{P}_{\text{L},i}$")
plt.title("Nominal load disturbances for each area")
# have tick marks that stand out at t = 1, 2, while leaving the tick marks 0 through 10 as is
plt.xticks(np.arange(0, 11, 1), labels=[f"{i}" for i in range(0, 11)])
plt.show()


