import numpy as np
import matplotlib.pyplot as plt
from lfc_model import Model
import pickle

def plotDualVars(info_dict):
    """
    Plots evolution of augmented state minus z for all agents,
    within one timestep, where 'admm_iters' iterations of ADMM are executed
    """

    # shapes: 
    # u_iters: (self.iters, self.n, self.nu_l, self.N)
    # y_iters: [(self.iters, self.nx_l * len(self.G[i]), self.N + 1) for i in range(self.n)]
    # z_iters: (self.iters, self.n, self.nx_l, self.N + 1)
    # augmented_x_iters: [(self.iters, self.nx_l * len(self.G[i]), self.N + 1) for i in range(self.n)] 

    x_aug_iters = info_dict['augmented_x_iters'] # list of (iters, 12, N+1) for n agents 
    y_iters = info_dict['y_iters'] # list of (iters, 12, N+1) for n agents
    iters = x_aug_iters[0].shape[0] 
    horizon = x_aug_iters[0].shape[2]
    numAgents = len(x_aug_iters)
    u_iters = info_dict['u_iters'] # (iters, 3, 1, N+1)
    u_iters = u_iters.reshape((iters, 3, -1))
    z_iters = info_dict['z_iters'] # (iters, 3, 4, N+1)
    z_iters = z_iters.reshape((iters, 12, -1))

    # Bad coding practice, but: since fully interconnected, we can skip aligning the augmented x's with the z: its always (12,) and sorted
    
    it = np.arange(1,iters+1)
    _, axs = plt.subplots(3,3, constrained_layout=True)
    for j in range(numAgents):
        dif = np.sum(np.abs(x_aug_iters[j] - z_iters), axis=1) # takes abs value between x-z and sums over all states -> shape (iters, timesteps)
        vars = np.sum(y_iters[j], axis=(1,2))
        for timestep in range(horizon):
            if timestep == 0 or timestep == 1 or timestep == horizon-1:
                axs[j, 0].plot(it, dif[:, timestep], label=f'timestep k={timestep}')
            else:
                axs[j, 0].plot(it, dif[:, timestep])
        axs[j, 0].set_title(r'|$\tilde{x}$-$\tilde{z}$|' + f' for Agent {j+1}')
        axs[j, 0].set_xlabel('ADMM iters')
        
        axs[j, 1].plot(it, np.sum(dif, axis=1)) # sums again, so now shape (iters,): shows sum of all states over all timesteps for one iteration
        axs[j, 1].set_title(r'|$\tilde{x}$-$\tilde{z}$|' + f' per iteration for Agent {j+1}')

        axs[j, 2].plot(it, vars)
        axs[j, 2].set_title(f'Dual vars for Agent {j+1}')
    axs[2, 0].legend()
    plt.show()
    
# with open(
#     "dualvarstest.pkl",
#     "rb",
# ) as file:
#     data = pickle.load(file)
# info_dict = data['info_dict']
# plotDualVars(info_dict)