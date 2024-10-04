import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model

def visualize(file:str) -> None:
    """Makes plots to visualize TD-error, rewards, states and inputs"""

    # Change filename below -> update: gets passed into visualize()
    filename = file + '.pkl'
    with open(filename, 'rb',) as file:
        data = pickle.load(file)

    # Note that .squeeze() will get rid of the (1,) in the first index for case 1 episode, making it incompatible
    param_dict = data.get('param_dict') # params V0_0, ... all have len = steps
    x = data.get('X') # shape = (4, 201, 12, 1)                 | (eps, steps, states, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1) # (4, 201, 12)    | (eps, steps, states)
    u = data.get('U') # shape = (4, 201, 3, 1)                  | (eps, steps, inputs, 1)
    u = u.reshape(u.shape[0], u.shape[1], -1) # (4, 201, 3)     | (eps, steps, inputs)
    R = data.get('R') # shape = (4, 200)                        | (eps, steps)

    # bit trickier, TD, Pl and Pl_noise are reshaped later if numEps > 1
    TD = np.asarray(data.get('TD')).reshape(1,-1) # e.g (1,800) for 4 eps at 200 steps  | (1, eps*steps)
    Pl = np.asarray(data.get('Pl')).squeeze().reshape(1, -1, 3)             #           | (1, eps*steps, 3)
    Pl_noise = np.asarray(data.get('Pl_noise')).squeeze().reshape(1, -1, 3) #           | (1, eps*steps, 3)
    

    if isinstance(data, dict):
        # Find and print all the keys where the value is a list
        list_keys = [key for key, value in data.items()]
        # param_keys = [key for key, value in data.get('param_dict').items()]
        
        print("Keys in data:", list_keys)
        # print("Keys inside param_dict:", param_keys)
    else:
        print("The loaded data is not a dictionary.")
    if (param_dict['A_0'][0] == param_dict['A_0'][-1]).all():
        print('\nNo learning of A_0')

    m = Model()
    numAgents = m.n
    stateDim = m.nx_l
    u_bnd = m.u_bnd_l
    x_bnd = m.x_bnd_l.T
    x_len = x.shape[1] if len(x.shape) == 3 else x.shape[0] # takes length of x   | if/else: shape of x depends on numEpisodes
    t = np.linspace(0, m.ts*(x_len-1), x_len) # time = sampling_time * num_samples
    numEpisodes = x.shape[0] if len(x.shape) == 3 else 1

    # plot states of all agents
    _, axs = plt.subplots(5, 3, figsize=(10,7.5),) # figsize: (width, height)
    for j in range(numAgents):
        for n in range(numEpisodes): # plot trajectories for every episode
            axs[0, j].plot(t, x[n, :, 4*j])
            axs[1, j].plot(t, x[n, :, 4*j+1])
            axs[2, j].plot(t, x[n, :, 4*j+2])
            axs[3, j].plot(t, x[n, :, 4*j+3])
            axs[4, j].plot(t[:-1], u[n, :, j])
        
        # only needs to be plotted once for each agent
        axs[0, j].hlines(x_bnd[0, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax)
        axs[0, j].set_title("Agent {}".format(j+1)) # sets agent-title on every top plot only
        axs[1, j].hlines(x_bnd[1, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax)
        axs[2, j].hlines(x_bnd[2, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax) 
        axs[3, j].hlines(x_bnd[3, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax) 
        axs[4, j].hlines(u_bnd, 0, t[-2], linestyles='--', color='r') # hlines(y_values, xmin, xmax)
        axs[4, j].set_xlabel(r"time $t$") # x-labels (only bottom row)

    # only set once | y-axis labels (states, input)
    axs[0, 0].set_ylabel(r"$\Delta f_i$")
    axs[1, 0].set_ylabel(r"$\Delta P_{m,i}$") 
    axs[2, 0].set_ylabel(r"$\Delta P_{g,i}$") 
    axs[3, 0].set_ylabel(r"$\Delta P_{tie,i}$")
    axs[4, 0].set_ylabel(r"$u$")
    
    wm = plt.get_current_fig_manager() # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    # print(wm.window.geometry()) # (x,y,dx,dy)
    figx, figy, figdx, figdy = wm.window.geometry().getRect()
    wm.window.setGeometry(-10, 0, figdx, figdy)

    if numEpisodes != 1:
        TD = TD.reshape((numEpisodes, -1)) # newShape = (numEps, steps)

    # plot TD error, reward and cumulative reward
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True, figsize=(5,5))
    for n in range(numEpisodes):
        axs[0].plot(t[:-1], TD[n, :], label='ep. {}'.format(n+1))
        axs[1].plot(t[:-1], R[n, :], label='ep. {}'.format(n+1))
        axs[2].plot(t[:-1], np.cumsum(R[n, :]), label='ep. {}'.format(n+1))
        
    # only set once
    axs[0].set_title("Temporal difference error (TD)")
    axs[0].set_xlabel(r"time $t$") 
    axs[0].legend()
    axs[1].set_title("Centralized rewards (R)")
    axs[1].set_xlabel(r"time $t$")
    axs[1].legend()
    axs[2].set_title("Cumulative reward")
    axs[2].set_xlabel(r"time $t$")
    axs[2].legend()
    
    
    wm = plt.get_current_fig_manager() # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    # print(wm.window.geometry()) # (x,y,dx,dy)
    figx, figy, figdx, figdy = wm.window.geometry().getRect()
    wm.window.setGeometry(900, 0, figdx, figdy)

    if numEpisodes != 1:
        Pl = Pl.reshape((numEpisodes, -1, 3)) # newShape = (numEps, steps)
        Pl_noise = Pl_noise.reshape((numEpisodes, -1, 3)) # newShape = (numEps, steps)

    # Plot loads and loads + noise
    _, axs = plt.subplots(1,3, figsize=(5,2)) # figsize: (width, height)
    for j in range(numAgents):
        for n in range(numEpisodes):
            axs[j].plot(t[:-1], Pl[n,:,j] + Pl_noise[n,:,j], label='ep. {}'.format(n+1))
            axs[j].plot(t[:-1], Pl[n,:,j], linestyle='--', color="black")

        # only set once for every agent
        axs[j].set_title("Agent {}".format(j+1)) # sets agent-title on every top plot only
        axs[j].set_xlabel(r"time $t$")
        axs[j].legend()
    
    # only set once
    axs[0].set_ylabel(r"Load $\Delta P_l$")

    wm = plt.get_current_fig_manager() # using pyqt5 allows .setGeometry() and changes behavior of geometry()
    # print(wm.window.geometry()) # (x,y,dx,dy)
    figx, figy, figdx, figdy = wm.window.geometry().getRect()
    wm.window.setGeometry(900, 550, figdx, figdy)

    plt.show()

# filename = 'cent'
# filename = 'cent_no_learning_1ep'
# filename = 'cent_no_learning_4ep'
# filename = 'cent_4ep'
# filename = 'cent_1ep'
# visualize(filename)
# print('debug')