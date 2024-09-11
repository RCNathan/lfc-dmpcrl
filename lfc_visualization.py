import numpy as np
import matplotlib.pyplot as plt
import pickle
from lfc_model import Model

# filename = 'cent.pkl'
filename = 'cent_no_learning.pkl'
with open(filename, 'rb',) as file:
    data = pickle.load(file)

# data.get('X') can be replaced with data['X']
# shape is (1, 1001, 12, 1) but can be made more workable with squeeze() -> (1001, 12)
x = data.get('X').squeeze() # shape = (1, 1001, 12, 1) -> (1001, 12)
u = data.get('U').squeeze() # shape = (1, 1000, 3, 1)
TD = data.get('TD') # len = 1000
R = data.get('R').squeeze()
param_dict = data.get('param_dict') # len = 30; V0_0, ... all have len = 501

if isinstance(data, dict):
    # Find and print all the keys where the value is a list
    list_keys = [key for key, value in data.items()]
    param_keys = [key for key, value in data.get('param_dict').items()]
    
    print("Keys in data:", list_keys)
    print("Keys inside param_dict:", param_keys)
else:
    print("The loaded data is not a dictionary.")

_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD)
axs[0].set_title("Temporal difference error (TD)")
axs[0].set_xlabel("Something time-related?") # probably episodes * sample_time used for discretization I'd imagine
axs[1].plot(R)
axs[1].set_title("Centralized rewards (R)")
axs[1].set_xlabel("Something time-related?")
axs[2].plot(np.cumsum(R))
axs[2].set_title("Cumulative reward")

m = Model()
numAgents = m.n
stateDim = m.nx_l
u_bnd = m.u_bnd_l
x_bnd = m.x_bnd_l.T
t = np.linspace(0, m.ts*(x.shape[0]-1), x.shape[0]) # time - I suppose connected to sampling time no?
_, axs = plt.subplots(4, 3)
for j in range(numAgents):
    axs[0, j].plot(t, x[:, 4*j])
    axs[0, j].hlines(x_bnd[0, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax)
    axs[0, j].set_title("Agent {}".format(j+1)) # sets agent-title on every top plot only

    axs[1, j].plot(t, x[:, 4*j+1])
    axs[1, j].hlines(x_bnd[1, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax) 

    axs[2, j].plot(t, x[:, 4*j+2])
    axs[2, j].hlines(x_bnd[2, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax) 

    axs[3, j].plot(t, x[:, 4*j+3])
    axs[3, j].hlines(x_bnd[3, :], 0, t[-1], linestyles='--', color='r') # hlines(y_values, xmin, xmax)
    axs[3, j].set_xlabel(r"time $t$") # x-labels (only bottom row) 

# y-axis labels (states)
axs[0, 0].set_ylabel(r"$\Delta$ f$_i$")
axs[1, 0].set_ylabel(r"$\Delta$ P$_{m,i}$") 
axs[2, 0].set_ylabel(r"$\Delta$ P$_{g,i}$") 
axs[3, 0].set_ylabel(r"$\Delta$ P$_{tie,i}$")

plt.show()

# # subplot index increases to the right ->
# plt.figure()
# for j in range(3): # numagents    
#     plt.subplot(4,3, j + 1)
#     plt.plot(x[:, 4*j + 0])

#     plt.subplot(4,3, j + 4)
#     plt.plot(x[:, 4*j + 1])

#     plt.subplot(4,3, j + 7)    
#     plt.plot(x[:, 4*j + 2])

#     plt.subplot(4,3, j + 10)
#     plt.plot(x[:, 4*j + 3])
# plt.show()


# for i in 
print('debug')