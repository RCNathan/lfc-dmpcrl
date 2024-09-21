import numpy as np
from casadi import *
from lfc_model import Model # NOTE: on importing, all code in module is executed, and an instance is created (always).
print("test")


N = 6
A = np.reshape(np.linspace(1, N, N), (2, 3))
B = np.ones((3, 2))
print(A, "\n", B)
print(A @ B)
print(B @ A)


class Graph:
    def __init__(self):
        self.adj = {}  # Adjacency list

    def add_edge(self, u, v):
        if u not in self.adj:
            self.adj[u] = []
        if v not in self.adj:
            self.adj[v] = []
        self.adj[u].append(v)
        self.adj[v].append(u)


# Usage
graph = Graph()
graph.add_edge(1, 2)
graph.add_edge(1, 3)
print(graph.adj)

print("the end lol")



x = MX.sym("x") # creates 1-by-1 matrix with symbolic primitive called x

# create vector- or matrix-valued symbolic variables by supplying additional arguments to SX.sym:
y = SX.sym("y", 5)  # creates vector 5 by 1
Z = SX.sym("Z", 4, 2)  # creates matrix 4 by 2 (indexes columnwise)

# after declaring, expressions can be formed in an intuitive way:
f = x**2 + 10
f = sqrt(f)
print(f)

import numpy as np

positions = np.array([
    # x,  y,   z
    [0.1, 0.2, 0.3], 
    [0.0, 0.1, 0.2], 
    [0.0, 0.4, 0.2], 
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3],    
]) # shape = (7,3)
masses = np.array([0, 1, 2, 3, 4, 5, 6]) # shape (7,)

def CoM(masses, positions):
    totalMass = 0
    totalFr = 0 
    if positions.shape[0] == masses.shape[0]: # checks if shapes match
        for i in range(positions.shape[0]):
            totalFr += masses[i]*positions[i, :] # ith row, all columns (i = all)
            totalMass += masses[i] # a += b         <==>    a = a + b 
        CoM = totalFr / totalMass
    else:
        print("Shapes do not match")
        CoM = 0    
    return CoM

print("Center of Mass is", CoM(masses, positions))