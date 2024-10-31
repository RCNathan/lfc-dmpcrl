import numpy as np
from casadi import *
from lfc_model import (
    Model,
)  # NOTE: on importing, all code in module is executed, and an instance is created (always).
import pickle

# print("test")


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


x = MX.sym("x")  # creates 1-by-1 matrix with symbolic primitive called x

# create vector- or matrix-valued symbolic variables by supplying additional arguments to SX.sym:
y = SX.sym("y", 5)  # creates vector 5 by 1
Z = SX.sym("Z", 4, 2)  # creates matrix 4 by 2 (indexes columnwise)

# after declaring, expressions can be formed in an intuitive way:
f = x**2 + 10
f = sqrt(f)
print(f)


def CoM(masses: np.ndarray, positions: np.ndarray[np.ndarray]) -> np.ndarray:
    """Calculates Center of Mass based on array of masses and corresponding x,y,z-matrix of positions"""
    totalMass = 0
    totalFr = 0
    if positions.shape[0] == masses.shape[0]:  # checks if shapes match
        for i in range(positions.shape[0]):
            totalFr += masses[i] * positions[i, :]  # ith row, all columns (: = all)
            totalMass += masses[i]  # a += b         <==>    a = a + b
        CoM = totalFr / totalMass
    else:
        print("Shapes do not match")
        return
    return CoM


positions = np.array(
    [
        # x,  y,   z
        [0.1, 0.2, 0.3],
        [0.0, 0.1, 0.2],
        [0.0, 0.4, 0.2],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]
)  # shape = (7,3)
masses = np.array([0, 1, 2, 3, 4, 5, 6])  # shape (7,)

print("Center of Mass is", CoM(masses, positions))


loads = []
for i in range(10):
    load = np.random.random((3, 1))
    loads.append(
        load
    )  # this way -> np.asarray(loads).shape = (10, 3, 1): (steps, dim, 1) which is similar to data as 'X'


file = "cent_no_learning"
filename = file + ".pkl"
with open(
    filename,
    "rb",
) as file:
    data = pickle.load(file)

x = data.get(
    "X"
).squeeze()  # shape = (ep, steps, states, 1) =  (2, 101, 12, 1) --squeeze-> (2, 101, 12)
Pl = data.get("Pl").squeeze()
Pl_noise = data.get("Pl_noise").squeeze()

print("o")
