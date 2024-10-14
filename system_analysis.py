import numpy as np
from lfc_model import Model
from controllability import ctrb
import control as ct

m = Model()

# controllability of (discretized) A,B:
K, rank = ctrb(m.A, m.B)
print("\nRank of controllability matrix is", rank)
if rank != m.A.shape[0]:
    print("Rank is smaller than dimension, meaning system (A,B) has uncontrollable modes")
if (np.isclose(ct.ctrb(m.A,m.B), K)).all() == True:
    print('Ctrb is implemented correctly \n\n')

eigvals, eigvec = np.linalg.eig(m.A)
print("Eigenvalues of A:", eigvals) # in DT: any eigenvalues outside unit circle are unstable
for i in range(len(eigvals)):
    eigval = eigvals[i]
    if abs(eigval) > 1:
        print("Unstable mode: i = ",i, ":", eigval) # unstable if outside unit circle
        # print("Corresponding eigvec: ", eigvec[:, i])
        K_prime = np.hstack([K, eigvec[:, i].reshape(12,1)]) # K' = [K vi], if rank K' = rank K, mode is controllable. If rank K' > K: uncontrollable.
        rank_K_p = np.linalg.matrix_rank(K_prime, tol=1e-8)
        print("rank of K':", rank_K_p) 
        if np.isclose(rank_K_p, rank, atol=1e-6):
            print("Mode is controllable")
        else:
            print("Mode is uncontrollable")


print("\n\n")
for i in range(12):
    testVec = np.zeros((12,1)) # unit vector with with a 1 in ith pos
    testVec[i] = 1
    K_prime = np.hstack([K, testVec])
    rank_K_p = np.linalg.matrix_rank(K_prime, tol=1e-8)
    if np.isclose(rank_K_p, rank, atol=1e-6):
        print("State {} is controllable".format(i+1))
    else:
        print("State {} is uncontrollable".format(i+1)) # states 4, 8 and 12 are uncontrollable -> Ptie's!

# sys = ct.ss(m.A, m.B, np.zeros((4, 12)), np.zeros((4,3)))
# ct.poles(sys) # ct.poles(sys) == eigvals

Kl, rankl = ctrb(m.A_l_1, m.B_l_1)
print("\nRank of smaller subsystem is", rankl)

print("Debug")


