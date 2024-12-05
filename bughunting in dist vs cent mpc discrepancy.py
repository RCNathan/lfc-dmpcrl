import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from dmpcrl.mpc.mpc_admm import MpcAdmm
from dmpcrl.utils.solver_options import SolverOptions

from lfc_model import Model
from lfc_discretization import lfc_forward_euler_cs, lfc_zero_order_hold
from block_diag import block_diag

from dmpcrl.core.admm import AdmmCoordinator
from lfc_learnable_mpc import CentralizedMpc, LocalMpc 


model = Model()
G = AdmmCoordinator.g_map(model.adj)
centralized_mpc = CentralizedMpc(model, prediction_horizon=10) # for comparison/debugging
distributed_mpcs = [LocalMpc(model, prediction_horizon=10, num_neighbours=2, my_index=G[i].index(i), global_index=i, G=G) for i in range(3)]

A,B,F = model.centralized_dynamics_from_local(
    np.array([centralized_mpc.learnable_pars_init['A_0'], centralized_mpc.learnable_pars_init['A_1'], centralized_mpc.learnable_pars_init['A_2']]),
    np.array([centralized_mpc.learnable_pars_init['B_0'], centralized_mpc.learnable_pars_init['B_1'], centralized_mpc.learnable_pars_init['B_2']]),
    np.array([
        [np.zeros((4,4)), centralized_mpc.learnable_pars_init['A_c_0_1'], centralized_mpc.learnable_pars_init['A_c_0_2']],
        [centralized_mpc.learnable_pars_init['A_c_1_0'], np.zeros((4,4)), centralized_mpc.learnable_pars_init['A_c_1_2']],
        [centralized_mpc.learnable_pars_init['A_c_2_0'], centralized_mpc.learnable_pars_init['A_c_2_1'], np.zeros((4,4))]
    ]),
    np.array([centralized_mpc.learnable_pars_init['F_0'], centralized_mpc.learnable_pars_init['F_1'], centralized_mpc.learnable_pars_init['F_2']]),
    model.ts
)
Adist = np.vstack([
        np.hstack([distributed_mpcs[0].learnable_pars_init['A'], distributed_mpcs[0].learnable_pars_init['A_c_1'], distributed_mpcs[0].learnable_pars_init['A_c_2']]),
        np.hstack([distributed_mpcs[1].learnable_pars_init['A_c_0'], distributed_mpcs[1].learnable_pars_init['A'], distributed_mpcs[1].learnable_pars_init['A_c_2']]),
        np.hstack([distributed_mpcs[2].learnable_pars_init['A_c_0'], distributed_mpcs[2].learnable_pars_init['A_c_1'], distributed_mpcs[2].learnable_pars_init['A']])
    ])

print("Dynamics ok?", np.allclose(A, Adist))