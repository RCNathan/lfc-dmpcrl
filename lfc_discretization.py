import numpy as np
import casadi as cs
from scipy.linalg import expm
from math import factorial


def lfc_forward_euler(
    A: np.ndarray,
    B: np.ndarray,
    F: np.ndarray,
    ts: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu + F Pl using forward euler method."""
    Ad = np.eye(A.shape[0]) + ts * A
    Bd = ts * B
    Fd = ts * F
    return Ad, Bd, Fd


def lfc_forward_euler_cs(
    A: cs.SX,
    B: cs.SX,
    F: cs.SX,
    ts: float,
) -> tuple[cs.SX, cs.SX, cs.SX]:
    Ad = cs.SX_eye(A.shape[0]) + ts * A
    Bd = ts * B
    Fd = ts * F
    return Ad, Bd, Fd


def lfc_zero_order_hold(
    A: np.ndarray, B: np.ndarray, F: np.ndarray, ts: float, N: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretise the continuous time system x_dot = Ax + Bu using ZOH"""
    n = A.shape[0]
    I = np.eye(n)
    if isinstance(A, np.ndarray):
        D = expm(ts * np.vstack((np.hstack([A, I]), np.zeros((n, 2 * n)))))
        Ad = D[:n, :n]
        Id = D[:n, n:]
        Bd = Id.dot(B)
        Fd = Id.dot(F)
    else:
        M = ts * cs.vertcat(cs.horzcat(A, I), np.zeros((n, 2 * n)))
        D = sum(cs.mpower(M, k) / factorial(k) for k in range(N))
        Ad = D[:n, :n]
        Id = D[:n, n:]
        Bd = Id @ B
        Fd = Id @ F
    return Ad, Bd, Fd
