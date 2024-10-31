import numpy as np


def ctrb(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, int]:
    """Calculates controllability/Kalman matrix [B AB ...] and returns additionally the rank"""
    n = A.shape[0]
    entry = B  # iteratively use previous result for next matmul

    ctrb = B
    for i in range(n - 1):
        entry = A @ entry
        ctrb = np.hstack([ctrb, entry])  # final shape = (n x nm)
    rank = np.linalg.matrix_rank(ctrb)
    return ctrb, rank
