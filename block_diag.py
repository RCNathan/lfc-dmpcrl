import numpy as np
import casadi as cs


def block_diag(
    *arrays: np.ndarray | cs.SX,
    n: int = 0,
) -> np.ndarray | cs.SX:
    """
    Returns block-diagonal matrix using either
        - any number of matrices provided, or
        - one matrix, n amount of times (use n=int)
    Works with both numpy and CasADi symbolic matrices.
    """
    # using n copies of first entry
    if n != 0:
        if type(n) != int:
            raise Exception("Use an integer for n")

        if isinstance(arrays[0], np.ndarray):
            # for np.ndarrays
            total_shape = np.sum(np.array([arrays[0].shape for i in range(n)]), axis=0)
            diag = np.zeros(total_shape, dtype=arrays[0].dtype)

            row, col = 0, 0
            for _ in range(n):
                r, c = arrays[0].shape
                diag[row : row + r, col : col + c] = arrays[0]
                row += r
                col += c
        elif isinstance(arrays[0], cs.SX):
            # for cs.SX symbolic matrices
            diag = None  # Initialize empty CasADi matrix
            row, col = 0, 0
            for _ in range(n):
                if diag is None:
                    diag = arrays[0]  # initialize with the first matrix
                else:
                    # Vertically concatenate zeros and arrays[0] for a block structure
                    diag = cs.diagcat(diag, arrays[0])
    # using all entries
    else:
        if isinstance(arrays[0], np.ndarray):
            # determine total shape
            shapes = np.array([a.shape for a in arrays])
            total_shape = np.sum(shapes, axis=0)

            # initialize block diagonal matrix with zeros
            diag = np.zeros(total_shape, dtype=arrays[0].dtype)

            row, col = 0, 0
            for a in arrays:
                r, c = a.shape
                diag[row : row + r, col : col + c] = a
                row += r
                col += c
        elif isinstance(arrays[0], cs.SX):
            # Case when multiple arrays are provided
            diag = None
            for a in arrays:
                if diag is None:
                    diag = a  # Initialize with the first matrix
                else:
                    diag = cs.diagcat(diag, a)
    return diag


# # Example usage with CasADi SX symbolic matrices
# A = cs.SX.sym('A', 2, 2)
# B = cs.SX.sym('B', 3, 3)

# # Block diagonal with multiple matrices
# result = block_diag(A, B)

# # Block diagonal with one matrix repeated n times
# result_n = block_diag(A, n=3)

# # Display the results
# print(result)
# print(result_n)
