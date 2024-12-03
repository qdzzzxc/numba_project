from numba import njit, prange
import numpy as np

@njit
def matmul_int64(A, B):
    n, k_A = A.shape
    k_B, m = B.shape
    if k_A != k_B:
        raise ValueError(f"Can't multiply {n}x{k_A} with {k_B}x{m}")
    result = np.zeros((n, m), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(k_A):
                result[i, j] += A[i, k] * B[k, j]
    return result

@njit(parallel=True)
def matmul_int64_parallel(A, B):
    n, k_A = A.shape
    k_B, m = B.shape

    if k_A != k_B:
        raise ValueError(f"Can't multiply {n}x{k_A} with {k_B}x{m}")
    
    result = np.zeros((n, m), dtype=np.int32)

    for i in prange(n):
        for j in prange(m):
            for k in prange(k_A):
                result[i, j] += A[i, k] * B[k, j]
    return result

matmul_int64(np.array([[1]]), np.array([[1]]))
matmul_int64_parallel(np.array([[1]]), np.array([[1]]))