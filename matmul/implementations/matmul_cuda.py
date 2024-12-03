import numpy as np
from numba import cuda

@cuda.jit
def matmul_gpu(A, B, result):
    """Умножение матриц на GPU."""
    row = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    col = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if row < result.shape[0] and col < result.shape[1]:
        temp = 0.0
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col]
        result[row, col] = temp

def matmul_int64(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Матрицы имеют несовместимые размеры для умножения."

    m, k = A.shape
    n = B.shape[1]

    result = np.zeros((m, n), dtype=np.int64)

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_result = cuda.device_array((m, n), dtype=np.int64)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (result.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (result.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    matmul_gpu[blocks_per_grid, threads_per_block](d_A, d_B, d_result)

    result = d_result.copy_to_host()
    return result

matmul_int64(np.array([[1]]), np.array([[1]]))