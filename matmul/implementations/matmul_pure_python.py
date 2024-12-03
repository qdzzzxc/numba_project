def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Функция матричного умножения"""
    n = len(A)
    k_A = len(A[0])
    k_B = len(B)
    m = len(B[0])

    assert k_A == k_B, f"Can't multiply {n}x{k_A} on {k_B}x{m}"

    result = [[0 for j in range(m)] for i in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(k_A):
                result[i][j] += A[i][k] * B[k][j]

    return result