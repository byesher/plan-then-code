"""Basic linear algebra utilities (nested lists)."""


def vector_add(a, b):
    return [x + y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = []
    for i in range(m):
        row = []
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            row.append(s)
        result.append(row)
    return result


def transpose(A):
    return [list(col) for col in zip(*A)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def determinant(A):
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    total = 0.0
    for j in range(n):
        minor = [[A[i][k] for k in range(n) if k != j] for i in range(1, n)]
        total += ((-1) ** j) * A[0][j] * determinant(minor)
    return total
