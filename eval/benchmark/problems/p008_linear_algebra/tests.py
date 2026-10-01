import pytest

from solution import vector_add, dot, matmul, transpose, identity, determinant


def test_vector_add():
    assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_dot():
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_matmul():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert matmul(A, B) == [[19, 22], [43, 50]]


def test_matmul_rectangular():
    A = [[1, 2, 3], [4, 5, 6]]   # 2x3
    B = [[7, 8], [9, 10], [11, 12]]  # 3x2
    assert matmul(A, B) == [[58, 64], [139, 154]]


def test_transpose():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_identity():
    assert identity(3) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_determinant_1x1():
    assert determinant([[5]]) == 5


def test_determinant_2x2():
    assert determinant([[1, 2], [3, 4]]) == -2


def test_determinant_3x3():
    A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    assert determinant(A) == pytest.approx(1.0)
