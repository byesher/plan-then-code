import math

import pytest

from solution import (
    evaluate_polynomial,
    derivative,
    bisection,
    newton,
    trapezoidal,
    simpson,
    riemann_sum,
)


def test_evaluate_polynomial():
    # 1 + 2x + 3x^2 at x=2 -> 17
    assert evaluate_polynomial([1, 2, 3], 2) == pytest.approx(17.0)


def test_derivative():
    # d/dx (1 + 2x + 3x^2) = 2 + 6x
    assert derivative([1, 2, 3]) == [2, 6]


def test_bisection():
    root = bisection(lambda x: x * x - 2, 0, 2)
    assert root == pytest.approx(math.sqrt(2), rel=1e-5)


def test_newton():
    root = newton(lambda x: x * x - 2, lambda x: 2 * x, 1.0)
    assert root == pytest.approx(math.sqrt(2), rel=1e-5)


def test_bisection_same_sign_raises():
    with pytest.raises(ValueError):
        bisection(lambda x: x * x + 1, 0, 2)


def test_trapezoidal():
    v = trapezoidal(lambda x: x * x, 0, 1, 1000)
    assert v == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_simpson():
    # Simpson's rule is exact for quadratics
    v = simpson(lambda x: x * x, 0, 1, 101)  # odd n -> auto even
    assert v == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_riemann_sum():
    v = riemann_sum(lambda x: x * x, 0, 1, 10000)
    assert v == pytest.approx(1.0 / 3.0, rel=1e-3)
