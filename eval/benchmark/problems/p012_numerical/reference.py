"""Numerical methods: polynomials, root-finding, integration."""


def evaluate_polynomial(coeffs, x):
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def derivative(coeffs):
    return [i * coeffs[i] for i in range(1, len(coeffs))]


def bisection(f, a, b, tol=1e-6, max_iter=100):
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fm = f(mid)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return mid
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return (a + b) / 2.0


def newton(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        d = df(x)
        if d == 0:
            raise ValueError("derivative is zero")
        x = x - fx / d
    return x


def trapezoidal(f, a, b, n):
    h = (b - a) / n
    total = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += 4.0 * f(x) if i % 2 == 1 else 2.0 * f(x)
    return total * h / 3.0


def riemann_sum(f, a, b, n):
    h = (b - a) / n
    return sum(f(a + i * h) for i in range(n)) * h
