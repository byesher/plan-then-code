import sys


def evaluate(coeffs, x):
    r = 0.0
    for c in reversed(coeffs):
        r = r * x + c
    return r


def bisection(a, b):
    f = lambda x: x * x - 2
    for _ in range(100):
        mid = (a + b) / 2.0
        if abs(f(mid)) < 1e-9 or (b - a) / 2.0 < 1e-9:
            return mid
        if f(mid) * f(a) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2.0


for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "poly":
        coeffs = list(map(float, parts[1:-1]))
        x = float(parts[-1])
        print(evaluate(coeffs, x))
    elif cmd == "bisect":
        a, b = float(parts[1]), float(parts[2])
        print(f"{bisection(a, b):.6f}")
