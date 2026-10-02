import sys


for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "det":
        a, b, c, d = map(float, parts[1:5])
        print(a * d - b * c)
    elif cmd == "matmul":
        a, b, c, d, e, f, g, h = map(float, parts[1:9])
        print(a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
