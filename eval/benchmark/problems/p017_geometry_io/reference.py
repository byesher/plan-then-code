import sys


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def polygon_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def point_in_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside


for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "dist":
        x1, y1, x2, y2 = map(float, parts[1:5])
        print(distance((x1, y1), (x2, y2)))
    elif cmd == "area":
        n = int(parts[1])
        coords = list(map(float, parts[2:2 + 2 * n]))
        pts = [(coords[i], coords[i + 1]) for i in range(0, 2 * n, 2)]
        print(polygon_area(pts))
    elif cmd == "inside":
        x, y = map(float, parts[1:3])
        n = int(parts[3])
        coords = list(map(float, parts[4:4 + 2 * n]))
        pts = [(coords[i], coords[i + 1]) for i in range(0, 2 * n, 2)]
        print("true" if point_in_polygon((x, y), pts) else "false")
