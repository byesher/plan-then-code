import sys
import heapq
from collections import defaultdict

g = defaultdict(dict)
edges = set()


def dijkstra(s, t):
    dist = {s: 0}
    prev = {}
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == t:
            break
        for v, w in g[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if t not in dist:
        return None, []
    path = []
    cur = t
    while cur != s:
        path.append(cur)
        cur = prev[cur]
    path.append(s)
    path.reverse()
    return dist[t], path


def mst():
    all_edges = sorted((g[u][v], u, v) for u, v in edges)
    parent = {v: v for v in g}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    cnt = 0
    n = len(g)
    for w, u, v in all_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
            total += w
            cnt += 1
            if cnt == n - 1:
                break
    return total


for line in sys.stdin:
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "add_edge":
        u, v, w = int(parts[1]), int(parts[2]), int(parts[3])
        g[u][v] = w
        g[v][u] = w
        edges.add(frozenset((u, v)))
    elif cmd == "dijkstra":
        s, t = int(parts[1]), int(parts[2])
        d, path = dijkstra(s, t)
        if d is None:
            print("inf")
        else:
            print(d, " ".join(str(x) for x in path))
    elif cmd == "mst":
        print(mst())
