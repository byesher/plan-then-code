"""An undirected weighted graph with common algorithms."""

from collections import deque
import heapq


class Graph:
    def __init__(self):
        self._adj = {}       # vertex -> {neighbor: weight}
        self._edges = set()  # set of frozenset({u, v})

    def add_vertex(self, v):
        if v not in self._adj:
            self._adj[v] = {}

    def add_edge(self, u, v, weight=1):
        self.add_vertex(u)
        self.add_vertex(v)
        self._adj[u][v] = weight
        self._adj[v][u] = weight
        self._edges.add(frozenset((u, v)))

    def has_vertex(self, v):
        return v in self._adj

    def has_edge(self, u, v):
        return frozenset((u, v)) in self._edges

    def neighbors(self, v):
        if v not in self._adj:
            return []
        return sorted(self._adj[v])

    def bfs(self, start):
        if start not in self._adj:
            return []
        visited = {start}
        order = []
        q = deque([start])
        while q:
            v = q.popleft()
            order.append(v)
            for nb in sorted(self._adj[v]):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return order

    def dfs(self, start):
        if start not in self._adj:
            return []
        visited = set()
        order = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            order.append(v)
            for nb in sorted(self._adj[v], reverse=True):
                if nb not in visited:
                    stack.append(nb)
        return order

    def dijkstra(self, start, end):
        if start not in self._adj or end not in self._adj:
            return (float('inf'), [])
        dist = {start: 0}
        prev = {}
        pq = [(0, start)]
        visited = set()
        while pq:
            d, v = heapq.heappop(pq)
            if v in visited:
                continue
            visited.add(v)
            if v == end:
                break
            for nb, w in self._adj[v].items():
                nd = d + w
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    prev[nb] = v
                    heapq.heappush(pq, (nd, nb))
        if end not in dist:
            return (float('inf'), [])
        path = []
        cur = end
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.append(start)
        path.reverse()
        return (dist[end], path)

    def connected_components(self):
        seen = set()
        comps = []
        for v in sorted(self._adj):
            if v in seen:
                continue
            comp = self.bfs(v)
            seen.update(comp)
            comps.append(comp)
        return comps

    def has_cycle(self):
        visited = set()
        for start in self._adj:
            if start in visited:
                continue
            stack = [(start, None)]
            while stack:
                v, parent = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                for nb in self._adj[v]:
                    if nb not in visited:
                        stack.append((nb, v))
                    elif nb != parent:
                        return True
        return False

    def mst_weight(self):
        edges = []
        for e in self._edges:
            u, v = tuple(e)
            edges.append((self._adj[u][v], u, v))
        edges.sort()

        parent = {v: v for v in self._adj}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
                return True
            return False

        total = 0
        count = 0
        n = len(self._adj)
        for w, u, v in edges:
            if union(u, v):
                total += w
                count += 1
                if count == n - 1:
                    break
        return total
