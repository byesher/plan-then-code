from solution import Graph


def test_add_vertex_and_edge():
    g = Graph()
    g.add_vertex(0)
    assert g.has_vertex(0) is True
    assert g.has_vertex(1) is False
    g.add_edge(0, 1)
    assert g.has_edge(0, 1) is True
    assert g.has_edge(1, 0) is True
    assert g.has_vertex(1) is True  # auto-created


def test_neighbors_sorted():
    g = Graph()
    g.add_edge(0, 3)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    assert g.neighbors(0) == [1, 2, 3]
    assert g.neighbors(99) == []


def test_bfs():
    g = Graph()
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 4)]:
        g.add_edge(u, v)
    assert g.bfs(0) == [0, 1, 2, 3, 4]


def test_dfs():
    g = Graph()
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 4)]:
        g.add_edge(u, v)
    assert g.dfs(0) == [0, 1, 3, 2, 4]


def test_dijkstra():
    g = Graph()
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 1)
    g.add_edge(2, 1, 2)
    g.add_edge(2, 3, 5)
    g.add_edge(1, 3, 1)
    dist, path = g.dijkstra(0, 3)
    assert dist == 4
    assert path == [0, 2, 1, 3]


def test_dijkstra_unreachable():
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    dist, path = g.dijkstra(0, 3)
    assert dist == float('inf')
    assert path == []


def test_connected_components():
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    g.add_vertex(4)
    assert g.connected_components() == [[0, 1], [2, 3], [4]]


def test_has_cycle_true():
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    assert g.has_cycle() is True


def test_has_cycle_false():
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert g.has_cycle() is False


def test_mst_weight():
    g = Graph()
    g.add_edge(0, 1, 4)
    g.add_edge(0, 2, 1)
    g.add_edge(2, 1, 2)
    g.add_edge(2, 3, 5)
    g.add_edge(1, 3, 1)
    assert g.mst_weight() == 4
