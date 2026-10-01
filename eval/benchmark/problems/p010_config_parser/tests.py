from solution import parse, dumps, get_path, set_path, flatten, unflatten, merge

SAMPLE = """
# comment
host = localhost
port = 8080
debug = true
ratio = 0.5
tags = [a, b, c]

[database]
host = db.local
port = 5432

[network.timeout]
connect = 30
read = 60
"""


def test_parse_types():
    cfg = parse(SAMPLE)
    assert cfg["host"] == "localhost"
    assert cfg["port"] == 8080
    assert cfg["debug"] is True
    assert cfg["ratio"] == 0.5
    assert cfg["tags"] == ["a", "b", "c"]


def test_parse_nested_sections():
    cfg = parse(SAMPLE)
    assert cfg["database"] == {"host": "db.local", "port": 5432}
    assert cfg["network"]["timeout"] == {"connect": 30, "read": 60}


def test_dumps_roundtrip():
    cfg = parse(SAMPLE)
    assert parse(dumps(cfg)) == cfg


def test_get_path():
    cfg = parse(SAMPLE)
    assert get_path(cfg, "database.host") == "db.local"
    assert get_path(cfg, "network.timeout.read") == 60
    assert get_path(cfg, "nope.missing", "x") == "x"


def test_set_path():
    cfg = {}
    set_path(cfg, "a.b.c", 1)
    assert cfg == {"a": {"b": {"c": 1}}}


def test_flatten_unflatten():
    cfg = parse(SAMPLE)
    flat = flatten(cfg)
    assert flat["database.host"] == "db.local"
    assert flat["network.timeout.connect"] == 30
    assert unflatten(flat) == cfg


def test_merge():
    a = {"x": 1, "y": {"p": 1, "q": 2}}
    b = {"y": {"q": 20, "r": 30}, "z": 3}
    assert merge(a, b) == {"x": 1, "y": {"p": 1, "q": 20, "r": 30}, "z": 3}
