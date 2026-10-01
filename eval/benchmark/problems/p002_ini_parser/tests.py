from solution import parse_ini, write_ini, get_value, get_int, get_bool, sections

SAMPLE = """
; comment
# another comment
name = example
port = 8080

[database]
host = localhost
port = 5432
use_ssl = true

[network]
timeout = 30
"""


def test_parse_sections():
    assert sections(parse_ini(SAMPLE)) == ["database", "network"]


def test_parse_global_keys():
    cfg = parse_ini(SAMPLE)
    assert get_value(cfg, "", "name") == "example"


def test_get_int():
    cfg = parse_ini(SAMPLE)
    assert get_int(cfg, "network", "timeout") == 30
    assert get_int(cfg, "network", "missing") == 0
    assert get_int(cfg, "network", "missing", default=7) == 7


def test_get_bool():
    cfg = parse_ini(SAMPLE)
    assert get_bool(cfg, "database", "use_ssl") is True


def test_get_value_default():
    cfg = parse_ini(SAMPLE)
    assert get_value(cfg, "database", "missing", "x") == "x"


def test_roundtrip():
    cfg = parse_ini(SAMPLE)
    cfg2 = parse_ini(write_ini(cfg))
    assert cfg2 == cfg
