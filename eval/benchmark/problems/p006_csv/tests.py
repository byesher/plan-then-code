from solution import parse_csv, write_csv, to_records, to_table, column

SAMPLE = 'name,age,city\n"Alice, A.",30,"New York"\n"Bob ""B""",25,LA\n'


def test_parse_simple():
    assert parse_csv("a,b,c\n1,2,3\n") == [["a", "b", "c"], ["1", "2", "3"]]


def test_parse_quoted_comma():
    assert parse_csv('"a,b",c\n') == [["a,b", "c"]]


def test_parse_escaped_quote():
    assert parse_csv('"a""b",c\n') == [['a"b', "c"]]


def test_parse_newline_in_quotes():
    assert parse_csv('"a\nb",c\n') == [["a\nb", "c"]]


def test_parse_full_sample():
    rows = parse_csv(SAMPLE)
    assert rows[0] == ["name", "age", "city"]
    assert rows[1] == ["Alice, A.", "30", "New York"]
    assert rows[2] == ['Bob "B"', "25", "LA"]


def test_roundtrip():
    rows = parse_csv(SAMPLE)
    assert parse_csv(write_csv(rows)) == rows


def test_to_records():
    rows = [["name", "age"], ["Alice", "30"], ["Bob", "25"]]
    assert to_records(rows) == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]


def test_to_table():
    records = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
    assert to_table(records) == [["name", "age"], ["Alice", "30"], ["Bob", "25"]]


def test_column():
    rows = [["name", "age"], ["Alice", "30"], ["Bob", "25"]]
    assert column(rows, "age") == ["30", "25"]
    assert column(rows, "nope") == []
