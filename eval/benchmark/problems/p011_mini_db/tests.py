import pytest

from solution import Database


def _make_db():
    db = Database()
    db.create_table("users", ["id", "name", "age", "city"])
    db.insert("users", {"id": 1, "name": "Alice", "age": 30, "city": "NY"})
    db.insert("users", {"id": 2, "name": "Bob", "age": 25, "city": "LA"})
    db.insert("users", {"id": 3, "name": "Carol", "age": 30, "city": "NY"})
    return db


def test_tables_columns():
    db = _make_db()
    assert db.tables() == ["users"]
    assert db.columns("users") == ["id", "name", "age", "city"]


def test_select_all():
    db = _make_db()
    assert len(db.select("users")) == 3


def test_select_where():
    db = _make_db()
    rows = db.select("users", where={"age": 30})
    assert sorted(r["name"] for r in rows) == ["Alice", "Carol"]


def test_select_order_by():
    db = _make_db()
    rows = db.select("users", order_by="age")
    assert [r["name"] for r in rows] == ["Bob", "Alice", "Carol"]
    rows_desc = db.select("users", order_by="age", reverse=True)
    assert rows_desc[0]["name"] == "Alice"


def test_update():
    db = _make_db()
    n = db.update("users", {"age": 30}, {"city": "SF"})
    assert n == 2
    assert db.count("users", where={"city": "SF"}) == 2


def test_delete():
    db = _make_db()
    n = db.delete("users", {"city": "NY"})
    assert n == 2
    assert db.count("users") == 1


def test_count():
    db = _make_db()
    assert db.count("users") == 3
    assert db.count("users", where={"age": 30}) == 2


def test_join():
    db = Database()
    db.create_table("users", ["id", "name"])
    db.create_table("orders", ["id", "amount"])
    db.insert("users", {"id": 1, "name": "Alice"})
    db.insert("users", {"id": 2, "name": "Bob"})
    db.insert("orders", {"id": 1, "amount": 100})
    db.insert("orders", {"id": 1, "amount": 50})
    db.insert("orders", {"id": 3, "amount": 99})
    rows = db.join("users", "orders", key="id")
    assert len(rows) == 2
    assert sorted(r["amount"] for r in rows) == [50, 100]
    assert all(r["name"] == "Alice" for r in rows)


def test_aggregate():
    db = _make_db()
    assert db.aggregate("users", "age", "sum") == pytest.approx(85.0)
    assert db.aggregate("users", "age", "avg") == pytest.approx(85.0 / 3)
    assert db.aggregate("users", "age", "min") == pytest.approx(25.0)
    assert db.aggregate("users", "age", "max") == pytest.approx(30.0)


def test_insert_missing_column():
    db = _make_db()
    with pytest.raises(ValueError):
        db.insert("users", {"id": 4, "name": "Dave"})


def test_missing_table():
    db = _make_db()
    with pytest.raises(ValueError):
        db.select("nope")
