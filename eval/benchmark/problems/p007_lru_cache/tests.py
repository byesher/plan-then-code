from solution import LRUCache


def test_put_get():
    c = LRUCache(2)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") == -1


def test_eviction():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # evicts a (least recently used)
    assert c.get("a") == -1
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_get_refreshes_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")      # a becomes most-recently-used
    c.put("c", 3)   # evicts b
    assert c.get("a") == 1
    assert c.get("b") == -1
    assert c.get("c") == 3


def test_peek_does_not_change_order():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.peek("a") == 1
    c.put("c", 3)   # a still least-recently-used, gets evicted
    assert c.get("a") == -1


def test_update_existing():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("a", 10)
    assert c.get("a") == 10
    assert c.size() == 1


def test_keys_order():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.keys() == ["b", "a"]  # most-recent first
    c.get("a")
    assert c.keys() == ["a", "b"]


def test_contains_size_clear():
    c = LRUCache(2)
    c.put("a", 1)
    assert c.contains("a") is True
    assert c.contains("b") is False
    assert c.size() == 1
    c.clear()
    assert c.size() == 0
    assert c.get("a") == -1


def test_zero_capacity():
    c = LRUCache(0)
    c.put("a", 1)
    assert c.size() == 0
    assert c.get("a") == -1
