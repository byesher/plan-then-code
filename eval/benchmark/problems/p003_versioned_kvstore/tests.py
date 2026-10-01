from solution import KVStore


def test_put_get():
    s = KVStore()
    assert s.put("a", 1) == 1
    assert s.get("a") == 1


def test_version_increments():
    s = KVStore()
    s.put("a", 1)
    s.put("a", 2)
    assert s.get_version("a") == 2


def test_get_version_missing():
    assert KVStore().get_version("x") == 0


def test_delete():
    s = KVStore()
    s.put("a", 1)
    assert s.delete("a") is True
    assert s.get("a") is None
    assert s.delete("a") is False


def test_cas_success():
    s = KVStore()
    s.put("a", 1)
    assert s.put_if_version("a", 10, 1) is True
    assert s.get("a") == 10
    assert s.get_version("a") == 2


def test_cas_failure():
    s = KVStore()
    s.put("a", 1)
    assert s.put_if_version("a", 10, 5) is False
    assert s.get("a") == 1


def test_cas_missing_key():
    s = KVStore()
    assert s.put_if_version("a", 1, 0) is True
    assert s.get("a") == 1


def test_list_keys_and_size():
    s = KVStore()
    s.put("b", 1)
    s.put("a", 2)
    s.put("c", 3)
    assert s.list_keys() == ["a", "b", "c"]
    assert s.size() == 3
