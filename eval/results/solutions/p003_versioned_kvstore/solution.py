"""A versioned key-value store with compare-and-set (CAS)."""


class KVStore:
    def __init__(self):
        self._data = {}
        self._version = {}

    def put(self, key, value):
        self._data[key] = value
        self._version[key] = self._version.get(key, 0) + 1
        return self._version[key]

    def get(self, key):
        return self._data.get(key)

    def get_version(self, key):
        return self._version.get(key, 0)

    def delete(self, key):
        if key not in self._data:
            return False
        del self._data[key]
        del self._version[key]
        return True

    def put_if_version(self, key, value, expected_version):
        if self._version.get(key, 0) != expected_version:
            return False
        self.put(key, value)
        return True

    def list_keys(self):
        return sorted(self._data)

    def size(self):
        return len(self._data)