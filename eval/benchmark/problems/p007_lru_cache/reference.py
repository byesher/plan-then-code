"""An LRU cache."""

from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self._capacity = capacity
        self._data = OrderedDict()

    def capacity(self):
        return self._capacity

    def get(self, key):
        if key not in self._data:
            return -1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if self._capacity <= 0:
            return
        if key in self._data:
            self._data[key] = value
            self._data.move_to_end(key)
            return
        if len(self._data) >= self._capacity:
            self._data.popitem(last=False)  # evict least-recently-used
        self._data[key] = value

    def peek(self, key):
        return self._data.get(key, -1)

    def contains(self, key):
        return key in self._data

    def size(self):
        return len(self._data)

    def keys(self):
        return list(reversed(self._data))

    def clear(self):
        self._data.clear()
