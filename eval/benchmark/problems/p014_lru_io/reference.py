import sys
from collections import OrderedDict


def main():
    lines = sys.stdin.read().splitlines()
    if not lines:
        return
    capacity = int(lines[0].strip())
    cache = OrderedDict()
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "put":
            key, value = parts[1], parts[2]
            if key in cache:
                cache.pop(key)
            elif len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = value
        elif parts[0] == "get":
            key = parts[1]
            if key in cache:
                print(cache[key])
                cache.move_to_end(key)
            else:
                print(-1)


if __name__ == "__main__":
    main()
