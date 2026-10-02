import sys


def main():
    data = {}
    version = {}
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "put":
            key, val = parts[1], int(parts[2])
            data[key] = val
            version[key] = version.get(key, 0) + 1
        elif cmd == "get":
            print(data.get(parts[1], -1))
        elif cmd == "get_version":
            print(version.get(parts[1], 0))
        elif cmd == "cas":
            key, val, expected = parts[1], int(parts[2]), int(parts[3])
            if version.get(key, 0) == expected:
                data[key] = val
                version[key] = version.get(key, 0) + 1
                print("true")
            else:
                print("false")
        elif cmd == "delete":
            key = parts[1]
            if key in data:
                del data[key]
                del version[key]
                print("true")
            else:
                print("false")
        elif cmd == "size":
            print(len(data))


if __name__ == "__main__":
    main()
