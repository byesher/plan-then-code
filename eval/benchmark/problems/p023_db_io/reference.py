import sys

tables = {}
cols = {}

for line in sys.stdin.read().splitlines():
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "create":
        name = parts[1]
        cols[name] = parts[2:]
        tables[name] = []
    elif cmd == "insert":
        name = parts[1]
        tables[name].append(parts[2:])
    elif cmd == "select":
        name = parts[1]
        for row in tables.get(name, []):
            print(" ".join(row))
    elif cmd == "count":
        name = parts[1]
        print(len(tables.get(name, [])))
