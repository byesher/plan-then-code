import sys

config = {}
section = ""
for line in sys.stdin.read().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("[") and line.endswith("]"):
        section = line[1:-1].strip()
    elif "=" in line:
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        full = f"{section}.{key}" if section else key
        config[full] = val
for k in sorted(config):
    print(f"{k} = {config[k]}")
