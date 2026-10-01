"""A typed, nestable config parser."""


def _parse_value(s):
    s = s.strip()
    if s == "true":
        return True
    if s == "false":
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x.strip()) for x in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _format_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _ensure_path(config, path):
    node = config
    for p in path.split("."):
        if p:
            node = node.setdefault(p, {})
    return node


def parse(text):
    config = {}
    current = config
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = _ensure_path(config, line[1:-1].strip())
        elif "=" in line:
            key, _, val = line.partition("=")
            current[key.strip()] = _parse_value(val)
    return config


def dumps(config):
    lines = []

    def emit_scalars(node):
        for k in sorted(node):
            if not isinstance(node[k], dict):
                lines.append(f"{k} = {_format_value(node[k])}")

    def rec(node, prefix):
        for k in sorted(node):
            if not isinstance(node[k], dict):
                continue
            full = f"{prefix}.{k}" if prefix else k
            lines.append(f"[{full}]")
            emit_scalars(node[k])
            rec(node[k], full)

    emit_scalars(config)
    rec(config, "")
    return "\n".join(lines)


def get_path(config, path, default=None):
    node = config
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def set_path(config, path, value):
    parts = [p for p in path.split(".") if p]
    if not parts:
        return
    node = config
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def flatten(config):
    result = {}

    def rec(node, prefix):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rec(v, key)
            else:
                result[key] = v

    rec(config, "")
    return result


def unflatten(flat):
    result = {}
    for key, value in flat.items():
        set_path(result, key, value)
    return result


def merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = v
    return result
