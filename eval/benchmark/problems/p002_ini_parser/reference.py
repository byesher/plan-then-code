"""A small INI config parser.

Format: [section] for sections, key = value for entries, # or ; for comments.
Keys before any section belong to the "" (global) section.
"""


def parse_ini(text):
    config = {}
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            config.setdefault(section, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            config.setdefault(section, {})[key] = value
    return config


def write_ini(config):
    lines = []
    for key in sorted(config.get("", {})):
        lines.append(f"{key} = {config[''][key]}")
    for section in sorted(s for s in config if s != ""):
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key in sorted(config[section]):
            lines.append(f"{key} = {config[section][key]}")
    return "\n".join(lines)


def get_value(config, section, key, default=None):
    return config.get(section, {}).get(key, default)


def get_int(config, section, key, default=0):
    raw = get_value(config, section, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def get_bool(config, section, key, default=False):
    raw = get_value(config, section, key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def sections(config):
    return sorted(s for s in config if s != "")
