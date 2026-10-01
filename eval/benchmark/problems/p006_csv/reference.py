"""CSV parsing/serialization with proper quoting."""


def parse_csv(text):
    if not text:
        return []
    rows = []
    row = []
    field = []
    in_quotes = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                else:
                    in_quotes = False
                    i += 1
            else:
                field.append(c)
                i += 1
        elif c == '"':
            in_quotes = True
            i += 1
        elif c == ',':
            row.append(''.join(field))
            field = []
            i += 1
        elif c == '\n':
            row.append(''.join(field))
            field = []
            rows.append(row)
            row = []
            i += 1
        elif c == '\r':
            i += 1
        else:
            field.append(c)
            i += 1
    if field or row:
        row.append(''.join(field))
        rows.append(row)
    return rows


def write_csv(rows):
    lines = [','.join(_escape(field) for field in row) for row in rows]
    return '\n'.join(lines) + ('\n' if lines else '')


def _escape(field):
    s = str(field)
    if any(ch in s for ch in (',', '"', '\n', '\r')):
        return '"' + s.replace('"', '""') + '"'
    return s


def to_records(rows):
    if not rows:
        return []
    header = rows[0]
    records = []
    for row in rows[1:]:
        rec = {}
        for j, name in enumerate(header):
            rec[name] = row[j] if j < len(row) else ""
        records.append(rec)
    return records


def to_table(records):
    if not records:
        return []
    header = list(records[0].keys())
    table = [header]
    for rec in records:
        table.append([rec.get(k, "") for k in header])
    return table


def column(rows, name):
    if not rows:
        return []
    header = rows[0]
    if name not in header:
        return []
    idx = header.index(name)
    return [row[idx] if idx < len(row) else "" for row in rows[1:]]
