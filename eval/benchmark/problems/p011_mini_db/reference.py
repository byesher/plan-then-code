"""A small in-memory relational database."""


class Database:
    def __init__(self):
        self._tables = {}  # name -> {"columns": [...], "rows": [...]}

    def _check_table(self, table):
        if table not in self._tables:
            raise ValueError(f"table does not exist: {table}")

    def create_table(self, name, columns):
        self._tables[name] = {"columns": list(columns), "rows": []}

    def tables(self):
        return sorted(self._tables)

    def columns(self, table):
        self._check_table(table)
        return list(self._tables[table]["columns"])

    def insert(self, table, row):
        self._check_table(table)
        if not isinstance(row, dict):
            raise ValueError("row must be a dict")
        cols = self._tables[table]["columns"]
        for c in cols:
            if c not in row:
                raise ValueError(f"missing column: {c}")
        self._tables[table]["rows"].append({c: row[c] for c in cols})

    def _matches(self, row, where):
        return all(row.get(k) == v for k, v in where.items())

    def select(self, table, where=None, order_by=None, reverse=False):
        self._check_table(table)
        rows = self._tables[table]["rows"]
        if where is not None:
            rows = [r for r in rows if self._matches(r, where)]
        result = [dict(r) for r in rows]
        if order_by is not None:
            result.sort(key=lambda r: r.get(order_by), reverse=reverse)
        return result

    def update(self, table, where, values):
        self._check_table(table)
        count = 0
        for r in self._tables[table]["rows"]:
            if self._matches(r, where):
                r.update(values)
                count += 1
        return count

    def delete(self, table, where):
        self._check_table(table)
        before = len(self._tables[table]["rows"])
        self._tables[table]["rows"] = [
            r for r in self._tables[table]["rows"] if not self._matches(r, where)
        ]
        return before - len(self._tables[table]["rows"])

    def count(self, table, where=None):
        self._check_table(table)
        rows = self._tables[table]["rows"]
        if where is None:
            return len(rows)
        return sum(1 for r in rows if self._matches(r, where))

    def join(self, t1, t2, key):
        self._check_table(t1)
        self._check_table(t2)
        result = []
        for r1 in self._tables[t1]["rows"]:
            for r2 in self._tables[t2]["rows"]:
                if r1.get(key) == r2.get(key):
                    merged = dict(r1)
                    for k, v in r2.items():
                        if k != key:
                            merged[k] = v
                    result.append(merged)
        return result

    def aggregate(self, table, column, func):
        self._check_table(table)
        rows = self._tables[table]["rows"]
        if not rows:
            return 0.0
        vals = [r.get(column) for r in rows if column in r]
        if not vals:
            return 0.0
        if func == "sum":
            return float(sum(vals))
        if func == "avg":
            return float(sum(vals)) / len(vals)
        if func == "min":
            return float(min(vals))
        if func == "max":
            return float(max(vals))
        raise ValueError(f"unknown aggregate function: {func}")
