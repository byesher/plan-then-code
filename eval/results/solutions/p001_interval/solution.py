"""Interval utilities. All intervals are [start, end) with start < end."""


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def intersection(a, b):
    if not overlaps(a, b):
        return None
    return (max(a[0], b[0]), min(a[1], b[1]))


def merge(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def coverage_length(intervals):
    return sum(end - start for start, end in merge(intervals))