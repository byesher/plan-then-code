from solution import overlaps, intersection, merge, coverage_length


def test_overlaps_true():
    assert overlaps((0, 5), (3, 8)) is True


def test_overlaps_touching_is_false():
    assert overlaps((0, 5), (5, 8)) is False


def test_intersection_basic():
    assert intersection((0, 5), (3, 8)) == (3, 5)


def test_intersection_none():
    assert intersection((0, 5), (5, 8)) is None


def test_merge_basic():
    assert merge([(1, 3), (2, 6), (8, 10), (15, 18)]) == [(1, 6), (8, 10), (15, 18)]


def test_merge_unsorted():
    assert merge([(8, 10), (1, 3), (15, 18), (2, 6)]) == [(1, 6), (8, 10), (15, 18)]


def test_merge_touching():
    assert merge([(0, 2), (2, 4)]) == [(0, 4)]


def test_merge_empty():
    assert merge([]) == []


def test_coverage_length():
    assert coverage_length([(1, 3), (2, 6)]) == 5
    assert coverage_length([]) == 0
