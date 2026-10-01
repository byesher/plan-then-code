import pytest

from solution import (
    distance,
    line_intersection,
    polygon_area,
    point_in_polygon,
    convex_hull,
    bounding_box,
    is_convex,
)


def test_distance():
    assert distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_line_intersection_cross():
    p = line_intersection((0, 0), (2, 2), (0, 2), (2, 0))
    assert p is not None
    assert p[0] == pytest.approx(1.0) and p[1] == pytest.approx(1.0)


def test_line_intersection_parallel():
    assert line_intersection((0, 0), (2, 0), (0, 2), (2, 2)) is None


def test_line_intersection_outside_segment():
    assert line_intersection((0, 0), (1, 0), (0, 2), (1, 3)) is None


def test_polygon_area_square():
    assert polygon_area([(0, 0), (2, 0), (2, 2), (0, 2)]) == pytest.approx(4.0)


def test_polygon_area_triangle():
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6.0)


def test_point_in_polygon():
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert point_in_polygon((1, 1), square) is True
    assert point_in_polygon((3, 3), square) is False


def test_convex_hull():
    pts = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 1)]
    assert sorted(convex_hull(pts)) == sorted([(0, 0), (2, 0), (2, 2), (0, 2)])


def test_bounding_box():
    assert bounding_box([(1, 3), (2, -1), (0, 5)]) == (0, -1, 2, 5)


def test_bounding_box_empty():
    assert bounding_box([]) is None


def test_is_convex_true():
    assert is_convex([(0, 0), (2, 0), (2, 2), (0, 2)]) is True


def test_is_convex_false():
    assert is_convex([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)]) is False
