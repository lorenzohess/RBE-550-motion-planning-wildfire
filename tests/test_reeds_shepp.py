import math

import numpy as np
import pytest

from wildfire import reeds_shepp as rs

RHO = 13.0


def random_pose(rng, spread=60.0):
    return (float(rng.uniform(-spread, spread)), float(rng.uniform(-spread, spread)),
            float(rng.uniform(-math.pi, math.pi)))


def close(p, q, tol=1e-6):
    return (abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol
            and abs(rs.mod2pi(p[2] - q[2])) < tol)


def test_every_candidate_word_reaches_the_goal():
    rng = np.random.default_rng(0)
    used = set()
    for _ in range(3000):
        x, y, phi = random_pose(rng, 4.0)
        for steers, lengths in rs.candidates(x, y, phi):
            path = tuple(zip(steers, lengths))
            assert close(rs.end_pose((0.0, 0.0, 0.0), path, 1.0), (x, y, phi)), (steers, lengths)
            used.add((steers, tuple(np.sign(np.round(lengths, 9)))))
    assert len(used) > 30          # most of the 48 words show up


def test_shortest_reaches_goal_in_world_frame():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        a, b = random_pose(rng), random_pose(rng)
        path = rs.shortest(a, b, RHO)
        assert close(rs.end_pose(a, path, RHO), b)
        x, y, th = rs.sample(a, path, RHO, 0.25)
        assert close((x[-1], y[-1], th[-1]), b)
        step = np.hypot(np.diff(x), np.diff(y))
        assert step.max() <= 0.25 + 1e-9


def test_metric_is_symmetric_and_bounded_below_by_distance():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        a, b = random_pose(rng), random_pose(rng)
        d_ab = rs.length(rs.shortest(a, b, RHO))
        d_ba = rs.length(rs.shortest(b, a, RHO))
        assert d_ab == pytest.approx(d_ba, abs=1e-6)
        assert d_ab >= math.hypot(b[0] - a[0], b[1] - a[1]) - 1e-9


def test_triangle_inequality():
    """A missing or broken word family shows up as a shortcut through a third pose."""
    rng = np.random.default_rng(3)
    for _ in range(1500):
        a, b, c = random_pose(rng, 30.0), random_pose(rng, 30.0), random_pose(rng, 30.0)
        d = lambda p, q: rs.length(rs.shortest(p, q, RHO))  # noqa: E731
        assert d(a, c) <= d(a, b) + d(b, c) + 1e-6


def test_known_cases():
    # Straight ahead.
    assert rs.length(rs.shortest((0, 0, 0), (20, 0, 0), RHO)) == pytest.approx(20.0)
    # Straight back: reverse, not a loop.
    path = rs.shortest((0, 0, 0), (-20, 0, 0), RHO)
    assert rs.length(path) == pytest.approx(20.0) and all(a < 0 for _, a in path)
    # A quarter turn on the circle: one arc of length rho * pi / 2.
    path = rs.shortest((0, 0, 0), (RHO, RHO, math.pi / 2), RHO)
    assert rs.length(path) == pytest.approx(RHO * math.pi / 2)
    assert len(path) == 1 and path[0][0] == rs.L


def test_reverse_path_is_the_same_curve_backwards():
    rng = np.random.default_rng(4)
    for _ in range(200):
        a, b = random_pose(rng), random_pose(rng)
        path = rs.shortest(a, b, RHO)
        assert close(rs.end_pose(b, rs.reverse(path), RHO), a)


@pytest.mark.parametrize("cusp", [0.0, 70.0])
def test_vectorised_matches_scalar(cusp):
    rng = np.random.default_rng(5)
    a = np.array([random_pose(rng) for _ in range(2000)])
    b = np.array([random_pose(rng) for _ in range(2000)])
    paths, lengths, n_cusps = rs.shortest_many(a, b, RHO, cusp)
    for i in range(len(a)):
        ref = rs.shortest(tuple(a[i]), tuple(b[i]), RHO, cusp)
        assert rs.length(paths[i]) + cusp * rs.cusps(paths[i]) == pytest.approx(
            rs.length(ref) + cusp * rs.cusps(ref), abs=1e-6)
        assert close(rs.end_pose(tuple(a[i]), paths[i], RHO), tuple(b[i]))
        assert lengths[i] == pytest.approx(rs.length(paths[i]))
        assert n_cusps[i] == rs.cusps(paths[i])
