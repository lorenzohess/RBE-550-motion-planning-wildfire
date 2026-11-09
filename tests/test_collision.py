import math

import numpy as np

from wildfire import reeds_shepp as rs
from wildfire.collision import Checker, exact_free
from wildfire.truck import TruckSpec
from wildfire.world import SIZE_M, build_world


def test_fast_checker_matches_exact_reference():
    world = build_world(5)
    spec = TruckSpec()
    checker = Checker(world, spec)
    rng = np.random.default_rng(0)
    # Bias samples towards obstacles so the exact stage gets exercised.
    centres = world.centres[rng.integers(world.n, size=600)]
    x = np.concatenate([centres[:, 0] + rng.uniform(-9, 9, 600), rng.uniform(0, SIZE_M, 200)])
    y = np.concatenate([centres[:, 1] + rng.uniform(-9, 9, 600), rng.uniform(0, SIZE_M, 200)])
    psi = rng.uniform(-math.pi, math.pi, len(x))
    fast = checker.free(x, y, psi)
    for i in range(len(x)):
        assert fast[i] == exact_free(world, checker.body, x[i], y[i], psi[i]), i
    assert checker.exact_checked < 0.8 * checker.poses_checked
    assert 0.2 < fast.mean() < 0.95


def test_paths_free_flags_a_path_through_a_cell():
    world = build_world(5)
    spec = TruckSpec()
    checker = Checker(world, spec)
    cx, cy = world.centres[0]
    start = (cx - 30.0, cy, 0.0)
    goal = (cx + 30.0, cy, 0.0)
    path = rs.shortest(start, goal, spec.rho)
    assert not checker.paths_free([start], [path])[0]
