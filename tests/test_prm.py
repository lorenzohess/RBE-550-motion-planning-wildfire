import numpy as np
import pytest

from wildfire import reeds_shepp as rs
from wildfire.collision import Checker
from wildfire.prm import BLOCKED, FREE, Roadmap
from wildfire.truck import TruckSpec, Trajectory
from wildfire.world import build_world, rng_for


@pytest.fixture(scope="module")
def built():
    world = build_world(2)
    spec = TruckSpec()
    checker = Checker(world, spec)
    rm = Roadmap(checker, spec, k=8, candidates=60)
    rm.build(800, rng_for(2, "prm"))
    return world, spec, rm


def test_edges_are_free_at_a_fine_step_without_margin(built):
    world, spec, rm = built
    fine = Checker(world, spec, margin=0.0, step=0.05)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(rm.edges), size=400, replace=False)
    starts = [rm.poses[rm.edges[e].u] for e in sample]
    paths = [rm.edges[e].path for e in sample]
    assert fine.paths_free(starts, paths).all()


def test_route_ends_at_the_target_and_times_agree(built):
    world, spec, rm = built
    lab = rm.components()
    big = np.flatnonzero(lab == np.bincount(lab).argmax())
    rng = np.random.default_rng(1)
    for _ in range(10):
        s, t = (int(v) for v in rng.choice(big, 2, replace=False))
        times = rm.times_from(s)
        route = rm.route(s, t)
        segs = rm.segments(route)
        end = rs.end_pose(rm.poses[s], segs, spec.rho)
        assert np.allclose(end[:2], rm.poses[t, :2], atol=1e-6)
        traj = Trajectory(spec, rm.poses[s], segs)
        x, y, _, v = traj.pose(traj.duration)
        assert (x, y) == pytest.approx(tuple(rm.poses[t, :2]), abs=1e-6) and v == 0.0
        # Graph cost is the long-run form of the trapezoid: never below the real time.
        assert times[t] >= traj.duration - 1e-6
        assert times[t] <= traj.duration + 4.0 * (rs.cusps(segs) + 1)


def test_lazy_roadmap_never_returns_a_colliding_edge():
    world = build_world(4)
    spec = TruckSpec()
    checker = Checker(world, spec)
    rm = Roadmap(checker, spec, k=8, candidates=60, lazy=True)
    ids = rm.add_vertices(rm.sample_free(600, rng_for(4, "prm")))
    rm.connect(ids)
    rng = np.random.default_rng(2)
    found = 0
    for _ in range(20):
        s, t = (int(v) for v in rng.choice(rm.n, 2, replace=False))
        while True:
            route = rm.route(s, t)
            if route is None:
                break
            if rm.check_edges([e for e, _ in route]):
                break
        if route is None:
            continue
        found += 1
        assert all(rm.edges[e].status == FREE for e, _ in route)
        starts = [rm.poses[rm.edges[e].u] for e, _ in route]
        assert checker.paths_free(starts, [rm.edges[e].path for e, _ in route]).all()
    assert found >= 10
    assert any(e.status == BLOCKED for e in rm.edges)


def test_oracle_best_stop_matches_brute_force(built):
    from wildfire.oracle import HERE, RoadmapOracle
    world, spec, rm = built
    lab = rm.components()
    comp = np.flatnonzero(lab == np.bincount(lab).argmax())
    o = RoadmapOracle(world, spec, rm, comp)
    o.update(int(comp[0]), rm.centres()[comp[0]])
    o.ban(3)
    times, stops = o.cell_best()
    t = np.where(o.banned, np.inf, o.times)
    masked = np.where(o.cover, t[None, :], np.inf)
    ref = masked.min(axis=1)
    ref[o.here_cells] = 0.0
    assert np.array_equal(times, ref)
    for c in range(world.n):
        if np.isfinite(times[c]) and stops[c] != HERE:
            assert o.cover[c, stops[c]] and t[stops[c]] == times[c]
