import numpy as np
import pytest

from wildfire.fire import BURNED, BURNING, EXTINGUISHED, INTACT, Fire, score
from wildfire.world import CELLS, World, build_world


def line_world(xs):
    """Obstacle cells at (x, 0) for each x in xs (cell units)."""
    occ = np.zeros((CELLS, CELLS), bool)
    for x in xs:
        occ[x, 0] = True
    return World(seed=0, rho=0.0, occupied=occ)


def test_field_coverage_and_ids():
    w = build_world(3)
    assert w.occupied.mean() >= 0.10
    assert w.n == w.occupied.sum()
    for i, (cx, cy) in enumerate(w.cells):
        assert w.index[cx, cy] == i


def test_spread_graph_is_30m_between_centres():
    # 6 cells apart = 30 m: neighbours; 7 apart = 35 m: not.
    w = line_world([0, 6, 13])
    assert list(w.spread[0]) == [1]
    assert list(w.spread[1]) == [0]
    assert len(w.spread[2]) == 0


def test_spread_once_at_10s_into_intact_only():
    w = line_world([0, 2, 4])
    f = Fire(w, burn_ticks=1000)
    f.ignite(0, 0, "arsonist")
    f.ignite(2, 0, "arsonist")
    f.extinguish(2, 50)                 # extinguished cells are not relit by spread
    f.advance(99)
    assert f.state[1] == INTACT
    f.advance(100)
    assert f.state[1] == BURNING and f.ignited[1] == 100
    assert f.state[2] == EXTINGUISHED


def test_extinguish_before_10s_stops_spread():
    w = line_world([0, 2])
    f = Fire(w, burn_ticks=1000)
    f.ignite(0, 0, "arsonist")
    f.extinguish(0, 99)
    f.advance(500)
    assert f.state[1] == INTACT


def test_burnout_and_terminal():
    w = line_world([0])
    f = Fire(w, burn_ticks=300)
    f.ignite(0, 10, "arsonist")
    f.advance(309)
    assert f.state[0] == BURNING
    f.advance(310)
    assert f.state[0] == BURNED
    with pytest.raises(ValueError):
        f.ignite(0, 400, "arsonist")


def test_relight_restarts_clocks():
    w = line_world([0, 3])
    f = Fire(w, burn_ticks=300)
    f.ignite(0, 0, "arsonist")
    f.extinguish(0, 50)
    f.ignite(0, 200, "arsonist")        # relight
    f.advance(299)
    assert f.state[1] == INTACT
    f.advance(300)
    assert f.state[0] == BURNING         # the old 300-tick burnout is stale
    assert f.state[1] == BURNING         # spread from the relight at 200 + 100
    f.advance(500)
    assert f.state[0] == BURNED
    s = score(f.events)
    assert s.relights == 1


def test_score_identities_on_random_play():
    w = build_world(1)
    rng = np.random.default_rng(0)
    f = Fire(w, burn_ticks=600)
    for tick in range(0, 5000, 7):
        f.advance(tick)
        c = int(rng.integers(w.n))
        if f.state[c] in (INTACT, EXTINGUISHED) and rng.random() < 0.05:
            f.ignite(c, tick, "arsonist")
        burning = np.flatnonzero(f.state == BURNING)
        if len(burning) and rng.random() < 0.5:
            f.extinguish(int(rng.choice(burning)), tick)
    s = score(f.events)
    counts = f.counts()
    assert s.arsonist == s.ignitions + s.burnouts
    assert s.truck == 2 * s.extinguishes
    assert counts["burned"] == s.burnouts
    assert s.ignitions == s.arsonist_ignitions + s.spread_ignitions
