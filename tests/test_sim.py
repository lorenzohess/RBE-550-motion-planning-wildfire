import numpy as np
import pytest

from wildfire.fire import BURNED, BURNING, EXTINGUISHED, score
from wildfire.sim import Config, Game
from wildfire.world import REACH

SHORT = dict(duration=400.0, burn=300.0, prm_n=700, prm_candidates=60)


@pytest.mark.parametrize("scenario,truck", [("S1", "T0"), ("S1", "T2"), ("S2", "T1")])
def test_run_is_reproducible_and_scores_add_up(scenario, truck):
    a = Game(Config(seed=3, scenario=scenario, truck=truck, **SHORT)).run()
    b = Game(Config(seed=3, scenario=scenario, truck=truck, **SHORT)).run()
    for k in ("score_truck", "score_arsonist", "ignitions", "extinguishes", "burnouts",
              "truck_distance", "truck_legs"):
        assert a[k] == b[k], k
    assert a["score_arsonist"] == a["ignitions"] + a["burnouts"]
    assert a["score_truck"] == 2 * a["extinguishes"]
    assert a["extinguishes"] > 0
    assert a["final_burned"] == a["burnouts"]


def test_every_extinguish_happens_parked_in_reach_after_5s():
    g = Game(Config(seed=1, truck="T0", **SHORT))
    while not g.done:
        before = len(g.fire.events)
        dwelling = g.dwell
        parked = g.traj is None
        centre = g.truck_centre()
        g.step()
        for e in g.fire.events[before:]:
            if e.kind == "extinguish":
                assert parked
                assert g.world.distances(centre)[0, e.cell] <= REACH + 1e-9
                assert dwelling is not None and dwelling[0] == e.cell
                assert e.tick - dwelling[1] >= 50
    assert score(g.fire.events).extinguishes > 0


def test_truck_never_collides_along_its_legs():
    g = Game(Config(seed=2, truck="T2", **SHORT))
    checker = g.checker
    legs = []
    while not g.done:
        had = g.traj
        g.step()
        if g.traj is not None and g.traj is not had:
            legs.append(g.traj)
    assert legs
    assert checker.paths_free([t.start for t in legs], [t.segments for t in legs]).all()


@pytest.mark.parametrize("scenario", ["S1", "S2"])
def test_catharsis_chase(scenario):
    """Once nothing can burn any more, the truck hunts the arsonist."""
    g = Game(Config(seed=0, scenario=scenario, truck="T0", duration=900.0, prm_n=1200,
                    prm_candidates=60))
    g.fire.state[:] = BURNED
    r = g.run()
    assert g.chasing and g.truck_stats["legs"] > 0
    assert r["end_reason"] in ("caught", "time")
    if r["caught"]:
        a = np.array(g.arsonist.xy())
        assert np.hypot(*(g.truck_centre() - a)) <= 5.0 + 1e-9
