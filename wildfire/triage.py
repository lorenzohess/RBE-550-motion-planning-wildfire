"""T2: the truck's decision rule. The design is the user's (notes, "The truck's problem"):

- Every burning cell is worth the same: putting it out before burnout is a 3-point swing. So the
  one goal is to put out as many cells as possible before they burn out.
- With free driving, oldest-first is optimal. Driving is never free, so the decision is how far
  to stray from oldest-first to save driving: work the cells close by, unless something across
  the map is about to burn out.
- Stages are about load: when the truck can keep up, any order saves everything and the cheapest
  driving wins; in triage, which cells to give up on is the whole game.

The rule, as a rollout (Bertsekas): for each candidate next stop, imagine committing to it, then
carrying on oldest-savable-first with quick straight-line estimates, up to a horizon. Choose the
stop whose future saves the most cells; between equals, the one that drives least. That one
rule covers every stage: when everything gets saved anyway, the counts tie and the cheapest
driving wins; under load, counting lost cells is exactly the trade between near and urgent.

Commit to the stop: the sim serves every cell planned there before asking again, so the truck
doesn't re-target mid-path the way the other submission's truck did.

Knobs: CANDIDATES (stops tried per decision), HORIZON (s simulated ahead), PARK (how far from a
cell the rollout assumes the truck parks).
"""

from __future__ import annotations

import numpy as np

from .fire import BURNING
from .oracle import HERE
from .policies import Plan, Snapshot, nearest, oldest
from .world import REACH

CANDIDATES = 12
HORIZON = 1200.0
PARK = 7.0


def rollout(snap: Snapshot) -> Plan | None:
    o = snap.oracle
    t, service, deadline = snap.t, snap.service, snap.deadline
    times, _ = o.cell_best()
    savable = (snap.state == BURNING) & np.isfinite(times) & (t + times + service <= deadline)
    if not savable.any():
        return None

    # Candidate stops: the ones that reach many savable cells per second, plus where the two
    # baselines would go, plus staying put.
    stop_times = np.where(o.banned, np.inf, o.times)
    k = o.cover[savable].sum(axis=0)
    rate = np.divide(k, stop_times + service * k, out=np.zeros(len(k)), where=k > 0)
    cands = list(np.argsort(-rate, kind="stable")[:CANDIDATES])
    for base in (oldest(snap), nearest(snap)):
        if base is not None:
            cands.append(base.stop)
    if o.here_cells[savable].any():
        cands.append(HERE)
    cands = list(dict.fromkeys(int(s) for s in cands if s == HERE or np.isfinite(stop_times[s])))

    best_key, best_plan = None, None
    for s in cands:
        first = 0.0 if s == HERE else float(stop_times[s])
        saved, driving, order = _simulate(snap, savable, s, first)
        if not order:
            continue
        key = (saved, -driving)
        if best_key is None or key > best_key:
            best_key, best_plan = key, Plan(s, order, f"rollout saves {saved}")
    return best_plan


def _simulate(snap: Snapshot, savable: np.ndarray, stop: int, first: float):
    """Saved cells and driving time if the truck goes to `stop` now, then works oldest-savable-
    first. Returns (saved, driving, cells served at `stop` in order)."""
    o, world = snap.oracle, snap.world
    service, deadline = snap.service, snap.deadline
    horizon = snap.t + HORIZON
    alive = savable.copy()
    clock = snap.t + first
    driving = first
    pos = o.stop_point(stop)
    saved = 0

    def serve_here(clock, saved):
        """Serve every live cell in reach of pos, oldest first."""
        near = np.flatnonzero(alive & (world.distances(pos)[0] <= REACH))
        for c in near[np.argsort(deadline[near], kind="stable")]:
            alive[c] = False
            if clock + service <= deadline[c]:
                clock += service
                saved += 1
        return clock, saved

    if stop == HERE:
        cells = np.flatnonzero(alive & o.here_cells)
    else:
        cells = np.flatnonzero(alive & o.cover[:, stop])
    order = []
    for c in cells[np.argsort(deadline[cells], kind="stable")]:
        alive[c] = False
        if clock + service <= deadline[c]:
            clock += service
            saved += 1
            order.append(int(c))

    while clock < horizon:
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        gap = np.hypot(*(world.centres[idx] - pos).T)
        travel = np.where(gap > REACH, o.drive_time(gap - PARK), 0.0)
        arrive = clock + travel
        ok = arrive + service <= deadline[idx]
        alive[idx[~ok]] = False          # written off: too far to make it from here
        if not ok.any():
            break
        j = np.flatnonzero(ok)[np.argmin(deadline[idx[ok]])]
        c = idx[j]
        driving += travel[j]
        if gap[j] > REACH:
            pos = world.centres[c] + (pos - world.centres[c]) * (PARK / gap[j])
        clock = arrive[j]
        clock, saved = serve_here(clock, saved)
    return saved, driving, order
