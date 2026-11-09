"""The free-travel bound (notes, "What the rules actually produce" 6): what a truck that needed no
time to drive could save. With equal 5 s service and deadlines in ignition order, serving the
oldest unexpired cell first and skipping lost ones is optimal (swap argument), so this is a hard
ceiling for any real truck.

- `saved_free_travel(ignitions, burn, start)`: the ceiling for a given list of ignition times.
  Used on each run's actual ignitions, as E3's ceiling.
- `save_all_threshold(world, seed)`: the burnout time the map needs for a free-travel truck to
  save a whole smart-arsonist cascade (largest spread group lit at t = 0, cells igniting 10 s per
  spread hop, truck arriving at 30 s). Stage-2 load is lambda = threshold / T_burn.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .world import World

SERVICE = 5.0
ARRIVAL = 30.0


def saved_free_travel(ignitions, burn: float, start: float = 0.0, service: float = SERVICE,
                      end: float = np.inf) -> int:
    t, n = start, 0
    for r in np.sort(np.asarray(ignitions, float)):
        t = max(t, r)
        if t + service <= min(r + burn, end):
            t += service
            n += 1
    return n


def cascade_times(world: World, seed: int) -> np.ndarray:
    adj = [(i, j) for i, nb in enumerate(world.spread) for j in nb]
    rows, cols = zip(*adj)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(world.n, world.n))
    _, lab = connected_components(g, directed=False)
    members = np.flatnonzero(lab == np.bincount(lab).argmax())
    src = np.random.default_rng(seed).choice(members)
    hops = shortest_path(g, unweighted=True, indices=src)
    return 10.0 * hops[np.isfinite(hops)]


def save_all_threshold(world: World, seed: int, per_cell: float = SERVICE) -> float:
    """Smallest burnout time (s, to 5 s) at which a free-travel truck saves the whole cascade."""
    r = cascade_times(world, seed)
    lo, hi = 10.0, 7200.0
    while hi - lo > 5.0:
        mid = 0.5 * (lo + hi)
        if saved_free_travel(r, mid, ARRIVAL, per_cell) == len(r):
            hi = mid
        else:
            lo = mid
    return hi
