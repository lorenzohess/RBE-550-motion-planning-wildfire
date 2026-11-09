"""Travel-time oracles: how long the truck needs to reach each candidate stop.

A *stop* is a place the truck can park (its geometric centre), with the cells within 10 m of it.
The truck's goal selection only ever sees stops, times and coverage, so the same policy runs on
either scenario:

- RoadmapOracle (S1): stops are the a-priori roadmap's vertices in the truck's component, and the
  times come from one gear-aware Dijkstra from the truck's vertex: Reeds–Shepp shapes, reversals
  and obstacles all included. The leg to the chosen stop is read off the same search.
- EuclidOracle: straight-line distance times a detour factor, plus the stop overhead. Used by S2
  (no roadmap exists to ask; stops are a static table of parking spots) and by the T2e policy on
  S1 (stops are the roadmap's vertices, times ignore kinematics).

Every oracle also offers HERE: stay parked, time 0, the cells in reach right now.
"""

from __future__ import annotations

import math

import numpy as np

from .collision import Checker
from .prm import Roadmap
from .truck import TruckSpec
from .world import REACH, SIZE_M, World

HERE = -1
DETOUR = 1.6       # roadmap travel time over straight-line time at v_max (median, measured)


class Oracle:
    def __init__(self, world: World, spec: TruckSpec, stop_xy: np.ndarray,
                 stop_vertex: np.ndarray | None = None):
        self.world = world
        self.spec = spec
        self.stop_xy = np.asarray(stop_xy, float)
        self.stop_vertex = stop_vertex                      # roadmap vertex per stop (S1)
        self.cover = (world.distances(self.stop_xy) <= REACH).T     # (cells, stops)
        self._owner, self._flat = np.nonzero(self.cover)           # (cell, stop) pairs by cell
        counts = np.bincount(self._owner, minlength=world.n)
        self._has = counts > 0
        self._starts = (np.cumsum(counts) - counts)[self._has]
        self.times = np.full(len(self.stop_xy), np.inf)
        self.here_xy = np.zeros(2)
        self.here_cells = np.zeros(world.n, dtype=bool)
        self.banned = np.zeros(len(self.stop_xy), dtype=bool)
        self.cell_banned = np.zeros(world.n, dtype=bool)   # legs to it keep failing (S2)
        self._best = None

    @property
    def n_stops(self) -> int:
        return len(self.stop_xy)

    def set_here(self, xy) -> None:
        self.here_xy = np.asarray(xy, float)
        self.here_cells = self.world.distances(self.here_xy)[0] <= REACH
        self._best = None

    def ban(self, stop: int, banned: bool = True) -> None:
        self.banned[stop] = banned
        self._best = None

    def ban_cell(self, cell: int, banned: bool = True) -> None:
        self.cell_banned[cell] = banned
        self._best = None

    def cells_at(self, stop: int) -> np.ndarray:
        if stop == HERE:
            return np.flatnonzero(self.here_cells)
        return np.flatnonzero(self.cover[:, stop])

    def stop_point(self, stop: int) -> np.ndarray:
        return self.here_xy if stop == HERE else self.stop_xy[stop]

    def cell_best(self) -> tuple[np.ndarray, np.ndarray]:
        """For every cell: the soonest the truck can be in reach of it, and the stop that does it
        (HERE for cells in reach now)."""
        if self._best is None:
            t = np.where(self.banned, np.inf, self.times)
            vals = t[self._flat]
            times = np.full(self.world.n, np.inf)
            times[self._has] = np.minimum.reduceat(vals, self._starts)
            hit = np.flatnonzero(vals <= times[self._owner])
            cells, first = np.unique(self._owner[hit], return_index=True)
            stops = np.zeros(self.world.n, dtype=int)
            stops[cells] = self._flat[hit[first]]
            times = np.where(self.here_cells, 0.0, times)
            stops = np.where(self.here_cells, HERE, stops)
            times[self.cell_banned] = np.inf
            self._best = (times, stops)
        return self._best

    def drive_time(self, d) -> np.ndarray:
        """Straight-line estimate of driving distance d (m), for rollouts and EuclidOracle."""
        d = np.asarray(d, float)
        return np.where(d > 0.5, self.spec.stop_overhead + DETOUR * d / self.spec.v_max, 0.0)

    def estimate(self, a_xy, b_xy) -> np.ndarray:
        """drive_time between points (broadcast)."""
        return self.drive_time(np.hypot(*(np.asarray(b_xy, float) - np.asarray(a_xy, float)).T))

    def update(self, vertex: int, xy) -> None:
        raise NotImplementedError


class RoadmapOracle(Oracle):
    def __init__(self, world: World, spec: TruckSpec, roadmap: Roadmap, vertices: np.ndarray):
        self.roadmap = roadmap
        super().__init__(world, spec, roadmap.centres()[vertices], stop_vertex=np.asarray(vertices))

    def update(self, vertex: int, xy) -> None:
        self.set_here(xy)
        self.times = self.roadmap.times_from(vertex)[self.stop_vertex]


class EuclidOracle(Oracle):
    def update(self, vertex: int, xy) -> None:
        self.set_here(xy)
        self.times = self.estimate(self.here_xy, self.stop_xy)


def parking_spots(checker: Checker, spec: TruckSpec, spacing: float = 2.5, headings: int = 8):
    """S2's stop table: lattice points where the truck's centre fits at one of `headings` headings.
    It knows nothing about reachability; the on-demand planner finds that out per leg."""
    g = np.arange(spacing / 2, SIZE_M, spacing)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    cx, cy = cx.ravel(), cy.ravel()
    ok = np.zeros(len(cx), dtype=bool)
    for k in range(headings):
        psi = math.pi * k / headings                      # a rectangle repeats every 180 deg
        x, y, _ = spec.pose_from_centre(cx, cy, psi)
        ok |= checker.free(x, y, np.full(len(cx), psi))
    return np.column_stack([cx[ok], cy[ok]])
