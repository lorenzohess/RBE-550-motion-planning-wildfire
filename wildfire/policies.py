"""Truck goal selection: which cells to put out next, and where to park for them.

A policy gets a Snapshot at each decision and returns a Plan (a stop and the cells to serve
there, in order) or None to wait. The sim drives to the stop, serves the cells that are still
burning and in reach (5 s each), and asks again when the plan is used up or its cell is lost.

Baselines (notes "Baseline meaning": goal selection only; both use the full PRM to get there):
- T0 nearest: the burning cell with the shortest travel time.
- T1 oldest: the oldest burning cell that can still be reached before it burns out. Optimal if
  driving took no time (notes, "oldest-first is optimal").
T2, the user's policy, is in triage.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fire import BURNING
from .oracle import HERE, Oracle
from .world import World

SERVICE = 5.0       # s stopped per cell (assignment §2)


@dataclass
class Plan:
    stop: int                        # oracle stop index, or HERE
    cells: list[int] = field(default_factory=list)
    note: str = ""


@dataclass
class Snapshot:
    t: float                         # s
    state: np.ndarray                # per cell
    deadline: np.ndarray             # s; burnout time if burning, inf otherwise
    oracle: Oracle
    world: World
    service: float = SERVICE


def _burning_with_times(snap: Snapshot):
    times, stops = snap.oracle.cell_best()
    cand = np.flatnonzero((snap.state == BURNING) & np.isfinite(times))
    return cand, times, stops


def nearest(snap: Snapshot) -> Plan | None:
    """T0."""
    cand, times, stops = _burning_with_times(snap)
    if len(cand) == 0:
        return None
    c = int(cand[np.lexsort((snap.deadline[cand], times[cand]))[0]])
    return Plan(int(stops[c]), [c], "nearest")


def oldest(snap: Snapshot) -> Plan | None:
    """T1: oldest first, skipping cells that would burn out before the truck finished them."""
    cand, times, stops = _burning_with_times(snap)
    ok = snap.t + times[cand] + snap.service <= snap.deadline[cand]
    cand = cand[ok]
    if len(cand) == 0:
        return None
    c = int(cand[np.lexsort((times[cand], snap.deadline[cand]))[0]])
    return Plan(int(stops[c]), [c], "oldest")


def get(name: str):
    from . import triage
    table = {"T0": (nearest, "roadmap"), "T1": (oldest, "roadmap"),
             "T2": (triage.rollout, "roadmap"), "T2e": (triage.rollout, "euclid")}
    if name not in table:
        raise ValueError(f"unknown truck policy {name}; choose from {', '.join(table)}")
    return table[name]


__all__ = ["HERE", "Plan", "Snapshot", "nearest", "oldest", "get", "SERVICE"]
