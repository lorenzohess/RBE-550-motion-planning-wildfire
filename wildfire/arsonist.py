"""The arsonist: walks the 5 m grid with A*, lights obstacles beside it.

Rules (assignment §2, §3.2; notes "Arsonist"):
- It moves node to node at `speed` m/s, one 5 m step every 5 / speed seconds.
- At a node it may light one intact or extinguished obstacle in a 4-adjacent cell, instantly,
  and *at most one per node it visits* (D15). Leaving and coming back is a new visit.
- It sees the map, every cell's state and the truck. It can't be burned and doesn't burn the
  truck.

Goal selection:
- W0 (baseline): the nearest cell it can light, by straight-line distance.
- W1: the cell that sets the most intact cells burning (itself plus every intact cell its spread
  would cascade into), per second of walking, discounted when the truck is near it. Its A* steps
  cost more near the truck. Under per-state-change scoring an ignition the truck puts out nets the
  truck +1, so only fires far from the truck pay.

En route it lights a beside-the-path cell at each new node (W1 only where the truck is far).
With nothing left to light, it walks away from the truck.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .astar import STEPS, astar, free_components
from .cpu import NULL, Cpu
from .fire import BURNING, EXTINGUISHED, INTACT, Fire
from .world import CELL_M, CELLS, World

TRUCK_SAFE = 80.0          # W1: ignitions closer than this to the truck are discounted (m)
AVOID_SCALE = 20.0         # W1: A* cost bump near the truck decays over this distance (m)
AVOID_GAIN = 2.0           # ... and is this many times the step length at the truck


class Arsonist:
    def __init__(self, world: World, fire: Fire, node: tuple[int, int], speed: float, dt: float,
                 policy: str = "W1", rng: np.random.Generator | None = None, cpu: Cpu = NULL):
        if policy not in ("W0", "W1"):
            raise ValueError(f"unknown arsonist policy {policy}")
        self.world, self.fire, self.policy, self.cpu = world, fire, policy, cpu
        self.rng = rng or np.random.default_rng(0)
        self.speed = speed
        self.step_ticks = max(1, round(CELL_M / speed / dt))
        self.free = ~world.occupied
        labels = free_components(self.free)
        self.component = labels == labels[node]
        # Free nodes of our component beside each obstacle.
        self.access: list[list[tuple[int, int]]] = []
        for cx, cy in world.cells:
            nodes = [(cx + dx, cy + dy) for dx, dy in STEPS
                     if 0 <= cx + dx < CELLS and 0 <= cy + dy < CELLS
                     and self.component[cx + dx, cy + dy]]
            self.access.append(nodes)
        self.reachable = np.array([bool(a) for a in self.access])
        adj = [(i, j) for i, nb in enumerate(world.spread) for j in nb]
        rows, cols = zip(*adj) if adj else ((), ())
        self._spread = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(world.n, world.n))

        self.node = node
        self.prev = node
        self.path: list[tuple[int, int]] = []     # nodes still to walk to
        self.progress = 0                          # ticks into the current step
        self.lit_here = False
        self.target: int | None = None
        self.goal_nodes: set = set()
        self.fleeing = False
        self.stats = {"plans": 0, "expanded": 0, "ignitions": 0, "steps": 0}

    # --- what it sees -------------------------------------------------------------------------

    def lightable(self) -> np.ndarray:
        s = self.fire.state
        return np.flatnonzero(self.reachable & ((s == INTACT) | (s == EXTINGUISHED)))

    def has_targets(self) -> bool:
        return len(self.lightable()) > 0

    def xy(self) -> tuple[float, float]:
        """Position for drawing: between the last node and the next one."""
        a = (np.array(self.prev) + 0.5) * CELL_M
        if not self.path:
            return tuple((np.array(self.node) + 0.5) * CELL_M)
        b = (np.array(self.path[0]) + 0.5) * CELL_M
        f = self.progress / self.step_ticks
        return tuple(a + f * (b - a))

    # --- each tick ----------------------------------------------------------------------------

    def update(self, tick: int, truck_xy) -> None:
        if self.path:
            self.progress += 1
            if self.progress < self.step_ticks:
                return
            self.prev = self.node = self.path.pop(0)
            self.progress = 0
            self.lit_here = False
            self.stats["steps"] += 1
        # At a node, at rest or just arrived.
        if not self.lit_here:
            self._light(tick, truck_xy)
        if self._needs_plan():
            with self.cpu("goal_selection"):
                self._choose(truck_xy)
            if self.goal_nodes:
                with self.cpu("search"):
                    cost = self._step_cost(truck_xy) if self.policy == "W1" else None
                    path = astar(self.free, self.node, self.goal_nodes, cost, self.stats)
                self.stats["plans"] += 1
                self.path = path[1:] if path else []
                if path is None:                       # walled off: never try that cell again
                    if self.target is not None:
                        self.reachable[self.target] = False
                    self.target = None
        if self.path:
            self.prev = self.node

    def _needs_plan(self) -> bool:
        if self.fleeing:
            if self.has_targets():
                return True
            return not self.path and self.rng.random() < 0.02     # dawdle, then move on
        if self.target is None:
            return True
        if self.fire.state[self.target] not in (INTACT, EXTINGUISHED):
            return True
        return not self.path

    def _adjacent(self) -> list[int]:
        cx, cy = self.node
        out = []
        for dx, dy in STEPS:
            x, y = cx + dx, cy + dy
            if 0 <= x < CELLS and 0 <= y < CELLS and self.world.index[x, y] >= 0:
                c = int(self.world.index[x, y])
                if self.fire.state[c] in (INTACT, EXTINGUISHED):
                    out.append(c)
        return out

    def _light(self, tick: int, truck_xy) -> None:
        options = self._adjacent()
        if not options:
            return
        if self.target in options:
            cell = self.target
            self.target = None
        else:
            if self.policy == "W1":
                d = np.hypot(*(self.world.centres[options] - np.asarray(truck_xy)).T)
                options = [c for c, di in zip(options, d) if di >= TRUCK_SAFE]
                if not options:
                    return
            # Intact first (it may spread), then the lowest id, so runs are reproducible.
            options.sort(key=lambda c: (self.fire.state[c] != INTACT, c))
            cell = options[0]
        self.fire.ignite(cell, tick, "arsonist")
        self.lit_here = True
        self.stats["ignitions"] += 1

    # --- goal selection -------------------------------------------------------------------------

    def _choose(self, truck_xy) -> None:
        cand = self.lightable()
        here = (np.array(self.node) + 0.5) * CELL_M
        self.fleeing = False
        if len(cand) == 0:
            self._flee(truck_xy)
            return
        d_here = np.hypot(*(self.world.centres[cand] - here).T)
        if self.policy == "W0":
            order = np.argsort(d_here, kind="stable")
        else:
            value = self._cascade_value(cand)
            walk = (np.abs(self.world.cells[cand] - np.array(self.node)).sum(axis=1) * CELL_M
                    / self.speed)
            d_truck = np.hypot(*(self.world.centres[cand] - np.asarray(truck_xy)).T)
            score = value * np.minimum(1.0, d_truck / TRUCK_SAFE) / (walk + 10.0)
            order = np.argsort(-score, kind="stable")
        for i in order[:8]:
            c = int(cand[i])
            goals = set(self.access[c])
            if self.lit_here:
                goals.discard(self.node)
            if goals:
                self.target, self.goal_nodes = c, goals
                return
        self.target, self.goal_nodes = None, set()

    def _cascade_value(self, cand: np.ndarray) -> np.ndarray:
        """Intact cells set burning by lighting each candidate, if nobody intervened: its whole
        group in the spread graph restricted to intact cells (an extinguished cell spreads into
        the intact groups within 30 m of it)."""
        intact = self.fire.state == INTACT
        idx = np.flatnonzero(intact)
        value = np.ones(len(cand))
        if len(idx) == 0:
            return value
        sub = self._spread[idx][:, idx]
        _, lab = connected_components(sub, directed=False)
        size = np.bincount(lab)
        group = np.full(self.world.n, -1)
        group[idx] = lab
        for k, c in enumerate(cand):
            if intact[c]:
                value[k] = size[group[c]]
            else:
                groups = {group[j] for j in self.world.spread[c] if intact[j]}
                value[k] = 1 + sum(size[g] for g in groups)
        return value

    def _step_cost(self, truck_xy):
        tx, ty = truck_xy

        def cost(n):
            d = math.hypot((n[0] + 0.5) * CELL_M - tx, (n[1] + 0.5) * CELL_M - ty)
            return CELL_M * (1.0 + AVOID_GAIN * math.exp(-d / AVOID_SCALE))
        return cost

    def _flee(self, truck_xy) -> None:
        """Nothing to light: walk to a random node among the tenth farthest from the truck."""
        nodes = np.argwhere(self.component)
        d = np.hypot(*((nodes + 0.5) * CELL_M - np.asarray(truck_xy)).T)
        far = nodes[d >= np.quantile(d, 0.9)]
        goal = tuple(int(v) for v in far[self.rng.integers(len(far))])
        self.target, self.goal_nodes, self.fleeing = None, {goal}, True
