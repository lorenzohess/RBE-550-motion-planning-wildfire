"""A* on the arsonist's grid (assignment §3.2, "combinatorial" planning; LaValle §2.2.2).

The graph: one node per free 5 m cell, edges to the 4 side neighbours (notes D7). The arsonist
stands at node centres, so a node is a cell of the obstacle grid that isn't an obstacle.

Costs and heuristic:
- Each step costs `step_cost(node)` metres for entering that node: 5 m by default. A cost may be
  raised (the W1 arsonist pays extra near the truck) but never lowered below 5 m.
- h(n) = 5 m x (Manhattan distance from n to the nearest goal node). One step changes the Manhattan
  distance by exactly 1 and costs at least 5 m, so h never overestimates (admissible) and
  h(n) <= cost(n, n') + h(n') (consistent). A* with a consistent heuristic expands each node at most
  once and returns a shortest path.
- Several goals are allowed (every free node beside the target cell); h uses the nearest.
- Ties in f go to the node with the larger g (deeper, nearer the goal), then to insertion order,
  so results are deterministic.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np
from scipy.ndimage import label

from .world import CELL_M, CELLS

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def astar(free: np.ndarray, start: tuple[int, int], goals, step_cost=None, stats: dict | None = None):
    """Shortest 4-connected path of nodes from `start` to any node in `goals`, both ends included;
    None if no goal is reachable."""
    goals = set(goals)
    if not goals:
        return None
    if start in goals:
        return [start]
    gx = [g[0] for g in goals]
    gy = [g[1] for g in goals]

    def h(n):
        x, y = n
        return CELL_M * min(abs(a - x) + abs(b - y) for a, b in zip(gx, gy))

    counter = itertools.count()
    g_cost = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), -0.0, next(counter), start)]
    closed = set()
    expanded = 0
    while heap:
        _, neg_g, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node in goals:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            if stats is not None:
                stats["expanded"] = stats.get("expanded", 0) + expanded
            return path
        g = -neg_g
        x, y = node
        for dx, dy in STEPS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < CELLS and 0 <= ny < CELLS) or not free[nx, ny]:
                continue
            nb = (nx, ny)
            if nb in closed:
                continue
            ng = g + (CELL_M if step_cost is None else step_cost(nb))
            if ng < g_cost.get(nb, np.inf):
                g_cost[nb] = ng
                parent[nb] = node
                heapq.heappush(heap, (ng + h(nb), -ng, next(counter), nb))
    if stats is not None:
        stats["expanded"] = stats.get("expanded", 0) + expanded
    return None


def free_components(free: np.ndarray) -> np.ndarray:
    """Label of each free node's 4-connected component (0 for obstacles)."""
    labels, _ = label(free, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    return labels
