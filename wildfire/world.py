"""The field: 250 x 250 m, a 50 x 50 grid of 5 m cells with tetromino obstacles.

Coordinates: x east, y north, metres, origin at the SW corner. Cells are indexed (cx, cy) the same
way, so `occupied[cx, cy]` covers x in [5 cx, 5 cx + 5], y in [5 cy, 5 cy + 5]. North is drawn at
the top of the window.

Every occupied cell is its own obstacle (notes D1). Obstacles are numbered 0..n-1 in the order
np.nonzero returns them, and `index[cx, cy]` maps back (-1 for free cells).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist

CELLS = 50
CELL_M = 5.0
SIZE_M = CELLS * CELL_M          # 250 m
SPREAD_RADIUS = 30.0             # assignment §2: obstacles within 30 m catch fire
REACH = 10.0                     # assignment §2: the truck extinguishes within 10 m

# Free tetrominoes, as in Valet (valet/world.py), which took them from the obstacleworld library
# of assignment 1. No O piece; a reflection counts as the same piece.
L_TET = ((0, 0), (1, 0), (1, -1), (1, -2))
I_TET = ((0, 0), (0, 1), (0, 2), (0, 3))
S_TET = ((0, 0), (0, -1), (1, -1), (1, -2))
T_TET = ((0, 0), (1, 0), (1, 1), (1, -1))


def _normalize(cells):
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    return tuple(sorted((c[0] - min_x, c[1] - min_y) for c in cells))


def _rotations(cells):
    """Distinct 90 degree rotations of a shape, each normalized to start at (0, 0)."""
    shapes = []
    current = list(cells)
    for _ in range(4):
        current = [(y, -x) for x, y in current]
        shape = _normalize(current)
        if shape not in shapes:
            shapes.append(shape)
    return shapes


# 12 orientations: L 4, I 2, S 2, T 4.
TETROMINOES = [shape for tet in (L_TET, I_TET, S_TET, T_TET) for shape in _rotations(tet)]


def place_tetrominoes(rng: random.Random, rho: float) -> np.ndarray:
    """Place random tetrominoes until at least a fraction rho of the cells is occupied (Valet's
    rule): a placement that would leave the field is drawn again, and pieces may overlap, so
    coverage is measured, not counted."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")
    occupied = np.zeros((CELLS, CELLS), dtype=bool)
    while occupied.mean() < rho:
        shape = rng.choice(TETROMINOES)
        cx0 = rng.randrange(CELLS)
        cy0 = rng.randrange(CELLS)
        cells = [(cx0 + dx, cy0 + dy) for dx, dy in shape]
        if all(0 <= cx < CELLS and 0 <= cy < CELLS for cx, cy in cells):
            for cx, cy in cells:
                occupied[cx, cy] = True
    return occupied


def square_distance(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Distance from each point (m, 2) to the nearest point of each 5 m square (n, 2 centres):
    (m, n). Zero inside a square (notes D11)."""
    d = np.abs(points[:, None, :] - centres[None, :, :]) - 0.5 * CELL_M
    d = np.maximum(d, 0.0)
    return np.hypot(d[..., 0], d[..., 1])


@dataclass
class World:
    seed: int
    rho: float
    occupied: np.ndarray                  # bool (CELLS, CELLS), indexed [cx, cy]
    cells: np.ndarray = field(init=False)     # (n, 2) int, (cx, cy) of each obstacle
    centres: np.ndarray = field(init=False)   # (n, 2) float, metres
    index: np.ndarray = field(init=False)     # (CELLS, CELLS) int, obstacle id or -1
    spread: list = field(init=False)          # spread[i]: obstacles within 30 m of i (centres)

    def __post_init__(self):
        cx, cy = np.nonzero(self.occupied)
        self.cells = np.stack([cx, cy], axis=1)
        self.centres = (self.cells + 0.5) * CELL_M
        self.index = np.full((CELLS, CELLS), -1, dtype=int)
        self.index[cx, cy] = np.arange(len(cx))
        d = cdist(self.centres, self.centres)
        adj = (d <= SPREAD_RADIUS + 1e-9) & ~np.eye(len(cx), dtype=bool)
        self.spread = [np.flatnonzero(row) for row in adj]

    @property
    def n(self) -> int:
        return len(self.cells)

    def cell_box(self, i: int) -> tuple[float, float, float, float]:
        cx, cy = self.cells[i]
        return (cx * CELL_M, (cx + 1) * CELL_M, cy * CELL_M, (cy + 1) * CELL_M)

    def distances(self, points: np.ndarray) -> np.ndarray:
        """Distance from points (m, 2) to every obstacle square: (m, n)."""
        return square_distance(np.atleast_2d(points), self.centres)

    def in_reach(self, point, reach: float = REACH) -> np.ndarray:
        """Obstacles within `reach` of a point (the truck's centre)."""
        return np.flatnonzero(self.distances(np.asarray(point, float))[0] <= reach)


def build_world(seed: int, rho: float = 0.10) -> World:
    return World(seed=seed, rho=rho, occupied=place_tetrominoes(random.Random(seed), rho))


def rng_for(seed: int, stream: str) -> np.random.Generator:
    """An independent random stream per purpose (spawns, roadmap, tie-breaks), all from the run
    seed, so changing one consumer never shifts another's draws."""
    return np.random.default_rng([seed, sum(ord(c) << (8 * i) for i, c in enumerate(stream)) % (2**32)])
