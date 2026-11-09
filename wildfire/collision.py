"""Collision checking: the truck rectangle against obstacle cells and the field boundary.

Two stages, both vectorised over many poses:
1. Prefilter. A distance field of the obstacle cells (scipy's Euclidean distance transform on a
   0.25 m raster) gives a lower bound on each pose's clearance. If that exceeds the radius of the
   circle around the body, the pose is free and nothing else is computed. Most poses in open
   space stop here.
2. Exact test for the rest: the separating axis theorem, rectangle against each occupied cell
   under the rectangle's axis-aligned extent (adapted from Valet's collision.py). The body grown by
   the margin spans at most 3 x 3 cells, so each pose has at most 9 candidates.

Along a motion, poses are checked every <= 0.25 m of rear-axle arc. With rho = 13 m no body point
moves more than 0.25 (1 + 4.15 / 13) = 0.33 m between samples, so growing the rectangle by 0.2 m on
every side covers the swept area between them.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import reeds_shepp as rs
from .truck import Body, TruckSpec
from .world import CELL_M, CELLS, SIZE_M, World

RES = 0.25          # distance field raster, m
STEP = 0.25         # max arc length between checked poses, m
MARGIN = 0.2        # rectangle growth, m


class Checker:
    def __init__(self, world: World, spec: TruckSpec, margin: float = MARGIN, step: float = STEP):
        self.world = world
        self.spec = spec
        self.step = step
        self.body: Body = spec.body.grown(margin)
        self.occupied = world.occupied
        k = int(round(CELL_M / RES))
        fine = np.kron(self.occupied, np.ones((k, k), dtype=bool))
        # Lower bound on the distance from any point of a raster pixel to the obstacles.
        self.clear = distance_transform_edt(~fine) * RES - RES * math.sqrt(2.0)
        b = self.body
        self.offset = 0.5 * (b.front - b.rear)
        self.half_len = 0.5 * (b.front + b.rear)
        self.radius = math.hypot(self.half_len, b.half_width)
        self.poses_checked = 0
        self.exact_checked = 0

    def free(self, x, y, psi) -> np.ndarray:
        """Boolean (S,): is the grown rectangle at each pose clear of every cell and inside the
        field?"""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        psi = np.atleast_1d(np.asarray(psi, float))
        self.poses_checked += len(x)
        b = self.body
        c, s = np.cos(psi), np.sin(psi)
        mx, my = x + self.offset * c, y + self.offset * s
        ac, as_ = np.abs(c), np.abs(s)
        hx = self.half_len * ac + b.half_width * as_
        hy = self.half_len * as_ + b.half_width * ac
        lo_x, hi_x, lo_y, hi_y = mx - hx, mx + hx, my - hy, my + hy
        out = (lo_x < 0.0) | (lo_y < 0.0) | (hi_x > SIZE_M) | (hi_y > SIZE_M)
        result = ~out
        ix = np.clip((mx / RES).astype(int), 0, self.clear.shape[0] - 1)
        iy = np.clip((my / RES).astype(int), 0, self.clear.shape[1] - 1)
        sure = self.clear[ix, iy] > self.radius
        todo = np.flatnonzero(result & ~sure)
        if len(todo) == 0:
            return result
        self.exact_checked += len(todo)

        # Candidate cells: the 3 x 3 block from the lower corner of each extent.
        cx0 = np.floor(lo_x[todo] / CELL_M).astype(int)
        cy0 = np.floor(lo_y[todo] / CELL_M).astype(int)
        cx1 = np.floor(hi_x[todo] / CELL_M).astype(int)
        cy1 = np.floor(hi_y[todo] / CELL_M).astype(int)
        di, dj = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        di, dj = di.ravel(), dj.ravel()
        ccx = cx0[:, None] + di[None, :]                     # (P, 9)
        ccy = cy0[:, None] + dj[None, :]
        valid = (ccx <= cx1[:, None]) & (ccy <= cy1[:, None]) & (ccx < CELLS) & (ccy < CELLS)
        ccx_c, ccy_c = np.clip(ccx, 0, CELLS - 1), np.clip(ccy, 0, CELLS - 1)
        valid &= self.occupied[ccx_c, ccy_c]
        p_idx, m_idx = np.nonzero(valid)
        if len(p_idx) == 0:
            return result
        boxes_lo = np.stack([ccx[p_idx, m_idx], ccy[p_idx, m_idx]], axis=1) * CELL_M     # (Q, 2)
        # World axes are already settled: the cell is under the extent. Rectangle axes: project
        # the square's corners into the vehicle frame.
        offsets = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * CELL_M
        sq = boxes_lo[:, None, :] + offsets                                           # (Q, 4, 2)
        pose = todo[p_idx]
        dx = sq[..., 0] - x[pose, None]
        dy = sq[..., 1] - y[pose, None]
        cc, ss = c[pose, None], s[pose, None]
        along = dx * cc + dy * ss
        across = -dx * ss + dy * cc
        overlap = ((along.max(axis=1) > -b.rear) & (along.min(axis=1) < b.front)
                   & (across.max(axis=1) > -b.half_width) & (across.min(axis=1) < b.half_width))
        result[pose[overlap]] = False
        return result

    def pose_free(self, pose) -> bool:
        return bool(self.free(pose[0], pose[1], pose[2])[0])

    def paths_free(self, starts, paths) -> np.ndarray:
        """Boolean per path: is every sampled pose along it free? One vectorised check for all."""
        if not paths:
            return np.zeros(0, dtype=bool)
        xs, ys, ts, owner = [], [], [], []
        for i, (start, path) in enumerate(zip(starts, paths)):
            x, y, t = rs.sample(start, path, self.spec.rho, self.step)
            xs.append(x)
            ys.append(y)
            ts.append(t)
            owner.append(np.full(len(x), i))
        ok = self.free(np.concatenate(xs), np.concatenate(ys), np.concatenate(ts))
        owner = np.concatenate(owner)
        bad = np.bincount(owner[~ok], minlength=len(paths))
        return bad == 0


def exact_free(world: World, body: Body, x: float, y: float, psi: float) -> bool:
    """Slow reference: SAT against every occupied cell and corners against the boundary (for
    tests)."""
    corners = body.corners(x, y, psi)
    if corners.min() < 0.0 or corners.max() > SIZE_M:
        return False
    for i in range(world.n):
        x0, x1, y0, y1 = world.cell_box(i)
        square = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        separated = False
        for poly in (corners, square):
            edges = np.roll(poly, -1, axis=0) - poly
            for ex, ey in edges:
                axis = np.array([-ey, ex])
                pp, qq = corners @ axis, square @ axis
                if pp.max() <= qq.min() or qq.max() <= pp.min():
                    separated = True
                    break
            if separated:
                break
        if not separated:
            return False
    return True
