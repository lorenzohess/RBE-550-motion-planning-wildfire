"""The fire truck: body geometry, timing along a Reeds–Shepp path, and the trajectory it drives.

Reference point: the centre of the rear axle, heading psi (rad, east = 0, counter-clockwise), as in
Valet. The body is a rectangle in the vehicle frame (x forward, y left): x in [-rear, front],
y in [-half_width, half_width]. The assignment gives 2.2 x 4.9 m and L = 3 m; the overhangs are
our assumption (notes Q2): front 1.0 m, so front = 3.0 + 1.0 = 4.0 m and rear = 4.9 - 4.0 = 0.9 m.

Speed profile (notes D9): each run of one gear is a trapezoid with acceleration a_max from rest to
rest, capped at v_max; at a gear change the truck stops and waits `gear_penalty` seconds (the
assignment's hint that instant acceleration is "destructive to the transmission"). a_max = inf
gives the instantaneous model.

The truck's pose at any time is read off the path in closed form, so there is no integration
error (D10). The extinguish range is measured from the geometric centre of the body (D11).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from . import reeds_shepp as rs


@dataclass(frozen=True)
class Body:
    rear: float
    front: float
    half_width: float

    @property
    def length(self) -> float:
        return self.rear + self.front

    @property
    def centre_offset(self) -> float:
        """Distance from the reference point forward to the geometric centre."""
        return 0.5 * (self.front - self.rear)

    def grown(self, margin: float) -> "Body":
        return Body(self.rear + margin, self.front + margin, self.half_width + margin)

    def corners(self, x, y, psi) -> np.ndarray:
        """Corners at one or many poses: (..., 4, 2), rear-right, front-right, front-left,
        rear-left (from Valet's vehicles.py)."""
        x, y, psi = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float),
                                        np.asarray(psi, float))
        local = np.array([[-self.rear, -self.half_width], [self.front, -self.half_width],
                          [self.front, self.half_width], [-self.rear, self.half_width]])
        c, s = np.cos(psi)[..., None], np.sin(psi)[..., None]
        cx = x[..., None] + local[:, 0] * c - local[:, 1] * s
        cy = y[..., None] + local[:, 0] * s + local[:, 1] * c
        return np.stack([cx, cy], axis=-1)


@dataclass(frozen=True)
class TruckSpec:
    body: Body = Body(rear=0.9, front=4.0, half_width=1.1)
    wheelbase: float = 3.0
    rho: float = 13.0              # minimum turning radius at the rear axle
    v_max: float = 10.0
    a_max: float = 2.0             # m/s^2; math.inf for instantaneous speed changes
    gear_penalty: float = 2.0      # s stopped at each gear change

    @property
    def max_steer(self) -> float:
        """Bicycle model: rho = L / tan(phi_max), so phi_max = atan(3 / 13) = 13.0 deg."""
        return math.atan(self.wheelbase / self.rho)

    @property
    def stop_overhead(self) -> float:
        """Extra time a long run takes over driving it at v_max: v / a (half to speed up, half to
        brake). Zero for the instantaneous model."""
        return 0.0 if math.isinf(self.a_max) else self.v_max / self.a_max

    def centre(self, x, y, psi):
        o = self.body.centre_offset
        return np.asarray(x) + o * np.cos(psi), np.asarray(y) + o * np.sin(psi)

    def pose_from_centre(self, cx, cy, psi):
        o = self.body.centre_offset
        return cx - o * np.cos(psi), cy - o * np.sin(psi), psi

    def run_time(self, d: float) -> float:
        """Rest-to-rest time over distance d in one gear."""
        v, a = self.v_max, self.a_max
        if math.isinf(a):
            return d / v
        if d >= v * v / a:
            return d / v + v / a
        return 2.0 * math.sqrt(d / a)

    def edge_cost(self, path) -> float:
        """Graph cost of a path for roadmap search: distance at v_max plus, for every gear change
        inside it, a full stop and the shift time. Joining edges adds the same charge when their
        gears differ (prm.py), and every leg pays stop_overhead once, so a leg's cost is the
        long-run form of its trapezoid time (exact for runs over v^2 / a = 50 m, a little high
        for shorter ones)."""
        changes = sum(1 for (_, a), (_, b) in zip(path, path[1:]) if (a > 0) != (b > 0))
        return rs.length(path) / self.v_max + changes * self.change_cost

    @property
    def change_cost(self) -> float:
        return self.stop_overhead + self.gear_penalty


def gear(a: float) -> int:
    """0 forward, 1 reverse."""
    return 0 if a > 0 else 1


@dataclass
class _Run:
    t0: float              # start time
    duration: float
    length: float
    starts: list           # pose at the start of each segment
    cum: list              # cumulative |length| at the start of each segment
    segs: list


class Trajectory:
    """A leg the truck drives: Reeds–Shepp segments from `start`, timed with the speed profile.
    pose(t) gives (x, y, psi, v) with v signed (negative in reverse)."""

    def __init__(self, spec: TruckSpec, start, segments):
        self.spec = spec
        self.start = tuple(start)
        self.segments = tuple(segments)
        self.runs: list[_Run] = []
        t, pose = 0.0, tuple(start)
        run: list = []
        for seg in self.segments + ((None, 0.0),):
            if run and (seg[0] is None or (seg[1] > 0) != (run[-1][1] > 0)):
                if self.runs:
                    t += spec.gear_penalty
                starts, cum, total = [], [], 0.0
                for st, a in run:
                    starts.append(pose)
                    cum.append(total)
                    pose = rs.end_pose(pose, ((st, a),), spec.rho)
                    total += abs(a)
                d = spec.run_time(total)
                self.runs.append(_Run(t, d, total, starts, cum, list(run)))
                t += d
                run = []
            if seg[0] is not None:
                run.append(seg)
        self.duration = t
        self.end = pose
        self.length = rs.length(self.segments)
        self._t0s = [r.t0 for r in self.runs]

    def _distance(self, run: _Run, tau: float) -> tuple[float, float]:
        """Distance travelled and speed tau seconds into a run."""
        v, a = self.spec.v_max, self.spec.a_max
        d, T = run.length, run.duration
        if math.isinf(a):
            return min(d, v * tau), v
        t_acc = min(v / a, 0.5 * T)
        peak = a * t_acc
        if tau < t_acc:
            return 0.5 * a * tau * tau, a * tau
        if tau > T - t_acc:
            r = max(0.0, T - tau)
            return d - 0.5 * a * r * r, a * r
        return 0.5 * a * t_acc * t_acc + peak * (tau - t_acc), peak

    def pose(self, t: float) -> tuple[float, float, float, float]:
        if not self.runs or t >= self.duration:
            return (*self.end, 0.0)
        if t <= 0.0:
            return (*self.start, 0.0)
        i = bisect.bisect_right(self._t0s, t) - 1
        run = self.runs[i]
        tau = t - run.t0
        if tau >= run.duration:                         # waiting at a gear change
            return (*rs.end_pose(run.starts[-1], (run.segs[-1],), self.spec.rho), 0.0)
        s, speed = self._distance(run, tau)
        j = bisect.bisect_right(run.cum, s) - 1
        st, a = run.segs[j]
        part = math.copysign(min(s - run.cum[j], abs(a)), a)
        x, y, psi = rs.end_pose(run.starts[j], ((st, part),), self.spec.rho)
        return x, y, psi, math.copysign(speed, a)

    def polyline(self, step: float = 0.5):
        """Sampled points for drawing, and the gear of each point."""
        x, y, th = rs.sample(self.start, self.segments, self.spec.rho, step)
        gears = []
        for st, a in self.segments:
            gears += [gear(a)] * max(1, math.ceil(abs(a) / step))
        gears = [gears[0] if gears else 0] + gears
        return x, y, np.array(gears)
