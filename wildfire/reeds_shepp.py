"""Reeds–Shepp curves: the shortest paths for a car that drives forward or in reverse, with the
steering at full lock left, full lock right, or centred (Reeds & Shepp 1990; LaValle §15.3.2).

That is the bicycle model of Valet (x' = v cos psi, y' = v sin psi, psi' = (v / L) tan phi) with
phi in {-phi_max, 0, +phi_max} and v = +-v_max: every piece is an arc of radius rho or a straight
line. The shortest path between any two poses is one of 48 words of at most 5 pieces, each with a
closed-form solution.

The formulas and their naming (LpSpLp = Left forward, Straight forward, Left forward; "m" for
reverse; "u" for equal arcs) follow the Reeds & Shepp paper as laid out in OMPL's
ReedsSheppStateSpace.cpp (Kavraki Lab, BSD licence), including its two corrections to typos in the
paper (formulas 8.4 and 8.11). Each base formula is reused four times through the paper's
symmetries: time flip (x -> -x, phi -> -phi, every length negated) and reflection (y -> -y,
phi -> -phi, left <-> right), and the CCC / CCSC families also backwards (start and goal swapped).

Units inside the formulas are radii (rho = 1). A path is a tuple of segments (steer, length):
steer +1 left, 0 straight, -1 right; length in metres, negative in reverse. The heading changes by
steer * length / rho along a segment.
"""

from __future__ import annotations

import math

import numpy as np

PI = math.pi
TWO_PI = 2.0 * PI
HALF_PI = 0.5 * PI
ZERO = 10.0 * 2.220446049250313e-16      # OMPL's tolerance on the sign of a length

L, S, R = 1, 0, -1


def mod2pi(x: float) -> float:
    """Wrap to [-pi, pi] (OMPL's mod2pi)."""
    v = math.fmod(x, TWO_PI)
    if v < -PI:
        v += TWO_PI
    elif v > PI:
        v -= TWO_PI
    return v


def _polar(x, y):
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u, v, xi, eta, phi):
    delta = mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = mod2pi(t1 + PI) if t2 < 0 else mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)
    return tau, omega


# --- base formulas: each returns (t, u, v) or None -----------------------------------------------

def _lpsplp(x, y, phi):                                   # 8.1
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -ZERO:
        v = mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _lpsprp(x, y, phi):                                   # 8.2
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _lprml(x, y, phi):                                    # 8.3 / 8.4
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = mod2pi(theta + 0.5 * u + PI)
        v = mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _lpruplumrm(x, y, phi):                               # 8.7
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _lprumlumrp(x, y, phi):                               # 8.8
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _lprmsmlm(x, y, phi):                                 # 8.9
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = mod2pi(theta + math.atan2(r, -2.0))
        v = mod2pi(phi - HALF_PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lprmsmrm(x, y, phi):                                 # 8.10
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = mod2pi(t + HALF_PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lprmslmrp(x, y, phi):                                # 8.11 (corrected)
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


# --- the 48 words ---------------------------------------------------------------------------------

def _variants(f, x, y, phi, steers, build):
    """The base formula f at (x, y, phi) and its time flip, reflection and both. `steers` is the
    word's steering for the base case; reflection swaps left and right. `build(t, u, v)` gives the
    base case's lengths; the time flip negates them all."""
    out = []
    flipped = tuple(-s for s in steers)
    for sx, sy, steer, sign in ((1, 1, steers, 1), (-1, 1, steers, -1),
                                (1, -1, flipped, 1), (-1, -1, flipped, -1)):
        r = f(sx * x, sy * y, sx * sy * phi)
        if r is not None:
            out.append((steer, tuple(sign * a for a in build(*r))))
    return out


def candidates(x: float, y: float, phi: float) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    """Every valid word from (0, 0, 0) to (x, y, phi), in radius units: [(steers, lengths)]."""
    out = []
    # CSC
    out += _variants(_lpsplp, x, y, phi, (L, S, L), lambda t, u, v: (t, u, v))
    out += _variants(_lpsprp, x, y, phi, (L, S, R), lambda t, u, v: (t, u, v))
    # CCC, forwards and backwards (the backwards case solves the reversed problem and reverses the
    # word: lengths v, u, t)
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
    out += _variants(_lprml, x, y, phi, (L, R, L), lambda t, u, v: (t, u, v))
    out += _variants(_lprml, xb, yb, phi, (L, R, L), lambda t, u, v: (v, u, t))
    # CCCC
    out += _variants(_lpruplumrm, x, y, phi, (L, R, L, R), lambda t, u, v: (t, u, -u, v))
    out += _variants(_lprumlumrp, x, y, phi, (L, R, L, R), lambda t, u, v: (t, u, u, v))
    # CCSC and CSCC
    out += _variants(_lprmsmlm, x, y, phi, (L, R, S, L), lambda t, u, v: (t, -HALF_PI, u, v))
    out += _variants(_lprmsmrm, x, y, phi, (L, R, S, R), lambda t, u, v: (t, -HALF_PI, u, v))
    out += _variants(_lprmsmlm, xb, yb, phi, (L, S, R, L), lambda t, u, v: (v, u, -HALF_PI, t))
    out += _variants(_lprmsmrm, xb, yb, phi, (R, S, R, L), lambda t, u, v: (v, u, -HALF_PI, t))
    # CCSCC
    out += _variants(_lprmslmrp, x, y, phi, (L, R, S, L, R),
                     lambda t, u, v: (t, -HALF_PI, u, -HALF_PI, v))
    return out


def shortest(start, goal, rho: float, cusp: float = 0.0) -> tuple[tuple[int, float], ...]:
    """The best Reeds–Shepp path from pose `start` to pose `goal` (x, y, psi), turning radius rho,
    as segments (steer, signed length in metres). Zero-length pieces are dropped.

    With cusp = 0 this is the shortest path (Reeds & Shepp's result). With cusp > 0, each gear
    change costs that many metres more, so among the 48 words the one that is fastest to drive
    wins: a slightly longer forward-only word beats a short one with a stop and reversal in it."""
    x0, y0, t0 = start
    dx, dy = goal[0] - x0, goal[1] - y0
    c, s = math.cos(t0), math.sin(t0)
    x = (c * dx + s * dy) / rho
    y = (-s * dx + c * dy) / rho
    phi = goal[2] - t0
    best, best_cost = None, math.inf
    cusp_r = cusp / rho
    for steers, lengths in candidates(x, y, phi):
        nz = [a for a in lengths if abs(a) > 1e-10]
        total = sum(abs(a) for a in nz)
        if cusp_r:
            total += cusp_r * sum(1 for a, b in zip(nz, nz[1:]) if (a > 0) != (b > 0))
        if total < best_cost:
            best, best_cost = (steers, lengths), total
    steers, lengths = best
    return tuple((st, a * rho) for st, a in zip(steers, lengths) if abs(a) > 1e-10)


def cusps(path) -> int:
    """Gear changes along a path."""
    return sum(1 for (_, a), (_, b) in zip(path, path[1:]) if (a > 0) != (b > 0))


def length(path) -> float:
    return sum(abs(a) for _, a in path)


def reverse(path):
    """The same curve driven from its end back to its start."""
    return tuple((st, -a) for st, a in reversed(path))


def end_pose(start, path, rho: float) -> tuple[float, float, float]:
    x, y, th = start
    for st, a in path:
        if st == 0:
            x += a * math.cos(th)
            y += a * math.sin(th)
        else:
            k = st / rho
            th1 = th + k * a
            x += (math.sin(th1) - math.sin(th)) / k
            y -= (math.cos(th1) - math.cos(th)) / k
            th = th1
    return x, y, th


def sample(start, path, rho: float, step: float, include_end: bool = True):
    """Poses along a path every <= step metres of arc: arrays x, y, psi (start included)."""
    xs, ys, ths = [np.array([start[0]])], [np.array([start[1]])], [np.array([start[2]])]
    x, y, th = start
    for st, a in path:
        n = max(1, math.ceil(abs(a) / step))
        s = np.linspace(a / n, a, n)
        if st == 0:
            px = x + s * math.cos(th)
            py = y + s * math.sin(th)
            pt = np.full(n, th)
        else:
            k = st / rho
            pt = th + k * s
            px = x + (np.sin(pt) - math.sin(th)) / k
            py = y - (np.cos(pt) - math.cos(th)) / k
        xs.append(px)
        ys.append(py)
        ths.append(pt)
        x, y, th = float(px[-1]), float(py[-1]), float(pt[-1])
    x, y, th = np.concatenate(xs), np.concatenate(ys), np.concatenate(ths)
    if not include_end:
        x, y, th = x[:-1], y[:-1], th[:-1]
    return x, y, th


# --- vectorised: many pose pairs at once --------------------------------------------------------
# The same formulas over numpy arrays, for building roadmaps (tens of thousands of pairs). Out-of-
# domain square roots and arcsines produce NaN where a word doesn't exist; the validity masks drop
# them. tests/test_reeds_shepp.py checks this against the scalar version above.

def _vmod2pi(x):
    v = np.fmod(x, TWO_PI)
    v = np.where(v < -PI, v + TWO_PI, v)
    return np.where(v > PI, v - TWO_PI, v)


def _vtau_omega(u, v, xi, eta, phi):
    delta = _vmod2pi(u - v)
    a = np.sin(u) - np.sin(delta)
    b = np.cos(u) - np.cos(delta) - 1.0
    t1 = np.arctan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (np.cos(delta) - np.cos(v) - np.cos(u)) + 3.0
    tau = np.where(t2 < 0, _vmod2pi(t1 + PI), _vmod2pi(t1))
    return tau, _vmod2pi(tau - u + v - phi)


def _v_lpsplp(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    u, t = np.hypot(xi, eta), np.arctan2(eta, xi)
    v = _vmod2pi(phi - t)
    return (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_lpsprp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    u1 = xi * xi + eta * eta
    t1 = np.arctan2(eta, xi)
    u = np.sqrt(u1 - 4.0)
    t = _vmod2pi(t1 + np.arctan2(2.0, u))
    v = _vmod2pi(t - phi)
    return (u1 >= 4.0) & (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_lprml(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    u1, theta = np.hypot(xi, eta), np.arctan2(eta, xi)
    u = -2.0 * np.arcsin(0.25 * u1)
    t = _vmod2pi(theta + 0.5 * u + PI)
    v = _vmod2pi(phi - t + u)
    return (u1 <= 4.0) & (t >= -ZERO) & (u <= ZERO), t, u, v


def _v_lpruplumrm(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = 0.25 * (2.0 + np.hypot(xi, eta))
    u = np.arccos(rho)
    t, v = _vtau_omega(u, -u, xi, eta, phi)
    return (rho <= 1.0) & (t >= -ZERO) & (v <= ZERO), t, u, v


def _v_lprumlumrp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    u = -np.arccos(rho)
    t, v = _vtau_omega(u, u, xi, eta, phi)
    return (rho >= 0.0) & (rho <= 1.0) & (u >= -HALF_PI) & (t >= -ZERO) & (v >= -ZERO), t, u, v


def _v_lprmsmlm(x, y, phi):
    xi, eta = x - np.sin(phi), y - 1.0 + np.cos(phi)
    rho, theta = np.hypot(xi, eta), np.arctan2(eta, xi)
    r = np.sqrt(rho * rho - 4.0)
    u = 2.0 - r
    t = _vmod2pi(theta + np.arctan2(r, -2.0))
    v = _vmod2pi(phi - HALF_PI - t)
    return (rho >= 2.0) & (t >= -ZERO) & (u <= ZERO) & (v <= ZERO), t, u, v


def _v_lprmsmrm(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho, theta = np.hypot(-eta, xi), np.arctan2(xi, -eta)
    t = theta
    u = 2.0 - rho
    v = _vmod2pi(t + HALF_PI - phi)
    return (rho >= 2.0) & (t >= -ZERO) & (u <= ZERO) & (v <= ZERO), t, u, v


def _v_lprmslmrp(x, y, phi):
    xi, eta = x + np.sin(phi), y - 1.0 - np.cos(phi)
    rho = np.hypot(xi, eta)
    u = 4.0 - np.sqrt(rho * rho - 4.0)
    t = _vmod2pi(np.arctan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
    v = _vmod2pi(t - phi)
    return (rho >= 2.0) & (u <= ZERO) & (t >= -ZERO) & (v >= -ZERO), t, u, v


# (formula, backwards, base steering, lengths from (t, u, v)), in the order of candidates().
_FAMILIES = (
    (_v_lpsplp, False, (L, S, L), lambda t, u, v: (t, u, v)),
    (_v_lpsprp, False, (L, S, R), lambda t, u, v: (t, u, v)),
    (_v_lprml, False, (L, R, L), lambda t, u, v: (t, u, v)),
    (_v_lprml, True, (L, R, L), lambda t, u, v: (v, u, t)),
    (_v_lpruplumrm, False, (L, R, L, R), lambda t, u, v: (t, u, -u, v)),
    (_v_lprumlumrp, False, (L, R, L, R), lambda t, u, v: (t, u, u, v)),
    (_v_lprmsmlm, False, (L, R, S, L), lambda t, u, v: (t, np.full_like(t, -HALF_PI), u, v)),
    (_v_lprmsmrm, False, (L, R, S, R), lambda t, u, v: (t, np.full_like(t, -HALF_PI), u, v)),
    (_v_lprmsmlm, True, (L, S, R, L), lambda t, u, v: (v, u, np.full_like(t, -HALF_PI), t)),
    (_v_lprmsmrm, True, (R, S, R, L), lambda t, u, v: (v, u, np.full_like(t, -HALF_PI), t)),
    (_v_lprmslmrp, False, (L, R, S, L, R),
     lambda t, u, v: (t, np.full_like(t, -HALF_PI), u, np.full_like(t, -HALF_PI), v)),
)


def shortest_many(starts: np.ndarray, goals: np.ndarray, rho: float, cusp: float = 0.0,
                  build: bool = True):
    """`shortest` for P pose pairs at once. Returns (paths, lengths in metres, cusp counts).
    With build=False, `paths` is a Words object instead, and only the paths asked for are made
    (building every tuple is most of the cost when only a few are kept)."""
    starts = np.asarray(starts, float).reshape(-1, 3)
    goals = np.asarray(goals, float).reshape(-1, 3)
    c, s = np.cos(starts[:, 2]), np.sin(starts[:, 2])
    dx, dy = goals[:, 0] - starts[:, 0], goals[:, 1] - starts[:, 1]
    x = (c * dx + s * dy) / rho
    y = (-s * dx + c * dy) / rho
    phi = goals[:, 2] - starts[:, 2]
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)
    p = len(x)
    best_cost = np.full(p, np.inf)
    best_len = np.zeros((p, 5))
    best_steer = np.zeros((p, 5), dtype=int)
    best_cusps = np.zeros(p, dtype=int)
    cusp_r = cusp / rho
    with np.errstate(invalid="ignore", divide="ignore"):
        for f, backwards, steers, make in _FAMILIES:
            bx, by = (xb, yb) if backwards else (x, y)
            flipped = tuple(-st for st in steers)
            for sx, sy, steer, sign in ((1, 1, steers, 1), (-1, 1, steers, -1),
                                        (1, -1, flipped, 1), (-1, -1, flipped, -1)):
                ok, t, u, v = f(sx * bx, sy * by, sx * sy * phi)
                lengths = np.stack([sign * a for a in make(t, u, v)], axis=1)      # (P, k)
                ok &= np.isfinite(lengths).all(axis=1)
                if not ok.any():
                    continue
                lengths = np.where(ok[:, None], lengths, 0.0)
                cost = np.abs(lengths).sum(axis=1)
                nz = np.abs(lengths) > 1e-10
                sg = np.where(nz, np.sign(lengths), 0)
                n_cusp = np.zeros(p, dtype=int)
                last = np.zeros(p)
                for i in range(lengths.shape[1]):
                    n_cusp += ((sg[:, i] != 0) & (last != 0) & (sg[:, i] != last)).astype(int)
                    last = np.where(sg[:, i] != 0, sg[:, i], last)
                cost = cost + cusp_r * n_cusp
                better = ok & (cost < best_cost)
                if not better.any():
                    continue
                k = lengths.shape[1]
                best_cost[better] = cost[better]
                best_len[better] = 0.0
                best_len[better, :k] = lengths[better]
                best_steer[better] = 0
                best_steer[better, :k] = steer
                best_cusps[better] = n_cusp[better]
    words = Words(best_steer, best_len * rho)
    paths = [words[i] for i in range(p)] if build else words
    return paths, np.abs(best_len).sum(axis=1) * rho, best_cusps


class Words:
    """Paths from shortest_many, kept as arrays until one is asked for."""

    def __init__(self, steers: np.ndarray, lengths: np.ndarray):
        self.steers = steers
        self.lengths = lengths

    def __getitem__(self, i: int) -> tuple:
        return tuple((int(st), float(a)) for st, a in zip(self.steers[i], self.lengths[i])
                     if abs(a) > 1e-10)
