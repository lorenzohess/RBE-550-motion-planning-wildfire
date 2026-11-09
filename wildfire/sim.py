"""One run of the game: the truck against the arsonist on a fresh seeded field.

Each tick (dt = 0.1 s, notes D10):
1. the arsonist moves and may light a cell;
2. the truck drives, dwells or decides, and may finish putting out a cell;
3. the fire's timers fire: spread, then burnout.
So a cell whose 5 s dwell ends on its burnout tick is saved, and putting a cell out on its 10 s
tick stops its spread.

The truck decides only when it's parked: on arrival, after each cell, when its cell is lost, and
while idle whenever a cell changes state. It never re-targets mid-leg (except when chasing).

Catharsis (notes D14): once nothing is burning and the arsonist has nothing left to light, the
truck learns where the arsonist is and chases it, replanning every leg; within 5 m ends the run.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import bound
from .arsonist import Arsonist
from .astar import free_components
from .collision import Checker
from .cpu import Cpu
from .fire import BURNED, BURNING, Fire, score
from .oracle import HERE, EuclidOracle, RoadmapOracle, parking_spots
from .policies import SERVICE, Plan, Snapshot, get
from .prm import FREE, Roadmap
from .truck import TruckSpec, Trajectory
from .world import CELLS, REACH, SIZE_M, build_world, rng_for, square_distance

DT = 0.1
CATCH = 5.0                 # m, catharsis ends when the truck's centre is this close
CHASE_EDGES = 3             # chase legs stop after this many roadmap edges, then replan
BAN_TICKS = 1200            # a stop whose leg failed is skipped for 120 s
CELL_BAN_TICKS = 6000       # a cell whose legs failed twice is skipped for 600 s


@dataclass
class Config:
    seed: int = 0
    scenario: str = "S1"            # S1 a-priori PRM, S2 on-demand PRM
    truck: str = "T2"               # T0 nearest, T1 oldest, T2 rollout, T2e rollout + Euclid
    arsonist: str = "W1"            # W0 nearest, W1 cascade-seeking
    burn: float = 300.0             # s, T_burn (E1: baseline vs baseline is closest to even here)
    arsonist_speed: float = 1.0     # m/s
    duration: float = 3600.0        # s
    rho: float = 0.10
    prm_n: int = 2000
    prm_k: int = 10
    prm_candidates: int = 120
    a_max: float = 2.0
    gear_penalty: float = 2.0
    catharsis: bool = True


class AprioriPlanner:
    def __init__(self, roadmap: Roadmap):
        self.roadmap = roadmap

    def leg(self, src: int, oracle, plan: Plan):
        v = int(oracle.stop_vertex[plan.stop])
        if v == src:
            return (), v
        route = self.roadmap.route(src, v)
        return None if route is None else (self.roadmap.segments(route), v)

    def chase(self, src: int, xy, component: np.ndarray):
        rm = self.roadmap
        c = rm.centres()[component]
        v = int(component[np.argmin(np.hypot(*(c - np.asarray(xy)).T))])
        route = rm.route(src, v)
        if route is None or v == src:
            return None
        route = route[:CHASE_EDGES]
        e, forward = route[-1]
        end = rm.edges[e].v if forward else rm.edges[e].u
        return rm.segments(route), int(end)


class OnDemandPlanner:
    """Lazy PRM queries (prm.py docstring): sample around start and goal, join lazily, search,
    check the path's edges, repeat; enlarge the box if the goal can't be reached."""

    def __init__(self, roadmap: Roadmap, world, spec: TruckSpec, rng, n_box=60, n_goal=20,
                 min_goals=5, pad=20.0, attempts=3):
        self.rm, self.world, self.spec, self.rng = roadmap, world, spec, rng
        self.n_box, self.n_goal, self.pad, self.attempts = n_box, n_goal, pad, attempts
        self.min_goals = min_goals
        self.stats = {"queries": 0, "failed": 0, "rounds": 0}

    def _goal_samples(self, xy, radius: float) -> np.ndarray:
        n = 4 * self.n_goal
        r = radius * np.sqrt(self.rng.uniform(0, 1, n))
        a = self.rng.uniform(-math.pi, math.pi, n)
        psi = self.rng.uniform(-math.pi, math.pi, n)
        cx, cy = xy[0] + r * np.cos(a), xy[1] + r * np.sin(a)
        x, y, _ = self.spec.pose_from_centre(cx, cy, psi)
        poses = np.column_stack([x, y, psi])
        ok = self.rm.checker.free(poses[:, 0], poses[:, 1], poses[:, 2])
        return poses[ok][: self.n_goal]

    def _query(self, src: int, xy, is_goal, sample_radius: float):
        """Round 0 tries the roadmap as it stands (adding goal samples only if fewer than
        `min_goals` vertices satisfy the goal). Each later round adds samples in a box around
        start and goal, twice as wide each time."""
        self.stats["queries"] += 1
        rm = self.rm
        start = rm.poses[src, :2]
        for attempt in range(self.attempts + 1):
            self.stats["rounds"] += 1
            goals = np.flatnonzero(is_goal(rm.centres()))
            new = []
            with rm.cpu("sampling"):
                if len(goals) < self.min_goals or attempt > 0:
                    new.append(self._goal_samples(xy, sample_radius))
                if attempt > 0:
                    pad = self.pad * 2 ** (attempt - 1)
                    box = (max(0.0, min(start[0], xy[0]) - pad), min(SIZE_M, max(start[0], xy[0]) + pad),
                           max(0.0, min(start[1], xy[1]) - pad), min(SIZE_M, max(start[1], xy[1]) + pad))
                    new.append(rm.sample_free(self.n_box, self.rng, box))
            if new:
                ids = rm.add_vertices(np.concatenate(new))
                rm.connect(ids)
                goals = np.flatnonzero(is_goal(rm.centres()))
            goals = goals[goals != src]
            for _ in range(50):
                if len(goals) == 0:
                    break
                t = rm.times_from(src)[goals]
                if not np.isfinite(t).any():
                    break
                g = int(goals[np.argmin(t)])
                route = rm.route(src, g)
                if rm.check_edges([e for e, _ in route]):
                    return route, g
        self.stats["failed"] += 1
        return None

    def leg(self, src: int, oracle, plan: Plan):
        xy = oracle.stop_point(plan.stop)
        want = np.array(plan.cells)

        def is_goal(centres):
            d = square_distance(centres, self.world.centres[want]) <= REACH
            ok = d[:, 0]
            count = d.sum(axis=1)
            return ok & (count >= count[ok].max()) if ok.any() else ok
        found = self._query(src, xy, is_goal, 3.0)
        if found is None:
            return None
        route, g = found
        return self.rm.segments(route), g

    def chase(self, src: int, xy, component=None):
        found = self._query(src, xy, lambda c: np.hypot(*(c - np.asarray(xy)).T) <= CATCH, CATCH)
        if found is None:
            return None
        route, g = found
        route = route[:CHASE_EDGES]
        e, forward = route[-1]
        end = self.rm.edges[e].v if forward else self.rm.edges[e].u
        return self.rm.segments(route), int(end)


class Game:
    def __init__(self, cfg: Config):
        wall0 = time.perf_counter()
        self.cfg = cfg
        self.world = build_world(cfg.seed, cfg.rho)
        self.spec = TruckSpec(a_max=cfg.a_max, gear_penalty=cfg.gear_penalty)
        self.checker = Checker(self.world, self.spec)
        self.fire = Fire(self.world, burn_ticks=round(cfg.burn / DT), spread_ticks=round(10.0 / DT))
        self.cpu = {"truck": Cpu("truck"), "arsonist": Cpu("arsonist")}
        self.policy, oracle_kind = get(cfg.truck)
        self.service_ticks = round(SERVICE / DT)
        self.end_tick = round(cfg.duration / DT)

        # --- arsonist spawn: a free node in the east 25 m band, in the largest free region ------
        labels = free_components(~self.world.occupied)
        main = np.bincount(labels[labels > 0]).argmax()
        east = [tuple(n) for n in np.argwhere(labels == main) if n[0] >= CELLS - 5]
        rng = rng_for(cfg.seed, "spawn")
        node = east[rng.integers(len(east))]
        self.arsonist = Arsonist(self.world, self.fire, node, cfg.arsonist_speed, DT, cfg.arsonist,
                                 rng_for(cfg.seed, "arsonist"), self.cpu["arsonist"])

        # --- truck spawn and roadmap -------------------------------------------------------------
        tcpu = self.cpu["truck"]
        lazy = cfg.scenario == "S2"
        if cfg.scenario not in ("S1", "S2"):
            raise ValueError("scenario is S1 (a priori) or S2 (on demand)")
        self.roadmap = Roadmap(self.checker, self.spec, k=cfg.prm_k, candidates=cfg.prm_candidates,
                               lazy=lazy, cpu=tcpu)
        # The spawn must join the main part of the a-priori roadmap, so the truck isn't boxed in.
        # S2 checks that on the same roadmap built untimed (the reference), so both scenarios
        # get the same spawn for a seed and runs pair up.
        tcpu.prefix = "build/"
        if lazy:
            self.reference = Roadmap(self.checker, self.spec, k=cfg.prm_k,
                                     candidates=cfg.prm_candidates)
            self.reference.build(cfg.prm_n, rng_for(cfg.seed, "prm"))
        else:
            self.roadmap.build(cfg.prm_n, rng_for(cfg.seed, "prm"))
            self.reference = self.roadmap
        for _ in range(100):
            pose = (float(rng.uniform(1.0, 20.0)), float(rng.uniform(5.0, SIZE_M - 5.0)), 0.0)
            if not self.checker.pose_free(pose):
                continue
            v = self.reference.connect_pose(pose)
            lab = self.reference.components()
            if (lab == lab[v]).sum() > 0.5 * self.reference.n:
                self.reference_vertex = v
                break
        else:
            raise RuntimeError("no truck spawn joins the roadmap")
        self.vertex = (int(self.roadmap.add_vertices(np.array([pose]))[0]) if lazy
                       else self.reference_vertex)
        self.spawn = pose
        if lazy:
            with tcpu("spots"):
                spots = parking_spots(self.checker, self.spec)
            self.oracle = EuclidOracle(self.world, self.spec, spots)
            self.planner = OnDemandPlanner(self.roadmap, self.world, self.spec,
                                           rng_for(cfg.seed, "prm"))
            self.component = None
        else:
            lab = self.roadmap.components()
            self.component = np.flatnonzero(lab == lab[self.vertex])
            if oracle_kind == "roadmap":
                self.oracle = RoadmapOracle(self.world, self.spec, self.roadmap, self.component)
            else:
                self.oracle = EuclidOracle(self.world, self.spec,
                                           self.roadmap.centres()[self.component],
                                           stop_vertex=self.component)
            self.planner = AprioriPlanner(self.roadmap)
        tcpu.prefix = ""
        self.build_wall = time.perf_counter() - wall0

        # --- truck state -------------------------------------------------------------------------
        self.pose = self.spawn
        self.traj: Trajectory | None = None
        self.traj_tick = 0
        self.traj_end = self.vertex
        self.queue: list[int] = []
        self.plan: Plan | None = None
        self.dwell: tuple[int, int] | None = None     # (cell, start tick)
        self.seen_events = -1
        self.bans: list[tuple[int, int]] = []
        self.cell_bans: list[tuple[int, int]] = []
        self.leg_failures: dict[int, int] = {}
        self.chasing = False
        self.chase_retry = 0
        self.caught = False
        self.tick = 0
        self.done = False
        self.end_reason = ""
        self.truck_stats = {"legs": 0, "distance": 0.0, "drive_time": 0.0, "gear_changes": 0,
                            "decisions": 0, "failed_legs": 0, "lost_while_dwelling": 0,
                            "idle_time": 0.0}
        self.series: list[tuple] = []
        self.wall0 = time.perf_counter()

    # --- geometry ------------------------------------------------------------------------------

    def truck_pose(self):
        if self.traj is not None:
            return self.traj.pose((self.tick - self.traj_tick) * DT)[:3]
        return self.pose

    def truck_centre(self) -> np.ndarray:
        x, y, psi = self.truck_pose()
        return np.array(self.spec.centre(x, y, psi), float)

    # --- the tick ------------------------------------------------------------------------------

    def step(self) -> None:
        if self.done:
            return
        tick = self.tick
        self.arsonist.update(tick, self.truck_centre())
        self._truck(tick)
        self.fire.advance(tick)
        if tick % 100 == 0:
            s = self.fire.counts()
            sc = score(self.fire.events) if tick % 600 == 0 else None
            self.series.append((tick * DT, s["intact"], s["burning"], s["extinguished"], s["burned"],
                                sc.truck if sc else None, sc.arsonist if sc else None))
        self.tick += 1
        self._check_end()

    def run(self) -> dict:
        while not self.done:
            self.step()
        return self.results()

    def _check_end(self) -> None:
        if self.tick >= self.end_tick:
            self.done, self.end_reason = True, "time"
        elif self.caught:
            self.done, self.end_reason = True, "caught"
        elif not self.chasing and not (self.fire.state == BURNING).any() \
                and not self.arsonist.has_targets() and self.traj is None:
            if self.cfg.catharsis:
                self.chasing = True
                self.queue, self.dwell = [], None
            else:
                self.done, self.end_reason = True, "steady"

    def _truck(self, tick: int) -> None:
        if self.traj is not None:
            if (tick - self.traj_tick) * DT < self.traj.duration:
                if self.chasing:
                    self._check_catch()
                return
            self.pose = self.traj.end
            self.vertex = self.traj_end
            self.traj = None
        if self.chasing:
            self._check_catch()
            if not self.caught:
                self._chase(tick)
            return
        if self.dwell is not None:
            cell, start = self.dwell
            if self.fire.state[cell] != BURNING:
                self.dwell = None
                self.truck_stats["lost_while_dwelling"] += int(self.fire.state[cell] == BURNED)
            elif tick - start >= self.service_ticks:
                self.fire.extinguish(cell, tick)
                self.dwell = None
            else:
                return
        centre = self.truck_centre()
        for _ in range(4):                        # a failed leg bans its stop and decides again
            while self.queue:
                c = self.queue.pop(0)
                if self.fire.state[c] == BURNING and \
                        self.world.distances(centre)[0, c] <= REACH:
                    self.dwell = (c, tick)
                    return
            if len(self.fire.events) == self.seen_events:
                self.truck_stats["idle_time"] += DT
                return
            plan = self._decide(tick, centre)
            if plan is None:
                self.seen_events = len(self.fire.events)
                self.truck_stats["idle_time"] += DT
                return
            self.seen_events = -1
            if plan.stop == HERE:
                self.queue = list(plan.cells)
                continue
            with self.cpu["truck"]("leg"):
                leg = self.planner.leg(self.vertex, self.oracle, plan)
            if leg is None:
                self.truck_stats["failed_legs"] += 1
                self.oracle.ban(plan.stop)
                self.bans.append((tick + BAN_TICKS, plan.stop))
                cell = plan.cells[0]
                self.leg_failures[cell] = self.leg_failures.get(cell, 0) + 1
                if self.leg_failures[cell] >= 2:      # S2 can't tell a pocket from bad luck:
                    self.leg_failures[cell] = 0       # give up on the cell for a while
                    self.oracle.ban_cell(cell)
                    self.cell_bans.append((tick + CELL_BAN_TICKS, cell))
                continue
            self._drive(tick, *leg)
            self.queue = list(plan.cells)
            return

    def _decide(self, tick: int, centre) -> Plan | None:
        while self.bans and self.bans[0][0] <= tick:
            self.oracle.ban(self.bans.pop(0)[1], False)
        while self.cell_bans and self.cell_bans[0][0] <= tick:
            self.oracle.ban_cell(self.cell_bans.pop(0)[1], False)
        cpu = self.cpu["truck"]
        with cpu("oracle"):
            self.oracle.update(self.vertex, centre)
        deadline = self.fire.deadline_ticks().astype(float) * DT
        deadline[self.fire.state != BURNING] = np.inf
        snap = Snapshot(t=tick * DT, state=self.fire.state, deadline=deadline, oracle=self.oracle,
                        world=self.world)
        with cpu("goal_selection"):
            plan = self.policy(snap)
        self.truck_stats["decisions"] += 1
        self.plan = plan
        return plan

    def _drive(self, tick: int, segments, end_vertex: int) -> None:
        self.traj = Trajectory(self.spec, self.pose, segments)
        self.traj_tick = tick
        self.traj_end = end_vertex
        st = self.truck_stats
        st["legs"] += 1
        st["distance"] += self.traj.length
        st["drive_time"] += self.traj.duration
        st["gear_changes"] += max(0, len(self.traj.runs) - 1)

    def _check_catch(self) -> None:
        a = np.array(self.arsonist.xy())
        if np.hypot(*(self.truck_centre() - a)) <= CATCH:
            self.caught = True

    def _chase(self, tick: int) -> None:
        if tick < self.chase_retry:
            return
        with self.cpu["truck"]("leg"):
            leg = self.planner.chase(self.vertex, self.arsonist.xy(), self.component)
        if leg is not None:
            self._drive(tick, *leg)
        else:
            self.chase_retry = tick + 50              # no way closer right now: retry in 5 s

    # --- results ---------------------------------------------------------------------------------

    def extinguishable(self) -> np.ndarray:
        """Cells with a roadmap vertex within 10 m in the truck's component (notes Scoring 4), on
        the a-priori roadmap: S1's own, or S2's untimed reference."""
        ref = self.reference
        lab = ref.components()
        centres = ref.centres()[lab == lab[self.reference_vertex]]
        return (self.world.distances(centres) <= REACH).any(axis=0)

    def results(self) -> dict:
        cfg = self.cfg
        sc = score(self.fire.events)
        counts = self.fire.counts()
        ign = [e.tick * DT for e in self.fire.events if e.kind == "ignite"]
        ext = self.extinguishable()
        threshold = bound.save_all_threshold(self.world, cfg.seed)
        st = self.truck_stats
        cpu = {p: dict(c.totals) for p, c in self.cpu.items()}
        rm = self.roadmap
        out = {
            **asdict(cfg),
            "end_reason": self.end_reason, "sim_time": self.tick * DT, "caught": self.caught,
            "score_truck": sc.truck, "score_arsonist": sc.arsonist, "winner": sc.winner,
            "margin": sc.truck - sc.arsonist,
            "ignitions": sc.ignitions, "arsonist_ignitions": sc.arsonist_ignitions,
            "spread_ignitions": sc.spread_ignitions, "relights": sc.relights,
            "extinguishes": sc.extinguishes, "burnouts": sc.burnouts,
            **{f"final_{k}": v for k, v in counts.items()},
            "cells": self.world.n,
            "extinguishable": int(ext.sum()) if ext is not None else None,
            "extinguish_rate": sc.extinguishes / max(1, sc.extinguishes + sc.burnouts),
            "saved_share": sc.extinguishes / max(1, sc.ignitions),
            "free_travel_ceiling": bound.saved_free_travel(ign, cfg.burn, end=self.tick * DT),
            "save_all_threshold": threshold, "lam": threshold / cfg.burn,
            **{f"truck_{k}": v for k, v in st.items()},
            "time_per_extinguish": (st["drive_time"] + SERVICE * sc.extinguishes) / max(1, sc.extinguishes),
            "cpu_truck": sum(cpu["truck"].values()), "cpu_arsonist": sum(cpu["arsonist"].values()),
            "cpu_truck_build": sum(v for k, v in cpu["truck"].items() if k.startswith("build/")),
            **{f"cpu_truck_{k.replace('/', '_')}": v for k, v in cpu["truck"].items()},
            **{f"cpu_arsonist_{k}": v for k, v in cpu["arsonist"].items()},
            "prm_vertices": rm.n, "prm_edges": sum(e.status == FREE for e in rm.edges),
            "prm_edges_total": len(rm.edges),
            **{f"prm_{k}": v for k, v in rm.stats.items()},
            **({f"ondemand_{k}": v for k, v in self.planner.stats.items()}
               if isinstance(self.planner, OnDemandPlanner) else {}),
            **{f"arsonist_{k}": v for k, v in self.arsonist.stats.items()},
            "wall_build": self.build_wall, "wall_run": time.perf_counter() - self.wall0,
        }
        out["series"] = self.series
        return out
