"""Probabilistic roadmaps over truck poses (x, y, psi), LaValle §5.6.1 (Fig. 5.25).

Build (a priori, scenario S1):
1. Sample poses uniformly over the field; keep the collision-free ones (N of them).
2. For each vertex, take its nearest vertices by position (scipy's cKDTree) as candidates.
3. Join each candidate pair with the local planner: the Reeds–Shepp word that is fastest to drive,
   reversals charged their stop and shift time (reeds_shepp.shortest with a cusp cost). Keep the K
   cheapest per vertex (see Roadmap.connect for why). Paths longer than `max_edge` are dropped.
4. Collision-check the truck along the path every 0.25 m (collision.py); keep the free ones as
   undirected edges. A path is drivable both ways, so one edge serves both directions.

LaValle's line 6 skips pairs already in the same component. We don't: that makes the roadmap a
forest and the travel times poor. Components are still reported.

On demand (scenario S2, `lazy=True`): vertices are added per query and joined to their K nearest
without any collision check (status UNKNOWN). A query searches, checks only the edges on the path
it found, deletes the colliding ones and searches again (Lazy PRM, Bohlin & Kavraki 2000). The
roadmap persists between queries, so later queries reuse earlier work.

Search: Dijkstra over (vertex, gear) states, so a gear change where two edges meet is charged
like one inside an edge (TruckSpec.edge_cost). scipy.sparse.csgraph.dijkstra does the search;
one call from the truck's vertex gives the travel time to every vertex at once, which the truck's
goal selection uses as its travel-time oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import reeds_shepp as rs
from .collision import Checker
from .cpu import NULL, Cpu
from .truck import TruckSpec, gear
from .world import SIZE_M

FREE, UNKNOWN, BLOCKED = 0, 1, 2


@dataclass
class Edge:
    u: int
    v: int
    path: tuple             # from u to v
    cost: float
    first: int              # gear of the first and last segment, from u to v
    last: int
    status: int


class Roadmap:
    def __init__(self, checker: Checker, spec: TruckSpec, k: int = 10, candidates: int = 120,
                 max_edge: float = 60.0, lazy: bool = False, cpu: Cpu = NULL):
        self.checker = checker
        self.spec = spec
        self.k = k
        self.candidates = candidates
        self.max_edge = max_edge
        self.lazy = lazy
        self.cpu = cpu
        self.poses = np.zeros((0, 3))
        self.edges: list[Edge] = []
        self.edge_of: dict[tuple[int, int], int] = {}
        self._cols = {"u": [], "v": [], "cost": [], "first": [], "last": [], "status": []}
        self._graph = None
        self._dijkstra_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.stats = {"samples_drawn": 0, "local_plans": 0, "edges_checked": 0, "edges_blocked": 0,
                      "searches": 0}

    # --- vertices and edges ---------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.poses)

    def sample_free(self, n: int, rng: np.random.Generator, box=None) -> np.ndarray:
        """n collision-free poses, uniform in `box` (x0, x1, y0, y1) or the whole field. Gives up
        (returning fewer) after 20 n draws, for boxes that are mostly obstacle."""
        x0, x1, y0, y1 = box or (0.0, SIZE_M, 0.0, SIZE_M)
        out, drawn = [], 0
        while sum(len(o) for o in out) < n and drawn < 20 * n:
            m = max(64, 2 * n)
            cand = np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m),
                                    rng.uniform(-math.pi, math.pi, m)])
            drawn += m
            with self.cpu("collision"):
                ok = self.checker.free(cand[:, 0], cand[:, 1], cand[:, 2])
            out.append(cand[ok])
        self.stats["samples_drawn"] += drawn
        poses = np.concatenate(out) if out else np.zeros((0, 3))
        return poses[:n]

    def add_vertices(self, poses: np.ndarray) -> np.ndarray:
        start = self.n
        self.poses = np.concatenate([self.poses, np.asarray(poses, float).reshape(-1, 3)])
        return np.arange(start, self.n)

    def connect(self, ids, k: int | None = None) -> int:
        """Join each vertex in `ids` to the k vertices it is cheapest to drive to or from, among
        its `candidates` nearest by position. Returns the number of new edges.

        Nearest-by-position alone joins mostly poses a few metres apart at random headings, and
        between those the local planner needs a reversal almost every time; routes through such a
        roadmap zigzag with a gear change per vertex. Ranking by the local planner's own cost
        (travel time, reversals included) keeps the edges that line up with the vertex's heading."""
        k = k or self.k
        ids = np.asarray(ids, int)
        if len(ids) == 0 or self.n < 2:
            return 0
        tree = cKDTree(self.poses[:, :2])
        _, nbrs = tree.query(self.poses[ids, :2], k=min(self.candidates + 1, self.n))
        nbrs = np.asarray(nbrs).reshape(len(ids), -1)
        src = np.repeat(ids, nbrs.shape[1])
        dst = nbrs.ravel()
        m = (dst != src) & (dst < self.n)
        src, dst = src[m], dst[m]
        lo, hi = np.minimum(src, dst), np.maximum(src, dst)
        keys = np.unique(lo * self.n + hi)
        keys = np.array([key for key in keys if (key // self.n, key % self.n) not in self.edge_of],
                        dtype=int)
        if len(keys) == 0:
            return 0
        u_all, v_all = keys // self.n, keys % self.n
        cusp_m = self.spec.change_cost * self.spec.v_max
        with self.cpu("local_planner"):
            words, lengths, cusps = rs.shortest_many(self.poses[u_all], self.poses[v_all],
                                                     self.spec.rho, cusp_m, build=False)
            self.stats["local_plans"] += len(keys)
            cost = lengths / self.spec.v_max + cusps * self.spec.change_cost
            cost[(lengths > self.max_edge) | (lengths <= 0.0)] = np.inf
            # The k cheapest per vertex in `ids`, looking at each pair from both ends.
            ends = np.concatenate([u_all, v_all])
            pair = np.concatenate([np.arange(len(keys))] * 2)
            c2 = np.concatenate([cost, cost])
            mine = np.isin(ends, ids) & np.isfinite(c2)
            ends, pair, c2 = ends[mine], pair[mine], c2[mine]
            order = np.lexsort((c2, ends))
            ends, pair = ends[order], pair[order]
            first = np.searchsorted(ends, ends, side="left")
            rank = np.arange(len(ends)) - first
            chosen = np.unique(pair[rank < k])
        keep = [(int(u_all[i]), int(v_all[i])) for i in chosen]
        starts = [self.poses[u] for u, _ in keep]
        paths = [words[i] for i in chosen]
        if self.lazy:
            ok = np.ones(len(paths), dtype=bool)
            status = UNKNOWN
        else:
            ok = self._check(starts, paths)
            status = FREE
        added = 0
        for (u, v), path, good in zip(keep, paths, ok):
            if good:
                self._add_edge(u, v, path, status)
                added += 1
        self._changed()
        return added

    def _check(self, starts, paths, chunk: int = 1500) -> np.ndarray:
        out = []
        with self.cpu("collision"):
            for i in range(0, len(paths), chunk):
                out.append(self.checker.paths_free(starts[i:i + chunk], paths[i:i + chunk]))
        self.stats["edges_checked"] += len(paths)
        return np.concatenate(out) if out else np.zeros(0, dtype=bool)

    def _add_edge(self, u: int, v: int, path, status: int) -> int:
        e = Edge(u, v, path, self.spec.edge_cost(path), gear(path[0][1]), gear(path[-1][1]), status)
        self.edges.append(e)
        for name, value in (("u", u), ("v", v), ("cost", e.cost), ("first", e.first),
                            ("last", e.last), ("status", status)):
            self._cols[name].append(value)
        self.edge_of[(min(u, v), max(u, v))] = len(self.edges) - 1
        return len(self.edges) - 1

    def connect_pose(self, pose, k: int | None = None) -> int:
        """Add one pose (the truck's spawn) as a vertex and join it. Returns its id."""
        (i,) = self.add_vertices(np.asarray(pose, float)[None, :])
        self.connect([i], k=k or 2 * self.k)
        return int(i)

    def check_edges(self, ids) -> bool:
        """Lazy PRM: collision-check UNKNOWN edges; block the bad ones. True if all were free."""
        ids = [e for e in ids if self.edges[e].status == UNKNOWN]
        if not ids:
            return True
        ok = self._check([self.poses[self.edges[e].u] for e in ids], [self.edges[e].path for e in ids])
        for e, good in zip(ids, ok):
            self.edges[e].status = FREE if good else BLOCKED
            self._cols["status"][e] = self.edges[e].status
            self.stats["edges_blocked"] += int(not good)
            if not good and self._graph is not None:
                self._graph.data[self._arc_pos[e]] = np.inf
        if not ok.all():
            self._dijkstra_cache.clear()
        return bool(ok.all())

    def _changed(self):
        self._graph = None
        self._dijkstra_cache.clear()

    # --- build ------------------------------------------------------------------------------------

    def build(self, n: int, rng: np.random.Generator) -> None:
        with self.cpu("sampling"):
            ids = self.add_vertices(self.sample_free(n, rng))
        self.connect(ids)

    # --- search -----------------------------------------------------------------------------------

    def graph(self) -> csr_matrix:
        """Directed graph over states 2 u + gear (0 forward, 1 reverse): the gear the truck arrived
        in. Traversing an edge from a state in the other gear costs a gear change. Rebuilt only
        when edges are added; a lazily blocked edge just gets infinite weight in place."""
        if self._graph is None:
            cols = self._cols
            u = np.asarray(cols["u"], int)
            v = np.asarray(cols["v"], int)
            c = np.asarray(cols["cost"], float)
            f = np.asarray(cols["first"], int)
            l = np.asarray(cols["last"], int)  # noqa: E741
            c = np.where(np.asarray(cols["status"], int) == BLOCKED, np.inf, c)
            ch = self.spec.change_cost
            # u -> v: starts in gear f, ends in gear l. v -> u: the reversed path starts in gear
            # 1 - l and ends in gear 1 - f. No two arcs share a (row, col).
            rows = np.concatenate([2 * u + f, 2 * u + 1 - f, 2 * v + 1 - l, 2 * v + l])
            dst = np.concatenate([2 * v + l, 2 * v + l, 2 * u + 1 - f, 2 * u + 1 - f])
            data = np.concatenate([c, c + ch, c, c + ch])
            order = np.lexsort((dst, rows))
            indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=2 * self.n))])
            self._graph = csr_matrix((data[order], dst[order], indptr), shape=(2 * self.n, 2 * self.n))
            pos = np.empty(len(order), dtype=int)
            pos[order] = np.arange(len(order))
            self._arc_pos = pos.reshape(4, -1).T                 # (edges, 4) positions in data
        return self._graph

    def search(self, source: int) -> tuple[np.ndarray, np.ndarray]:
        """Dijkstra from a vertex where the truck is at rest (either gear). Returns the cost of every
        state and the predecessor array. Cached until the roadmap changes."""
        if source not in self._dijkstra_cache:
            with self.cpu("search"):
                g = self.graph()
                dist, pred, _ = dijkstra(g, directed=True, indices=[2 * source, 2 * source + 1],
                                         min_only=True, return_predecessors=True)
            self.stats["searches"] += 1
            self._dijkstra_cache = {source: (dist, pred)}
        return self._dijkstra_cache[source]

    def times_from(self, source: int) -> np.ndarray:
        """Estimated travel time (s) from `source` to every vertex, inf if unreachable: the graph
        cost plus the one stop_overhead every leg pays."""
        dist, _ = self.search(source)
        t = np.minimum(dist[0::2], dist[1::2]) + self.spec.stop_overhead
        t[source] = 0.0
        return t

    def route(self, source: int, target: int) -> list[tuple[int, bool]] | None:
        """Edges (id, forward) from source to target, from the cached Dijkstra."""
        dist, pred = self.search(source)
        state = 2 * target + int(dist[2 * target + 1] < dist[2 * target])
        if not np.isfinite(dist[state]):
            return None
        states = [state]
        while pred[states[-1]] >= 0:
            states.append(int(pred[states[-1]]))
        states.reverse()
        out = []
        for a, b in zip(states, states[1:]):
            u, w = a // 2, b // 2
            e = self.edge_of[(min(u, w), max(u, w))]
            out.append((e, self.edges[e].u == u))
        return out

    def segments(self, route) -> tuple:
        segs = []
        for e, forward in route:
            p = self.edges[e].path
            segs.extend(p if forward else rs.reverse(p))
        return tuple(segs)

    def components(self) -> np.ndarray:
        """Component label per vertex over the edges not known to collide."""
        live = np.asarray(self._cols["status"], int) != BLOCKED
        if not live.any():
            return np.arange(self.n)
        u = np.asarray(self._cols["u"], int)[live]
        v = np.asarray(self._cols["v"], int)[live]
        g = csr_matrix((np.ones(len(u)), (u, v)), shape=(self.n, self.n))
        return connected_components(g, directed=False)[1]

    def centres(self) -> np.ndarray:
        cx, cy = self.spec.centre(self.poses[:, 0], self.poses[:, 1], self.poses[:, 2])
        return np.column_stack([cx, cy])
