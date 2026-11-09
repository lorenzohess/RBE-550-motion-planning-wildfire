from collections import deque

import numpy as np

from wildfire.arsonist import Arsonist
from wildfire.astar import STEPS, astar
from wildfire.fire import BURNING, INTACT, Fire
from wildfire.world import CELL_M, CELLS, build_world


def bfs_length(free, start, goals):
    seen = {start: 0}
    q = deque([start])
    while q:
        n = q.popleft()
        if n in goals:
            return seen[n]
        for dx, dy in STEPS:
            m = (n[0] + dx, n[1] + dy)
            if 0 <= m[0] < CELLS and 0 <= m[1] < CELLS and free[m] and m not in seen:
                seen[m] = seen[n] + 1
                q.append(m)
    return None


def test_astar_is_optimal_against_bfs():
    rng = np.random.default_rng(0)
    for seed in range(5):
        free = ~build_world(seed, rho=0.3).occupied
        nodes = np.argwhere(free)
        for _ in range(30):
            s = tuple(nodes[rng.integers(len(nodes))])
            goals = {tuple(nodes[rng.integers(len(nodes))]) for _ in range(3)}
            path = astar(free, s, goals)
            ref = bfs_length(free, s, goals)
            if ref is None:
                assert path is None
                continue
            assert len(path) - 1 == ref
            assert path[0] == s and path[-1] in goals
            for a, b in zip(path, path[1:]):
                assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and free[b]


def test_raised_costs_keep_paths_valid_and_no_cheaper():
    free = ~build_world(1).occupied
    nodes = np.argwhere(free)
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = tuple(nodes[rng.integers(len(nodes))])
        g = {tuple(nodes[rng.integers(len(nodes))])}
        plain = astar(free, s, g)
        bumped = astar(free, s, g, step_cost=lambda n: CELL_M * (1 + (n[0] % 3 == 0)))
        assert (plain is None) == (bumped is None)
        if plain:
            assert len(bumped) >= len(plain) and bumped[-1] in g


def test_one_ignition_per_node_visit_and_only_adjacent():
    world = build_world(0)
    for policy in ("W0", "W1"):
        fire = Fire(world, burn_ticks=10**7)
        node = tuple(np.argwhere(~world.occupied)[-1])
        a = Arsonist(world, fire, node, speed=1.0, dt=0.1, policy=policy)
        visits = {}
        rng = np.random.default_rng(0)
        for tick in range(30000):
            n_before = len(fire.events)
            a.update(tick, truck_xy=(10.0, 10.0))
            fire.advance(tick)
            burning = np.flatnonzero(fire.state == BURNING)
            if len(burning) and tick % 20 == 0:          # a stand-in truck, so relights happen
                fire.extinguish(int(rng.choice(burning)), tick)
            mine = [e for e in fire.events[n_before:] if e.cause == "arsonist"]
            assert len(mine) <= 1
            visit = (a.node, a.stats["steps"])
            for e in mine:
                cx, cy = world.cells[e.cell]
                assert abs(cx - a.node[0]) + abs(cy - a.node[1]) == 1
                visits[visit] = visits.get(visit, 0) + 1
                assert visits[visit] == 1
        assert a.stats["ignitions"] > 20
        assert sum(e.relight for e in fire.events) > 10
