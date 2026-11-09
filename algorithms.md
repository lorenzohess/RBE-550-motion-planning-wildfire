# Wildfire — planners in pseudocode

For the report's §5.2 / §5.3 flow descriptions. Names match the code.

## Arsonist (`arsonist.py`, `astar.py`)

```
every tick:
    if walking: advance; on reaching the next node: lit_here ← false
    at a node:
        if not lit_here and a lightable cell is 4-adjacent:          # D15: one per node visit
            light the target if adjacent, else an adjacent intact cell (W1: only if far from truck)
            lit_here ← true
        if no target, target no longer lightable, or path used up:
            target ← SELECT_GOAL()
            goals  ← free nodes 4-adjacent to target (minus here, if lit_here)
            path   ← A*(here, goals)

SELECT_GOAL:
    lightable ← intact or extinguished cells beside a node the arsonist can reach
    if none: walk to a random node among the 10 % farthest from the truck
    W0: nearest lightable cell (straight line)
    W1: argmax  value(c) · min(1, d_truck(c) / 80 m) / (walk_time(c) + 10 s)
        value(c) = intact cells its spread would cascade into (spread graph restricted to intact)

A*(start, goals):                     # 4-connected 5 m grid, LaValle §2.2.2
    h(n) = 5 m · min over goals of Manhattan(n, goal)     # admissible, consistent
    open ← {start}, g(start) = 0
    pop min f = g + h (ties: larger g, then FIFO); if a goal: return path
    for each free 4-neighbour m: g' = g + step_cost(m)   # 5 m; W1: 5 m · (1 + 2 e^(-d_truck/20 m))
        if g' < g(m): g(m) ← g', parent(m) ← n, push m
```

## Truck: roadmap (`prm.py`, `reeds_shepp.py`, `collision.py`)

```
BUILD (S1, before t = 0):                                    # LaValle §5.6.1, Fig. 5.25
    V ← N = 2000 collision-free poses (x, y, ψ), uniform
    for each v: candidates ← its 120 nearest by position (k-d tree)
        for each candidate u: path ← LOCAL_PLAN(v, u); cost ← travel time incl. gear changes
        keep the 10 cheapest
    check every kept path for collisions every 0.25 m; add the free ones as undirected edges

LOCAL_PLAN(a, b):                                            # Reeds–Shepp, ρ = 13 m
    evaluate all 48 words in closed form; return the one minimising length + 70 m · cusps

QUERY(source vertex):                                        # one call serves every target
    Dijkstra over states (vertex, gear); crossing an edge in the other gear costs a stop + shift
    → travel time to every vertex, and predecessors for the leg to whichever is chosen

ON-DEMAND QUERY (S2, roadmap starts empty and persists):
    round 0: use the roadmap as it is (add ~20 goal samples if < 5 vertices reach the goal)
    round k ≥ 1: add 60 samples in a box around start and goal (padded 20 · 2^(k-1) m) + 20 near goal
    join new vertices to their 10 cheapest of 120 nearest WITHOUT collision checks (lazy)
    repeat: Dijkstra → best goal → check only that route's unchecked edges
            all free → return route;   else switch the colliding edges off and search again
```

## Truck: goal selection (`policies.py`, `triage.py`)

```
when parked and the plan is used up, its cell is lost, or (idle) any cell changed state:
    times ← travel time from here to every stop (S1: roadmap Dijkstra; S2 / T2e: straight line)
    plan  ← POLICY(times, cell states, deadlines)
    drive the leg to plan.stop; serve plan.cells in order, 5 s each, skipping lost ones

T0: nearest burning cell by travel time
T1: oldest burning cell with  now + travel + 5 s ≤ burnout
T2 (rollout):
    savable ← burning cells reachable before burnout
    candidates ← 12 stops with the most savable cells per second, + T0's and T1's stops, + here
    for each candidate s:
        clock ← now + travel(s); serve s's cells oldest first
        until the 1200 s horizon: go to the oldest cell still savable under straight-line
            estimates, serve everything in reach there; write off cells that can't be made
        score(s) ← (cells saved, − driving time)
    return the best s and its cells
```
