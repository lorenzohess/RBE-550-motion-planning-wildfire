# Wildfire

RBE-550 assignment 5: a fire truck (Ackermann, PRM over Reeds–Shepp curves) against an arsonist
(grid A*) on a 250 × 250 m field of tetromino brush. 

**Status:** version 1 runs end to end: both PRM variants, every policy, window, mp4, batch runs
and figures; 32 tests pass.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`ffmpeg` must be on `PATH` for `--record`.

## Running

```bash
.venv/bin/python main.py run --seed 0                         # window
.venv/bin/python main.py run --seed 0 --scenario S2           # on-demand PRM
.venv/bin/python main.py run --seed 0 --truck T0 --arsonist W0 --burn 900
.venv/bin/python main.py run --seed 0 --headless              # summary only
.venv/bin/python main.py run --seed 0 --headless --record runs/seed0.mp4 --record-every 2
.venv/bin/python main.py run --seed 0 --headless --screenshot-at 60 600 1300 --show roadmap
.venv/bin/python main.py batch e1     # calibration: T_burn x arsonist speed, baseline vs baseline
.venv/bin/python main.py batch e2     # the assignment's scenarios: S1 vs S2, 5 seeds each
.venv/bin/python main.py batch e3     # truck policies, 20 seeds x 3 burnout times
.venv/bin/python main.py plots        # figures and tables into runs/figures/
.venv/bin/python -m pytest -q
```

Options (all of `run`): `--seed`, `--scenario S1|S2`, `--truck T0|T1|T2|T2e`,
`--arsonist W0|W1`, `--burn` (s), `--arsonist-speed` (m/s), `--duration` (s), `--rho`,
`--prm-n`, `--prm-k`, `--prm-candidates`, `--a-max` (m/s², `inf` for instant), `--gear-penalty`
(s), `--catharsis 0|1`.

Keys: `space` pause, `+` / `-` speed, `r` roadmap, `g` spread links from burning cells, `x` cells
the truck can't reach, `s` screenshot, `esc` quit.

## What you see

- Cells: green intact, orange burning (darkening towards burnout, a white bar for the time left,
  a yellow outline in the first 10 s before it spreads), blue extinguished, black burned.
- The truck: red rectangle, yellow front, its 10 m reach ring, the current leg (blue forward,
  orange reverse), the cells it plans to put out (white outlines) and the 5 s dwell as an arc.
- The arsonist: magenta dot, its A* path and its target cell.
- The HUD: time, scores, cell counts, ignitions by cause, CPU per player, the truck's legs and
  roadmap size.

## The game

| | |
|---|---|
| Field | 50 × 50 cells of 5 m, 10 % tetromino coverage, a new seeded field every run; every cell is its own obstacle |
| Fire | the arsonist lights intact or extinguished cells; after 10 s a burning cell lights every intact cell within 30 m; it burns out after T_burn (5 min by default, from E1); burned is final |
| Truck | stopped 5 s with its centre within 10 m of a burning cell puts it out, one cell at a time |
| Arsonist | 5 m 4-connected grid at 1 m/s, lights a 4-adjacent cell, at most one per node it visits |
| Scoring | every state change: arsonist +1 per ignition and +1 per burnout, truck +2 per extinguish |
| Run | 3600 s sim time at dt = 0.1 s; 5 runs per scenario; trophy to whoever wins 3 |

## How it works

- **Arsonist (`astar.py`, `arsonist.py`)**: A* on the free cells, heuristic 5 m × Manhattan
  distance to the nearest goal node (admissible and consistent). W0 heads for the nearest cell it
  can light; W1 for the most intact cells its spread would cascade into per second of walking,
  discounted near the truck, with A* steps costing more near the truck.
- **Truck motion (`reeds_shepp.py`, `collision.py`, `prm.py`)**: a PRM over poses (x, y, ψ). The
  local planner is the Reeds–Shepp word that is fastest to drive (each gear change costs its stop
  and a 2 s shift). Each vertex keeps its 10 cheapest edges among its 120 nearest, so edges line
  up with headings and routes rarely reverse. Collisions: distance-field prefilter, then exact
  separating axes, every 0.25 m with a 0.2 m margin. Search: Dijkstra over (vertex, gear).
- **Scenarios**: S1 builds 2000 vertices before t = 0 (≈ 3 s); S2 grows a lazy roadmap per query
  and checks only the edges a path uses.
- **Truck goal selection (`policies.py`, `triage.py`)**: T0 nearest, T1 oldest savable, T2 a
  rollout: try a dozen candidate stops, simulate oldest-savable-first from each with straight-line
  estimates, pick the one that saves the most cells, ties to the least driving. T2e is T2 with
  straight-line travel times instead of the roadmap's.
- **Catharsis**: once nothing burns and the arsonist has nothing left to light, the truck is told
  where the arsonist is and chases it; within 5 m ends the run.

## Third-party code and AI use

- numpy, scipy (cKDTree, sparse Dijkstra, distance transform), pygame, Pillow, matplotlib.
- Reeds–Shepp formulas transcribed from the Reeds & Shepp (1990) paper as laid out in OMPL's
  `ReedsSheppStateSpace.cpp` (BSD), with its two typo corrections.
- Tetromino placement, the SAT collision test, the recorder and the Pillow text renderer come from
  assignments 2 and 3 (Flatland, Valet).
- Code written with Claude (Anthropic); the truck's decision rule (T2) is the author's design.
