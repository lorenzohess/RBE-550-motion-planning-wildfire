"""Batch runs (notes, "Experiments"). Every run is one process-pool task; every scenario uses the
same seeds, so runs pair up across scenarios and policies.

- E1 calibration: baseline truck T0 vs baseline arsonist W0 on the a-priori PRM, over
  T_burn x arsonist speed, 5 seeds each, full hour. Picks the defaults.
- E2 required scenarios: S1 a-priori vs S2 on-demand PRM, T2 vs W1, 5 seeds (the assignment's
  "5 runs"), with event logs.
- E3 novelty: truck policies T0, T1, T2, T2e vs W1 on S1, 20 seeds, at four burnout times.

Outputs: runs/<exp>/results.csv (one row per run), runs/<exp>/series.jsonl (state counts over time),
runs/e2/events/*.csv (every state change, enough to rebuild any score).
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from .sim import Config

E1_BURNS = (300.0, 450.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0)
E1_SPEEDS = (0.5, 1.0, 2.0)
E3_BURNS = (300.0, 600.0, 1200.0, 1800.0)
E3_POLICIES = ("T0", "T1", "T2", "T2e")
DEFAULT_BURN = 300.0
KEY = ("seed", "scenario", "truck", "arsonist", "burn", "arsonist_speed")
DEFAULT_SPEED = 1.0


def configs(name: str, seeds: int | None, burn: float | None, speed: float | None) -> list[Config]:
    burn = burn or DEFAULT_BURN
    speed = speed or DEFAULT_SPEED
    if name == "e1":
        return [Config(seed=s, truck="T0", arsonist="W0", burn=b, arsonist_speed=v)
                for b in E1_BURNS for v in E1_SPEEDS for s in range(seeds or 5)]
    if name == "e2":
        return [Config(seed=s, scenario=sc, truck="T2", arsonist="W1", burn=burn, arsonist_speed=speed)
                for sc in ("S1", "S2") for s in range(seeds or 5)]
    if name == "e3":
        return [Config(seed=s, truck=p, arsonist="W1", burn=b, arsonist_speed=speed)
                for b in E3_BURNS for p in E3_POLICIES for s in range(seeds or 20)]
    raise ValueError(name)


def _one(task):
    cfg_dict, events_dir = task
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from .sim import Game
    t = time.perf_counter()
    game = Game(Config(**cfg_dict))
    r = game.run()
    if events_dir:
        path = Path(events_dir) / f"seed{cfg_dict['seed']}-{cfg_dict['scenario']}-{cfg_dict['truck']}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "cell", "cx", "cy", "event", "cause", "relight"])
            for e in game.fire.events:
                cx, cy = game.world.cells[e.cell]
                w.writerow([f"{e.tick * 0.1:.1f}", e.cell, cx, cy, e.kind, e.cause, int(e.relight)])
    r["wall_total"] = time.perf_counter() - t
    return r


def run(name: str, workers: int = 8, seeds: int | None = None, out: Path = Path("runs"),
        burn: float | None = None, speed: float | None = None) -> Path:
    cfgs = configs(name, seeds, burn, speed)
    out_dir = out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    events_dir = None
    if name == "e2":
        events_dir = out_dir / "events"
        events_dir.mkdir(exist_ok=True)
    # Resume: keep rows already in results.csv for the same configuration and seed.
    key = lambda d: tuple(str(d[k]) for k in KEY)  # noqa: E731
    rows = []
    done = set()
    if (out_dir / "results.csv").exists():
        with (out_dir / "results.csv").open() as f:
            old = list(csv.DictReader(f))
        wanted = {key({k: str(v) for k, v in asdict(c).items()}) for c in cfgs}
        rows = [r for r in old if key(r) in wanted]
        done = {key(r) for r in rows}
    cfgs = [c for c in cfgs if key({k: str(v) for k, v in asdict(c).items()}) not in done]
    tasks = [(asdict(c), str(events_dir) if events_dir else None) for c in cfgs]
    t0 = time.perf_counter()
    print(f"{name}: {len(tasks)} runs on {workers} workers ({len(rows)} already done) -> {out_dir}")
    with get_context("spawn").Pool(workers, maxtasksperchild=4) as pool, \
            (out_dir / "series.jsonl").open("a") as series:
        for i, r in enumerate(pool.imap_unordered(_one, tasks), 1):
            series.write(json.dumps({k: r[k] for k in ("seed", "scenario", "truck", "arsonist", "burn",
                                                        "arsonist_speed")} | {"series": r.pop("series")})
                         + "\n")
            rows.append(r)
            print(f"  [{i}/{len(tasks)}] seed {r['seed']} {r['scenario']} {r['truck']} vs {r['arsonist']} "
                  f"burn {r['burn']:.0f} v {r['arsonist_speed']:g}: {r['winner']} "
                  f"{r['score_truck']}-{r['score_arsonist']} ({r['wall_total']:.0f} s)", flush=True)
    keys = sorted({k for r in rows for k in r})
    first = [k for k in ("seed", "scenario", "truck", "arsonist", "burn", "arsonist_speed", "winner",
                         "score_truck", "score_arsonist", "margin") if k in keys]
    keys = first + [k for k in keys if k not in first]
    rows.sort(key=lambda r: (str(r["scenario"]), str(r["truck"]), float(r["burn"]),
                             float(r["arsonist_speed"]), int(r["seed"])))
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: done in {time.perf_counter() - t0:.0f} s -> {out_dir / 'results.csv'}")
    return out_dir / "results.csv"
