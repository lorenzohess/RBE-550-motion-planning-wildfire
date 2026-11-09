"""Command line: `python main.py run|batch|plots ...`. See `python main.py <command> --help`."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .sim import Config

SUMMARY = ("winner", "score_truck", "score_arsonist", "margin", "ignitions", "arsonist_ignitions",
           "relights", "spread_ignitions", "extinguishes", "burnouts", "final_burning",
           "extinguishable", "cells", "extinguish_rate", "free_travel_ceiling", "lam",
           "time_per_extinguish", "truck_legs", "truck_distance", "truck_gear_changes",
           "truck_failed_legs", "caught", "end_reason", "cpu_truck", "cpu_truck_build",
           "cpu_arsonist", "wall_build", "wall_run")


def add_config_args(p: argparse.ArgumentParser) -> None:
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"),
                           default=f.default, metavar="0|1")
        else:
            kind = {"int": int, "float": float, "str": str}.get(str(f.type), type(f.default))
            p.add_argument(flag, type=kind, default=f.default)


def config_from(args) -> Config:
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})


def cmd_run(args) -> int:
    from .sim import Game
    cfg = config_from(args)
    game = Game(cfg)
    print(f"seed {cfg.seed} {cfg.scenario}: {game.world.n} cells, roadmap {game.roadmap.n} vertices "
          f"{len(game.roadmap.edges)} edges, built in {game.build_wall:.1f} s")
    if args.headless and not args.record and args.screenshot_at is None:
        r = game.run()
    else:
        from .view import View
        view = View(game, headless=args.headless, record=args.record,
                    record_every=args.record_every, record_from=args.record_from,
                    record_to=args.record_to)
        if args.screenshot_at is not None:
            shots = sorted(args.screenshot_at)
            for i, t in enumerate(shots):
                while not game.done and game.tick * 0.1 < t:
                    game.step()
                for flag in args.show:
                    setattr(view, f"show_{flag}", True)
                view.render()
                out = Path(args.screenshot_dir) / f"seed{cfg.seed}-{cfg.scenario}-{cfg.truck}-t{int(t)}.png"
                view.screenshot(out)
            view.close()
            r = game.results() if game.done else None
        elif args.headless:
            view.run_record()
            view.close()
            r = game.results()
        else:
            for flag in args.show:
                setattr(view, f"show_{flag}", True)
            view.run_window()
            view.close()
            r = game.results()
    if r is not None:
        for k in SUMMARY:
            v = r.get(k)
            print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(r, indent=1, default=float))
    return 0


def cmd_batch(args) -> int:
    from . import experiments
    experiments.run(args.experiment, workers=args.workers, seeds=args.seeds, out=Path(args.out),
                    burn=args.burn, speed=args.arsonist_speed)
    return 0


def cmd_plots(args) -> int:
    from . import plots
    plots.make_all(Path(args.runs), Path(args.out))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="wildfire", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="one run: window, headless summary, recording or screenshots")
    add_config_args(r)
    r.add_argument("--headless", action="store_true", help="no window")
    r.add_argument("--record", help="write an mp4 here")
    r.add_argument("--record-every", type=float, default=1.0, help="sim seconds per video frame")
    r.add_argument("--record-from", type=float, default=0.0, help="sim time to start recording")
    r.add_argument("--record-to", type=float, default=None, help="sim time to stop recording")
    r.add_argument("--screenshot-at", type=float, nargs="+", help="headless PNGs at these sim times")
    r.add_argument("--screenshot-dir", default="runs/screenshots")
    r.add_argument("--show", nargs="*", default=[], choices=["roadmap", "spread", "unreachable"],
                   help="overlays to start with")
    r.add_argument("--json", help="write the full result dict here")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="experiments E1, E2, E3 (see experiments.py)")
    b.add_argument("experiment", choices=["e1", "e2", "e3"])
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--seeds", type=int, default=None, help="number of seeds (default per experiment)")
    b.add_argument("--burn", type=float, default=None, help="T_burn for E2 (s)")
    b.add_argument("--arsonist-speed", type=float, default=None, help="arsonist speed for E2 / E3")
    b.add_argument("--out", default="runs")
    b.set_defaults(func=cmd_batch)

    q = sub.add_parser("plots", help="figures from runs/")
    q.add_argument("--runs", default="runs")
    q.add_argument("--out", default="runs/figures")
    q.set_defaults(func=cmd_plots)

    args = p.parse_args(argv)
    return args.func(args)
