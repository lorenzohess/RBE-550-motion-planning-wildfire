"""Report figures from runs/<exp>/results.csv (static PNGs for the PDF report).

Colours: categorical slots in fixed order from the reference palette of the dataviz method (blue,
orange, aqua, yellow); truck = slot 1, arsonist = slot 2 everywhere. Magnitude uses one hue light
to dark; the margin uses orange (arsonist) - gray - blue (truck).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TRUCK, ARSONIST = SLOTS[0], SLOTS[1]
POLICY_COLOUR = {"T0": SLOTS[0], "T1": SLOTS[1], "T2": SLOTS[2], "T2e": SLOTS[3]}
POLICY_NAME = {"T0": "T0 nearest", "T1": "T1 oldest savable", "T2": "T2 rollout",
               "T2e": "T2e rollout, straight-line times"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("margin", [ARSONIST, "#ecebe7", TRUCK])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ee", TRUCK, "#123f7a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v) if v not in ("", "None") else None
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}")


# --- E1 ------------------------------------------------------------------------------------------

def e1_heatmap(rows, out: Path):
    burns = sorted({r["burn"] for r in rows})
    speeds = sorted({r["arsonist_speed"] for r in rows})
    win = np.full((len(speeds), len(burns)), np.nan)
    margin = np.full_like(win, np.nan)
    for i, v in enumerate(speeds):
        for j, b in enumerate(burns):
            sel = [r for r in rows if r["burn"] == b and r["arsonist_speed"] == v]
            if sel:
                win[i, j] = np.mean([r["winner"] == "truck" for r in sel])
                margin[i, j] = np.mean([r["margin"] for r in sel])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    lim = np.nanmax(np.abs(margin))
    for ax, data, cmap, title, fmt, vmin, vmax in (
            (axes[0], win, SEQUENTIAL, "Truck win rate (T0 vs W0)", "{:.0%}", 0, 1),
            (axes[1], margin, DIVERGING, "Mean margin, truck minus arsonist", "{:+.0f}", -lim, lim)):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(burns)), [f"{b / 60:g}" for b in burns])
        ax.set_yticks(range(len(speeds)), [f"{v:g}" for v in speeds])
        ax.set_xlabel("burnout time T_burn (min)")
        ax.set_ylabel("arsonist speed (m/s)")
        ax.set_title(title, loc="left")
        ax.grid(False)
        for i in range(len(speeds)):
            for j in range(len(burns)):
                if np.isfinite(data[i, j]):
                    v = (data[i, j] - vmin) / (vmax - vmin)
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if v > 0.7 or (cmap is DIVERGING and v < 0.2) else INK)
        fig.colorbar(im, ax=ax, shrink=0.85)
    n = len(rows) // max(1, len(burns) * len(speeds))
    fig.suptitle(f"E1 calibration: baseline vs baseline, a-priori PRM, {n} seeds per cell",
                 x=0.01, y=1.04, ha="left", fontsize=11)
    _save(fig, out / "e1_balance.png")


# --- E2 ------------------------------------------------------------------------------------------

def trophy(ax, label: str, colour="#b08d2a"):
    """A plain cup (bowl, stem, base) in an inset at the top right of `ax`."""
    t = ax.inset_axes([0.80, 0.70, 0.17, 0.28])
    t.set_xlim(0, 1)
    t.set_ylim(0, 1.25)
    t.axis("off")
    t.add_patch(Polygon([(0.2, 1.0), (0.8, 1.0), (0.62, 0.55), (0.38, 0.55)], closed=True,
                        color=colour))
    t.add_patch(Rectangle((0.46, 0.3), 0.08, 0.25, color=colour))
    t.add_patch(Rectangle((0.3, 0.18), 0.4, 0.12, color=colour))
    t.text(0.5, 0.05, label, ha="center", va="center", fontsize=9, color=INK2)


def e2_scoreboard(rows, out: Path):
    scenarios = sorted({r["scenario"] for r in rows})
    lines = ["# E2 scores", ""]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 3.6))
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scenarios):
        sel = sorted([r for r in rows if r["scenario"] == sc], key=lambda r: r["seed"])
        wins = sum(r["winner"] == "truck" for r in sel)
        losses = sum(r["winner"] == "arsonist" for r in sel)
        champ = "truck" if wins >= 3 else "arsonist" if losses >= 3 else "nobody"
        name = "S1 a-priori PRM" if sc == "S1" else "S2 on-demand PRM"
        lines += [f"## {name}: trophy to the {champ} ({wins} truck wins of {len(sel)})", "",
                  "| seed | truck | arsonist | winner | extinguished | burned | relights | "
                  "truck CPU (s) | arsonist CPU (s) |", "|---|---|---|---|---|---|---|---|---|"]
        for r in sel:
            lines.append(f"| {r['seed']:.0f} | {r['score_truck']:.0f} | {r['score_arsonist']:.0f} | "
                         f"{r['winner']} | {r['extinguishes']:.0f} | {r['burnouts']:.0f} | "
                         f"{r['relights']:.0f} | {r['cpu_truck']:.1f} | {r['cpu_arsonist']:.2f} |")
        lines.append("")
        x = np.arange(len(sel))
        ax.bar(x - 0.2, [r["score_truck"] for r in sel], 0.38, color=TRUCK, label="truck")
        ax.bar(x + 0.2, [r["score_arsonist"] for r in sel], 0.38, color=ARSONIST, label="arsonist")
        ax.set_xticks(x, [f"seed {r['seed']:.0f}" for r in sel])
        ax.set_ylabel("points")
        ax.set_title(f"{name}: truck wins {wins} of {len(sel)}", loc="left")
        ax.grid(axis="x", visible=False)
        top = max(max(r["score_truck"], r["score_arsonist"]) for r in sel)
        ax.set_ylim(0, top * 1.45)
        if champ != "nobody":
            trophy(ax, champ)
        ax.legend(loc="upper left", ncols=2)
    _save(fig, out / "e2_scores.png")
    (out / "e2_scores.md").write_text("\n".join(lines) + "\n")
    print(f"table: {out / 'e2_scores.md'}")


CPU_GROUPS = ["roadmap build", "parking spots", "local planner + collision", "graph search",
              "goal selection", "A* search"]


def cpu_group(player: str, category: str) -> str:
    """Report bucket for a cpu.py category (timers nest, so the buckets add up to the total)."""
    if player == "arsonist":
        return "A* search" if category == "search" else "goal selection"
    if category == "build_spots":
        return "parking spots"
    if category.startswith("build_"):
        return "roadmap build"
    if category in ("local_planner", "collision", "sampling"):
        return "local planner + collision"
    if category in ("search", "oracle", "leg"):
        return "graph search"
    return "goal selection"


def cpu_bars(rows, out: Path):
    scenarios = sorted({r["scenario"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels, stacks = [], []
    for sc in scenarios:
        sel = [r for r in rows if r["scenario"] == sc]
        for player in ("truck", "arsonist"):
            prefix = f"cpu_{player}_"
            parts: dict[str, float] = {}
            for k in sel[0]:
                if k.startswith(prefix) and k != "cpu_truck_build":
                    g = cpu_group(player, k[len(prefix):])
                    parts[g] = parts.get(g, 0.0) + sum(r[k] or 0.0 for r in sel)
            labels.append(f"{sc}\n{player}")
            stacks.append(parts)
    colours = dict(zip(CPU_GROUPS, SLOTS))
    bottom = np.zeros(len(labels))
    for n in CPU_GROUPS:
        vals = np.array([st.get(n, 0.0) for st in stacks])
        if not vals.any():
            continue
        ax.bar(range(len(labels)), vals, 0.6, bottom=bottom, color=colours[n], label=n,
               edgecolor=SURFACE, linewidth=1.5)
        bottom += vals
    for i, b in enumerate(bottom):
        ax.text(i, b, f"{b:.1f} s", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, bottom.max() * 1.12)
    n = len({r["seed"] for r in rows})
    ax.set_ylabel(f"CPU time summed over {n} runs (s)")
    ax.set_title("CPU time per player and scenario (time.process_time)", loc="left", pad=12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    _save(fig, out / "e2_cpu.png")


# --- E3 ------------------------------------------------------------------------------------------

def e3_curves(rows, out: Path):
    policies = [p for p in POLICY_NAME if any(r["truck"] == p for r in rows)]
    lam = np.array([r["lam"] for r in rows])
    edges = np.quantile(lam, np.linspace(0, 1, 7))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, key, ylabel in ((axes[0], "save", "extinguished / ignited"),
                            (axes[1], "margin", "margin, truck minus arsonist")):
        if key == "save":
            ceil = [(r["lam"], r["free_travel_ceiling"] / max(1, r["ignitions"])) for r in rows]
            xs, ys = _binned(ceil, edges)
            ax.plot(xs, ys, color=MUTED, lw=2, ls="--", label="free-travel ceiling")
        ends = []
        for p in policies:
            pts = [(r["lam"], r["extinguishes"] / max(1, r["ignitions"]) if key == "save" else r["margin"])
                   for r in rows if r["truck"] == p]
            xs, ys = _binned(pts, edges)
            ax.plot(xs, ys, color=POLICY_COLOUR[p], lw=2, marker="o", ms=5, label=POLICY_NAME[p])
            ends.append([ys[-1], xs[-1], p])
        _end_labels(ax, ends)
        if key == "margin":
            ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("stage-2 load λ = map's free-travel save-all burnout / T_burn")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Share of ignitions put out", loc="left")
    axes[1].set_title("Score margin", loc="left")
    axes[0].legend(loc="lower left", fontsize=8)
    n = len({r["seed"] for r in rows})
    nb = len({r["burn"] for r in rows})
    fig.suptitle(f"E3: truck policies vs W1 on the a-priori PRM, {n} seeds x {nb} burnout times "
                 f"(means in λ sextiles)", x=0.01, y=1.04, ha="left", fontsize=11)
    _save(fig, out / "e3_policies.png")

    burns = sorted({r["burn"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(policies)
    for k, p in enumerate(policies):
        vals = [np.mean([r["winner"] == "truck" for r in rows if r["truck"] == p and r["burn"] == b])
                for b in burns]
        x = np.arange(len(burns)) + (k - (len(policies) - 1) / 2) * w
        ax.bar(x, vals, w * 0.92, color=POLICY_COLOUR[p], label=POLICY_NAME[p])
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(burns)), [f"T_burn {b / 60:g} min" for b in burns])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("truck win rate")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=2, fontsize=8, loc="upper left")
    ax.set_title("Truck win rate by policy", loc="left")
    _save(fig, out / "e3_winrate.png")

    lines = ["# E3 summary", "", "| T_burn (min) | policy | win rate | mean margin | put out / ignited | "
             "burned | time per cell (s) | CPU per decision (ms) |", "|---|---|---|---|---|---|---|---|"]
    for b in burns:
        for p in policies:
            sel = [r for r in rows if r["truck"] == p and r["burn"] == b]
            dec = sum(r["truck_decisions"] for r in sel)
            gs = sum((r.get("cpu_truck_goal_selection") or 0) + (r.get("cpu_truck_oracle") or 0) for r in sel)
            lines.append(f"| {b / 60:g} | {p} | {np.mean([r['winner'] == 'truck' for r in sel]):.0%} | "
                         f"{np.mean([r['margin'] for r in sel]):+.0f} | "
                         f"{np.mean([r['extinguishes'] / max(1, r['ignitions']) for r in sel]):.2f} | "
                         f"{np.mean([r['burnouts'] for r in sel]):.0f} | "
                         f"{np.mean([r['time_per_extinguish'] for r in sel]):.1f} | "
                         f"{1000 * gs / max(1, dec):.1f} |")
    (out / "e3_summary.md").write_text("\n".join(lines) + "\n")
    print(f"table: {out / 'e3_summary.md'}")


def _end_labels(ax, ends):
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, name in ends:
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", color=INK2,
                    fontsize=9, va="center")


def _binned(points, edges):
    pts = np.array(points, float)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
        if m.any():
            xs.append(pts[m, 0].mean())
            ys.append(pts[m, 1].mean())
    return np.array(xs), np.array(ys)


# --- one map, one run ---------------------------------------------------------------------------

def roadmap_figure(seed: int, out: Path):
    from . import reeds_shepp as rs
    from .collision import Checker
    from .prm import Roadmap
    from .truck import TruckSpec
    from .world import CELL_M, SIZE_M, build_world, rng_for
    world = build_world(seed)
    spec = TruckSpec()
    rm = Roadmap(Checker(world, spec), spec)
    rm.build(2000, rng_for(seed, "prm"))
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for e in rm.edges:
        x, y, _ = rs.sample(rm.poses[e.u], e.path, spec.rho, 1.0)
        ax.plot(x, y, color="#9cc0ee", lw=0.35, alpha=0.8)
    for cx, cy in world.cells:
        ax.add_patch(Rectangle((cx * CELL_M, cy * CELL_M), CELL_M, CELL_M, color="#5c8040"))
    ax.plot(rm.poses[:, 0], rm.poses[:, 1], ".", color=TRUCK, ms=1.2)
    ax.set_xlim(0, SIZE_M)
    ax.set_ylim(0, SIZE_M)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Seed {seed}: {world.n} obstacle cells, roadmap {rm.n} vertices, "
                 f"{len(rm.edges)} Reeds–Shepp edges", loc="left", fontsize=10)
    _save(fig, out / f"roadmap_seed{seed}.png")


def series_figure(series_path: Path, out: Path, seed: int = 0, scenario: str = "S1"):
    with series_path.open() as f:
        runs = [json.loads(line) for line in f]
    run = next((r for r in runs if r["seed"] == seed and r["scenario"] == scenario), None)
    if run is None:
        return
    s = np.array([[np.nan if v is None else v for v in row] for row in run["series"]], float)
    t = s[:, 0] / 60
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for col, name, colour in ((1, "intact", "#5c8040"), (2, "burning", SLOTS[1]),
                              (3, "extinguished", SLOTS[0]), (4, "burned", "#3a3936")):
        axes[0].plot(t, s[:, col], color=colour, lw=2, label=name)
        axes[0].annotate(name, (t[-1], s[-1, col]), xytext=(4, 0), textcoords="offset points",
                         fontsize=8, color=INK2, va="center")
    axes[0].set_ylabel("cells")
    axes[0].set_title(f"Seed {seed}, {scenario}, {run['truck']} vs {run['arsonist']}: cell states",
                      loc="left")
    axes[0].legend(ncols=4, fontsize=8, loc="upper center")
    m = np.isfinite(s[:, 5])
    axes[1].plot(t[m], s[m, 5], color=TRUCK, lw=2, label="truck")
    axes[1].plot(t[m], s[m, 6], color=ARSONIST, lw=2, label="arsonist")
    axes[1].set_ylabel("points")
    axes[1].set_xlabel("sim time (min)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_title("Scores", loc="left")
    _save(fig, out / f"series_seed{seed}_{scenario}.png")


def make_all(runs: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    if (runs / "e1" / "results.csv").exists():
        e1_heatmap(load(runs / "e1" / "results.csv"), out)
    if (runs / "e2" / "results.csv").exists():
        rows = load(runs / "e2" / "results.csv")
        e2_scoreboard(rows, out)
        cpu_bars(rows, out)
        series_figure(runs / "e2" / "series.jsonl", out, 0, "S1")
    if (runs / "e3" / "results.csv").exists():
        e3_curves(load(runs / "e3" / "results.csv"), out)
    roadmap_figure(0, out)
