"""The pygame window: the field, the fire, both players and a HUD.

Keys: space pause, + / - speed, r roadmap, g spread links, x cells the truck can't reach,
s screenshot, esc or q quit. Recording (--record) writes frames to an mp4 at a fixed sim-time step.
Text is drawn with Pillow (this pygame build has no SDL_ttf), as in Valet.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pygame

from . import reeds_shepp as rs
from .fire import BURNED, BURNING, EXTINGUISHED, INTACT, score
from .prm import BLOCKED
from .recorder import Recorder
from .sim import CATCH, DT, Game
from .world import CELL_M, REACH, SIZE_M

PX_PER_M = 3
MARGIN = 12
PANEL = 330
FPS = 30
RATES = (0.25, 0.5, 1, 2, 5, 10, 20, 60, 120)     # sim seconds per frame

BACKGROUND = (236, 234, 228)
FIELD = (250, 249, 245)
GRID = (226, 224, 216)
INTACT_C = (92, 128, 64)
BURNING_C = (240, 120, 20)
LATE_C = (150, 20, 20)
EXT_C = (80, 130, 190)
BURNED_C = (48, 46, 44)
TRUCK_C = (200, 30, 40)
FRONT = (255, 210, 0)
FORWARD = (30, 110, 230)
REVERSE = (240, 140, 20)
ARSONIST_C = (190, 40, 190)
ROADMAP_C = (200, 200, 205)
TEXT = (32, 32, 36)
DIM = (110, 110, 118)


class Font:
    """Text rendered with Pillow's bundled FreeType font, cached (from Valet's view.py)."""

    def __init__(self, size: int):
        from PIL import ImageFont
        self.font = ImageFont.load_default(size=size)
        self._cache: dict[tuple[str, tuple], pygame.Surface] = {}

    def render(self, text: str, colour) -> pygame.Surface:
        key = (text, tuple(colour))
        if key not in self._cache:
            from PIL import Image, ImageDraw
            left, top, right, bottom = self.font.getbbox(text)
            image = Image.new("RGBA", (max(1, right), max(1, bottom)), (0, 0, 0, 0))
            ImageDraw.Draw(image).text((0, 0), text, font=self.font, fill=tuple(colour)[:3] + (255,))
            self._cache[key] = pygame.image.frombytes(image.tobytes(), image.size, "RGBA")
            if len(self._cache) > 4000:
                self._cache.clear()
        return self._cache[key]


def lerp(a, b, f):
    return tuple(int(x + (y - x) * f) for x, y in zip(a, b))


class View:
    def __init__(self, game: Game, headless: bool = False, record: str | None = None,
                 record_every: float = 1.0, record_from: float = 0.0, record_to: float | None = None,
                 rate_index: int = 4, screenshot_dir: Path = Path("runs/screenshots")):
        self.game = game
        self.headless = headless
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        cfg = game.cfg
        pygame.display.set_caption(f"Wildfire: seed {cfg.seed}, {cfg.scenario}, {cfg.truck} vs {cfg.arsonist}")
        self.s = PX_PER_M
        self.map_px = int(SIZE_M * self.s)
        w = self.map_px + 2 * MARGIN + PANEL
        h = self.map_px + 2 * MARGIN
        self.size = (w + w % 2, h + h % 2)
        self.screen = pygame.display.set_mode(self.size)
        self.font = Font(14)
        self.small = Font(12)
        self.big = Font(26)
        self.clock = pygame.time.Clock()
        self.rate_index = rate_index
        self.paused = False
        self.show_roadmap = False
        self.show_spread = False
        self.show_unreachable = False
        self.recorder = Recorder(record, self.size, FPS) if record else None
        self.record_every = record_every
        self.record_from = record_from
        self.record_to = record_to
        self.screenshot_dir = screenshot_dir
        self.screenshots = 0
        self._static = self._draw_static()
        self._roadmap_layer = None
        self._roadmap_edges = -1
        self._leg_cache = (None, None)
        self._unreachable = None

    # --- geometry -----------------------------------------------------------------------------

    def px(self, x, y):
        return (MARGIN + x * self.s, MARGIN + (SIZE_M - y) * self.s)

    def _draw_static(self) -> pygame.Surface:
        surf = pygame.Surface(self.size)
        surf.fill(BACKGROUND)
        pygame.draw.rect(surf, FIELD, (MARGIN, MARGIN, self.map_px, self.map_px))
        for k in range(0, int(SIZE_M) + 1, 25):
            pygame.draw.line(surf, GRID, self.px(k, 0), self.px(k, SIZE_M))
            pygame.draw.line(surf, GRID, self.px(0, k), self.px(SIZE_M, k))
        return surf

    def _roadmap(self) -> pygame.Surface:
        rm = self.game.roadmap
        if self._roadmap_layer is None or self._roadmap_edges != len(rm.edges):
            layer = pygame.Surface(self.size, pygame.SRCALPHA)
            for e in rm.edges:
                if e.status == BLOCKED:
                    continue
                x, y, _ = rs.sample(rm.poses[e.u], e.path, rm.spec.rho, 2.0)
                pts = [self.px(a, b) for a, b in zip(x, y)]
                if len(pts) > 1:
                    pygame.draw.lines(layer, ROADMAP_C + (150,), False, pts, 1)
            for x, y, _ in rm.poses:
                layer.set_at(tuple(int(v) for v in self.px(x, y)), (120, 120, 130, 255))
            self._roadmap_layer = layer
            self._roadmap_edges = len(rm.edges)
        return self._roadmap_layer

    # --- frame --------------------------------------------------------------------------------

    def render(self) -> None:
        g = self.game
        surf = self.screen
        surf.blit(self._static, (0, 0))
        if self.show_roadmap:
            surf.blit(self._roadmap(), (0, 0))
        self._cells(surf)
        if self.show_spread:
            self._spread(surf)
        if self.show_unreachable:
            self._mark_unreachable(surf)
        self._leg(surf)
        self._arsonist(surf)
        self._truck(surf)
        self._panel(surf)
        if g.done:
            self._banner(surf)

    def _cells(self, surf) -> None:
        g = self.game
        fire = g.fire
        tick = g.tick
        side = CELL_M * self.s
        for i, (cx, cy) in enumerate(g.world.cells):
            st = fire.state[i]
            x, y = self.px(cx * CELL_M, (cy + 1) * CELL_M)
            rect = pygame.Rect(int(x), int(y), math.ceil(side), math.ceil(side))
            if st == BURNING:
                age = (tick - fire.ignited[i]) / fire.burn_ticks
                pygame.draw.rect(surf, lerp(BURNING_C, LATE_C, min(1.0, age)), rect)
                if tick - fire.ignited[i] < fire.spread_ticks:          # about to spread
                    pygame.draw.rect(surf, FRONT, rect, 2)
                bar = int(side * (1.0 - min(1.0, age)))
                pygame.draw.line(surf, (255, 255, 255), (rect.x, rect.bottom - 2),
                                 (rect.x + bar, rect.bottom - 2), 2)
            else:
                colour = {INTACT: INTACT_C, EXTINGUISHED: EXT_C, BURNED: BURNED_C}[int(st)]
                pygame.draw.rect(surf, colour, rect)

    def _spread(self, surf) -> None:
        g = self.game
        for i in np.flatnonzero(g.fire.state == BURNING):
            a = self.px(*g.world.centres[i])
            for j in g.world.spread[i]:
                if g.fire.state[j] == INTACT:
                    pygame.draw.line(surf, (250, 170, 60), a, self.px(*g.world.centres[j]), 1)

    def _mark_unreachable(self, surf) -> None:
        g = self.game
        if self._unreachable is None and g.component is not None:
            covered = (g.world.distances(g.roadmap.centres()[g.component]) <= REACH).any(axis=0)
            self._unreachable = np.flatnonzero(~covered)
        for i in self._unreachable if self._unreachable is not None else []:
            cx, cy = g.world.centres[i]
            a, b = self.px(cx - 2, cy - 2), self.px(cx + 2, cy + 2)
            pygame.draw.line(surf, (255, 255, 255), a, b, 2)
            pygame.draw.line(surf, (255, 255, 255), (a[0], b[1]), (b[0], a[1]), 2)

    def _leg(self, surf) -> None:
        traj = self.game.traj
        if traj is None:
            return
        if self._leg_cache[0] is not traj:
            x, y, gears = traj.polyline(1.0)
            self._leg_cache = (traj, (x, y, gears))
        x, y, gears = self._leg_cache[1]
        for k in range(len(x) - 1):
            colour = FORWARD if gears[k + 1] == 0 else REVERSE
            pygame.draw.line(surf, colour, self.px(x[k], y[k]), self.px(x[k + 1], y[k + 1]), 2)

    def _truck(self, surf) -> None:
        g = self.game
        x, y, psi = g.truck_pose()
        corners = g.spec.body.corners(x, y, psi)
        pts = [self.px(a, b) for a, b in corners]
        pygame.draw.polygon(surf, TRUCK_C, pts)
        pygame.draw.line(surf, FRONT, pts[1], pts[2], 2)
        c = g.truck_centre()
        pygame.draw.circle(surf, TRUCK_C, self.px(*c), int(REACH * self.s), 1)
        if g.dwell is not None:
            cell, start = g.dwell
            cx, cy = g.world.centres[cell]
            r = pygame.Rect(0, 0, int(CELL_M * self.s) + 6, int(CELL_M * self.s) + 6)
            r.center = self.px(cx, cy)
            frac = min(1.0, (g.tick - start) / g.service_ticks)
            pygame.draw.rect(surf, (255, 255, 255), r, 2)
            pygame.draw.arc(surf, EXT_C, r.inflate(8, 8), math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 3)
        for cell in g.queue[:6]:
            cx, cy = g.world.centres[cell]
            r = pygame.Rect(0, 0, int(CELL_M * self.s) + 4, int(CELL_M * self.s) + 4)
            r.center = self.px(cx, cy)
            pygame.draw.rect(surf, (255, 255, 255), r, 1)
        if g.chasing:
            pygame.draw.circle(surf, ARSONIST_C, self.px(*c), int(CATCH * self.s), 1)

    def _arsonist(self, surf) -> None:
        a = self.game.arsonist
        x, y = a.xy()
        pts = [self.px(x, y)] + [self.px((n[0] + 0.5) * CELL_M, (n[1] + 0.5) * CELL_M) for n in a.path]
        for k in range(0, len(pts) - 1, 2):
            pygame.draw.line(surf, ARSONIST_C, pts[k], pts[k + 1], 1)
        if a.target is not None:
            cx, cy = self.game.world.centres[a.target]
            r = pygame.Rect(0, 0, int(CELL_M * self.s) + 4, int(CELL_M * self.s) + 4)
            r.center = self.px(cx, cy)
            pygame.draw.rect(surf, ARSONIST_C, r, 1)
        pygame.draw.circle(surf, ARSONIST_C, self.px(x, y), 6)
        pygame.draw.circle(surf, (255, 255, 255), self.px(x, y), 6, 1)

    def _text(self, surf, text, pos, font=None, colour=TEXT):
        surf.blit((font or self.font).render(text, colour), pos)

    def _panel(self, surf) -> None:
        g, cfg = self.game, self.game.cfg
        x0 = self.map_px + 2 * MARGIN
        y = MARGIN
        t = g.tick * DT
        sc = score(g.fire.events)
        self._text(surf, "Wildfire", (x0, y), self.big)
        y += 36
        self._text(surf, f"seed {cfg.seed}   {cfg.scenario} "
                         f"({'a-priori' if cfg.scenario == 'S1' else 'on-demand'} PRM)", (x0, y))
        y += 18
        self._text(surf, f"truck {cfg.truck}   arsonist {cfg.arsonist} @ {cfg.arsonist_speed:g} m/s",
                   (x0, y))
        y += 18
        self._text(surf, f"burnout {cfg.burn / 60:g} min   spread 30 m after 10 s", (x0, y), colour=DIM)
        y += 26
        state = "paused" if self.paused else f"x{RATES[self.rate_index] * FPS:g}"
        self._text(surf, f"{int(t // 60):02d}:{int(t % 60):02d} / {int(cfg.duration // 60)}:00   {state}",
                   (x0, y), self.big)
        y += 40
        self._text(surf, "Truck", (x0, y), colour=TRUCK_C)
        self._text(surf, "Arsonist", (x0 + 160, y), colour=ARSONIST_C)
        y += 18
        self._text(surf, f"{sc.truck}", (x0, y), self.big, TRUCK_C)
        self._text(surf, f"{sc.arsonist}", (x0 + 160, y), self.big, ARSONIST_C)
        y += 38
        self._text(surf, f"+2 x {sc.extinguishes} put out", (x0, y), self.small, DIM)
        self._text(surf, f"{sc.ignitions} lit + {sc.burnouts} burned", (x0 + 160, y), self.small, DIM)
        y += 26
        counts = g.fire.counts()
        for name, colour in (("intact", INTACT_C), ("burning", BURNING_C),
                             ("extinguished", EXT_C), ("burned", BURNED_C)):
            pygame.draw.rect(surf, colour, (x0, y + 2, 12, 12))
            self._text(surf, f"{name:13s} {counts[name]:4d}", (x0 + 18, y))
            y += 18
        y += 6
        self._text(surf, f"ignitions: arsonist {sc.arsonist_ignitions} (relights {sc.relights}),"
                         f" spread {sc.spread_ignitions}", (x0, y), self.small, DIM)
        y += 22
        ct = sum(g.cpu["truck"].totals.values())
        cb = sum(v for k, v in g.cpu["truck"].totals.items() if k.startswith("build/"))
        ca = sum(g.cpu["arsonist"].totals.values())
        self._text(surf, f"CPU truck {ct:.2f} s (roadmap {cb:.2f} s)", (x0, y))
        y += 18
        self._text(surf, f"CPU arsonist {ca:.3f} s", (x0, y))
        y += 22
        st = g.truck_stats
        self._text(surf, f"truck: {st['legs']} legs, {st['distance'] / 1000:.2f} km, "
                         f"{st['gear_changes']} gear changes", (x0, y), self.small)
        y += 16
        rm = g.roadmap
        self._text(surf, f"roadmap: {rm.n} vertices, {sum(e.status != BLOCKED for e in rm.edges)} edges",
                   (x0, y), self.small)
        y += 16
        status = ("chasing the arsonist" if g.chasing else "driving" if g.traj is not None
                  else "putting out a cell" if g.dwell is not None else "waiting")
        self._text(surf, f"truck is {status}", (x0, y), self.small)
        y += 16
        if g.plan is not None and g.plan.note:
            self._text(surf, f"last plan: {g.plan.note}", (x0, y), self.small, DIM)
        y += 26
        legend = [("forward", FORWARD), ("reverse", REVERSE), ("about to spread", FRONT)]
        for name, colour in legend:
            pygame.draw.line(surf, colour, (x0, y + 7), (x0 + 16, y + 7), 3)
            self._text(surf, name, (x0 + 22, y), self.small)
            y += 16
        y += 8
        keys = "space pause  +/- speed  r roadmap  g spread  x unreachable  s screenshot  q quit"
        words, line = keys.split("  "), ""
        for w in words:
            if len(line) + len(w) > 40:
                self._text(surf, line, (x0, y), self.small, DIM)
                y += 15
                line = ""
            line += w + "  "
        self._text(surf, line, (x0, y), self.small, DIM)

    def _banner(self, surf) -> None:
        g = self.game
        sc = score(g.fire.events)
        text = {"truck": "The truck wins", "arsonist": "The arsonist wins", "draw": "Draw"}[sc.winner]
        if g.caught:
            text += " - and caught the arsonist"
        img = self.big.render(f"{text}  {sc.truck} : {sc.arsonist}", (255, 255, 255))
        box = img.get_rect()
        box.center = (MARGIN + self.map_px // 2, MARGIN + self.map_px // 2)
        bg = box.inflate(30, 20)
        pygame.draw.rect(surf, (30, 30, 34), bg)
        surf.blit(img, box)

    # --- loops --------------------------------------------------------------------------------

    def _advance(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / DT))):
            if self.game.done:
                return
            self.game.step()

    def _pump(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.rate_index = min(self.rate_index + 1, len(RATES) - 1)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.rate_index = max(self.rate_index - 1, 0)
            elif event.key == pygame.K_r:
                self.show_roadmap = not self.show_roadmap
            elif event.key == pygame.K_g:
                self.show_spread = not self.show_spread
            elif event.key == pygame.K_x:
                self.show_unreachable = not self.show_unreachable
            elif event.key == pygame.K_s:
                self.screenshot()
        return True

    def screenshot(self, path: str | Path | None = None) -> Path:
        from PIL import Image
        if path is None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            self.screenshots += 1
            path = self.screenshot_dir / f"seed{self.game.cfg.seed}-{self.screenshots:02d}.png"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGB", self.size, pygame.image.tobytes(self.screen, "RGB")).save(path)
        print(f"screenshot: {path}")
        return path

    def run_window(self) -> None:
        """Live window until the run ends and the window is closed."""
        while self._pump():
            if not self.paused and not self.game.done:
                self._advance(RATES[self.rate_index])
            self.render()
            pygame.display.flip()
            if self.recorder is not None and not self.paused:
                self.recorder.add(self.screen)
            self.clock.tick(FPS)

    def run_record(self) -> None:
        """Headless: step the game, writing one frame every `record_every` sim seconds inside
        [record_from, record_to]."""
        g = self.game
        while not g.done:
            t = g.tick * DT
            in_window = t >= self.record_from and (self.record_to is None or t <= self.record_to)
            if in_window:
                self.render()
                self.recorder.add(self.screen)
                self._advance(self.record_every)
            else:
                self._advance(min(10.0, max(DT, self.record_from - t)) if t < self.record_from
                              else 60.0)
            if self.record_to is not None and t > self.record_to:
                break
        self.render()
        self.recorder.add(self.screen)
        self.recorder.hold(3.0)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            print(f"recorded: {self.recorder.path} ({self.recorder.frames} frames)")
        pygame.quit()
