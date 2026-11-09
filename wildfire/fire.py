"""Cell states, fire timers, scoring and the event log.

Time is counted in integer ticks (dt = 0.1 s, notes D10) so timers never drift.

Rules (assignment §2, notes "Fire"):
- A cell is intact, burning, extinguished or burned.
- The arsonist lights intact or extinguished cells (relighting). Spread lights intact cells only
  (D3), so a single spread at +10 s and continuous spread give the same result.
- `spread_ticks` after a cell starts burning, every intact cell within 30 m starts burning. If the
  cell was extinguished first, nothing spreads.
- `burn_ticks` after its latest ignition a burning cell is burned, for good. Each ignition restarts
  both clocks (D4).
- The truck extinguishes one burning cell at a time (the sim enforces the 5 s dwell and the 10 m
  range).

Scoring counts every state change (D5): arsonist +1 per ignition (its own or spread, relights
included) and +1 per burnout; truck +2 per extinguish.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .world import World

INTACT, BURNING, EXTINGUISHED, BURNED = 0, 1, 2, 3
STATE_NAMES = ("intact", "burning", "extinguished", "burned")

_SPREAD, _BURNOUT = 0, 1        # heap order within a tick: spread before burnout


@dataclass(frozen=True)
class Event:
    tick: int
    cell: int
    kind: str          # "ignite", "extinguish", "burnout"
    cause: str         # ignite: "arsonist" or "spread"; extinguish: "truck"; burnout: "fire"
    relight: bool = False


class Fire:
    def __init__(self, world: World, burn_ticks: int, spread_ticks: int = 100):
        if burn_ticks <= spread_ticks:
            raise ValueError("burnout must be longer than the spread delay, or nothing spreads")
        self.world = world
        self.burn_ticks = burn_ticks
        self.spread_ticks = spread_ticks
        n = world.n
        self.state = np.zeros(n, dtype=np.int8)
        self.ignited = np.full(n, -1, dtype=np.int64)      # tick of the latest ignition
        self._version = np.zeros(n, dtype=np.int64)        # bumps on every change, so stale
        self._heap: list[tuple[int, int, int, int]] = []   # timers are skipped
        self.events: list[Event] = []
        self.changed = False          # set on any state change; the sim clears it

    # --- actions ----------------------------------------------------------------------------

    def ignite(self, cell: int, tick: int, cause: str) -> None:
        s = self.state[cell]
        if cause == "spread":
            if s != INTACT:
                raise ValueError(f"spread can only light intact cells, cell {cell} is {STATE_NAMES[s]}")
        elif s not in (INTACT, EXTINGUISHED):
            raise ValueError(f"cannot light cell {cell}: it is {STATE_NAMES[s]}")
        self.state[cell] = BURNING
        self.ignited[cell] = tick
        self._version[cell] += 1
        v = self._version[cell]
        heapq.heappush(self._heap, (tick + self.spread_ticks, _SPREAD, cell, v))
        heapq.heappush(self._heap, (tick + self.burn_ticks, _BURNOUT, cell, v))
        self.events.append(Event(tick, cell, "ignite", cause, relight=bool(s == EXTINGUISHED)))
        self.changed = True

    def extinguish(self, cell: int, tick: int) -> None:
        if self.state[cell] != BURNING:
            raise ValueError(f"cell {cell} is not burning")
        self.state[cell] = EXTINGUISHED
        self._version[cell] += 1
        self.events.append(Event(tick, cell, "extinguish", "truck"))
        self.changed = True

    def advance(self, tick: int) -> None:
        """Fire every timer due at or before `tick`, in time order."""
        heap = self._heap
        while heap and heap[0][0] <= tick:
            t, kind, cell, v = heapq.heappop(heap)
            if v != self._version[cell]:
                continue
            if kind == _SPREAD:
                for nb in self.world.spread[cell]:
                    if self.state[nb] == INTACT:
                        self.ignite(int(nb), t, "spread")
            else:
                self.state[cell] = BURNED
                self._version[cell] += 1
                self.events.append(Event(t, cell, "burnout", "fire"))
                self.changed = True

    # --- queries ------------------------------------------------------------------------------

    def deadline_ticks(self) -> np.ndarray:
        """Burnout tick of each burning cell; a huge number for the others."""
        out = np.full(self.world.n, np.iinfo(np.int64).max // 4, dtype=np.int64)
        burning = self.state == BURNING
        out[burning] = self.ignited[burning] + self.burn_ticks
        return out

    def counts(self) -> dict[str, int]:
        return {name: int((self.state == k).sum()) for k, name in enumerate(STATE_NAMES)}

    def next_timer(self) -> int | None:
        """Tick of the next live timer, or None. Used to skip idle stretches."""
        while self._heap and self._heap[0][3] != self._version[self._heap[0][2]]:
            heapq.heappop(self._heap)
        return self._heap[0][0] if self._heap else None


@dataclass
class Score:
    ignitions: int = 0
    arsonist_ignitions: int = 0
    spread_ignitions: int = 0
    relights: int = 0
    extinguishes: int = 0
    burnouts: int = 0

    @property
    def arsonist(self) -> int:
        return self.ignitions + self.burnouts

    @property
    def truck(self) -> int:
        return 2 * self.extinguishes

    @property
    def winner(self) -> str:
        if self.truck > self.arsonist:
            return "truck"
        if self.arsonist > self.truck:
            return "arsonist"
        return "draw"


def score(events: list[Event]) -> Score:
    """Rebuild the score from the event log (notes Scoring 6: the log is enough)."""
    s = Score()
    for e in events:
        if e.kind == "ignite":
            s.ignitions += 1
            s.relights += e.relight
            if e.cause == "spread":
                s.spread_ignitions += 1
            else:
                s.arsonist_ignitions += 1
        elif e.kind == "extinguish":
            s.extinguishes += 1
        else:
            s.burnouts += 1
    return s
