"""CPU time per player and category (assignment §5.4: "total CPU time for each player").

`with cpu("search"): ...` adds the process time spent inside to that category. Timers nest: the
time inside an inner timer counts for the inner category only, so the categories add up to the
total. time.process_time() counts this process's CPU, not wall clock, so a busy machine or a
paused window doesn't inflate it.
"""

from __future__ import annotations

import time
from collections import defaultdict


class _Timer:
    __slots__ = ("cpu", "category", "t")

    def __init__(self, cpu: "Cpu", category: str):
        self.cpu = cpu
        self.category = category

    def __enter__(self):
        now = time.process_time()
        stack = self.cpu._stack
        if stack:
            parent = stack[-1]
            self.cpu.totals[parent.category] += now - parent.t
        self.t = now
        stack.append(self)
        self.cpu.calls[self.category] += 1
        return self

    def __exit__(self, *exc):
        now = time.process_time()
        self.cpu.totals[self.category] += now - self.t
        stack = self.cpu._stack
        stack.pop()
        if stack:
            stack[-1].t = now
        return False


class Cpu:
    def __init__(self, player: str):
        self.player = player
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._stack: list[_Timer] = []
        self.prefix = ""            # e.g. "build/" while the roadmap is built before t = 0

    def __call__(self, category: str) -> _Timer:
        return _Timer(self, self.prefix + category)

    @property
    def total(self) -> float:
        return sum(self.totals.values())

    def as_dict(self) -> dict[str, float]:
        return dict(self.totals)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NullCpu(Cpu):
    def __call__(self, category: str):
        return _NULL_TIMER


_NULL_TIMER = _Null()
NULL = _NullCpu("none")
