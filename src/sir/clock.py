"""Time, injected.

The scheduler reasons about a 30-second minimum residency and a 120-second starvation
ceiling. Testing that against wall-clock time means either waiting two minutes or scaling
every constant down until the test is really measuring the event loop's scheduling jitter.

So nothing in `sir` calls `time` or `asyncio.sleep` directly. Everything takes a `Clock`.
Production gets `RealClock`; tests get `VirtualClock`, where two minutes pass instantly and
in exactly the order you'd expect.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

# How many event-loop yields `VirtualClock` gives woken tasks to run to their next park
# before it considers time advanceable again. Generous: a yield costs microseconds, and
# under-settling would silently reorder events.
_SETTLE_ROUNDS = 50


@runtime_checkable
class Clock(Protocol):
    """Monotonic time plus the ability to wait on it."""

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Wall-clock time. Monotonic, so it survives NTP steps."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)
        else:
            await asyncio.sleep(0)


class VirtualClock:
    """Simulated time that only moves when a test tells it to.

    Sleepers park on futures held in a deadline-ordered heap. `advance()` fires them in
    order, letting each batch of woken tasks run to its next park before time moves again,
    so causality holds: a task sleeping until t=5 always observes everything that happened
    at t<5.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return

        deadline = self._now + seconds
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (deadline, next(self._seq), waiter)
        heapq.heappush(self._sleepers, entry)
        try:
            await waiter
        except asyncio.CancelledError:
            # A cancelled sleeper must not hold time back at its deadline.
            self._discard(entry)
            raise

    async def settle(self) -> None:
        """Let every runnable task advance without moving the clock."""
        for _ in range(_SETTLE_ROUNDS):
            await asyncio.sleep(0)

    async def advance(self, seconds: float) -> None:
        """Move time forward by `seconds`, firing sleepers as their deadlines arrive."""
        await self.advance_to(self._now + seconds)

    async def advance_to(self, target: float) -> None:
        await self.settle()
        while self._sleepers and self._sleepers[0][0] <= target:
            deadline = self._sleepers[0][0]
            self._now = max(self._now, deadline)
            # Everything due at this instant wakes together, in registration order.
            while self._sleepers and self._sleepers[0][0] <= self._now:
                _, _, waiter = heapq.heappop(self._sleepers)
                if not waiter.done():
                    waiter.set_result(None)
            await self.settle()
        self._now = max(self._now, target)
        await self.settle()

    async def advance_to_next(self) -> float | None:
        """Jump to the next scheduled wakeup. Returns the new time, or None if idle."""
        await self.settle()
        if not self._sleepers:
            return None
        await self.advance_to(self._sleepers[0][0])
        return self._now

    async def drive[T](self, awaitable: Awaitable[T]) -> T:
        """Run `awaitable` to completion, advancing time whenever it blocks.

        Needed for anything that sleeps outside the test's own timeline — shutdown being
        the obvious case, where unloading a backend takes simulated seconds that no
        `advance()` call is left to supply. Without this the teardown just hangs.
        """
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            await self.settle()
            if task.done():
                break
            if await self.advance_to_next() is None:
                raise RuntimeError(
                    "virtual clock has no pending wakeups but the task is still "
                    "blocked — it is waiting on something other than this clock"
                )
        return await task

    @property
    def pending_wakeups(self) -> int:
        return len(self._sleepers)

    def _discard(self, entry: tuple[float, int, asyncio.Future[None]]) -> None:
        try:
            self._sleepers.remove(entry)
        except ValueError:
            return
        heapq.heapify(self._sleepers)
