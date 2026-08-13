"""The virtual clock. If this is wrong, every timing assertion above it is meaningless."""

from __future__ import annotations

import asyncio

import pytest

from sir.clock import RealClock, VirtualClock


async def test_sleepers_wake_in_deadline_order():
    clock = VirtualClock()
    woken: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woken.append(name)

    tasks = [
        asyncio.create_task(sleeper("late", 30)),
        asyncio.create_task(sleeper("early", 5)),
        asyncio.create_task(sleeper("middle", 10)),
    ]
    await clock.advance(60)
    await asyncio.gather(*tasks)

    assert woken == ["early", "middle", "late"]
    assert clock.now() == 60


async def test_time_only_moves_when_advanced():
    clock = VirtualClock()
    done = False

    async def sleeper() -> None:
        nonlocal done
        await clock.sleep(10)
        done = True

    task = asyncio.create_task(sleeper())
    await clock.advance(9)
    assert not done
    await clock.advance(1)
    assert done
    await task


async def test_a_task_woken_mid_advance_can_schedule_more_sleep():
    """Causality: a task that wakes at t=5 must see the world as of t=5, not t=60."""
    clock = VirtualClock()
    observed: list[float] = []

    async def chain() -> None:
        await clock.sleep(5)
        observed.append(clock.now())
        await clock.sleep(5)
        observed.append(clock.now())

    task = asyncio.create_task(chain())
    await clock.advance(60)
    await task
    assert observed == [5, 10]


async def test_a_cancelled_sleeper_does_not_hold_time_at_its_deadline():
    clock = VirtualClock()
    task = asyncio.create_task(clock.sleep(1000))
    await clock.settle()
    assert clock.pending_wakeups == 1

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert clock.pending_wakeups == 0


async def test_advance_to_next_jumps_straight_to_the_next_wakeup():
    clock = VirtualClock()
    task = asyncio.create_task(clock.sleep(42))
    await clock.settle()

    assert await clock.advance_to_next() == 42
    await task
    assert await clock.advance_to_next() is None


async def test_drive_runs_work_that_sleeps_past_the_test_timeline():
    """This is what makes shutdown — which unloads a backend — terminate."""
    clock = VirtualClock()

    async def slow_teardown() -> str:
        await clock.sleep(500)
        return "done"

    assert await clock.drive(slow_teardown()) == "done"
    assert clock.now() >= 500


async def test_drive_refuses_to_hang_on_work_it_cannot_wake():
    clock = VirtualClock()
    never = asyncio.Event()
    with pytest.raises(RuntimeError, match="no pending wakeups"):
        await clock.drive(never.wait())


async def test_real_clock_is_monotonic_and_sleeps():
    clock = RealClock()
    before = clock.now()
    await clock.sleep(0.01)
    assert clock.now() >= before
