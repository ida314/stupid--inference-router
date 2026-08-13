"""Swaps are ordered, and they never cut a response in half.

`drain -> unload -> load -> dispatch` is the sequence the README promises. Getting it
wrong is how you end up with a truncated response, or two models briefly co-resident on a
GPU that has room for one.
"""

from __future__ import annotations

from sir.types import EventKind, RequestState
from tests.sim import Arrival, alternating, build_config, generation, running_engine, simulate


async def test_the_swap_sequence_is_drain_unload_load():
    config = build_config(min_residency_seconds=5, max_wait_seconds=60)
    result = await simulate(
        config, alternating(("chat", "translate"), every=2.0, count=8), until=120
    )

    print(result.report())
    order = [
        event.kind
        for event in result.events
        if event.kind
        in (
            EventKind.DRAIN_START,
            EventKind.DRAIN_END,
            EventKind.UNLOAD,
            EventKind.LOAD_START,
            EventKind.LOAD_END,
        )
    ]

    # Find the first real swap and check the five steps land in order.
    unload_at = order.index(EventKind.UNLOAD)
    assert order[unload_at - 2] is EventKind.DRAIN_START
    assert order[unload_at - 1] is EventKind.DRAIN_END
    assert order[unload_at + 1] is EventKind.LOAD_START
    assert order[unload_at + 2] is EventKind.LOAD_END


async def test_in_flight_generation_survives_the_swap_that_follows_it():
    config = build_config(min_residency_seconds=5, max_wait_seconds=60)
    result = await simulate(
        config, alternating(("chat", "translate"), every=2.0, count=8), until=120
    )

    print(result.report())
    assert result.swaps >= 1
    # Nothing was killed mid-response: every dispatched request reached a clean finish.
    assert all(req.state is RequestState.DONE for req in result.requests)


async def test_only_one_model_is_ever_resident(clock):
    config = build_config(min_residency_seconds=2, max_wait_seconds=30)

    async with running_engine(config, clock) as engine:
        for index in range(12):
            model_name = "chat" if index % 2 == 0 else "translate"
            engine.submit(
                generation(model_name, max_tokens=8)
            )
            await clock.advance(2.0)
            # The engine tracks exactly one resident model, and nothing else is loaded.
            loaded = [
                name
                for name in config.model_names
                if engine._backends[name].loaded  # noqa: SLF001 - invariant check
            ]
            assert len(loaded) <= 1
            assert engine.resident in (None, *loaded)


async def test_a_drain_waits_for_generation_but_not_for_a_departed_client(clock):
    """A client that hung up must not hold the GPU hostage during a swap."""
    config = build_config(
        min_residency_seconds=0, max_wait_seconds=60, tick_interval_seconds=0.1
    )

    async with running_engine(config, clock) as engine:
        slow = engine.submit(
            generation("chat", max_tokens=100000)
        )
        await clock.advance(10.0)  # load (8s) then dispatch
        assert slow.state is RequestState.RUNNING

        slow.cancel()
        engine.submit(generation("translate", max_tokens=8))

        # Without cancellation-aware draining, the 100k-token request would block this
        # swap for the better part of an hour.
        await clock.advance(60.0)
        assert engine.resident == "translate"
        assert slow.state is RequestState.CANCELLED


async def test_the_resident_model_stays_loaded_when_nothing_is_queued():
    """Idle is not a reason to unload — the reload would cost more than it saves."""
    config = build_config(min_residency_seconds=5, max_wait_seconds=60)
    result = await simulate(config, [Arrival(at=0.0, model="chat")], until=200)

    print(result.report())
    assert result.loads == 1
    assert result.swaps == 0
    assert not [e for e in result.events if e.kind is EventKind.UNLOAD]
