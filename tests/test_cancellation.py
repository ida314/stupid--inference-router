"""A client that hangs up stops costing GPU time.

Two cases, and they behave differently on purpose. A queued request is simply dropped —
nobody paid for it yet. An in-flight one is cancelled, and crucially is not allowed to
hold up the next drain.
"""

from __future__ import annotations

from sir.types import RequestState
from tests.sim import build_config, generation, running_engine


async def test_a_queued_request_that_is_cancelled_is_never_dispatched(clock):
    config = build_config(min_residency_seconds=0, max_wait_seconds=60)

    async with running_engine(config, clock) as engine:
        doomed = engine.submit(
            generation("chat", max_tokens=8)
        )
        keeper = engine.submit(
            generation("chat", max_tokens=8)
        )
        doomed.cancel()  # while the backend is still loading

        await clock.advance(20.0)
        assert doomed.started_at is None
        assert doomed.state is RequestState.CANCELLED
        assert keeper.state is RequestState.DONE


async def test_a_cancelled_queue_cannot_trigger_a_swap(clock):
    """Ghost requests must not be able to move the GPU on their way out."""
    config = build_config(min_residency_seconds=0, max_wait_seconds=30)

    async with running_engine(config, clock) as engine:
        engine.submit(generation("chat", max_tokens=8))
        await clock.advance(12.0)
        assert engine.resident == "chat"

        ghosts = [
            engine.submit(generation("translate", max_tokens=8))
            for _ in range(20)
        ]
        for ghost in ghosts:
            ghost.cancel()

        # Twenty cancelled requests, well past the starvation ceiling, must not buy a swap.
        await clock.advance(120.0)
        assert engine.resident == "chat"
        assert engine.swaps == 0


async def test_cancelling_in_flight_work_frees_the_slot(clock):
    config = build_config(
        min_residency_seconds=0, max_wait_seconds=600, max_concurrent_requests=1
    )

    async with running_engine(config, clock) as engine:
        hog = engine.submit(
            generation("chat", max_tokens=100000)
        )
        waiting = engine.submit(
            generation("chat", max_tokens=8)
        )
        await clock.advance(10.0)
        assert hog.state is RequestState.RUNNING
        assert waiting.state is RequestState.QUEUED  # blocked behind the concurrency cap

        hog.cancel()
        await clock.advance(10.0)
        assert hog.state is RequestState.CANCELLED
        assert waiting.state is RequestState.DONE


async def test_the_queue_depth_the_scheduler_sees_excludes_cancelled_requests(clock):
    config = build_config(min_residency_seconds=0, max_wait_seconds=600)

    async with running_engine(config, clock) as engine:
        requests = [
            engine.submit(generation("translate", max_tokens=8))
            for _ in range(5)
        ]
        assert engine.queues.depth("translate") == 5

        for request in requests[:3]:
            request.cancel()
        assert engine.queues.depth("translate") == 2
        assert engine.queues.oldest_enqueued_at("translate") == requests[3].enqueued_at
