"""A backend crash is a degraded workload, not an outage.

The router owns several services' inference. If one model's process dying took the router
down with it, every service would lose inference over one bad model — which is strictly
worse than the per-service vLLM setup this project exists to replace.
"""

from __future__ import annotations

from sir.config import ModelConfig
from sir.types import GenerationRequest, QueuedRequest, RequestState, StreamEvent, StreamError
from tests.sim import build_config, model, running_engine


def crashing(name: str, after: int, **kw: object) -> ModelConfig:
    return model(name, crash_after_n_requests=after, **kw)


def collected(queued: QueuedRequest) -> list[StreamEvent]:
    """Everything currently sitting in a request's event queue."""
    events: list[StreamEvent] = []
    while not queued.events.empty():
        events.append(queued.events.get_nowait())
    return events


async def test_a_crash_fails_its_own_requests_and_leaves_the_router_serving(clock):
    config = build_config(
        models=[
            crashing("chat", after=1, load_seconds=4, tokens_per_second=40),
            model("translate", load_seconds=4, tokens_per_second=40),
        ],
        min_residency_seconds=1,
        max_wait_seconds=60,
        backend_retry_seconds=5,
        tick_interval_seconds=0.1,
    )

    async with running_engine(config, clock) as engine:
        first = engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=32))
        second = engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=32))
        await clock.advance(6.0)  # load, dispatch, crash

        # The dead backend is no longer resident and no longer trusted.
        assert engine.resident is None
        assert second.state in (RequestState.FAILED, RequestState.CANCELLED)
        assert first.state in (RequestState.FAILED, RequestState.CANCELLED, RequestState.DONE)

        # The router itself is fine, and the other model is scheduled as normal.
        healthy = engine.submit(
            GenerationRequest(model="translate", prompt="x", max_tokens=8)
        )
        await clock.advance(20.0)
        assert healthy.state is RequestState.DONE
        assert engine.resident == "translate"


async def test_a_load_failure_leaves_the_request_queued_for_a_retry(clock):
    """A model that won't load is a reason to back off, not to fail the client."""
    config = build_config(
        models=[
            crashing("chat", after=0, load_seconds=1, tokens_per_second=40),
            model("translate", load_seconds=1),
        ],
        min_residency_seconds=1,
        max_wait_seconds=60,
        tick_interval_seconds=0.1,
    )
    # crash_after_n_requests=0 disables the crash, so force it via load failure instead.
    config.models[0].mock.fail_on_load_every_n = 1

    async with running_engine(config, clock) as engine:
        request = engine.submit(
            GenerationRequest(model="chat", prompt="x", max_tokens=8)
        )
        await clock.advance(5.0)

        # The load failed, so the model was never resident and the request is still
        # queued rather than errored — the router will retry it after the backoff.
        assert engine.resident is None
        assert request.state is RequestState.QUEUED


async def test_a_crashed_model_is_retried_and_recovers(clock):
    config = build_config(
        models=[
            crashing("chat", after=1, load_seconds=2, tokens_per_second=100),
            model("translate", load_seconds=2),
        ],
        min_residency_seconds=1,
        max_wait_seconds=300,
        backend_retry_seconds=5,
        tick_interval_seconds=0.1,
    )

    async with running_engine(config, clock) as engine:
        engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        await clock.advance(6.0)
        assert engine.resident is None  # crashed

        # A restart gives the backend a fresh process, and one more request fits before
        # the injected fault triggers again.
        later = engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        await clock.advance(30.0)
        assert later.state is RequestState.DONE
        assert engine.loads >= 2


async def test_an_error_event_carries_a_503_to_the_client(clock):
    config = build_config(
        models=[
            crashing("chat", after=0, load_seconds=1, tokens_per_second=100),
            model("translate", load_seconds=1),
        ],
        min_residency_seconds=1,
        max_wait_seconds=60,
        tick_interval_seconds=0.1,
    )
    config.models[0].mock.crash_after_n_requests = 1

    async with running_engine(config, clock) as engine:
        engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        doomed = engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        await clock.advance(6.0)

        events = collected(doomed)
        errors = [e for e in events if isinstance(e, StreamError)]
        assert errors, f"expected an error event, got {events}"
        assert errors[0].status_code == 503


async def test_the_control_loop_survives_a_backend_raising_something_unexpected(clock):
    """A bug in a backend must not silently stop the scheduler."""
    config = build_config(min_residency_seconds=1, max_wait_seconds=60)

    async with running_engine(config, clock) as engine:
        backend = engine._backends["chat"]  # noqa: SLF001 - fault injection

        async def exploding(request: object):
            raise ValueError("backend bug")
            yield  # pragma: no cover - makes this an async generator

        backend.stream = exploding  # type: ignore[method-assign]

        broken = engine.submit(GenerationRequest(model="chat", prompt="x", max_tokens=8))
        await clock.advance(12.0)
        assert broken.state is RequestState.FAILED
        assert engine.is_running

        # Still scheduling.
        ok = engine.submit(GenerationRequest(model="translate", prompt="x", max_tokens=8))
        await clock.advance(30.0)
        assert ok.state is RequestState.DONE
        assert engine.is_running
