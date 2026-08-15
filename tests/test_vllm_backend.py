"""The vLLM backend, driven against a fake vLLM and a fake Docker daemon.

No sockets, no daemon, no weights: `httpx.MockTransport` answers both clients and the
virtual clock drives every wait, so the timeout cases cost microseconds like the rest of
the suite.

What is worth testing here is the seam. That a `start()` which never becomes healthy is
reported as a crash rather than as success. That an adopted backend cannot stop the
container — the safety property the `manage_lifecycle` flag exists for. That `stop()` waits
for the GPU to actually come back before the next load races it. That a stream which dies
halfway raises `BackendError` so the engine's containment can fire. And that the client's
body arrives unaltered, because that is the contract the whole README rests on.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from sir.backend import BackendError
from sir.backends.vllm import VllmBackend
from sir.clock import VirtualClock
from sir.config import VllmParams
from sir.types import Chunk, GenerationRequest, StreamEnd

BASE = "http://vllm:8000"


def make_backend(
    handler,
    clock: VirtualClock,
    docker_handler=None,
    **overrides,
) -> VllmBackend:
    params = VllmParams(
        base_url=BASE,
        ready_poll_seconds=1.0,
        start_timeout_seconds=60.0,
        stop_timeout_seconds=30.0,
        **overrides,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    docker = (
        httpx.AsyncClient(
            transport=httpx.MockTransport(docker_handler), base_url="http://docker"
        )
        if docker_handler
        else None
    )
    return VllmBackend("chat", params, clock, client, docker)


def owning(handler, clock: VirtualClock, docker_handler, **overrides) -> VllmBackend:
    return make_backend(
        handler,
        clock,
        docker_handler,
        manage_lifecycle=True,
        container_name="vllm-qwen",
        **overrides,
    )


def request(**payload) -> GenerationRequest:
    body = {"model": "nvidia/Qwen3.6-27B-NVFP4", "messages": [], **payload}
    return GenerationRequest(
        model="chat", served_model="nvidia/Qwen3.6-27B-NVFP4", payload=body
    )


def sse(*events: dict | str) -> bytes:
    lines = []
    for event in events:
        body = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {body}\n\n")
    return "".join(lines).encode()


def delta(content: str, finish_reason: str | None = None) -> dict:
    return {
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
        ]
    }


async def drain(backend: VllmBackend, req: GenerationRequest) -> list:
    return [event async for event in backend.stream(req)]


async def run_with_clock(clock: VirtualClock, coro, step: float = 1.0, limit: int = 200):
    """Run `coro` while pushing the virtual clock along until it finishes."""
    task = asyncio.ensure_future(coro)
    for _ in range(limit):
        if task.done():
            break
        await clock.advance(step)
    return await task


def healthy(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


# ------------------------------------------------------------------------- config


def test_owning_a_container_requires_naming_it() -> None:
    """Fail at startup, where a config error is cheap to read."""
    with pytest.raises(ValueError, match="requires container_name"):
        VllmParams(manage_lifecycle=True)


# ------------------------------------------------------------------------ adopting


async def test_adopted_start_only_waits_for_health(clock: VirtualClock) -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(200)

    backend = make_backend(handler, clock)
    await backend.start()

    assert seen == ["/health"]


async def test_adopted_stop_does_nothing(clock: VirtualClock) -> None:
    """The safety property: a backend that does not own the container cannot stop it.

    While one model has the GPU, this is what keeps a scheduling decision from being able
    to take down the box's only inference server.
    """
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        return httpx.Response(200)

    def docker(req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"an adopted backend must not call docker: {req.url}")

    backend = make_backend(handler, clock, docker)
    await backend.stop()

    assert calls == []


async def test_start_fails_when_the_server_never_becomes_healthy(
    clock: VirtualClock,
) -> None:
    """A load that never finishes must be reported as a crash, not waited on forever.

    Believing a backend is serving when it is not means every request routed to it fails,
    one at a time, instead of the engine backing the model off and scheduling around it.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    backend = make_backend(handler, clock)
    with pytest.raises(BackendError, match="did not become healthy"):
        await run_with_clock(clock, backend.start())


async def test_start_waits_out_a_slow_load(clock: VirtualClock) -> None:
    """Five minutes of weight loading is normal here, not a failure."""
    polls = {"count": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        polls["count"] += 1
        return httpx.Response(200 if polls["count"] > 5 else 503)

    backend = make_backend(handler, clock)
    await run_with_clock(clock, backend.start())

    assert polls["count"] == 6


async def test_health_is_false_when_the_server_is_gone(clock: VirtualClock) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = make_backend(handler, clock)
    assert await backend.health() is False


# -------------------------------------------------------------------------- owning


async def test_owned_start_starts_the_container_then_waits(clock: VirtualClock) -> None:
    docker_calls: list[str] = []

    def docker(req: httpx.Request) -> httpx.Response:
        docker_calls.append(f"{req.method} {req.url.path}")
        return httpx.Response(204)

    backend = owning(healthy, clock, docker)
    await backend.start()

    assert docker_calls == ["POST /containers/vllm-qwen/start"]


async def test_owned_start_treats_already_running_as_success(clock: VirtualClock) -> None:
    """Docker answers 304 when the container is already up. That is not an error."""

    def docker(req: httpx.Request) -> httpx.Response:
        return httpx.Response(304)

    backend = owning(healthy, clock, docker)
    await backend.start()  # must not raise


async def test_owned_start_reports_a_refused_start_as_a_crash(clock: VirtualClock) -> None:
    def docker(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no such container")

    backend = owning(healthy, clock, docker)
    with pytest.raises(BackendError, match="refused to start"):
        await backend.start()


async def test_owned_start_reports_an_unreachable_daemon_as_a_crash(
    clock: VirtualClock,
) -> None:
    def docker(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no such file or directory")

    backend = owning(healthy, clock, docker)
    with pytest.raises(BackendError, match="could not start"):
        await backend.start()


async def test_owned_stop_waits_for_the_gpu_to_come_back(clock: VirtualClock) -> None:
    """Docker returns on SIGTERM delivery; the GPU memory outlives the API call.

    Waiting for the port to go quiet is what stops the next model's load from racing this
    one's teardown — the exact moment a swap would otherwise become an OOM crash.
    """
    alive = {"value": True}
    polls = {"count": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        polls["count"] += 1
        if polls["count"] >= 3:
            alive["value"] = False
        return httpx.Response(200 if alive["value"] else 503)

    def docker(req: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    backend = owning(handler, clock, docker)
    await run_with_clock(clock, backend.stop())

    assert polls["count"] >= 3


async def test_owned_stop_survives_an_unreachable_daemon(clock: VirtualClock) -> None:
    """`stop()` is mostly called while cleaning up a crash. It cannot raise."""

    def docker(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no such file or directory")

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = owning(handler, clock, docker)
    await run_with_clock(clock, backend.stop())  # must not raise


async def test_owned_stop_passes_a_kill_deadline(clock: VirtualClock) -> None:
    """A wedged server must not hold the GPU forever; docker escalates to SIGKILL."""
    seen: dict = {}

    def docker(req: httpx.Request) -> httpx.Response:
        seen["query"] = dict(req.url.params)
        return httpx.Response(204)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    backend = owning(handler, clock, docker)
    await backend.stop()

    assert seen["query"] == {"t": "30"}


# ---------------------------------------------------------------------- generation


async def test_stream_yields_chunks_then_a_terminator(clock: VirtualClock) -> None:
    body = sse(
        delta(""),
        delta("Hello"),
        delta(" world", finish_reason="stop"),
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}},
        "[DONE]",
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = make_backend(handler, clock)
    events = await drain(backend, request())

    assert events[:-1] == [Chunk(text="Hello", index=0), Chunk(text=" world", index=1)]
    assert events[-1] == StreamEnd(
        finish_reason="stop",
        usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )


async def test_stream_reports_real_token_counts_not_word_counts(clock: VirtualClock) -> None:
    """The backend tokenised the prompt; `sir` only guessed. The backend's numbers win."""
    body = sse(
        delta("one two three"),
        {"choices": [], "usage": {"prompt_tokens": 41, "completion_tokens": 3, "total_tokens": 44}},
        "[DONE]",
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = make_backend(handler, clock)
    events = await drain(backend, request())

    assert events[-1].usage == {
        "prompt_tokens": 41,
        "completion_tokens": 3,
        "total_tokens": 44,
    }


async def test_stream_without_usage_leaves_it_unset(clock: VirtualClock) -> None:
    """No usage reported means fall back to the word count, not report zero."""
    body = sse(delta("hi"), "[DONE]")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = make_backend(handler, clock)
    events = await drain(backend, request())
    assert events[-1] == StreamEnd(finish_reason="stop", usage=None)


async def test_stream_forwards_the_body_untouched(clock: VirtualClock) -> None:
    """The wire contract: vLLM serves the tag the client asked for, so nothing is renamed.

    `response_format` in particular has to arrive intact — it is what makes the callers on
    this box get JSON instead of prose, and the failure when it is dropped is silent.
    """
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, content=sse("[DONE]"))

    backend = make_backend(handler, clock)
    await drain(
        backend,
        request(
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "verdict"}},
            some_future_vllm_flag={"nested": True},
        ),
    )

    # Not rewritten: the API layer already routed it to the tag vLLM serves.
    assert captured["model"] == "nvidia/Qwen3.6-27B-NVFP4"
    # `sir` always consumes a stream, whatever the client asked for.
    assert captured["stream"] is True
    assert captured["stream_options"]["include_usage"] is True
    # Everything else, including fields `sir` has never heard of.
    assert captured["temperature"] == 0
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict"},
    }
    assert captured["some_future_vllm_flag"] == {"nested": True}


async def test_stream_keeps_a_clients_own_stream_options(clock: VirtualClock) -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, content=sse("[DONE]"))

    backend = make_backend(handler, clock)
    await drain(backend, request(stream_options={"continuous_usage_stats": True}))

    assert captured["stream_options"] == {
        "continuous_usage_stats": True,
        "include_usage": True,
    }


async def test_stream_reports_a_rejected_request_as_a_crash(clock: VirtualClock) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model is not ready")

    backend = make_backend(handler, clock)
    with pytest.raises(BackendError, match="rejected generation"):
        await drain(backend, request())


async def test_stream_reports_a_mid_stream_death_as_a_crash(clock: VirtualClock) -> None:
    """The case the engine's containment exists for: dead after some output, not before."""

    async def dying() -> bytes:
        yield sse(delta("half an ans"))
        raise httpx.ReadError("connection reset")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=dying())

    backend = make_backend(handler, clock)
    with pytest.raises(BackendError, match="failed"):
        await drain(backend, request())


async def test_stream_skips_a_malformed_line_rather_than_dying(clock: VirtualClock) -> None:
    """Losing one chunk is recoverable. Failing the whole request over it is not."""
    body = b"data: {not json}\n\n" + sse(delta("fine"), "[DONE]")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = make_backend(handler, clock)
    events = await drain(backend, request())
    assert events[0] == Chunk(text="fine", index=0)


async def test_stream_is_cancellable_between_chunks(clock: VirtualClock) -> None:
    """The engine cancels this task when a client disconnects; it must not hang."""
    started = asyncio.Event()

    async def slow() -> bytes:
        yield sse(delta("first"))
        started.set()
        await asyncio.sleep(3600)  # never arrives

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=slow())

    backend = make_backend(handler, clock)
    task = asyncio.ensure_future(drain(backend, request()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
