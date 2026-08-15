"""The single endpoint every service talks to.

Handlers here do four things and nothing else: validate, enqueue, stream back, and notice
when a client hangs up. No scheduling logic lives in this file — the API has no opinion
about which model is resident, which is exactly what lets clients have none either.

It has no opinion about the request body either. A service sends `sir` byte-for-byte what
it would send a vLLM or SGLang server, including that server's non-standard extras, and
`sir` reads only `model` and `stream` before forwarding the rest. The point is that
pointing a service at `:8000` is a change of host and nothing else.

That is also why asynchronous submission is opted into with a *header* rather than a body
field. A client that sends `Prefer: respond-async` gets `202` and a job to poll instead of
a held-open socket; a client that doesn't gets exactly what it got before. Keeping the
switch out of the body means `sir` still reads only `model` and `stream`, and it means the
same client code works against a plain vLLM — which ignores the unknown header and answers
`200` — with no capability negotiation at all.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sir import logging as sir_logging
from sir.clock import Clock
from sir.config import AppConfig, JobsConfig
from sir.engine import BackendFactory, Engine, default_backend_factory
from sir.jobs import Job, JobStore, JobStoreFull
from sir.schemas import (
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatDelta,
    JobDocument,
    ModelCard,
    ModelList,
    WaitInfo,
    build_response,
    error_body,
)
from sir.types import Chunk, QueuedRequest, RequestState, StreamEnd, StreamError

# How often a handler checks whether its client is still there. Uvicorn does not cancel
# handlers on disconnect, so we ask.
_DISCONNECT_POLL_SECONDS = 0.2

# RFC 7240's token for "don't make me wait on this connection".
_ASYNC_PREFERENCE = "respond-async"


def create_app(
    config: AppConfig,
    clock: Clock | None = None,
    backend_factory: BackendFactory = default_backend_factory,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = Engine(
            config,
            clock=clock,
            backend_factory=backend_factory,
            on_decision=sir_logging.log_decision,
            on_event=sir_logging.log_event,
        )
        jobs = JobStore(config.jobs, clock=engine.clock)
        app.state.engine = engine
        app.state.jobs = jobs
        await engine.start()
        await jobs.start()
        try:
            yield
        finally:
            # Jobs first: their pumps are readers of queues the engine is about to drain.
            await jobs.stop()
            await engine.stop()

    app = FastAPI(title="sir", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Router liveness, deliberately independent of backend state.

        A crashed model must not make the router look dead — that's the difference
        between one degraded workload and an outage.
        """
        return {"status": "ok", "models": config.served_model_names}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return _engine(app).status()

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        # Every alias gets its own card, as vLLM does with repeated
        # `--served-model-name` values, so client-side discovery finds all of them.
        return ModelList(
            data=[
                ModelCard(id=tag, root=model.served_as)
                for model in config.models
                for tag in model.served_names
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, http: Request) -> Any:
        engine = _engine(app)
        # Resolution is by served tag only. Internal names are a config and logging
        # convenience, and a client that guessed one should get the same 404 as a client
        # that guessed anything else.
        target = config.model_by_served_name(body.model)
        if target is None:
            return JSONResponse(
                status_code=404,
                content=error_body(
                    f"model {body.model!r} is not configured; "
                    f"available: {', '.join(config.served_model_names)}",
                    kind="not_found_error",
                    code="model_not_found",
                ),
            )

        wants_async = config.jobs.enabled and _prefers_async(http)
        if wants_async and body.stream:
            # Advisory though `Prefer` is, silently streaming at a client that asked for a
            # job handle would hang it on a response shape it isn't parsing.
            return JSONResponse(
                status_code=400,
                content=error_body(
                    "stream and Prefer: respond-async are mutually exclusive; "
                    "poll the job for a complete response, or stream synchronously"
                ),
            )

        # An idempotent resubmission must not buy a second generation. Checked before
        # `submit`, so the duplicate never reaches a queue at all.
        key = http.headers.get("idempotency-key")
        if wants_async and key:
            existing = _jobs(app).find_by_key(key)
            if existing is not None:
                return _accepted(engine, existing, config.jobs)

        # Accepted regardless of what is currently loaded. The wait, if any, happens in
        # the queue behind this call — the client never learns the difference.
        queued = engine.submit(
            body.to_generation_request(target.name, target.served_as)
        )

        if wants_async:
            try:
                job = _jobs(app).create(queued, body, idempotency_key=key)
            except JobStoreFull as exc:
                queued.cancel()
                return JSONResponse(
                    status_code=503,
                    content=error_body(str(exc), kind="server_error", code="jobs_full"),
                )
            return _accepted(engine, job, config.jobs)

        if body.stream:
            return StreamingResponse(
                _sse(queued, http, body.model),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )
        return await _collect(queued, http, body)

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> Any:
        # Reading is what renews the lease, so this is also how a client says "still here".
        job = _jobs(app).get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content=_job_lost(job_id))
        return JSONResponse(content=_document(_engine(app), job, config.jobs).model_dump(mode="json"))

    @app.delete("/v1/jobs/{job_id}")
    async def cancel_job(job_id: str) -> Any:
        job = _jobs(app).cancel(job_id)
        if job is None:
            return JSONResponse(status_code=404, content=_job_lost(job_id))
        return JSONResponse(content=_document(_engine(app), job, config.jobs).model_dump(mode="json"))

    return app


def _engine(app: FastAPI) -> Engine:
    return app.state.engine


def _jobs(app: FastAPI) -> JobStore:
    return app.state.jobs


# ------------------------------------------------------------------ async submission


def _prefers_async(http: Request) -> bool:
    """Is `respond-async` among the client's `Prefer` tokens? (RFC 7240)"""
    for header in http.headers.getlist("prefer"):
        for token in header.split(","):
            if token.strip().lower() == _ASYNC_PREFERENCE:
                return True
    return False


def _completion_id(request_id: str) -> str:
    """One request, one completion id, whichever path rendered it."""
    return f"chatcmpl-{request_id}"


def _job_lost(job_id: str) -> dict[str, Any]:
    # The two causes are indistinguishable from here, and a client's response differs
    # between them, so name both rather than implying only the first.
    return error_body(
        f"job {job_id!r} not found; it has expired or the router restarted",
        kind="not_found_error",
        code="job_not_found",
    )


def _accepted(engine: Engine, job: Job, jobs: JobsConfig) -> JSONResponse:
    document = _document(engine, job, jobs)
    return JSONResponse(
        status_code=202,
        content=document.model_dump(mode="json"),
        headers={
            "location": f"/v1/jobs/{job.id}",
            # Tells the client the preference was actually honoured, so one that also
            # talks to a plain vLLM can tell `200` -> ignored from `200` -> finished.
            "preference-applied": _ASYNC_PREFERENCE,
            "retry-after": str(max(1, round(document.retry_after))),
        },
    )


def _document(engine: Engine, job: Job, jobs: JobsConfig) -> JobDocument:
    """Render a job at whatever stage of its life it happens to be in."""
    estimate = engine.estimate_wait(job.queued) if not job.is_terminal else None

    if estimate is not None:
        retry_after = estimate.retry_after
    elif job.is_terminal:
        retry_after = 0.0
    else:
        retry_after = engine.running_poll_seconds

    # Never advise a cadence that would cost the client its lease. Polling is what proves
    # a client is still there, so a router that says "come back in 30s" while cancelling
    # anything unseen for 20s would cancel the clients doing exactly as they were told.
    # Both numbers are known here, which makes this the place to keep them consistent —
    # cheaper than a config rule the operator has to get right.
    if jobs.lease_seconds:
        retry_after = min(retry_after, jobs.lease_seconds / 2)

    response = None
    error = None
    if job.state is RequestState.DONE:
        response = build_response(
            job.body, job.texts, job.finish_reason, _completion_id(job.id), job.usage
        )
    elif job.state is RequestState.FAILED and isinstance(job.terminal, StreamError):
        # The inner object, not the envelope: the job document is the envelope here, and
        # nesting `{"error": {"error": ...}}` would just make clients unwrap twice.
        error = error_body(job.terminal.message, kind="server_error")["error"]
    elif job.state is RequestState.CANCELLED:
        error = error_body(
            "job cancelled", kind="cancelled_error", code="job_cancelled"
        )["error"]

    return JobDocument(
        id=job.id,
        status=job.state.value,
        model=job.body.model,
        wait=(
            WaitInfo(
                position=estimate.position,
                resident=estimate.resident,
                needs_swap=estimate.needs_swap,
                load_seconds=estimate.load_seconds,
                dispatch_within_seconds=estimate.dispatch_within_seconds,
                estimated_seconds=estimate.estimated_seconds,
            )
            if estimate is not None
            else None
        ),
        retry_after=retry_after,
        response=response,
        error=error,
    )


# ------------------------------------------------------------------ client liveness


async def _watch_disconnect(http: Request) -> None:
    while not await http.is_disconnected():
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _next_event(queued: QueuedRequest, watcher: asyncio.Task[None]) -> Any | None:
    """Wait for the next stream event, or None if the client vanished first."""
    getter = asyncio.create_task(queued.events.get())
    done, _ = await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
    if getter in done:
        return getter.result()
    getter.cancel()
    return None


# ------------------------------------------------------------------ response paths


async def _collect(
    queued: QueuedRequest, http: Request, body: ChatCompletionRequest
) -> Any:
    """Non-streaming: the same stream, accumulated. One dispatch path, two renderings."""
    watcher = asyncio.create_task(_watch_disconnect(http))
    parts: list[str] = []
    try:
        while True:
            event = await _next_event(queued, watcher)
            if event is None:
                queued.cancel()
                return JSONResponse(status_code=499, content=error_body("client closed request"))
            if isinstance(event, Chunk):
                parts.append(event.text)
                continue
            if isinstance(event, StreamError):
                return JSONResponse(
                    status_code=event.status_code,
                    content=error_body(event.message, kind="server_error"),
                )
            break  # StreamEnd
    except asyncio.CancelledError:
        queued.cancel()
        raise
    finally:
        watcher.cancel()

    ended = event if isinstance(event, StreamEnd) else None
    finish_reason = ended.finish_reason if ended else "stop"
    usage = ended.usage if ended else None
    return build_response(body, parts, finish_reason, _completion_id(queued.id), usage)


async def _sse(
    queued: QueuedRequest, http: Request, model: str
) -> AsyncIterator[str]:
    """Server-sent events in OpenAI's chunk format, terminated by `[DONE]`."""
    stream_id = _completion_id(queued.id)
    watcher = asyncio.create_task(_watch_disconnect(http))

    def chunk(delta: ChatDelta, finish_reason: str | None = None) -> str:
        payload = ChatCompletionChunk(
            id=stream_id,
            model=model,
            choices=[ChatChunkChoice(delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    try:
        yield chunk(ChatDelta(role="assistant"))
        while True:
            event = await _next_event(queued, watcher)
            if event is None:
                queued.cancel()
                return
            if isinstance(event, Chunk):
                yield chunk(ChatDelta(content=event.text))
                continue
            if isinstance(event, StreamError):
                # Headers are long gone, so the status code can't change. Deliver the
                # error in-band rather than truncating the stream silently.
                yield f"data: {json.dumps(error_body(event.message, kind='server_error'))}\n\n"
                break
            yield chunk(ChatDelta(), finish_reason=event.finish_reason)
            break
        yield "data: [DONE]\n\n"
    finally:
        # Covers GeneratorExit too: if the response task dies, the request stops
        # costing GPU time.
        watcher.cancel()
        if queued.finished_at is None:
            queued.cancel()
