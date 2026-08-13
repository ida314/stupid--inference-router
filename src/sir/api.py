"""The single endpoint every service talks to.

Handlers here do four things and nothing else: validate, enqueue, stream back, and notice
when a client hangs up. No scheduling logic lives in this file — the API has no opinion
about which model is resident, which is exactly what lets clients have none either.

It has no opinion about the request body either. A service sends `sir` byte-for-byte what
it would send a vLLM or SGLang server, including that server's non-standard extras, and
`sir` reads only `model` and `stream` before forwarding the rest. The point is that
pointing a service at `:8000` is a change of host and nothing else.
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
from sir.config import AppConfig
from sir.engine import BackendFactory, Engine, default_backend_factory
from sir.schemas import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatMessage,
    ModelCard,
    ModelList,
    Usage,
    error_body,
    render_prompt,
)
from sir.types import Chunk, QueuedRequest, StreamEnd, StreamError

# How often a handler checks whether its client is still there. Uvicorn does not cancel
# handlers on disconnect, so we ask.
_DISCONNECT_POLL_SECONDS = 0.2


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
        app.state.engine = engine
        await engine.start()
        try:
            yield
        finally:
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

        # Accepted regardless of what is currently loaded. The wait, if any, happens in
        # the queue behind this call — the client never learns the difference.
        queued = engine.submit(
            body.to_generation_request(target.name, target.served_as)
        )

        if body.stream:
            return StreamingResponse(
                _sse(queued, http, body.model),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )
        return await _collect(queued, http, body)

    return app


def _engine(app: FastAPI) -> Engine:
    return app.state.engine


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

    text = "".join(parts)
    finish_reason = event.finish_reason if isinstance(event, StreamEnd) else "stop"
    # A word count standing in for tokenisation, which belongs to the backend. Phase 2
    # takes these numbers from the real response instead.
    prompt_tokens = len(render_prompt(body.messages).split())
    return ChatCompletionResponse(
        model=body.model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(parts),
            total_tokens=prompt_tokens + len(parts),
        ),
    )


async def _sse(
    queued: QueuedRequest, http: Request, model: str
) -> AsyncIterator[str]:
    """Server-sent events in OpenAI's chunk format, terminated by `[DONE]`."""
    stream_id = f"chatcmpl-{queued.id}"
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
