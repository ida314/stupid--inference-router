"""One call that returns a completion, however long the router had to queue it.

The problem this exists to solve is that a swap-scheduled GPU can leave a request waiting
minutes before it starts generating, and holding an HTTP connection open that long is a
good way to discover every idle timeout between a service and the router. So `sir` accepts
work asynchronously and hands back a job to poll, and this hides that: `run` looks like an
ordinary request/response call and there is no queue in the signature.

Three things it is careful about, in rough order of how much they matter:

1. **Cancellation reaches the GPU.** If the caller's task is cancelled or its deadline
   passes, the job is cancelled too. Holding a socket open is what tells the router today
   that a client still wants its answer; once nobody is holding one, this is what carries
   that signal, and without it an abandoned request keeps a model resident for a response
   no one will read.
2. **Poll cadence comes from the server.** The router knows the queue depth, whether a
   swap is pending, and how long its requests have been taking. Its `retry_after` is a
   better guess than any interval hardcoded here.
3. **A vanished job is not silently retried.** Resubmitting a request that may already
   have run is a decision with a cost, so it is the caller's to make.

It is also careful about what it does *not* do: it never reads or rewrites the request
body. See `registry.py` for why.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from sir_client.errors import (
    JobCancelled,
    JobFailed,
    JobLost,
    RequestTimeout,
    TransportError,
)
from sir_client.registry import Registry

_TERMINAL = frozenset({"done", "failed", "cancelled"})

# Bounds on how long to wait between polls, whatever the server suggests. The floor keeps
# a bad `retry_after` from turning into a busy loop; the ceiling keeps a finished response
# from sitting unread for long enough to hit its retention window.
_MIN_POLL_SECONDS = 0.05
_MAX_POLL_SECONDS = 10.0

# How many times `run` will resubmit after a lost job, when asked to. One retry covers a
# router restart; more would just replay the request into a router that is still down.
_MAX_ATTEMPTS = 2


@dataclass
class Job:
    """A handle on work the router has accepted."""

    id: str
    model: str
    base_url: str
    client: AsyncClient
    document: dict[str, Any] = field(default_factory=dict)
    # Set when the endpoint answered synchronously — a plain vLLM, or `sir` with the async
    # path disabled. There is nothing to poll, but the caller gets the same handle either
    # way rather than two shapes to branch on.
    completion: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.completion is not None:
            return "done"
        return str(self.document.get("status", "queued"))

    @property
    def wait(self) -> dict[str, Any] | None:
        """The router's own account of the wait: position, residency, swap, estimate."""
        return self.document.get("wait")

    async def refresh(self) -> dict[str, Any]:
        """Re-read the job, renewing its lease. Raises `JobLost` if it is gone."""
        if self.completion is not None:
            return self.document
        self.document = await self.client._fetch(self.base_url, self.id)
        return self.document

    async def result(self, *, timeout: float | None = None) -> dict[str, Any]:
        if self.completion is not None:
            return self.completion
        return await self.client._await_job(self, timeout)

    async def cancel(self) -> None:
        if self.completion is None:
            await self.client._cancel(self.base_url, self.id)


class AsyncClient:
    """Talks to one or more inference endpoints. Safe to share across tasks."""

    def __init__(
        self,
        registry: Registry | None = None,
        *,
        base_url: str | None = None,
        endpoints: dict[str, str] | None = None,
        http: httpx.AsyncClient | None = None,
        timeout: float | None = None,
        max_poll_seconds: float = _MAX_POLL_SECONDS,
    ) -> None:
        if registry is None:
            registry = (
                Registry(endpoints, default=base_url)
                if (endpoints or base_url)
                else Registry.from_env()
            )
        self.registry = registry
        self.timeout = timeout
        self.max_poll_seconds = max_poll_seconds
        # An injected client is the caller's to close — that is also how the test suite
        # points this at an in-process ASGI app instead of a socket.
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ the common case

    async def run(
        self,
        model: str,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
        idempotency_key: str | None = None,
        resubmit_on_loss: bool = False,
    ) -> dict[str, Any]:
        """Submit `body` for `model` and return the completion.

        `body` is forwarded as written, except that `model` is set from the argument so
        routing and payload cannot disagree. Whatever extras the target backend accepts,
        put them in `body`; nothing here inspects them.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        attempts = _MAX_ATTEMPTS if (resubmit_on_loss and idempotency_key) else 1

        for attempt in range(attempts):
            job = await self.submit(model, body, idempotency_key=idempotency_key)
            try:
                return await self._await_job(job, timeout, deadline=deadline)
            except JobLost:
                if attempt == attempts - 1:
                    raise
        raise AssertionError("unreachable")

    async def submit(
        self,
        model: str,
        body: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> Job:
        """Accept-and-return-a-handle, for work the caller doesn't want to wait on."""
        base_url = self.registry.resolve(model)
        headers = {"prefer": "respond-async"}
        if idempotency_key:
            headers["idempotency-key"] = idempotency_key

        response = await self._http.post(
            f"{base_url}/v1/chat/completions",
            json={**body, "model": model},
            headers=headers,
        )

        # 202 means the preference was honoured. 200 means it wasn't — a plain vLLM
        # ignoring an unknown header, or `sir` with jobs turned off — and the body is
        # already the answer. Branching on the status code is the entire compatibility
        # story: no probing, no configuration, no per-backend client.
        if response.status_code == 200:
            return Job(
                id="",
                model=model,
                base_url=base_url,
                client=self,
                completion=response.json(),
            )
        if response.status_code != 202:
            raise TransportError(response.status_code, _body_of(response))

        document = response.json()
        return Job(
            id=document["id"],
            model=model,
            base_url=base_url,
            client=self,
            document=document,
        )

    # ------------------------------------------------------------------ polling

    async def _await_job(
        self,
        job: Job,
        timeout: float | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        if job.completion is not None:
            return job.completion
        if deadline is None:
            effective = self.timeout if timeout is None else timeout
            deadline = None if effective is None else time.monotonic() + effective

        try:
            while True:
                status = job.status
                if status == "done":
                    return job.document["response"]
                if status == "failed":
                    raise JobFailed(job.id, job.document.get("error"))
                if status == "cancelled":
                    raise JobCancelled(job.id)

                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    await self._cancel_quietly(job.base_url, job.id)
                    raise RequestTimeout(job.id, timeout if timeout else 0.0)

                await asyncio.sleep(self._delay(job.document, remaining))
                job.document = await self._fetch(job.base_url, job.id)
        except asyncio.CancelledError:
            # The caller gave up. Tell the router before unwinding, so the GPU stops
            # working on an answer that now has nowhere to go.
            await self._cancel_quietly(job.base_url, job.id)
            raise

    def _delay(self, document: dict[str, Any], remaining: float | None) -> float:
        suggested = document.get("retry_after")
        delay = float(suggested) if isinstance(suggested, (int, float)) else 1.0
        delay = min(self.max_poll_seconds, max(_MIN_POLL_SECONDS, delay))
        if remaining is not None:
            # Never sleep past the deadline; wake up in time to cancel the job.
            delay = min(delay, max(_MIN_POLL_SECONDS, remaining))
        return delay

    async def _fetch(self, base_url: str, job_id: str) -> dict[str, Any]:
        response = await self._http.get(f"{base_url}/v1/jobs/{job_id}")
        if response.status_code == 404:
            raise JobLost(job_id)
        if response.status_code != 200:
            raise TransportError(response.status_code, _body_of(response))
        return response.json()

    async def _cancel(self, base_url: str, job_id: str) -> None:
        response = await self._http.delete(f"{base_url}/v1/jobs/{job_id}")
        if response.status_code not in (200, 404):
            raise TransportError(response.status_code, _body_of(response))

    async def _cancel_quietly(self, base_url: str, job_id: str) -> None:
        """Best-effort cancel from a path that is already unwinding.

        Run as a separate task and shielded, because an `await` in a task that is being
        cancelled is cancelled in turn — without this the DELETE would never leave the
        process, which is the one thing this call exists to guarantee.
        """
        cleanup = asyncio.create_task(self._swallow(self._cancel(base_url, job_id)))
        with contextlib.suppress(BaseException):
            await asyncio.shield(cleanup)

    @staticmethod
    async def _swallow(coro: Any) -> None:
        with contextlib.suppress(Exception):
            await coro


def _body_of(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text
