"""Held results for clients that would rather poll than hold a socket open.

The router's one dispatch path already has two renderings: `_collect` accumulates the
event stream into a single response, `_sse` forwards it chunk by chunk. This is the third.
A job drains the same `QueuedRequest.events` channel into a buffer that outlives the
request that created it, so the client can come back for it later.

Nothing here schedules anything, and nothing here reaches into the engine. The one thing a
job can do to the scheduler is cancel its own request — through `QueuedRequest.cancel()`,
the same call the disconnect watcher makes — which the control loop picks up on its next
pass. Single writer, unchanged.

The lease is why that matters. Today an abandoned request is noticed because its socket
closed; a polled job has no socket, so polling itself becomes the liveness signal. Without
it, a service that restarts leaves queued work behind that goes on competing for the GPU
indefinitely — the thing `ModelQueues` is careful to prevent for cancelled requests.

What the lease buys, precisely: it bounds how long that can go on. Since it is shorter than
`scheduler.max_wait_seconds`, abandoned work is cancelled before it reaches the starvation
ceiling, where a swap stops being a judgement call and becomes mandatory. It does not
prevent a *scored* swap — a challenger can out-score an idle incumbent within seconds of
arriving, and no lease short enough to catch that would be long enough to be safe. The
guarantee is a ceiling on waste, not immunity from it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sir.clock import Clock
from sir.config import JobsConfig
from sir.schemas import ChatCompletionRequest
from sir.types import Chunk, QueuedRequest, RequestState, StreamEnd, StreamError


class JobStoreFull(RuntimeError):
    """Raised when the retention ceiling is reached and nothing can be evicted."""


@dataclass
class Job:
    """One asynchronously-submitted request, from acceptance to expiry."""

    queued: QueuedRequest
    body: ChatCompletionRequest
    created_at: float
    last_polled_at: float
    idempotency_key: str | None = None

    # Chunks are kept whole rather than concatenated. Costs nothing, and it is what would
    # let `?since=<index>` stream a job over polling later without changing this shape.
    chunks: list[Chunk] = field(default_factory=list)
    terminal: StreamEnd | StreamError | None = None

    finished_at: float | None = None
    first_read_at: float | None = None
    pump: asyncio.Task[None] | None = None

    @property
    def id(self) -> str:
        return self.queued.id

    @property
    def state(self) -> RequestState:
        """Derived, never stored — there is only one state machine and the engine owns it."""
        if isinstance(self.terminal, StreamError):
            return RequestState.FAILED
        if isinstance(self.terminal, StreamEnd):
            return RequestState.DONE
        if self.queued.cancelled.is_set():
            return RequestState.CANCELLED
        return self.queued.state

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            RequestState.DONE,
            RequestState.FAILED,
            RequestState.CANCELLED,
        )

    @property
    def texts(self) -> list[str]:
        return [chunk.text for chunk in self.chunks]

    @property
    def finish_reason(self) -> str:
        return self.terminal.finish_reason if isinstance(self.terminal, StreamEnd) else "stop"


class JobStore:
    """Buffers job results, expires them, and cancels the ones nobody is waiting for."""

    def __init__(self, config: JobsConfig, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._sweeper: asyncio.Task[None] | None = None
        self._running = False

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._sweeper = asyncio.create_task(self._sweep(), name="sir-job-sweeper")

    async def stop(self) -> None:
        self._running = False
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
            self._sweeper = None

        pumps = [job.pump for job in self._jobs.values() if job.pump is not None]
        for pump in pumps:
            pump.cancel()
        if pumps:
            await asyncio.gather(*pumps, return_exceptions=True)
        self._jobs.clear()
        self._by_key.clear()

    # ---------------------------------------------------------------- submission

    def create(
        self,
        queued: QueuedRequest,
        body: ChatCompletionRequest,
        idempotency_key: str | None = None,
    ) -> Job:
        """Start buffering `queued`'s output under a job handle."""
        now = self.clock.now()
        job = Job(
            queued=queued,
            body=body,
            created_at=now,
            last_polled_at=now,
            idempotency_key=idempotency_key,
        )
        self._make_room()
        self._jobs[job.id] = job
        if idempotency_key is not None:
            self._by_key[idempotency_key] = job.id
        job.pump = asyncio.create_task(self._drain(job), name=f"sir-job-{job.id}")
        return job

    def find_by_key(self, idempotency_key: str) -> Job | None:
        """Resolve a repeated submission to the job it already created.

        Without this, an SDK that resubmits after a network blip pays for the same
        generation twice and only ever sees one of the answers.
        """
        job_id = self._by_key.get(idempotency_key)
        return self._jobs.get(job_id) if job_id else None

    # ---------------------------------------------------------------- reads

    def get(self, job_id: str) -> Job | None:
        """Fetch a job and renew its lease. The only thing that keeps a job alive."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        now = self.clock.now()
        job.last_polled_at = now
        if job.is_terminal and job.first_read_at is None:
            job.first_read_at = now
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        self._abandon(job)
        return job

    # ---------------------------------------------------------------- internals

    async def _drain(self, job: Job) -> None:
        """Move one request's events out of its queue and into the job's buffer.

        The mirror of `_collect`, except that nothing is waiting on the far end. Stops on
        the first terminal event; a cancelled request never produces one, which is why
        `_abandon` cancels this task rather than waiting for it to end on its own.
        """
        try:
            while True:
                event = await job.queued.events.get()
                if isinstance(event, Chunk):
                    job.chunks.append(event)
                    continue
                job.terminal = event
                break
        except asyncio.CancelledError:
            raise
        finally:
            job.finished_at = self.clock.now()

    def _abandon(self, job: Job) -> None:
        """Cancel a job's request and stop pumping it. Idempotent."""
        job.queued.cancel()
        if job.pump is not None and not job.pump.done():
            job.pump.cancel()
        if job.finished_at is None:
            job.finished_at = self.clock.now()

    def _make_room(self) -> None:
        if len(self._jobs) < self.config.max_jobs:
            return
        # Finished jobs are evictable; in-flight ones are not. Dropping a running job
        # would strand a client that is still polling for an answer it is going to get.
        evictable = sorted(
            (job for job in self._jobs.values() if job.is_terminal),
            key=lambda job: job.finished_at or job.created_at,
        )
        if not evictable:
            raise JobStoreFull(
                f"{len(self._jobs)} jobs in flight, at the configured max_jobs ceiling"
            )
        self._forget(evictable[0])

    def _forget(self, job: Job) -> None:
        self._jobs.pop(job.id, None)
        if job.idempotency_key is not None:
            self._by_key.pop(job.idempotency_key, None)

    async def _sweep(self) -> None:
        """Expire finished results and cancel jobs nobody is polling for."""
        try:
            while self._running:
                await self.clock.sleep(self.config.sweep_interval_seconds)
                self.sweep_once()
        except asyncio.CancelledError:
            raise

    def sweep_once(self) -> None:
        """One expiry pass. Separate from the loop so tests can drive it directly."""
        now = self.clock.now()
        lease = self.config.lease_seconds

        for job in list(self._jobs.values()):
            if (
                lease
                and not job.is_terminal
                and now - job.last_polled_at > lease
            ):
                self._abandon(job)
                continue

            if not job.is_terminal or job.finished_at is None:
                continue

            # A read result is on a shorter clock than an unread one: the client has what
            # it asked for, and only needs the window to survive a retry.
            if job.first_read_at is not None:
                expires_at = job.first_read_at + self.config.read_ttl_seconds
            else:
                expires_at = job.finished_at + self.config.result_ttl_seconds
            if now >= expires_at:
                self._forget(job)

    def __len__(self) -> int:
        return len(self._jobs)
