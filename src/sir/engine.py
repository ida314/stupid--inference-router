"""The mechanism: one control loop that carries out whatever the policy decided.

Residency, drains, loads, dispatch, cancellation, and crash containment all live here. The
loop is deliberately the only thing that mutates scheduling state — request tasks report
failures by dropping a model name in a set, and the loop picks it up on its next pass.
Single writer, no locks, no half-applied swaps.

The invariants this file exists to keep:

- Exactly one model is resident at a time.
- A switch drains first. In-flight generation always finishes; nothing is cut mid-response.
- A backend crash fails that backend's in-flight requests and nothing else. The router
  stays up, the other model keeps being scheduled, and the dead model is retried after a
  backoff.
- A client that disconnects stops costing GPU time, and never delays a drain.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from sir.backend import Backend, BackendError
from sir.backends.mock import MockBackend
from sir.clock import Clock, RealClock
from sir.config import AppConfig, ModelConfig
from sir.policy import decide
from sir.queues import ModelQueues
from sir.types import (
    Decision,
    DecisionKind,
    EngineEvent,
    EventKind,
    GenerationRequest,
    ModelState,
    QueuedRequest,
    RequestState,
    SchedulerSnapshot,
    StreamEnd,
    StreamError,
)

BackendFactory = Callable[[ModelConfig, Clock], Backend]

_HISTORY = 200


def default_backend_factory(model: ModelConfig, clock: Clock) -> Backend:
    if model.backend == "mock":
        return MockBackend(model.name, model.mock, clock)
    raise ValueError(f"unsupported backend kind: {model.backend!r}")


class Engine:
    """Owns model residency and the queues in front of it."""

    def __init__(
        self,
        config: AppConfig,
        clock: Clock | None = None,
        backend_factory: BackendFactory = default_backend_factory,
        on_decision: Callable[[Decision], None] | None = None,
        on_event: Callable[[EngineEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or RealClock()
        self._backend_factory = backend_factory
        self._on_decision = on_decision
        self._on_event = on_event

        self.queues = ModelQueues(config.model_names)
        self._backends: dict[str, Backend] = {
            model.name: backend_factory(model, self.clock) for model in config.models
        }
        self._model_config = {model.name: model for model in config.models}

        self._resident: str | None = None
        self._resident_since: float | None = None
        self._idle_since: float | None = None
        self._unavailable_until: dict[str, float] = {}
        self._in_flight: dict[asyncio.Task[None], QueuedRequest] = {}
        self._failed: set[str] = set()

        self._draining = False
        self._switching = False
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._wakeup = asyncio.Event()

        self.decisions: deque[Decision] = deque(maxlen=_HISTORY)
        self.events: deque[EngineEvent] = deque(maxlen=_HISTORY)
        self.swaps = 0
        self.loads = 0
        self.time_loading = 0.0

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run(), name="sir-scheduler")

    async def stop(self) -> None:
        self._running = False
        self._wakeup.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None

        for task in list(self._in_flight):
            task.cancel()
        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)
        self._in_flight.clear()

        for name in self.queues.model_names:
            for request in self.queues.drain(name):
                request.state = RequestState.CANCELLED
                await request.events.put(StreamError("router shutting down", 503))

        if self._resident is not None:
            await self._backends[self._resident].stop()
            self._resident = None
            self._resident_since = None

    # ---------------------------------------------------------------- submission

    def submit(self, request: GenerationRequest) -> QueuedRequest:
        """Accept a request for any configured model, loaded or not.

        This never blocks and never waits for a swap — that's the point of the whole
        design. Clients address logical model names and stay ignorant of residency.
        """
        if request.model not in self.queues:
            raise KeyError(f"unknown model: {request.model!r}")

        queued = QueuedRequest(request=request, enqueued_at=self.clock.now())
        self.queues.enqueue(queued)
        self._emit(EventKind.ENQUEUE, request.model, detail=queued.id)
        self._wakeup.set()
        return queued

    # ---------------------------------------------------------------- control loop

    async def _run(self) -> None:
        try:
            while self._running:
                # Cleared before the snapshot is taken, so a request arriving at any
                # point after this line still wakes the loop rather than waiting a tick.
                self._wakeup.clear()

                await self._handle_failures()
                self.queues.purge_cancelled()
                self._cancel_disconnected()
                self._reap_finished()
                self._track_idleness()

                decision = decide(
                    self._snapshot(), self.clock.now(), self.config.scheduler
                )
                self._record(decision)

                if decision.kind is DecisionKind.SWITCH and decision.target:
                    await self._switch(decision.target)
                    continue

                if decision.kind in (DecisionKind.SERVE, DecisionKind.HOLD):
                    self._dispatch()

                await self._wait(self.config.scheduler.tick_interval_seconds)
        except asyncio.CancelledError:
            raise

    async def _wait(self, timeout: float) -> None:
        """Sleep until the tick expires or a request arrives, whichever comes first."""
        sleeper = asyncio.create_task(self.clock.sleep(timeout))
        waiter = asyncio.create_task(self._wakeup.wait())
        _, pending = await asyncio.wait(
            {sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ---------------------------------------------------------------- dispatch

    def _dispatch(self) -> None:
        """Hand queued work to the resident backend, up to the concurrency cap."""
        if self._resident is None:
            return
        capacity = self.config.scheduler.max_concurrent_requests - len(self._in_flight)
        if capacity <= 0:
            return

        backend = self._backends[self._resident]
        now = self.clock.now()
        for queued in self.queues.take_while(self._resident, now, capacity):
            self._emit(
                EventKind.DISPATCH,
                queued.model,
                detail=queued.id,
                duration=queued.wait_time(now),
            )
            task = asyncio.create_task(
                self._serve(backend, queued), name=f"sir-req-{queued.id}"
            )
            self._in_flight[task] = queued

    async def _serve(self, backend: Backend, queued: QueuedRequest) -> None:
        """Pump one request's chunks back to its client.

        Every exit path except client cancellation puts a terminator on the event queue —
        the API handler is blocked reading it and would otherwise hang forever.
        """
        try:
            async for chunk in backend.stream(queued.request):
                if queued.cancelled.is_set():
                    break
                await queued.events.put(chunk)

            if queued.cancelled.is_set():
                queued.state = RequestState.CANCELLED
            else:
                queued.state = RequestState.DONE
                await queued.events.put(StreamEnd("stop"))
                self._emit(EventKind.COMPLETE, queued.model, detail=queued.id)
        except asyncio.CancelledError:
            queued.state = RequestState.CANCELLED
            self._emit(EventKind.CANCELLED, queued.model, detail=queued.id)
            raise
        except BackendError as exc:
            queued.state = RequestState.FAILED
            # Reported here, acted on by the control loop — single writer.
            self._failed.add(queued.model)
            await queued.events.put(StreamError(str(exc), 503))
            self._emit(EventKind.FAILED, queued.model, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - a bug here must not kill the router
            queued.state = RequestState.FAILED
            await queued.events.put(StreamError(f"internal error: {exc}", 500))
            self._emit(EventKind.FAILED, queued.model, detail=repr(exc))
        finally:
            queued.finished_at = self.clock.now()

    def _cancel_disconnected(self) -> None:
        """Stop paying for clients that went away."""
        for task, queued in list(self._in_flight.items()):
            if queued.cancelled.is_set() and not task.done():
                task.cancel()

    def _reap_finished(self) -> None:
        for task in [task for task in self._in_flight if task.done()]:
            del self._in_flight[task]

    def _track_idleness(self) -> None:
        """Record when the resident model last ran completely dry.

        Hysteresis needs to distinguish a model that is between requests from one that
        has genuinely finished, and queue depth alone can't: a busy model's queue is
        empty most of the time because work is dispatched the instant it arrives.
        """
        if self._resident is None:
            self._idle_since = None
            return
        busy = self.queues.depth(self._resident) > 0 or bool(self._in_flight)
        if busy:
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = self.clock.now()

    # ---------------------------------------------------------------- transitions

    async def _switch(self, target: str) -> None:
        """Drain, unload, load, and hand the GPU to `target`."""
        await self._drain()

        self._switching = True
        try:
            if self._resident is not None:
                previous = self._resident
                self._resident = None
                self._resident_since = None
                await self._backends[previous].stop()
                self._emit(EventKind.UNLOAD, previous)
                self.swaps += 1

            backend = self._backends[target]
            started = self.clock.now()
            self._emit(EventKind.LOAD_START, target)
            try:
                await backend.start()
            except BackendError as exc:
                elapsed = self.clock.now() - started
                self.time_loading += elapsed
                self._emit(
                    EventKind.LOAD_FAILED, target, detail=str(exc), duration=elapsed
                )
                await backend.stop()
                self._mark_unavailable(target)
                return

            elapsed = self.clock.now() - started
            self.loads += 1
            self.time_loading += elapsed
            self._resident = target
            self._resident_since = self.clock.now()
            self._idle_since = None
            self._unavailable_until.pop(target, None)
            self._emit(EventKind.LOAD_END, target, duration=elapsed)
        finally:
            self._switching = False

    async def _drain(self) -> None:
        """Let in-flight generation finish. Nothing is killed mid-response.

        Requests whose clients already disconnected are cancelled first, so a departed
        client can't hold the GPU hostage while everyone else waits for the swap.
        """
        if not self._in_flight:
            return

        self._draining = True
        started = self.clock.now()
        self._emit(EventKind.DRAIN_START, self._resident or "-", detail=f"{len(self._in_flight)} in flight")
        try:
            self._cancel_disconnected()
            await asyncio.gather(*self._in_flight, return_exceptions=True)
            self._in_flight.clear()
        finally:
            self._draining = False
            self._emit(
                EventKind.DRAIN_END,
                self._resident or "-",
                duration=self.clock.now() - started,
            )

    async def _handle_failures(self) -> None:
        """Contain a crashed backend without taking the router down with it."""
        if not self._failed:
            return

        failed = self._failed
        self._failed = set()
        for model in failed:
            self._mark_unavailable(model)
            self._emit(EventKind.CRASH, model, detail="backend marked unavailable")

            if self._resident != model:
                continue

            # The resident process is gone. Abandon its in-flight work — it cannot
            # succeed — stop the backend, and let the policy pick a fresh model.
            for task, queued in list(self._in_flight.items()):
                if queued.model == model and not task.done():
                    task.cancel()
            if self._in_flight:
                await asyncio.gather(*self._in_flight, return_exceptions=True)
                self._in_flight.clear()

            self._resident = None
            self._resident_since = None
            await self._backends[model].stop()
            self._emit(EventKind.UNLOAD, model, detail="after crash")

    def _mark_unavailable(self, model: str) -> None:
        self._unavailable_until[model] = (
            self.clock.now() + self.config.scheduler.backend_retry_seconds
        )

    # ---------------------------------------------------------------- introspection

    def _snapshot(self) -> SchedulerSnapshot:
        states = tuple(
            ModelState(
                name=model.name,
                priority=model.priority,
                depth=self.queues.depth(model.name),
                oldest_enqueued_at=self.queues.oldest_enqueued_at(model.name),
                estimated_load_seconds=model.estimated_load_seconds,
                resident=model.name == self._resident,
                resident_since=(
                    self._resident_since if model.name == self._resident else None
                ),
                in_flight=(
                    len(self._in_flight) if model.name == self._resident else 0
                ),
                idle_since=(
                    self._idle_since if model.name == self._resident else None
                ),
                unavailable_until=self._unavailable_until.get(model.name),
            )
            for model in self.config.models
        )
        return SchedulerSnapshot(
            models=states,
            resident=self._resident,
            draining=self._draining,
            switching=self._switching,
        )

    def _record(self, decision: Decision) -> None:
        # Only log a decision when it says something new — an idle router ticking four
        # times a second must not bury the one line that explains a swap.
        previous = self.decisions[-1] if self.decisions else None
        self.decisions.append(decision)
        if (
            previous is None
            or previous.kind is not decision.kind
            or previous.reason is not decision.reason
            or previous.target != decision.target
            or previous.resident != decision.resident
        ):
            if self._on_decision is not None:
                self._on_decision(decision)

    def _emit(
        self,
        kind: EventKind,
        model: str,
        detail: str = "",
        duration: float | None = None,
    ) -> None:
        event = EngineEvent(
            at=self.clock.now(), kind=kind, model=model, detail=detail, duration=duration
        )
        self.events.append(event)
        if self._on_event is not None:
            self._on_event(event)

    @property
    def is_running(self) -> bool:
        """Is the control loop alive? A crashed backend must never make this False."""
        return self._loop_task is not None and not self._loop_task.done()

    @property
    def resident(self) -> str | None:
        return self._resident

    @property
    def resident_since(self) -> float | None:
        return self._resident_since

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    def status(self) -> dict[str, Any]:
        """The debug surface behind `GET /status`."""
        now = self.clock.now()
        last = self.decisions[-1] if self.decisions else None
        return {
            "now": round(now, 3),
            "resident": self._resident,
            "resident_for_seconds": (
                round(now - self._resident_since, 3)
                if self._resident_since is not None
                else None
            ),
            "idle_for_seconds": (
                round(now - self._idle_since, 3)
                if self._idle_since is not None
                else None
            ),
            "draining": self._draining,
            "switching": self._switching,
            "in_flight": len(self._in_flight),
            "swaps": self.swaps,
            "loads": self.loads,
            "time_loading_seconds": round(self.time_loading, 3),
            "models": [
                {
                    "name": model.name,
                    "priority": model.priority,
                    "queue_depth": self.queues.depth(model.name),
                    "oldest_wait_seconds": round(
                        now - (self.queues.oldest_enqueued_at(model.name) or now), 3
                    ),
                    "resident": model.name == self._resident,
                    "available_in_seconds": max(
                        0.0,
                        round(self._unavailable_until.get(model.name, now) - now, 3),
                    ),
                }
                for model in self.config.models
            ],
            "last_decision": (
                {
                    "kind": last.kind.value,
                    "reason": last.reason.value,
                    "target": last.target,
                    "at": round(last.at, 3),
                    "scores": [score.format() for score in last.scores],
                }
                if last
                else None
            ),
        }
