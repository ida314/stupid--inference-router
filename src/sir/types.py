"""The vocabulary shared by the queues, the policy, the engine, and the API.

Nothing here does work. These are the shapes that let the pure policy talk to the async
mechanism without either importing the other.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """A client request, carried in the shape the backend will actually receive.

    `payload` is the client's own JSON body, forwarded verbatim except that `model` is
    rewritten to the tag the backend serves. `sir` deliberately does not model the
    sampling parameters: vLLM and SGLang each add their own non-standard fields, those
    sets drift between releases, and a router that parses them would have to be updated
    every time one does. Whatever it doesn't route on, it doesn't interpret.
    """

    model: str
    """Internal scheduler key — the config's `name`. Never appears on the wire."""

    served_model: str
    """The tag the client asked for, echoed back so responses match the request."""

    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    @property
    def max_tokens(self) -> int | None:
        """The token budget, if the client set one. Read, but still passed through."""
        value = self.payload.get("max_completion_tokens")
        if value is None:
            value = self.payload.get("max_tokens")
        return int(value) if value is not None else None

    @property
    def stream(self) -> bool:
        return bool(self.payload.get("stream", False))


@dataclass(frozen=True)
class Chunk:
    """One increment of generated text."""

    text: str
    index: int


@dataclass(frozen=True)
class StreamEnd:
    finish_reason: str = "stop"


@dataclass(frozen=True)
class StreamError:
    message: str
    status_code: int = 503


StreamEvent = Chunk | StreamEnd | StreamError


class RequestState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedRequest:
    """A client request from arrival to completion.

    `events` is the single channel back to the API handler — it carries chunks whether or
    not the client asked for streaming, because a non-streaming response is just the
    stream collected. One dispatch path, not two.
    """

    request: GenerationRequest
    enqueued_at: float
    events: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    state: RequestState = RequestState.QUEUED
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def model(self) -> str:
        return self.request.model

    def wait_time(self, now: float) -> float:
        """How long the client has been waiting to be *dispatched*."""
        return (self.started_at if self.started_at is not None else now) - self.enqueued_at

    def cancel(self) -> None:
        self.cancelled.set()


# --------------------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelState:
    """Everything the policy is allowed to know about one model."""

    name: str
    priority: float
    depth: int
    oldest_enqueued_at: float | None
    estimated_load_seconds: float
    resident: bool = False
    resident_since: float | None = None
    in_flight: int = 0
    # When this model last ran dry — nothing queued, nothing generating. `None` means it
    # has work right now. Hysteresis uses this to tell "briefly between requests" apart
    # from "genuinely finished", which a queue-depth reading alone cannot do.
    idle_since: float | None = None
    # Set after a crash or failed load; the policy will not choose this model until then.
    unavailable_until: float | None = None

    @property
    def has_pending(self) -> bool:
        return self.depth > 0

    @property
    def is_busy(self) -> bool:
        """Queued *or* generating. A model streaming its last response is not idle."""
        return self.depth > 0 or self.in_flight > 0

    def idle_for(self, now: float) -> float:
        if self.idle_since is None:
            return 0.0
        return max(0.0, now - self.idle_since)

    def oldest_wait(self, now: float) -> float:
        if self.oldest_enqueued_at is None:
            return 0.0
        return max(0.0, now - self.oldest_enqueued_at)

    def is_available(self, now: float) -> bool:
        return self.unavailable_until is None or now >= self.unavailable_until


@dataclass(frozen=True)
class SchedulerSnapshot:
    """A consistent read of the world, handed to `decide()`."""

    models: tuple[ModelState, ...]
    resident: str | None = None
    draining: bool = False
    switching: bool = False

    def by_name(self, name: str) -> ModelState | None:
        for model in self.models:
            if model.name == name:
                return model
        return None

    @property
    def resident_state(self) -> ModelState | None:
        return self.by_name(self.resident) if self.resident else None


class DecisionKind(StrEnum):
    IDLE = "idle"
    """No pending work anywhere. The resident model stays loaded — unloading buys nothing."""

    SERVE = "serve"
    """Dispatch the resident model's queue."""

    HOLD = "hold"
    """Something else wants the GPU, but the resident model keeps it."""

    SWITCH = "switch"
    """Drain, unload, load `target`, then serve it."""


class DecisionReason(StrEnum):
    NO_WORK = "no_work"
    COLD_START = "cold_start"
    STARVATION = "starvation"
    MIN_RESIDENCY = "min_residency"
    RESIDENT_WINS = "resident_wins"
    CHALLENGER_WINS = "challenger_wins"
    RESIDENT_IDLE = "resident_idle"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    IN_TRANSITION = "in_transition"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Why a model scored what it scored. Logged verbatim; this is the audit trail."""

    model: str
    priority: float
    depth: int
    oldest_wait: float
    age_term: float
    depth_term: float
    switch_cost: float
    score: float
    eligible: bool = True
    note: str = ""

    def format(self) -> str:
        if not self.eligible:
            return f"{self.model} — ({self.note})"
        if self.depth == 0:
            return f"{self.model} 0.0 (idle)"
        cost = f" - switch {self.switch_cost:.1f}" if self.switch_cost else ""
        return (
            f"{self.model} {self.score:.1f} "
            f"[prio {self.priority:g} x (age {self.age_term:.1f} "
            f"+ depth {self.depth_term:.1f}){cost}]"
        )


class EventKind(StrEnum):
    """Lifecycle events, distinct from scheduling decisions.

    A `Decision` is what the policy chose; an `EngineEvent` is what the mechanism actually
    did about it, and when. Both get logged; together they explain any minute of runtime.
    """

    ENQUEUE = "enqueue"
    DISPATCH = "dispatch"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DRAIN_START = "drain_start"
    DRAIN_END = "drain_end"
    LOAD_START = "load_start"
    LOAD_END = "load_end"
    LOAD_FAILED = "load_failed"
    UNLOAD = "unload"
    CRASH = "crash"


@dataclass(frozen=True)
class EngineEvent:
    at: float
    kind: EventKind
    model: str
    detail: str = ""
    duration: float | None = None

    def format(self) -> str:
        line = f"t={self.at:7.1f}  {self.kind.value:<11} {self.model:<12}"
        if self.duration is not None:
            line += f" {self.duration:.1f}s"
        if self.detail:
            line += f"  {self.detail}"
        return line


@dataclass(frozen=True)
class Decision:
    """The output of the policy: what to do, and the full arithmetic behind it."""

    kind: DecisionKind
    reason: DecisionReason
    at: float
    resident: str | None = None
    target: str | None = None
    scores: tuple[ScoreBreakdown, ...] = ()

    @property
    def is_swap(self) -> bool:
        return self.kind is DecisionKind.SWITCH and self.resident is not None

    def format(self) -> str:
        """One human-readable line. The README's 'read a log line and understand why'."""
        head = f"t={self.at:7.1f}  {self.kind.value.upper():<6}"
        if self.kind is DecisionKind.SWITCH:
            move = f"{self.resident} -> {self.target}" if self.resident else f"{self.target}"
            head += f" {move:<24}"
        else:
            head += f" {self.resident or '-':<24}"
        return f"{head} {self.reason.value}"
