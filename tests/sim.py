"""A traffic simulator for the scheduler.

The test suite asserts invariants — nothing starves, alternating traffic doesn't thrash.
Those are pass/fail. This is the other half: run a workload and see the *shape* of the
trade-off the current weights produce. Per-model wait percentiles, swap count, and how
much of the wall clock the GPU spent loading instead of generating.

Every number here is downstream of the mock's `load_seconds` and `tokens_per_second`,
which are guesses until Phase 3 measures a real swap. Treat it as a sanity check on the
policy's shape, not as a tuning oracle.

Not a test file. Used by tests, and printed with `pytest -s`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import groupby

from sir.clock import VirtualClock
from sir.config import AppConfig, JobsConfig, ModelConfig, MockParams, SchedulerConfig
from sir.engine import Engine
from sir.types import (
    Decision,
    DecisionKind,
    EngineEvent,
    EventKind,
    GenerationRequest,
    QueuedRequest,
    RequestState,
)

_TIMELINE_EVENTS = {
    EventKind.LOAD_START,
    EventKind.LOAD_END,
    EventKind.LOAD_FAILED,
    EventKind.UNLOAD,
    EventKind.DRAIN_START,
    EventKind.DRAIN_END,
    EventKind.CRASH,
}


def model(name: str, priority: float = 1.0, **mock: object) -> ModelConfig:
    return ModelConfig(name=name, priority=priority, mock=MockParams(**mock))


def generation(
    name: str,
    *,
    max_tokens: int | None = None,
    served_model: str | None = None,
    content: str = "simulated request",
    **extras: object,
) -> GenerationRequest:
    """Build a request the way the API layer would, for tests that skip HTTP.

    `extras` land in the payload untouched, which is how pass-through gets exercised
    below the API — the scheduler must not care what's in there.
    """
    tag = served_model or name
    payload: dict[str, object] = {
        "model": tag,
        "messages": [{"role": "user", "content": content}],
        **extras,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return GenerationRequest(model=name, served_model=tag, payload=payload)


def build_config(
    models: list[ModelConfig] | None = None,
    jobs: JobsConfig | None = None,
    **scheduler: object,
) -> AppConfig:
    """A two-model config with fast defaults, overridable per test.

    Tests build configs in code rather than from YAML so each scenario states exactly the
    scheduler constants it depends on. `test_config.py` covers the YAML path.
    """
    if models is None:
        models = [
            model("chat", priority=1.0, load_seconds=8, tokens_per_second=40),
            model("translate", priority=1.0, load_seconds=6, tokens_per_second=40),
        ]
    return AppConfig(
        models=models,
        scheduler=SchedulerConfig(**scheduler),
        jobs=jobs or JobsConfig(),
    )


@asynccontextmanager
async def running_engine(
    config: AppConfig, clock: VirtualClock, **kwargs: object
) -> AsyncIterator[Engine]:
    """An engine with its control loop running, for tests that drive it by hand."""
    engine = Engine(config, clock=clock, **kwargs)
    await engine.start()
    try:
        yield engine
    finally:
        await clock.drive(engine.stop())


@dataclass(frozen=True)
class Arrival:
    """One client request, fired at `at` seconds into the simulation."""

    at: float
    model: str
    max_tokens: int | None = None


def alternating(
    models: tuple[str, str], every: float, count: int, start: float = 0.0
) -> list[Arrival]:
    """A B A B traffic — the pattern the README says must not produce a swap per request."""
    return [
        Arrival(at=start + index * every, model=models[index % 2])
        for index in range(count)
    ]


def steady(model: str, every: float, count: int, start: float = 0.0, **kw: object) -> list[Arrival]:
    """A continuous stream for one model, used to try to starve another."""
    return [Arrival(at=start + index * every, model=model, **kw) for index in range(count)]  # type: ignore[arg-type]


@dataclass
class SimResult:
    config: AppConfig
    requests: list[QueuedRequest]
    decisions: list[Decision]
    events: list[EngineEvent]
    swaps: int
    loads: int
    duration: float
    time_loading: float
    idle_fraction: float = field(default=0.0)

    # -------------------------------------------------------------- queries

    def for_model(self, model: str) -> list[QueuedRequest]:
        return [req for req in self.requests if req.model == model]

    def waits(self, model: str) -> list[float]:
        """Time from arrival to dispatch — what a client actually experiences as delay."""
        return [
            req.wait_time(self.duration)
            for req in self.for_model(model)
            if req.started_at is not None
        ]

    def max_wait(self, model: str | None = None) -> float:
        models = [model] if model else self.config.model_names
        waits = [w for name in models for w in self.waits(name)]
        return max(waits) if waits else 0.0

    def unserved(self, model: str | None = None) -> list[QueuedRequest]:
        pool = self.for_model(model) if model else self.requests
        return [req for req in pool if req.started_at is None]

    def completed(self, model: str | None = None) -> list[QueuedRequest]:
        pool = self.for_model(model) if model else self.requests
        return [req for req in pool if req.state is RequestState.DONE]

    def switches(self) -> list[Decision]:
        return [d for d in self.decisions if d.kind is DecisionKind.SWITCH]

    def residency_order(self) -> list[str]:
        """The sequence of models that actually became resident."""
        return [e.model for e in self.events if e.kind is EventKind.LOAD_END]

    # -------------------------------------------------------------- reporting

    def timeline(self) -> str:
        lines: list[tuple[float, str]] = [
            (d.at, d.format()) for d in self.decisions
        ]
        lines += [
            (e.at, "          " + e.format().split("  ", 1)[1])
            for e in self.events
            if e.kind in _TIMELINE_EVENTS
        ]
        lines.sort(key=lambda item: item[0])
        return "\n".join(text for _, text in lines)

    def summary(self) -> str:
        rows = []
        for name in self.config.model_names:
            waits = sorted(self.waits(name))
            served = len(self.completed(name))
            total = len(self.for_model(name))
            if waits:
                rows.append(
                    f"  {name:<12} wait p50 {_pct(waits, 50):6.1f}s  "
                    f"p99 {_pct(waits, 99):6.1f}s  max {waits[-1]:6.1f}s  "
                    f"served {served}/{total}"
                )
            else:
                rows.append(f"  {name:<12} never served ({total} queued)")
        return "\n".join(
            [
                *rows,
                f"  swaps {self.swaps}   loads {self.loads}   "
                f"loading {self.time_loading:.1f}s   "
                f"gpu idle {self.idle_fraction * 100:.0f}%   "
                f"elapsed {self.duration:.1f}s",
            ]
        )

    def report(self) -> str:
        return f"\n{self.timeline()}\n\n{self.summary()}\n"


async def simulate(
    config: AppConfig,
    arrivals: list[Arrival],
    until: float,
    clock: VirtualClock | None = None,
) -> SimResult:
    """Run a workload against the mock backend in virtual time."""
    clock = clock or VirtualClock()
    decisions: list[Decision] = []
    events: list[EngineEvent] = []

    engine = Engine(
        config, clock=clock, on_decision=decisions.append, on_event=events.append
    )
    await engine.start()

    submitted: list[QueuedRequest] = []
    try:
        # Arrivals sharing a timestamp are submitted as a batch. Advancing between them
        # would let the scheduler decide on a partial queue, which is an artifact of the
        # harness rather than anything a real client would produce.
        for at, batch in groupby(sorted(arrivals, key=_at), key=_at):
            await clock.advance_to(at)
            for arrival in batch:
                submitted.append(
                    engine.submit(
                        generation(arrival.model, max_tokens=arrival.max_tokens)
                    )
                )
        await clock.advance_to(until)
        # Clients are never reading, so nothing drains the per-request event queues; that
        # is fine, they're unbounded. Requests still complete normally.
    finally:
        # Shutdown unloads a backend, which costs simulated seconds no `advance()` call
        # is left to supply — so the clock has to drive it.
        await clock.drive(engine.stop())

    duration = clock.now()
    busy = _union_duration(
        [
            (req.started_at, req.finished_at or duration)
            for req in submitted
            if req.started_at is not None
        ]
        + [
            (event.at - (event.duration or 0.0), event.at)
            for event in events
            if event.kind in (EventKind.LOAD_END, EventKind.LOAD_FAILED)
        ]
    )

    return SimResult(
        config=config,
        requests=submitted,
        decisions=decisions,
        events=events,
        swaps=engine.swaps,
        loads=engine.loads,
        duration=duration,
        time_loading=engine.time_loading,
        idle_fraction=max(0.0, 1.0 - busy / duration) if duration else 0.0,
    )


def _at(arrival: Arrival) -> float:
    return arrival.at


def _pct(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        int(round((percentile / 100.0) * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by overlapping intervals, counted once."""
    spans = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    if not spans:
        return 0.0
    total = 0.0
    current_lo, current_hi = spans[0]
    for lo, hi in spans[1:]:
        if lo > current_hi:
            total += current_hi - current_lo
            current_lo, current_hi = lo, hi
        else:
            current_hi = max(current_hi, hi)
    return total + (current_hi - current_lo)
