"""A simulated inference backend.

Its purpose is to make GPU behavior *configurable* rather than real: a load that takes
eight seconds, generation that trickles out at forty tokens a second, a process that dies
on its third request. That's what lets the scheduler's interesting properties — no thrash,
no starvation, clean drains, crash survival — be proven before any weights are downloaded.

Every wait goes through the injected clock, so the whole suite runs in virtual time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sir.backend import BackendError
from sir.clock import Clock
from sir.config import MockParams
from sir.types import Chunk, GenerationRequest

_VOCAB = (
    "the quick brown fox jumps over the lazy dog while the router "
    "decides which model deserves the gpu right now"
).split()


class MockBackend:
    """Stands in for vLLM. Deterministic, instrumented, and able to fail on cue."""

    def __init__(self, model_name: str, params: MockParams, clock: Clock) -> None:
        self._model_name = model_name
        self._params = params
        self._clock = clock

        self._loaded = False
        self._crashed = False
        self.load_attempts = 0
        self.requests_started = 0
        self.requests_completed = 0

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def start(self) -> None:
        self.load_attempts += 1
        await self._clock.sleep(self._params.load_seconds)

        every = self._params.fail_on_load_every_n
        if every and self.load_attempts % every == 0:
            raise BackendError(
                f"mock backend for {self._model_name!r} failed to load "
                f"(injected failure on attempt {self.load_attempts})"
            )

        self._loaded = True
        self._crashed = False
        # A reload is a fresh process, so the crash counter starts over.
        self.requests_started = 0

    async def stop(self) -> None:
        # Must tolerate being called on a crashed backend — that's the common case.
        if self._loaded or self._crashed:
            await self._clock.sleep(self._params.unload_seconds)
        self._loaded = False
        self._crashed = False

    async def health(self) -> bool:
        return self._loaded and not self._crashed

    async def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk]:
        if self._crashed:
            raise BackendError(f"backend for {self._model_name!r} has crashed")
        if not self._loaded:
            raise BackendError(f"backend for {self._model_name!r} is not loaded")

        self.requests_started += 1
        crash_after = self._params.crash_after_n_requests
        should_crash = bool(crash_after) and self.requests_started > crash_after

        await self._clock.sleep(self._params.first_token_seconds)

        budget = request.max_tokens or self._params.default_max_tokens
        interval = 1.0 / self._params.tokens_per_second
        # Echo the tag the request was routed to, so the response shows what a real
        # backend would have generated from rather than `sir`'s internal label.
        tag = str(request.payload.get("model") or self._model_name)
        # Die a couple of tokens in, so the test exercises a *mid-stream* failure and not
        # merely a rejected request.
        crash_at = 2 if should_crash else None

        for index in range(budget):
            if crash_at is not None and index >= crash_at:
                self._crashed = True
                self._loaded = False
                raise BackendError(
                    f"backend for {self._model_name!r} died mid-generation "
                    f"(injected crash after {crash_after} request(s))"
                )
            yield Chunk(text=_token(tag, index), index=index)
            await self._clock.sleep(interval)

        self.requests_completed += 1


def _token(model: str, index: int) -> str:
    """Deterministic filler text, prefixed so responses identify their model."""
    if index == 0:
        return f"[{model}]"
    return " " + _VOCAB[(index - 1) % len(_VOCAB)]
