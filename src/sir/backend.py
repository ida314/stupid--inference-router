"""The policy/mechanism boundary.

Above this line the scheduler decides *what should run*. Below it, a backend decides *how*.
The scheduler never touches a process, a port, or a CUDA context — so replacing the mock
with vLLM, or vLLM with Triton, changes nothing above this file.

Keeping `stream` as the only generation primitive is deliberate: a non-streaming response
is the stream collected. One dispatch path means drain, cancellation, and error handling
have one implementation each instead of two that drift.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from sir.types import Chunk, GenerationRequest


class BackendError(RuntimeError):
    """A backend failed in a way the engine should treat as a crash.

    Raised on load failure or mid-generation death. The engine's job on seeing this is to
    fail the affected requests, mark the model unavailable for a backoff period, and keep
    scheduling everything else. The router does not go down with a backend.
    """


@runtime_checkable
class Backend(Protocol):
    """One model's serving process. Exactly one backend is started at a time in v1."""

    @property
    def model_name(self) -> str: ...

    async def start(self) -> None:
        """Load weights and become ready to serve. Raises `BackendError` on failure."""
        ...

    async def stop(self) -> None:
        """Unload and free memory. Must be safe to call on a crashed backend."""
        ...

    async def health(self) -> bool:
        """Is this backend able to serve right now?"""
        ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[Chunk]:
        """Generate a response as a sequence of chunks.

        Must be cancellable between chunks: the engine cancels this task when a client
        disconnects. Raises `BackendError` if the backend dies mid-generation.
        """
        ...
