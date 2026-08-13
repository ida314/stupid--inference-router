"""`sir-client` — call a model without caring whether it is loaded.

    from sir_client import run_llm

    completion = await run_llm(
        "Qwen/Qwen3-8B",
        {"messages": [{"role": "user", "content": "hello"}]},
    )

Endpoints come from `SIR_BASE_URL` (a catch-all) or `SIR_ENDPOINTS` (`model=url` pairs).
For a service that makes more than the occasional call, build an `AsyncClient` once and
reuse it — the module-level helpers exist for scripts and one-offs, and hold a shared
client bound to whichever event loop first used them.

Only an async client is provided. Every caller here is already inside an event loop, and a
synchronous wrapper that cannot be called from one is a footgun with no user.
"""

from __future__ import annotations

from typing import Any

from sir_client.client import AsyncClient, Job
from sir_client.errors import (
    JobCancelled,
    JobFailed,
    JobLost,
    ModelNotRouted,
    RequestTimeout,
    SirClientError,
    TransportError,
)
from sir_client.registry import ENV_BASE_URL, ENV_ENDPOINTS, Registry

__all__ = [
    "AsyncClient",
    "ENV_BASE_URL",
    "ENV_ENDPOINTS",
    "Job",
    "JobCancelled",
    "JobFailed",
    "JobLost",
    "ModelNotRouted",
    "Registry",
    "RequestTimeout",
    "SirClientError",
    "TransportError",
    "close_default_client",
    "run_llm",
    "submit_llm",
]

_default: AsyncClient | None = None


def default_client() -> AsyncClient:
    """The lazily-built shared client, configured from the environment."""
    global _default
    if _default is None:
        _default = AsyncClient()
    return _default


async def close_default_client() -> None:
    """Release the shared client. Worth calling from a service's shutdown hook."""
    global _default
    if _default is not None:
        await _default.aclose()
        _default = None


async def run_llm(
    model: str,
    body: dict[str, Any],
    *,
    timeout: float | None = None,
    idempotency_key: str | None = None,
    resubmit_on_loss: bool = False,
) -> dict[str, Any]:
    """Submit a request and return the completion, queueing transparently."""
    return await default_client().run(
        model,
        body,
        timeout=timeout,
        idempotency_key=idempotency_key,
        resubmit_on_loss=resubmit_on_loss,
    )


async def submit_llm(
    model: str,
    body: dict[str, Any],
    *,
    idempotency_key: str | None = None,
) -> Job:
    """Submit without waiting, for work to be collected later."""
    return await default_client().submit(
        model, body, idempotency_key=idempotency_key
    )
