"""Failure modes a caller might reasonably want to tell apart.

Deliberately few. The distinctions that exist are the ones that change what a service
should do next: retry, give up, or resubmit.
"""

from __future__ import annotations

from typing import Any


class SirClientError(Exception):
    """Base for everything this package raises."""


class ModelNotRouted(SirClientError):
    """No endpoint is configured for the requested model.

    A configuration mistake, not a runtime condition — raised before anything is sent.
    """


class TransportError(SirClientError):
    """A non-2xx response that isn't a recognised job condition."""

    def __init__(self, status_code: int, body: Any) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class JobLost(SirClientError):
    """A job vanished mid-poll.

    Either its result expired before it was read, or the router restarted and lost the
    in-memory store. The two are indistinguishable from out here, which is why this is
    raised rather than silently resubmitted: replaying a request that may already have run
    is the caller's decision, not the SDK's. Pass `resubmit_on_loss=True` (with an
    idempotency key) to opt into it.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(
            f"job {job_id!r} is gone; it expired or the router restarted"
        )
        self.job_id = job_id


class JobFailed(SirClientError):
    """The router accepted the request and generation failed."""

    def __init__(self, job_id: str, error: Any) -> None:
        super().__init__(f"job {job_id!r} failed: {error}")
        self.job_id = job_id
        self.error = error


class JobCancelled(SirClientError):
    """The job was cancelled — by this client, by another, or by lease expiry."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id!r} was cancelled")
        self.job_id = job_id


class RequestTimeout(SirClientError):
    """The caller's deadline passed before the job finished.

    The job is cancelled on the way out, so a timeout stops costing GPU time rather than
    leaving orphaned work behind.
    """

    def __init__(self, job_id: str, timeout: float) -> None:
        super().__init__(f"job {job_id!r} did not finish within {timeout}s")
        self.job_id = job_id
        self.timeout = timeout
