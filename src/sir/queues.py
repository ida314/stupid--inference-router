"""Per-model FIFO queues, and the wait-age accounting the policy scores on.

One queue per model, not one queue for the router. That is the whole point: scheduling
happens at the model level first, and a request's position only matters once its model has
won the GPU.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from sir.types import QueuedRequest, RequestState


class ModelQueues:
    """FIFO per model, with cancelled requests treated as though they were never there.

    Cancelled entries are filtered out of every read rather than only being purged on a
    timer: a client that disconnected must not be able to trigger a model swap on its way
    out the door.
    """

    def __init__(self, model_names: Iterable[str]) -> None:
        self._queues: dict[str, deque[QueuedRequest]] = {
            name: deque() for name in model_names
        }

    def __contains__(self, model: str) -> bool:
        return model in self._queues

    @property
    def model_names(self) -> list[str]:
        return list(self._queues)

    def enqueue(self, request: QueuedRequest) -> None:
        if request.model not in self._queues:
            raise KeyError(f"unknown model: {request.model!r}")
        self._queues[request.model].append(request)

    def depth(self, model: str) -> int:
        return sum(1 for req in self._queues[model] if not req.cancelled.is_set())

    def oldest_enqueued_at(self, model: str) -> float | None:
        for req in self._queues[model]:
            if not req.cancelled.is_set():
                return req.enqueued_at
        return None

    def position_of(self, request: QueuedRequest) -> int | None:
        """Where `request` sits in its model's queue, counting only live entries.

        Cancelled entries ahead of it are skipped for the same reason `depth` skips them:
        they will never be dispatched, so counting them would overstate the wait. `None`
        once the request has been popped — by then it is running, not waiting.
        """
        position = 0
        for queued in self._queues.get(request.model, ()):
            if queued is request:
                return position
            if not queued.cancelled.is_set():
                position += 1
        return None

    def total_pending(self) -> int:
        return sum(self.depth(name) for name in self._queues)

    def models_with_work(self) -> list[str]:
        return [name for name in self._queues if self.depth(name) > 0]

    def next_for(self, model: str, now: float) -> QueuedRequest | None:
        """Pop the oldest live request for `model`, dropping cancelled ones on the way."""
        queue = self._queues[model]
        while queue:
            request = queue.popleft()
            if request.cancelled.is_set():
                request.state = RequestState.CANCELLED
                continue
            request.state = RequestState.RUNNING
            request.started_at = now
            return request
        return None

    def take_while(self, model: str, now: float, limit: int) -> Iterator[QueuedRequest]:
        """Pop up to `limit` live requests for `model`."""
        for _ in range(max(0, limit)):
            request = self.next_for(model, now)
            if request is None:
                return
            yield request

    def purge_cancelled(self) -> list[QueuedRequest]:
        """Drop cancelled requests still sitting in queues. Returns what was removed."""
        removed: list[QueuedRequest] = []
        for name, queue in self._queues.items():
            if not any(req.cancelled.is_set() for req in queue):
                continue
            live: deque[QueuedRequest] = deque()
            for request in queue:
                if request.cancelled.is_set():
                    request.state = RequestState.CANCELLED
                    removed.append(request)
                else:
                    live.append(request)
            self._queues[name] = live
        return removed

    def drain(self, model: str) -> list[QueuedRequest]:
        """Remove and return everything queued for `model` (used on shutdown)."""
        queue = self._queues[model]
        pending = list(queue)
        queue.clear()
        return pending

    def snapshot_depths(self) -> dict[str, int]:
        return {name: self.depth(name) for name in self._queues}
