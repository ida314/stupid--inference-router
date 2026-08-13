"""Per-model queues and the wait accounting the policy scores on."""

from __future__ import annotations

import pytest

from sir.queues import ModelQueues
from sir.types import QueuedRequest, RequestState
from tests.sim import generation


def request(model: str, at: float) -> QueuedRequest:
    return QueuedRequest(
        request=generation(model), enqueued_at=at
    )


def test_queues_are_per_model_and_independent():
    queues = ModelQueues(["chat", "translate"])
    queues.enqueue(request("chat", 0.0))
    queues.enqueue(request("chat", 1.0))
    queues.enqueue(request("translate", 2.0))

    assert queues.depth("chat") == 2
    assert queues.depth("translate") == 1
    assert queues.total_pending() == 3
    assert sorted(queues.models_with_work()) == ["chat", "translate"]


def test_an_unknown_model_is_rejected():
    queues = ModelQueues(["chat"])
    with pytest.raises(KeyError):
        queues.enqueue(request("translate", 0.0))


def test_the_oldest_arrival_drives_the_wait_age():
    queues = ModelQueues(["chat"])
    queues.enqueue(request("chat", 10.0))
    queues.enqueue(request("chat", 20.0))
    assert queues.oldest_enqueued_at("chat") == 10.0


def test_cancelled_requests_are_invisible_to_the_scheduler():
    """A departed client must not be able to move the GPU."""
    queues = ModelQueues(["chat"])
    doomed = request("chat", 0.0)
    keeper = request("chat", 5.0)
    queues.enqueue(doomed)
    queues.enqueue(keeper)

    doomed.cancel()
    assert queues.depth("chat") == 1
    assert queues.oldest_enqueued_at("chat") == 5.0


def test_dispatch_skips_cancelled_requests_and_stamps_start_time():
    queues = ModelQueues(["chat"])
    doomed = request("chat", 0.0)
    keeper = request("chat", 1.0)
    queues.enqueue(doomed)
    queues.enqueue(keeper)
    doomed.cancel()

    taken = queues.next_for("chat", now=30.0)
    assert taken is keeper
    assert taken.state is RequestState.RUNNING
    assert taken.started_at == 30.0
    assert taken.wait_time(now=99.0) == 29.0  # measured to dispatch, not to now
    assert doomed.state is RequestState.CANCELLED
    assert queues.depth("chat") == 0


def test_take_while_respects_the_concurrency_limit():
    queues = ModelQueues(["chat"])
    for index in range(5):
        queues.enqueue(request("chat", float(index)))

    taken = list(queues.take_while("chat", now=10.0, limit=3))
    assert len(taken) == 3
    assert queues.depth("chat") == 2

    assert list(queues.take_while("chat", now=10.0, limit=0)) == []


def test_purging_removes_cancelled_entries_and_reports_them():
    queues = ModelQueues(["chat"])
    entries = [request("chat", float(i)) for i in range(4)]
    for entry in entries:
        queues.enqueue(entry)
    entries[1].cancel()
    entries[3].cancel()

    removed = queues.purge_cancelled()
    assert {r.id for r in removed} == {entries[1].id, entries[3].id}
    assert queues.depth("chat") == 2
    assert queues.purge_cancelled() == []


def test_draining_empties_a_queue():
    queues = ModelQueues(["chat"])
    queues.enqueue(request("chat", 0.0))
    queues.enqueue(request("chat", 1.0))
    assert len(queues.drain("chat")) == 2
    assert queues.depth("chat") == 0
    assert queues.oldest_enqueued_at("chat") is None
