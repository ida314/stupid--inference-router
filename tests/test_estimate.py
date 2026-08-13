"""What a queued request is told about its own wait.

The estimate is half fact and half projection, and these tests are mostly about keeping
the line between them where it belongs. Position, residency and the starvation ceiling are
read off scheduler state and must be exact. `estimated_seconds` is a guess, so it is
asserted for shape — it grows with the queue, it includes a swap when one is coming — and
never for a particular number.

Pure, like `test_policy.py`: no engine, no backend, no clock.
"""

from __future__ import annotations

from sir.config import SchedulerConfig
from sir.policy import estimate_wait
from sir.types import ModelState, SchedulerSnapshot

CONFIG = SchedulerConfig(
    min_residency_seconds=30,
    max_wait_seconds=120,
    max_concurrent_requests=8,
)
MEAN = 4.0


def snapshot(
    *,
    resident: str | None = "chat",
    resident_since: float | None = 0.0,
    chat_depth: int = 0,
    translate_depth: int = 0,
    oldest: float | None = None,
    in_flight: int = 0,
    unavailable_until: float | None = None,
) -> SchedulerSnapshot:
    return SchedulerSnapshot(
        models=(
            ModelState(
                name="chat",
                priority=1.0,
                depth=chat_depth,
                oldest_enqueued_at=oldest if chat_depth else None,
                estimated_load_seconds=8.0,
                resident=resident == "chat",
                resident_since=resident_since if resident == "chat" else None,
                in_flight=in_flight if resident == "chat" else 0,
            ),
            ModelState(
                name="translate",
                priority=1.0,
                depth=translate_depth,
                oldest_enqueued_at=oldest if translate_depth else None,
                estimated_load_seconds=6.0,
                resident=resident == "translate",
                resident_since=resident_since if resident == "translate" else None,
                unavailable_until=unavailable_until,
            ),
        ),
        resident=resident,
    )


# ---------------------------------------------------------------- the resident case


def test_the_resident_model_with_a_free_slot_waits_only_for_generation():
    estimate = estimate_wait(
        snapshot(chat_depth=1, oldest=0.0), "chat", 0, 1.0, CONFIG, MEAN
    )

    assert estimate.resident is True
    assert estimate.needs_swap is False
    assert estimate.load_seconds == 0.0
    # Nothing queued ahead and a free slot: the only wait left is the request itself.
    assert estimate.estimated_seconds == MEAN


def test_a_resident_model_reports_no_starvation_ceiling():
    """The ceiling says when a model wins the GPU. A resident model already has it."""
    estimate = estimate_wait(
        snapshot(chat_depth=1, oldest=0.0), "chat", 0, 60.0, CONFIG, MEAN
    )

    assert estimate.dispatch_within_seconds is None


# ---------------------------------------------------------------- batching


def test_waits_step_in_batches_of_max_concurrent_requests_not_in_requests():
    """The queue drains eight at a time, so being 1st and being 8th are the same wait.

    This is the detail most likely to be got wrong by a naive `position * mean`, and it is
    the difference between an estimate that tracks reality and one that is 8x pessimistic.
    """
    state = snapshot(chat_depth=20, oldest=0.0)

    first = estimate_wait(state, "chat", 0, 1.0, CONFIG, MEAN)
    last_of_batch = estimate_wait(state, "chat", 7, 1.0, CONFIG, MEAN)
    next_batch = estimate_wait(state, "chat", 8, 1.0, CONFIG, MEAN)

    assert first.estimated_seconds == last_of_batch.estimated_seconds
    assert next_batch.estimated_seconds == first.estimated_seconds + MEAN


def test_requests_already_generating_occupy_slots_the_queued_ones_wait_for():
    """Seven in flight means position 1 is in the second batch, not the first."""
    busy = estimate_wait(
        snapshot(chat_depth=5, oldest=0.0, in_flight=7), "chat", 1, 1.0, CONFIG, MEAN
    )
    idle = estimate_wait(
        snapshot(chat_depth=5, oldest=0.0, in_flight=0), "chat", 1, 1.0, CONFIG, MEAN
    )

    assert busy.estimated_seconds == idle.estimated_seconds + MEAN


# ---------------------------------------------------------------- the swap case


def test_a_model_that_is_not_resident_is_charged_for_its_own_load():
    estimate = estimate_wait(
        snapshot(translate_depth=1, oldest=0.0), "translate", 0, 1.0, CONFIG, MEAN
    )

    assert estimate.needs_swap is True
    assert estimate.load_seconds == 6.0
    assert estimate.estimated_seconds >= 6.0 + MEAN


def test_the_incumbents_remaining_hysteresis_is_part_of_the_wait():
    """A challenger can't have the GPU until min residency lapses, so say so."""
    fresh = estimate_wait(
        snapshot(resident_since=0.0, translate_depth=1, oldest=10.0),
        "translate",
        0,
        10.0,
        CONFIG,
        MEAN,
    )
    settled = estimate_wait(
        snapshot(resident_since=0.0, translate_depth=1, oldest=10.0),
        "translate",
        0,
        40.0,
        CONFIG,
        MEAN,
    )

    # At t=10 the incumbent holds another 20s; at t=40 its hold has long lapsed.
    assert fresh.estimated_seconds == settled.estimated_seconds + 20.0


def test_the_starvation_ceiling_caps_the_hysteresis_wait():
    """Step 1 of `decide` outranks step 5, and the estimate has to agree.

    A request that has already waited past the ceiling is dispatched next tick regardless
    of how much residency the incumbent has left, so the hold must not be added on top.
    """
    overdue = estimate_wait(
        # Resident just loaded (a full 30s hold), but this request has waited 119s.
        snapshot(resident_since=119.0, translate_depth=1, oldest=0.0),
        "translate",
        0,
        119.0,
        CONFIG,
        MEAN,
    )

    assert overdue.dispatch_within_seconds == 1.0
    # Load time still applies — the ceiling makes the swap mandatory, not instant.
    assert overdue.estimated_seconds == 1.0 + 6.0 + MEAN


def test_the_ceiling_shrinks_as_a_request_waits_and_never_goes_negative():
    waits = [
        estimate_wait(
            snapshot(translate_depth=1, oldest=0.0), "translate", 0, now, CONFIG, MEAN
        ).dispatch_within_seconds
        for now in (0.0, 60.0, 120.0, 500.0)
    ]

    assert waits == [120.0, 60.0, 0.0, 0.0]


def test_a_crashed_backends_retry_backoff_is_part_of_the_wait():
    """Queue depth says nothing while the backend can't be started at all."""
    estimate = estimate_wait(
        snapshot(translate_depth=1, oldest=0.0, unavailable_until=45.0),
        "translate",
        0,
        5.0,
        CONFIG,
        MEAN,
    )

    # 40s of backoff left, then the load, then the request.
    assert estimate.estimated_seconds == 40.0 + 6.0 + MEAN


# ---------------------------------------------------------------- poll pacing


def test_retry_after_tracks_the_estimate_within_sane_bounds():
    prompt = estimate_wait(
        snapshot(chat_depth=1, oldest=0.0), "chat", 0, 1.0, CONFIG, MEAN
    )
    distant = estimate_wait(
        snapshot(chat_depth=500, oldest=0.0), "chat", 499, 1.0, CONFIG, MEAN
    )

    assert 0.5 <= prompt.retry_after <= 10.0
    assert prompt.retry_after < distant.retry_after
    assert distant.retry_after == 10.0
