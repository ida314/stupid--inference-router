"""The policy, tested directly.

`decide()` is pure, so every scheduling property the README claims can be asserted
synchronously, with no event loop and no backends. These are the tests that would catch a
bad change to the trade-off; the engine tests below them check that the mechanism carries
the decision out.
"""

from __future__ import annotations

from sir.config import SchedulerConfig
from sir.policy import decide, score_model
from sir.types import DecisionKind, DecisionReason, ModelState, SchedulerSnapshot

NOW = 1000.0


def state(
    name: str,
    *,
    depth: int = 0,
    waiting: float | None = None,
    priority: float = 1.0,
    load: float = 8.0,
    resident: bool = False,
    resident_for: float | None = None,
    in_flight: int = 0,
    idle_for: float | None = None,
    unavailable_for: float = 0.0,
) -> ModelState:
    return ModelState(
        name=name,
        priority=priority,
        depth=depth,
        oldest_enqueued_at=None if waiting is None else NOW - waiting,
        estimated_load_seconds=load,
        resident=resident,
        resident_since=None if resident_for is None else NOW - resident_for,
        in_flight=in_flight,
        idle_since=None if idle_for is None else NOW - idle_for,
        unavailable_until=NOW + unavailable_for if unavailable_for else None,
    )


def snapshot(*models: ModelState, **kwargs: object) -> SchedulerSnapshot:
    resident = next((m.name for m in models if m.resident), None)
    return SchedulerSnapshot(models=models, resident=resident, **kwargs)  # type: ignore[arg-type]


def config(**kwargs: object) -> SchedulerConfig:
    return SchedulerConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------- the quiet cases


def test_no_pending_work_is_idle_and_keeps_the_model_loaded():
    result = decide(
        snapshot(state("chat", resident=True, resident_for=100), state("translate")),
        NOW,
        config(),
    )
    assert result.kind is DecisionKind.IDLE
    assert result.reason is DecisionReason.NO_WORK
    # Unloading an idle model buys back memory nobody is waiting for.
    assert result.target is None


def test_cold_start_loads_the_best_scoring_model():
    result = decide(
        snapshot(
            state("chat", depth=1, waiting=1.0, load=8),
            state("translate", depth=5, waiting=10.0, load=6),
        ),
        NOW,
        config(),
    )
    assert result.kind is DecisionKind.SWITCH
    assert result.reason is DecisionReason.COLD_START
    assert result.target == "translate"


def test_mid_transition_decides_nothing():
    """Changing our mind during a drain is how you get a half-unloaded model."""
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=5, depth=1, waiting=1),
            state("translate", depth=99, waiting=999),
            switching=True,
        ),
        NOW,
        config(),
    )
    assert result.kind is DecisionKind.HOLD
    assert result.reason is DecisionReason.IN_TRANSITION


# ---------------------------------------------------------------- the core trade-off


def test_a_fresh_request_cannot_outbid_the_swap_it_would_cause():
    """Switch cost is hysteresis in its own right, before min-residency even applies."""
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=999, depth=2, waiting=3),
            state("translate", depth=1, waiting=0.5, load=8),
        ),
        NOW,
        config(min_residency_seconds=0),
    )
    assert result.kind is DecisionKind.HOLD
    assert result.reason is DecisionReason.RESIDENT_WINS


def test_a_challenger_that_has_waited_long_enough_takes_the_gpu():
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=999, depth=1, waiting=1),
            state("translate", depth=3, waiting=40, load=8),
        ),
        NOW,
        config(min_residency_seconds=0),
    )
    assert result.kind is DecisionKind.SWITCH
    assert result.reason is DecisionReason.CHALLENGER_WINS
    assert result.target == "translate"


def test_ties_go_to_the_incumbent():
    """An exactly even trade is not worth paying a swap for."""
    identical = dict(depth=2, waiting=10.0, load=0.0)
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=999, **identical),
            state("translate", **identical),
        ),
        NOW,
        config(min_residency_seconds=0, switch_cost_weight=0),
    )
    assert result.kind is not DecisionKind.SWITCH


def test_priority_breaks_an_otherwise_even_contest():
    even = dict(depth=2, waiting=20.0, load=6.0)
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=999, priority=1.0, **even),
            state("translate", priority=3.0, **even),
        ),
        NOW,
        config(min_residency_seconds=0),
    )
    assert result.kind is DecisionKind.SWITCH
    assert result.target == "translate"


# ---------------------------------------------------------------- hysteresis


def test_min_residency_holds_a_busy_model_against_a_better_offer():
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=5, depth=1, waiting=1),
            state("translate", depth=9, waiting=60, load=6),
        ),
        NOW,
        config(min_residency_seconds=30, max_wait_seconds=600),
    )
    assert result.kind is DecisionKind.HOLD
    assert result.reason is DecisionReason.MIN_RESIDENCY


def test_min_residency_holds_through_the_gaps_between_requests():
    """A model streaming a response has an empty queue and is emphatically not idle."""
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=5, depth=0, in_flight=3),
            state("translate", depth=4, waiting=30, load=6),
        ),
        NOW,
        config(min_residency_seconds=30, max_wait_seconds=600),
    )
    assert result.kind is DecisionKind.HOLD
    assert result.reason is DecisionReason.MIN_RESIDENCY


def test_min_residency_lapses_once_idling_costs_more_than_the_swap():
    """Holding an idle GPU longer than the swap would take is a losing trade."""
    # The challenger's load time is 6s, so 6s of idling is the break-even point.
    held = decide(
        snapshot(
            state("chat", resident=True, resident_for=5, idle_for=4.0),
            state("translate", depth=4, waiting=30, load=6),
        ),
        NOW,
        config(min_residency_seconds=30, max_wait_seconds=600),
    )
    assert held.reason is DecisionReason.MIN_RESIDENCY

    lapsed = decide(
        snapshot(
            state("chat", resident=True, resident_for=5, idle_for=7.0),
            state("translate", depth=4, waiting=30, load=6),
        ),
        NOW,
        config(min_residency_seconds=30, max_wait_seconds=600),
    )
    assert lapsed.kind is DecisionKind.SWITCH
    assert lapsed.reason is DecisionReason.RESIDENT_IDLE
    assert lapsed.target == "translate"


# ---------------------------------------------------------------- the backstop


def test_the_starvation_ceiling_outranks_min_residency():
    """Fairness is a correctness property, not a weight. This is the whole point."""
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=1, depth=50, waiting=5),
            state("translate", depth=1, waiting=121, load=600),
        ),
        NOW,
        config(min_residency_seconds=30, max_wait_seconds=120),
    )
    assert result.kind is DecisionKind.SWITCH
    assert result.reason is DecisionReason.STARVATION
    assert result.target == "translate"


def test_the_starvation_ceiling_serves_the_longest_waiter_first():
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=1, depth=1, waiting=1),
            state("translate", depth=1, waiting=200),
            state("embed", depth=1, waiting=400),
        ),
        NOW,
        config(max_wait_seconds=120),
    )
    assert result.reason is DecisionReason.STARVATION
    assert result.target == "embed"


def test_the_resident_model_cannot_starve_itself():
    """A slow resident queue is a throughput problem, not a residency one."""
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=1, depth=10, waiting=999),
            state("translate"),
        ),
        NOW,
        config(max_wait_seconds=120),
    )
    assert result.kind is not DecisionKind.SWITCH


# ---------------------------------------------------------------- crash containment


def test_an_unavailable_backend_is_never_chosen():
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=999),
            state("translate", depth=5, waiting=500, unavailable_for=5),
        ),
        NOW,
        config(),
    )
    assert result.kind is DecisionKind.IDLE
    assert result.reason is DecisionReason.BACKEND_UNAVAILABLE


def test_a_dead_resident_hands_the_gpu_to_someone_who_can_use_it():
    result = decide(
        snapshot(
            state("chat", resident=True, resident_for=1, depth=3, waiting=5, unavailable_for=5),
            state("translate", depth=1, waiting=1),
        ),
        NOW,
        config(min_residency_seconds=30),
    )
    assert result.kind is DecisionKind.SWITCH
    assert result.reason is DecisionReason.BACKEND_UNAVAILABLE
    assert result.target == "translate"


# ---------------------------------------------------------------- scoring arithmetic


def test_the_score_breakdown_explains_itself():
    breakdown = score_model(
        state("translate", depth=4, waiting=10.0, priority=2.0, load=6.0),
        NOW,
        config(age_weight=1.0, depth_weight=0.5, switch_cost_weight=1.0),
        is_resident=False,
    )
    # 2.0 * (1.0*10 + 0.5*4) - 1.0*6
    assert breakdown.score == 18.0
    assert breakdown.age_term == 10.0
    assert breakdown.depth_term == 2.0
    assert breakdown.switch_cost == 6.0
    assert "translate" in breakdown.format()


def test_the_resident_pays_no_switch_cost():
    args = dict(depth=4, waiting=10.0, priority=2.0, load=6.0)
    resident = score_model(state("chat", **args), NOW, config(), is_resident=True)
    challenger = score_model(state("chat", **args), NOW, config(), is_resident=False)
    assert resident.score - challenger.score == 6.0
