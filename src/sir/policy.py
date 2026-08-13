"""The scheduler's policy: which model gets to be resident right now.

This module is pure. No I/O, no async, no clock reads, no knowledge that vLLM exists. It
takes a snapshot and a timestamp and returns a `Decision` carrying the full arithmetic
behind it. Everything expensive and stateful lives in `engine.py`.

That split is what makes the interesting behavior testable: thrash, starvation, and
priority are all properties of this function, and they can be asserted synchronously.

The scoring model
-----------------

    demand(m) = age_weight * oldest_wait(m) + depth_weight * depth(m)
    score(m)  = priority(m) * demand(m) - switch_cost(m)

where `switch_cost` is zero for the resident model and
`switch_cost_weight * estimated_load_seconds(m)` for everyone else. That asymmetry *is*
the core tension from the README — serving the resident model is free, serving anything
else costs a swap — expressed as arithmetic rather than as a special case.

A useful consequence: even when the resident model's queue is empty, a challenger still
has to out-score the swap it would cause. With an 8-second load time, a single fresh
request won't steal the GPU; it has to wait about eight seconds first. Hysteresis falls
out of the cost term, and `min_residency_seconds` (step 5 below) is the harder backstop
on top of it.
"""

from __future__ import annotations

from sir.config import SchedulerConfig
from sir.types import (
    Decision,
    DecisionKind,
    DecisionReason,
    ModelState,
    ScoreBreakdown,
    SchedulerSnapshot,
)


def score_model(
    state: ModelState,
    now: float,
    config: SchedulerConfig,
    *,
    is_resident: bool,
) -> ScoreBreakdown:
    """Score one model. Resident models pay no switch cost; that is the whole asymmetry."""
    if not state.is_available(now):
        return ScoreBreakdown(
            model=state.name,
            priority=state.priority,
            depth=state.depth,
            oldest_wait=state.oldest_wait(now),
            age_term=0.0,
            depth_term=0.0,
            switch_cost=0.0,
            score=float("-inf"),
            eligible=False,
            note="backend unavailable",
        )

    if not state.has_pending:
        return ScoreBreakdown(
            model=state.name,
            priority=state.priority,
            depth=0,
            oldest_wait=0.0,
            age_term=0.0,
            depth_term=0.0,
            switch_cost=0.0,
            score=0.0,
            eligible=True,
            note="no pending work",
        )

    oldest_wait = state.oldest_wait(now)
    age_term = config.age_weight * oldest_wait
    depth_term = config.depth_weight * state.depth
    switch_cost = (
        0.0 if is_resident else config.switch_cost_weight * state.estimated_load_seconds
    )
    score = state.priority * (age_term + depth_term) - switch_cost

    return ScoreBreakdown(
        model=state.name,
        priority=state.priority,
        depth=state.depth,
        oldest_wait=oldest_wait,
        age_term=age_term,
        depth_term=depth_term,
        switch_cost=switch_cost,
        score=score,
    )


def decide(
    snapshot: SchedulerSnapshot, now: float, config: SchedulerConfig
) -> Decision:
    """Choose what the GPU does next.

    The branches below are ordered by precedence, and the order is the policy. Read them
    top to bottom: the starvation ceiling outranks hysteresis, which outranks scoring.
    """
    scores = tuple(
        score_model(state, now, config, is_resident=state.name == snapshot.resident)
        for state in snapshot.models
    )

    def decision(
        kind: DecisionKind, reason: DecisionReason, target: str | None = None
    ) -> Decision:
        return Decision(
            kind=kind,
            reason=reason,
            at=now,
            resident=snapshot.resident,
            target=target,
            scores=scores,
        )

    # The mechanism is mid-drain or mid-load. Nothing to decide until it lands; changing
    # our mind now is how you get a half-unloaded model.
    if snapshot.switching or snapshot.draining:
        return decision(DecisionKind.HOLD, DecisionReason.IN_TRANSITION)

    resident = snapshot.resident_state
    with_work = [state for state in snapshot.models if state.has_pending]
    servable = [state for state in with_work if state.is_available(now)]

    # 1. The starvation ceiling. Past `max_wait_seconds` a model becomes mandatory,
    #    outranking hysteresis, priority, and switch cost alike. Fairness is a
    #    correctness property here, not a weight to be tuned — this is the backstop for
    #    when the weights above turn out to be wrong.
    overdue = [
        state
        for state in servable
        if state.name != snapshot.resident
        and state.oldest_wait(now) >= config.max_wait_seconds
    ]
    if overdue:
        target = max(overdue, key=lambda state: state.oldest_wait(now))
        return decision(DecisionKind.SWITCH, DecisionReason.STARVATION, target.name)

    # 2. Nothing wants the GPU. Keep whatever is loaded — unloading an idle model buys
    #    no memory back that anyone is waiting for, and costs a reload if it's next.
    if not with_work:
        return decision(DecisionKind.IDLE, DecisionReason.NO_WORK)

    # 3. Work exists, but only for backends we can't currently start. Wait them out.
    if not servable:
        return decision(DecisionKind.IDLE, DecisionReason.BACKEND_UNAVAILABLE)

    best = max(servable, key=lambda state: _score_of(scores, state.name))

    # 4. Cold start, or the resident backend died under us. Load the best candidate.
    if resident is None:
        return decision(DecisionKind.SWITCH, DecisionReason.COLD_START, best.name)
    if not resident.is_available(now):
        return decision(
            DecisionKind.SWITCH, DecisionReason.BACKEND_UNAVAILABLE, best.name
        )

    # 5. Hysteresis. A freshly loaded model keeps the GPU for `min_residency_seconds`,
    #    so alternating A B A B traffic buys one swap instead of one per request.
    #
    #    The subtlety is what counts as "still wants the GPU". Queue depth alone is the
    #    wrong test: a model serving 4 requests a second has an empty queue most of the
    #    time, and reading those gaps as "done" hands the GPU away mid-workload — exactly
    #    the thrash this rule exists to prevent.
    #
    #    So the hold lapses early only once the model has been *continuously* idle —
    #    nothing queued, nothing generating — for longer than the swap it would save.
    #    Below that threshold, swapping is a losing trade by definition: you'd spend more
    #    seconds loading than the GPU sat unused. No new tuning knob; the switch cost the
    #    challenger already pays sets the bar. Step 1 bounds the damage either way.
    if best.name != resident.name and resident.resident_since is not None:
        if now - resident.resident_since < config.min_residency_seconds:
            swap_cost = config.switch_cost_weight * best.estimated_load_seconds
            if resident.is_busy or resident.idle_for(now) < swap_cost:
                return decision(DecisionKind.HOLD, DecisionReason.MIN_RESIDENCY)

    # 6. Scoring. Ties go to the incumbent — an exactly-even trade isn't worth a swap.
    if best.name == snapshot.resident:
        kind = DecisionKind.SERVE if len(servable) == 1 else DecisionKind.HOLD
        return decision(kind, DecisionReason.RESIDENT_WINS)

    resident_score = _score_of(scores, resident.name)
    if _score_of(scores, best.name) > resident_score:
        # Label the swap by what actually happened. "Resident idle" is reserved for a
        # model that genuinely ran out of work — measured the same way hysteresis
        # measures it — so it can't be pinned on a busy model just because the tick
        # landed in the gap between two of its requests.
        swap_cost = config.switch_cost_weight * best.estimated_load_seconds
        finished = not resident.is_busy and resident.idle_for(now) >= swap_cost
        reason = (
            DecisionReason.RESIDENT_IDLE
            if finished
            else DecisionReason.CHALLENGER_WINS
        )
        return decision(DecisionKind.SWITCH, reason, best.name)

    return decision(DecisionKind.HOLD, DecisionReason.RESIDENT_WINS)


def _score_of(scores: tuple[ScoreBreakdown, ...], model: str) -> float:
    for breakdown in scores:
        if breakdown.model == model:
            return breakdown.score
    return float("-inf")
