"""No model waits forever. This is the non-negotiable one.

The README calls fairness a correctness property rather than a tuning parameter, which
means these tests must hold for *any* weighting — including weights chosen badly. So they
deliberately stack the deck against the starving model: constant competing traffic, worse
priority, and a more expensive load.
"""

from __future__ import annotations

from tests.sim import Arrival, build_config, model, simulate, steady

# The ceiling only bounds the wait until the scheduler *commits*. Actually dispatching
# still costs a drain plus a load, so assertions allow for that.
SWAP_SLACK = 20.0


async def test_a_lone_request_is_served_despite_constant_competing_traffic():
    config = build_config(min_residency_seconds=30, max_wait_seconds=60)

    arrivals = steady("chat", every=1.0, count=200)
    arrivals.append(Arrival(at=5.0, model="translate"))

    result = await simulate(config, arrivals, until=400)

    print(result.report())
    assert result.completed("translate"), "the low-traffic model never ran"
    assert result.max_wait("translate") < 60 + SWAP_SLACK


async def test_the_ceiling_beats_priority_stacked_against_the_waiter():
    """Even a 20x priority advantage cannot buy indefinite residency."""
    config = build_config(
        models=[
            model("chat", priority=20.0, load_seconds=8, tokens_per_second=40),
            model("translate", priority=1.0, load_seconds=12, tokens_per_second=40),
        ],
        min_residency_seconds=30,
        max_wait_seconds=60,
    )

    arrivals = steady("chat", every=0.5, count=400)
    arrivals.append(Arrival(at=2.0, model="translate"))

    result = await simulate(config, arrivals, until=400)

    print(result.report())
    assert result.completed("translate")
    assert result.max_wait("translate") < 60 + SWAP_SLACK


async def test_every_model_gets_a_turn_under_permanent_contention():
    """Three models, endless traffic on all of them: nobody may be shut out."""
    config = build_config(
        models=[
            model("chat", priority=1.0, load_seconds=6, tokens_per_second=40),
            model("translate", priority=1.0, load_seconds=6, tokens_per_second=40),
            model("embed", priority=1.0, load_seconds=6, tokens_per_second=40),
        ],
        min_residency_seconds=20,
        max_wait_seconds=60,
    )

    arrivals = (
        steady("chat", every=1.0, count=300)
        + steady("translate", every=3.0, count=100)
        + steady("embed", every=7.0, count=45)
    )

    result = await simulate(config, arrivals, until=400)

    print(result.report())
    for name in ("chat", "translate", "embed"):
        assert result.completed(name), f"{name} was never served"
        assert result.max_wait(name) < 60 + SWAP_SLACK, f"{name} waited too long"


async def test_the_ceiling_does_not_fire_when_scoring_already_behaves():
    """A backstop that trips constantly is a broken policy, not a safe one."""
    config = build_config(min_residency_seconds=10, max_wait_seconds=120)

    arrivals = steady("chat", every=4.0, count=20) + steady(
        "translate", every=4.0, count=20, start=2.0
    )
    result = await simulate(config, arrivals, until=300)

    print(result.report())
    starvations = [d for d in result.switches() if d.reason.value == "starvation"]
    assert not starvations, (
        "normal scoring should have handled this workload without the backstop:\n"
        + result.timeline()
    )
    assert len(result.completed()) == 40
