"""Alternating traffic must not buy a swap per request.

This is the failure mode the whole hysteresis design exists to prevent: A B A B arrivals
where the naive answer — always serve whoever is waiting longest — spends the GPU's life
loading weights.
"""

from __future__ import annotations

from tests.sim import Arrival, alternating, build_config, model, simulate


async def test_alternating_traffic_produces_one_swap_not_one_per_request():
    config = build_config(min_residency_seconds=30, max_wait_seconds=300)

    # Sixteen requests alternating chat/translate every two seconds.
    result = await simulate(
        config, alternating(("chat", "translate"), every=2.0, count=16), until=120
    )

    print(result.report())

    assert result.swaps == 1, (
        f"expected a single swap, got {result.swaps}:\n{result.timeline()}"
    )
    assert len(result.completed()) == 16
    assert result.residency_order() == ["chat", "translate"]


async def test_a_gap_between_requests_is_not_an_invitation_to_swap():
    """The resident's queue empties constantly. That isn't the same as being done."""
    config = build_config(min_residency_seconds=30, max_wait_seconds=300)

    # chat is served faster than it is fed, so its queue sits at zero between arrivals.
    arrivals = [Arrival(at=float(i) * 5.0, model="chat", max_tokens=8) for i in range(6)]
    arrivals += [Arrival(at=3.0, model="translate")]

    result = await simulate(config, arrivals, until=120)

    print(result.report())
    assert result.swaps == 1
    assert result.residency_order() == ["chat", "translate"]


async def test_hysteresis_lapses_when_the_resident_is_genuinely_finished():
    """The flip side: a model with nothing left to do shouldn't sit on the GPU."""
    config = build_config(min_residency_seconds=120, max_wait_seconds=600)

    result = await simulate(
        config,
        [Arrival(at=0.0, model="chat", max_tokens=4), Arrival(at=1.0, model="translate")],
        until=60,
    )

    print(result.report())
    # chat finishes around t=8.3; translate should not have to wait out the full
    # 120s min-residency for a model that has nothing left to serve.
    assert result.residency_order() == ["chat", "translate"]
    assert result.max_wait("translate") < 30


async def test_a_burst_for_one_model_is_served_in_a_single_window():
    config = build_config(min_residency_seconds=30, max_wait_seconds=300)

    burst = [Arrival(at=0.1 * i, model="chat", max_tokens=8) for i in range(10)]
    result = await simulate(config, burst, until=60)

    print(result.report())
    assert result.loads == 1
    assert result.swaps == 0
    assert len(result.completed("chat")) == 10


async def test_a_high_priority_model_still_cannot_thrash_the_gpu():
    config = build_config(
        models=[
            model("chat", priority=1.0, load_seconds=8, tokens_per_second=40),
            model("translate", priority=10.0, load_seconds=6, tokens_per_second=40),
        ],
        min_residency_seconds=30,
        max_wait_seconds=300,
    )

    result = await simulate(
        config, alternating(("chat", "translate"), every=2.0, count=16), until=150
    )

    print(result.report())
    # Priority decides who goes first, not how often the GPU changes hands.
    assert result.swaps <= 2
    assert len(result.completed()) == 16
