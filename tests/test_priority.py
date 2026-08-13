"""Priority orders the contest. It does not exempt anyone from the rules."""

from __future__ import annotations

from tests.sim import Arrival, build_config, model, simulate


async def test_the_higher_priority_model_wins_a_dead_heat():
    config = build_config(
        models=[
            model("chat", priority=1.0, load_seconds=6, tokens_per_second=40),
            model("translate", priority=4.0, load_seconds=6, tokens_per_second=40),
        ],
        min_residency_seconds=30,
        max_wait_seconds=300,
    )

    # Identical arrival times, identical queue depths, identical load costs.
    arrivals = [Arrival(at=0.0, model="chat"), Arrival(at=0.0, model="translate")]
    result = await simulate(config, arrivals, until=120)

    print(result.report())
    assert result.residency_order()[0] == "translate"


async def test_priority_does_not_override_the_starvation_ceiling():
    config = build_config(
        models=[
            model("chat", priority=50.0, load_seconds=6, tokens_per_second=40),
            model("translate", priority=1.0, load_seconds=6, tokens_per_second=40),
        ],
        min_residency_seconds=20,
        max_wait_seconds=45,
    )

    arrivals = [Arrival(at=float(i), model="chat") for i in range(120)]
    arrivals.append(Arrival(at=1.0, model="translate"))

    result = await simulate(config, arrivals, until=300)

    print(result.report())
    assert result.completed("translate")
    assert result.max_wait("translate") < 45 + 20


async def test_a_deeper_queue_can_outweigh_a_priority_edge():
    """Depth and age are real demand signals; priority only tilts the scale."""
    config = build_config(
        models=[
            model("chat", priority=1.0, load_seconds=6, tokens_per_second=40),
            model("translate", priority=2.0, load_seconds=6, tokens_per_second=40),
        ],
        min_residency_seconds=30,
        max_wait_seconds=300,
        depth_weight=2.0,
    )

    arrivals = [Arrival(at=0.0, model="chat") for _ in range(20)]
    arrivals.append(Arrival(at=0.0, model="translate"))

    result = await simulate(config, arrivals, until=120)

    print(result.report())
    # chat: 1 * (0 + 2*20) = 40 vs translate: 2 * (0 + 2*1) = 4, both minus a 6s load.
    assert result.residency_order()[0] == "chat"
