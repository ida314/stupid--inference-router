"""The scheduler's behavior, shown rather than only asserted.

Run with `pytest -s tests/test_policy_sim.py` to see the decision timeline and the
resulting wait/swap trade-off for each scenario. The assertions here are deliberately
loose — the tests above pin the invariants; this file exists so the policy's *shape* is
something you can look at when tuning weights.

Caveat worth repeating: these numbers inherit the mock's guessed load and generation
times. They tell you whether the policy is sane, not what it will do on the Spark.
"""

from __future__ import annotations

from tests.sim import alternating, build_config, model, simulate, steady

REALISTIC = dict(min_residency_seconds=30, max_wait_seconds=120)


def homelab_models():
    """Roughly the README's setup: a busy general model and a latency-sensitive one."""
    return [
        model("chat", priority=1.0, load_seconds=8, tokens_per_second=40),
        model("translate", priority=1.5, load_seconds=6, tokens_per_second=60),
    ]


async def test_alternating_workload():
    """The thrash scenario, with production-ish constants."""
    config = build_config(models=homelab_models(), **REALISTIC)
    result = await simulate(
        config, alternating(("chat", "translate"), every=3.0, count=20), until=300
    )

    print("\n=== alternating chat/translate every 3s ===")
    print(result.report())

    assert result.swaps <= 2
    assert len(result.completed()) == 20


async def test_busy_model_with_a_quiet_neighbour():
    """The starvation scenario: constant chat, occasional translate."""
    config = build_config(models=homelab_models(), **REALISTIC)
    arrivals = steady("chat", every=2.0, count=150) + steady(
        "translate", every=45.0, count=7, start=10.0
    )
    result = await simulate(config, arrivals, until=400)

    print("\n=== steady chat every 2s, translate every 45s ===")
    print(result.report())

    assert result.completed("translate")
    assert result.max_wait("translate") < 120 + 20


async def test_bursty_traffic_groups_into_windows():
    """Bursts should be batched into one residency window each, not spread across swaps."""
    config = build_config(models=homelab_models(), **REALISTIC)
    arrivals = []
    for burst_index in range(4):
        base = burst_index * 60.0
        name = "chat" if burst_index % 2 == 0 else "translate"
        arrivals += steady(name, every=0.5, count=12, start=base)
    result = await simulate(config, arrivals, until=400)

    print("\n=== four 12-request bursts, alternating model, 60s apart ===")
    print(result.report())

    # Four bursts across two models: three swaps is the floor, and we should be at it.
    assert result.swaps == 3
    assert len(result.completed()) == 48


async def test_three_models_sharing_one_gpu():
    config = build_config(
        models=[
            model("chat", priority=1.0, load_seconds=8, tokens_per_second=40),
            model("translate", priority=1.5, load_seconds=6, tokens_per_second=60),
            model("embed", priority=0.5, load_seconds=3, tokens_per_second=200),
        ],
        **REALISTIC,
    )
    arrivals = (
        steady("chat", every=2.0, count=150)
        + steady("translate", every=15.0, count=20, start=5.0)
        + steady("embed", every=30.0, count=10, start=12.0)
    )
    result = await simulate(config, arrivals, until=400)

    print("\n=== three models, one GPU ===")
    print(result.report())

    for name in ("chat", "translate", "embed"):
        assert result.completed(name), f"{name} never ran"
