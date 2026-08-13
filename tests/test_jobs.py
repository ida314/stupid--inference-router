"""The async submission path.

Two claims are under test here. The first is that it changes nothing: a job produces the
same completion the synchronous path would have produced for the same body, and a client
that doesn't ask for a job still gets what it always got.

The second is the one that justifies the lease. Holding a socket open is what tells `sir`
today that a client still wants its answer, and a polled job has no socket. Without
something to replace that signal, a service that restarts leaves queued work behind that
can still win the GPU and force a swap for a response nobody will ever read — which is
exactly what `ModelQueues` is careful to prevent for cancelled requests.
"""

from __future__ import annotations

import asyncio

import pytest

from sir.config import JobsConfig
from sir.jobs import JobStore
from sir.schemas import ChatCompletionRequest
from sir.types import EventKind
from tests.sim import build_config, generation, model, running_engine
from tests.test_api import chat_body, fast_config, serving

TERMINAL = {"done", "failed", "cancelled"}


def jobs_config(**overrides: object) -> JobsConfig:
    """Job retention scaled down so HTTP tests run in milliseconds."""
    return JobsConfig(
        **{
            "lease_seconds": 0.3,
            "result_ttl_seconds": 5.0,
            "read_ttl_seconds": 5.0,
            "sweep_interval_seconds": 0.05,
            **overrides,
        }
    )


def slow_config(**job_overrides: object):
    """A backend slow enough that a job is still running when the lease matters."""
    return build_config(
        models=[
            model(
                "chat",
                load_seconds=0.05,
                unload_seconds=0.01,
                tokens_per_second=20,
                first_token_seconds=0.0,
                default_max_tokens=20,
            )
        ],
        jobs=jobs_config(**job_overrides),
        min_residency_seconds=0.05,
        max_wait_seconds=1.0,
        tick_interval_seconds=0.01,
    )


async def poll(http, job_id: str, tries: int = 300, delay: float = 0.02) -> dict:
    """Poll to a terminal state, the way the SDK does but without the pacing."""
    for _ in range(tries):
        response = await http.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        document = response.json()
        if document["status"] in TERMINAL:
            return document
        await asyncio.sleep(delay)
    raise AssertionError(f"job {job_id} never reached a terminal state: {document}")


# ---------------------------------------------------------------- acceptance


async def test_an_async_submission_is_accepted_with_a_job_to_poll():
    async with serving(fast_config()) as (http, _):
        response = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("translate"),
        )

    assert response.status_code == 202
    assert response.headers["preference-applied"] == "respond-async"
    document = response.json()
    assert response.headers["location"] == f"/v1/jobs/{document['id']}"
    assert document["status"] in {"queued", "running"}
    assert document["response"] is None


async def test_the_job_reports_the_wait_it_is_in_for():
    """The whole point of the queue being visible: say what is actually happening."""
    async with serving(fast_config()) as (http, _):
        # `translate` is not resident on a cold router, so this one has to wait for a load.
        response = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("translate"),
        )

    wait = response.json()["wait"]
    assert wait["position"] == 0
    assert wait["resident"] is False
    assert wait["needs_swap"] is True
    assert wait["load_seconds"] > 0
    # A head-of-queue bound, and it must be a real number rather than a null placeholder.
    assert wait["dispatch_within_seconds"] > 0
    assert response.json()["retry_after"] > 0


async def test_a_job_produces_exactly_what_the_synchronous_path_would_have():
    """One dispatch path, one rendering. If these drift, the async path is a fork."""
    async with serving(fast_config()) as (http, _):
        synchronous = await http.post("/v1/chat/completions", json=chat_body("chat"))
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        document = await poll(http, accepted.json()["id"])

    assert document["status"] == "done"
    assert document["response"]["choices"] == synchronous.json()["choices"]
    assert document["response"]["usage"] == synchronous.json()["usage"]
    assert document["response"]["model"] == "chat"
    # Nothing left to wait for, so nothing left to say about the wait.
    assert document["wait"] is None
    assert document["retry_after"] == 0


async def test_a_finished_job_can_be_read_more_than_once():
    """A blip during the fetch must not be what loses the answer."""
    async with serving(fast_config()) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        job_id = accepted.json()["id"]
        first = await poll(http, job_id)
        second = await http.get(f"/v1/jobs/{job_id}")

    assert second.status_code == 200
    assert second.json()["response"] == first["response"]


# ---------------------------------------------------------------- the contract holds


async def test_without_the_header_nothing_changes():
    """The premise of the whole project: pointing a service at :8000 changes the host."""
    async with serving(fast_config()) as (http, _):
        response = await http.post("/v1/chat/completions", json=chat_body("chat"))

    assert response.status_code == 200
    assert "preference-applied" not in response.headers
    assert response.json()["object"] == "chat.completion"


async def test_streaming_and_async_submission_are_refused_together():
    """Silently streaming at a client that asked for a job would just hang it."""
    async with serving(fast_config()) as (http, _):
        response = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat", stream=True),
        )

    assert response.status_code == 400
    assert "mutually exclusive" in response.json()["error"]["message"]


async def test_the_async_path_can_be_turned_off_entirely():
    """Disabled, the preference is ignored rather than rejected — as RFC 7240 allows."""
    config = fast_config()
    config.jobs = JobsConfig(enabled=False)
    async with serving(config) as (http, _):
        response = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )

    assert response.status_code == 200
    assert "preference-applied" not in response.headers


async def test_an_unknown_job_is_reported_as_lost_not_merely_missing():
    async with serving(fast_config()) as (http, _):
        response = await http.get("/v1/jobs/req_nonexistent")

    assert response.status_code == 404
    message = response.json()["error"]["message"]
    # A client's response differs between the two causes, so both are named.
    assert "expired" in message and "restarted" in message


async def test_the_same_idempotency_key_buys_one_generation():
    async with serving(fast_config()) as (http, seen):
        headers = {"prefer": "respond-async", "idempotency-key": "abc-123"}
        first = await http.post(
            "/v1/chat/completions", headers=headers, json=chat_body("chat")
        )
        second = await http.post(
            "/v1/chat/completions", headers=headers, json=chat_body("chat")
        )
        assert first.json()["id"] == second.json()["id"]
        await poll(http, first.json()["id"])

    # The decisive assertion: the backend saw the work once, not twice.
    assert len(seen) == 1


# ---------------------------------------------------------------- cancellation


async def test_deleting_a_job_cancels_it():
    async with serving(slow_config()) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        job_id = accepted.json()["id"]
        deleted = await http.delete(f"/v1/jobs/{job_id}")
        follow_up = await http.get(f"/v1/jobs/{job_id}")

    assert deleted.status_code == 200
    assert deleted.json()["status"] == "cancelled"
    assert follow_up.json()["status"] == "cancelled"
    assert follow_up.json()["error"]["code"] == "job_cancelled"


async def test_polling_keeps_a_job_alive_past_its_lease():
    """Polling is the liveness signal, so a client that polls is a client that is there."""
    async with serving(slow_config()) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        # Lease is 0.3s; poll through several of them without ever going quiet.
        document = await poll(http, accepted.json()["id"], delay=0.05)

    assert document["status"] == "done"


async def test_the_advised_poll_cadence_never_outlasts_the_lease():
    """A client doing exactly as it is told must not be cancelled for it.

    `retry_after` is derived from the estimated wait, which on a long queue can exceed a
    short lease. The two numbers are only both known on the server, so keeping them
    consistent is the server's job — a client that trusts `retry_after` should never have
    to also know what the lease is set to.
    """
    async with serving(slow_config(lease_seconds=0.4)) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )

    assert 0 < accepted.json()["retry_after"] <= 0.2


async def test_a_job_nobody_polls_for_is_cancelled():
    async with serving(slow_config()) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        job_id = accepted.json()["id"]
        await asyncio.sleep(0.6)  # two lease windows of silence
        document = (await http.get(f"/v1/jobs/{job_id}")).json()

    assert document["status"] == "cancelled"


async def test_a_read_result_expires_on_the_shorter_clock():
    async with serving(slow_config(read_ttl_seconds=0.1, lease_seconds=0)) as (http, _):
        accepted = await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
        job_id = accepted.json()["id"]
        await poll(http, job_id, delay=0.02)
        await asyncio.sleep(0.4)
        gone = await http.get(f"/v1/jobs/{job_id}")

    assert gone.status_code == 404


# ---------------------------------------------------------------- the scheduler


async def abandoned_job_scenario(clock, lease_seconds: float) -> tuple[bool, int]:
    """A busy incumbent, plus one async request for the other model that nobody polls for.

    Because `chat` is kept continuously busy, the abandoned request cannot win on score —
    its only route to the GPU is the starvation ceiling at `max_wait_seconds`. That is the
    case the lease is sized to catch, and the same 1:2 ratio the defaults ship with (60s
    against 120s).

    Returns whether `translate` was ever loaded, and its queue depth at the end. Asking
    whether the swap *happened* rather than who is resident right now keeps the assertion
    off the exact instant the clock stops — a swap is a drain, an unload and a load, and
    sampling in the middle of one says nothing.
    """
    config = build_config(
        models=[
            model("chat", load_seconds=1, tokens_per_second=10, default_max_tokens=20),
            model(
                "translate", load_seconds=1, tokens_per_second=10, default_max_tokens=20
            ),
        ],
        jobs=JobsConfig(lease_seconds=lease_seconds, sweep_interval_seconds=0.5),
        min_residency_seconds=5,
        max_wait_seconds=20,
        max_concurrent_requests=1,
        tick_interval_seconds=0.1,
    )

    async with running_engine(config, clock) as engine:
        store = JobStore(config.jobs, clock)
        await store.start()
        try:
            # Two seconds of generation apiece, one at a time: `chat` has work for the
            # whole window and never falls idle long enough to be out-scored.
            for _ in range(30):
                engine.submit(generation("chat"))
            await clock.advance(3)
            assert engine.resident == "chat"

            # Now a job for the other model, submitted asynchronously and then abandoned:
            # the service that asked for it has gone away.
            queued = engine.submit(generation("translate"))
            store.create(
                queued,
                ChatCompletionRequest(
                    model="translate", messages=[{"role": "user", "content": "hola"}]
                ),
            )

            # Past the lease (10s) and past the starvation ceiling (20s).
            await clock.advance(25)
            loaded = any(
                event.kind is EventKind.LOAD_END and event.model == "translate"
                for event in engine.events
            )
            return loaded, engine.queues.depth("translate")
        finally:
            await clock.drive(store.stop())


async def test_an_abandoned_job_is_cancelled_before_it_can_force_a_swap(clock):
    """The invariant the lease is sized to preserve.

    With no socket to close, an abandoned request would otherwise sit in its queue
    accruing wait until the starvation ceiling made a swap mandatory — and the GPU would
    stop serving live traffic to generate a response with nowhere to go.
    """
    swapped, depth = await abandoned_job_scenario(clock, lease_seconds=10)

    assert swapped is False
    assert depth == 0  # cancelled by the sweeper, and gone from the queue


async def test_without_a_lease_the_same_abandoned_job_takes_the_gpu(clock):
    """The control. Proves the test above measures the lease and not the scheduler."""
    swapped, _ = await abandoned_job_scenario(clock, lease_seconds=0)

    assert swapped is True


# ---------------------------------------------------------------- shutdown


async def test_shutdown_does_not_leave_job_pumps_running():
    config = slow_config()
    async with serving(config) as (http, _):
        await http.post(
            "/v1/chat/completions",
            headers={"prefer": "respond-async"},
            json=chat_body("chat"),
        )
    # Leaving the context ran the lifespan's shutdown. A pump that outlived it would show
    # up as a task still holding a reference to a queue nobody is filling.
    lingering = [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("sir-job") and not task.done()
    ]
    assert lingering == []


@pytest.mark.parametrize("header", ["respond-async", "wait=10, respond-async", "RESPOND-ASYNC"])
async def test_the_preference_is_parsed_the_way_rfc_7240_writes_it(header):
    async with serving(fast_config()) as (http, _):
        response = await http.post(
            "/v1/chat/completions",
            headers={"prefer": header},
            json=chat_body("chat"),
        )

    assert response.status_code == 202
