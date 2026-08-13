"""The SDK against the real router, in-process.

Both halves of the wire format live in this repo precisely so this test can exist: the
client is driven against the actual ASGI app through `httpx.ASGITransport`, with no
sockets, no ports, and no stub of the server's behaviour to drift out of date. If the
router changes what a job document looks like, this fails in the same commit.

What's asserted here is the SDK's three jobs — hide the queue, don't touch the body, and
make sure a caller that gives up stops costing GPU time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import sir_client
from sir_client import AsyncClient, JobLost, ModelNotRouted, Registry, RequestTimeout

from sir.api import create_app
from sir.config import JobsConfig
from tests.test_api import RecordingBackend, chat_body, fast_config
from tests.test_jobs import slow_config

BASE_URL = "http://sir.test"


@asynccontextmanager
async def sdk(config=None) -> AsyncIterator[tuple[AsyncClient, object, list[dict]]]:
    """A client wired straight into the app, plus what the backends actually received."""
    seen: list[dict] = []

    def factory(model_config, clock):
        return RecordingBackend(model_config, clock, seen)

    app = create_app(config or fast_config(), backend_factory=factory)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, timeout=30) as http:
            client = AsyncClient(http=http, base_url=BASE_URL)
            yield client, app, seen


# ---------------------------------------------------------------- the common case


async def test_run_returns_a_completion_and_never_mentions_the_queue():
    async with sdk() as (client, _, _seen):
        completion = await client.run(
            "translate", {"messages": [{"role": "user", "content": "hola"}]}
        )

    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"]["role"] == "assistant"
    assert completion["choices"][0]["message"]["content"]


async def test_the_request_body_reaches_the_backend_as_written():
    """The SDK routes between endpoints; it does not translate between dialects.

    Provider extras are exactly what a translating client would mangle, so they are what
    this checks.
    """
    async with sdk() as (client, _, seen):
        await client.run(
            "chat",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": 40,
                "guided_json": {"type": "object"},
                "some_field_invented_next_release": True,
            },
        )

    assert seen[0]["top_k"] == 40
    assert seen[0]["guided_json"] == {"type": "object"}
    assert seen[0]["some_field_invented_next_release"] is True


async def test_a_synchronous_endpoint_needs_no_special_handling():
    """The compatibility story: `200` means the preference was ignored, and that's fine.

    A plain vLLM ignores `Prefer` and answers with the completion. Standing in for one
    here is `sir` with jobs disabled, which behaves identically from the client's side.
    """
    config = fast_config()
    config.jobs = JobsConfig(enabled=False)

    async with sdk(config) as (client, _, _seen):
        completion = await client.run("chat", {"messages": [{"role": "user", "content": "hi"}]})

    assert completion["object"] == "chat.completion"


async def test_submit_hands_back_the_routers_account_of_the_wait():
    async with sdk() as (client, _, _seen):
        job = await client.submit(
            "translate", {"messages": [{"role": "user", "content": "hola"}]}
        )
        assert job.status in {"queued", "running"}
        assert job.wait["needs_swap"] is True

        completion = await job.result()

    assert completion["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- giving up


async def test_cancelling_the_caller_cancels_the_job():
    """The invariant this SDK exists to carry.

    A held-open socket is what tells the router today that a client is still there. Once
    nobody is holding one, this is the only thing that stops the GPU generating an answer
    with nowhere to go.
    """
    async with sdk(slow_config()) as (client, app, _seen):
        job = await client.submit("chat", chat_body("chat"))
        waiter = asyncio.create_task(job.result())
        await asyncio.sleep(0.1)  # let it get into the poll loop

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        await asyncio.sleep(0.05)  # let the DELETE land
        document = app.state.jobs.get(job.id)

    assert document.queued.cancelled.is_set()


async def test_a_deadline_cancels_the_job_rather_than_orphaning_it():
    async with sdk(slow_config()) as (client, app, _seen):
        job = await client.submit("chat", chat_body("chat"))
        with pytest.raises(RequestTimeout):
            await job.result(timeout=0.15)

        await asyncio.sleep(0.05)
        document = app.state.jobs.get(job.id)

    assert document.queued.cancelled.is_set()


# ---------------------------------------------------------------- a job that vanishes


async def test_a_vanished_job_is_raised_not_silently_replayed():
    """Resubmitting work that may already have run is the caller's call, not the SDK's."""
    async with sdk(slow_config()) as (client, app, _seen):
        job = await client.submit("chat", chat_body("chat"))
        # Stand in for a result that expired, or a router that restarted.
        app.state.jobs._forget(app.state.jobs.get(job.id))

        with pytest.raises(JobLost):
            await job.result()


async def test_resubmission_after_a_loss_is_opt_in():
    async with sdk(slow_config()) as (client, app, seen):
        original = await client.run(
            "chat",
            chat_body("chat"),
            idempotency_key="key-1",
            resubmit_on_loss=True,
        )
        assert original["object"] == "chat.completion"

    # One completed generation; the opt-in path costs nothing when nothing goes wrong.
    assert len(seen) == 1


async def test_an_idempotency_key_collapses_a_duplicate_submission():
    async with sdk() as (client, _, seen):
        first = await client.submit("chat", chat_body("chat"), idempotency_key="k")
        second = await client.submit("chat", chat_body("chat"), idempotency_key="k")
        assert first.id == second.id
        await first.result()

    assert len(seen) == 1


# ---------------------------------------------------------------- routing


def test_an_unrouted_model_fails_before_anything_is_sent():
    registry = Registry({"chat": "http://gpu:8000"})

    with pytest.raises(ModelNotRouted) as raised:
        registry.resolve("something-else")

    assert "chat" in str(raised.value)  # tells you what it does know


def test_endpoints_come_from_the_environment():
    registry = Registry.from_env(
        {
            "SIR_ENDPOINTS": "Qwen/Qwen3-8B=http://gpu:8000, bge-m3=http://cpu:8001/",
            "SIR_BASE_URL": "http://router:8000",
        }
    )

    assert registry.resolve("Qwen/Qwen3-8B") == "http://gpu:8000"
    assert registry.resolve("bge-m3") == "http://cpu:8001"
    assert registry.resolve("anything-else") == "http://router:8000"


def test_a_malformed_endpoint_list_is_rejected_at_startup():
    with pytest.raises(ValueError):
        Registry.from_env({"SIR_ENDPOINTS": "no-url-here"})


async def test_discovery_reads_the_tags_the_router_advertises():
    async with sdk() as (client, _, _seen):
        registry = await Registry().discover(client._http, [BASE_URL])

    # Every tag `/v1/models` advertises, and only those.
    assert registry.models == ["chat", "translate"]
    assert registry.resolve("translate") == BASE_URL


async def test_two_hosts_claiming_one_tag_is_an_error_not_a_coin_flip():
    """Same reasoning as the router's duplicate `served_model_name` check."""
    async with sdk() as (client, _, _seen):
        registry = await Registry().discover(client._http, [BASE_URL])
        with pytest.raises(ValueError, match="routing would depend on discovery order"):
            await registry.discover(client._http, ["http://other.test"])


# ---------------------------------------------------------------- module surface


def test_the_package_exports_what_the_readme_promises():
    for name in ("run_llm", "submit_llm", "AsyncClient", "Registry", "JobLost"):
        assert hasattr(sir_client, name), name
