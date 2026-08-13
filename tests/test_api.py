"""The contract clients actually see.

The MVP claim under test: a service points at `:8000`, asks for a model by name, and gets
a response — whether or not that model was loaded when it asked, and without ever learning
which one is resident.

These run on the real clock with a deliberately tiny mock backend, because what's being
checked here is the HTTP surface, not the scheduling policy.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI

from sir.api import create_app
from tests.sim import build_config, model


def fast_config():
    """Everything scaled down so the API tests run in milliseconds."""
    return build_config(
        models=[
            model(
                "chat",
                priority=1.0,
                load_seconds=0.05,
                unload_seconds=0.01,
                tokens_per_second=2000,
                first_token_seconds=0.0,
                default_max_tokens=8,
            ),
            model(
                "translate",
                priority=1.0,
                load_seconds=0.05,
                unload_seconds=0.01,
                tokens_per_second=2000,
                first_token_seconds=0.0,
                default_max_tokens=8,
            ),
        ],
        min_residency_seconds=0.05,
        max_wait_seconds=1.0,
        tick_interval_seconds=0.01,
        backend_retry_seconds=0.1,
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(fast_config())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://sir.test", timeout=30
        ) as http:
            yield http


def chat_body(model_name: str, **extra: object) -> dict[str, object]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


# ---------------------------------------------------------------- discovery


async def test_healthz_reports_the_router_not_the_backends(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": ["chat", "translate"]}


async def test_models_are_listed_in_openai_shape(client):
    body = (await client.get("/v1/models")).json()
    assert body["object"] == "list"
    assert [card["id"] for card in body["data"]] == ["chat", "translate"]
    assert all(card["object"] == "model" for card in body["data"])


# ---------------------------------------------------------------- completions


async def test_a_chat_completion_comes_back_in_openai_shape(client):
    response = await client.post("/v1/chat/completions", json=chat_body("chat"))
    assert response.status_code == 200

    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "chat"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert "[chat]" in message["content"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] > 0


async def test_a_request_for_a_model_that_is_not_loaded_is_still_served(client):
    """The point of the whole project: clients don't know about residency."""
    first = await client.post("/v1/chat/completions", json=chat_body("chat"))
    assert first.status_code == 200
    assert (await client.get("/status")).json()["resident"] == "chat"

    # translate is not resident. The client neither knows nor cares.
    second = await client.post("/v1/chat/completions", json=chat_body("translate"))
    assert second.status_code == 200
    assert "[translate]" in second.json()["choices"][0]["message"]["content"]


async def test_concurrent_requests_for_both_models_all_complete(client):
    """Several services firing asynchronously at both models — the MVP scenario."""
    responses = await asyncio.gather(
        *(
            client.post("/v1/chat/completions", json=chat_body(name))
            for name in ("chat", "translate", "chat", "translate", "chat")
        )
    )
    assert [r.status_code for r in responses] == [200] * 5
    for response, expected in zip(
        responses, ("chat", "translate", "chat", "translate", "chat")
    ):
        assert f"[{expected}]" in response.json()["choices"][0]["message"]["content"]


async def test_max_tokens_is_honoured(client):
    response = await client.post(
        "/v1/chat/completions", json=chat_body("chat", max_tokens=3)
    )
    assert response.json()["usage"]["completion_tokens"] == 3


async def test_unknown_fields_from_real_clients_are_tolerated(client):
    response = await client.post(
        "/v1/chat/completions", json=chat_body("chat", top_p=0.9, seed=1, user="svc")
    )
    assert response.status_code == 200


# ---------------------------------------------------------------- streaming


async def test_streaming_emits_sse_chunks_and_terminates_with_done(client):
    lines: list[str] = []
    async with client.stream(
        "POST", "/v1/chat/completions", json=chat_body("chat", stream=True)
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                lines.append(line.removeprefix("data: "))

    assert lines[-1] == "[DONE]"

    payloads = [json.loads(line) for line in lines[:-1]]
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[0]["object"] == "chat.completion.chunk"

    text = "".join(
        p["choices"][0]["delta"].get("content") or "" for p in payloads
    )
    assert "[chat]" in text
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_works_for_a_model_that_has_to_be_loaded_first(client):
    await client.post("/v1/chat/completions", json=chat_body("chat"))

    chunks: list[str] = []
    async with client.stream(
        "POST", "/v1/chat/completions", json=chat_body("translate", stream=True)
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(line)
    assert any("[translate]" in chunk for chunk in chunks)


# ---------------------------------------------------------------- errors and status


async def test_an_unknown_model_is_a_404_in_openai_error_shape(client):
    response = await client.post("/v1/chat/completions", json=chat_body("gpt-9"))
    assert response.status_code == 404

    error = response.json()["error"]
    assert "gpt-9" in error["message"]
    assert "chat" in error["message"]  # tells the caller what it could have asked for
    assert error["code"] == "model_not_found"


async def test_a_malformed_request_is_a_422(client):
    response = await client.post("/v1/chat/completions", json={"model": "chat"})
    assert response.status_code == 422


async def test_status_exposes_residency_queues_and_the_last_decision(client):
    await client.post("/v1/chat/completions", json=chat_body("chat"))
    status = (await client.get("/status")).json()

    assert status["resident"] == "chat"
    assert status["resident_for_seconds"] >= 0
    assert status["loads"] == 1
    assert {m["name"] for m in status["models"]} == {"chat", "translate"}
    assert all(m["queue_depth"] == 0 for m in status["models"])

    decision = status["last_decision"]
    assert decision["kind"] in {"idle", "serve", "hold", "switch"}
    assert decision["reason"]
    # The score breakdown is the audit trail; it must survive to the debug surface.
    assert any("chat" in line for line in decision["scores"])
