# Stupid Inference Router (`sir`)

> A model-aware local inference scheduler for my homelab that presents one API endpoint while dynamically
> allocating a memory-constrained GPU between multiple LLM workloads.

## Motivation

I have one DGX Spark and several homelab services that all want LLM inference. Things like a general chat
model, translation models, an embedding model, a coding model. The box has plenty of compute, but not enough memory to keep everything resident at once.

The naive fix is to run a vLLM server per model. That breaks down quickly:

- The models collectively don't fit in memory.
- Every service ends up owning inference-server lifecycle, which is not its job.
- Whoever swaps models last wins, and swaps are expensive (tens of seconds).
- A low-traffic model can sit blocked forever behind a busy one.
- Services have to know whether their model is currently loaded.

The real problem isn't serving inference — vLLM already does that well. It's deciding **which model gets to be resident right now**, given asynchronous requests from independent clients. That decision needs a single owner.

`sir` is that owner: one endpoint in front, one scheduler deciding model residency behind it.

## What it does

- Exposes a single OpenAI-compatible endpoint that every internal service talks to.
- Accepts requests for any configured model, whether or not that model is currently loaded.
- Queues requests per model, and schedules at the model level before the request level.
- Loads, drains, and unloads inference backends on demand.
- Batches work into model-serving windows so it doesn't thrash between models.
- Guarantees no model waits indefinitely.

Optimization target is not raw throughput. It's minimizing user-visible latency while avoiding unnecessary model swaps and keeping worst-case wait time bounded.

## The core tension

Everything interesting in this project lives in one trade-off:
Serving the resident model is free. Serving anything else costs a model swap.

Lean too far toward the resident model and low-traffic models starve. Lean too far the other way and the GPU spends its life loading weights instead of generating tokens. The scheduler's whole job is sitting in the middle of that, and the design choices follow from it:

- Score models, not requests. Each model with pending work gets a score from queue age, queue depth, and priority, minus the cost of switching to it. Highest score wins the GPU.
- Hysteresis. A freshly loaded model stays resident for a minimum period, so alternating
  `A B A B` traffic doesn't produce four swaps.
- A hard starvation ceiling. Past a fixed maximum wait, a model becomes mandatory regardless of score. Fairness is a correctness property, not a tuning parameter — this is the backstop for when I get the weights wrong.
- Drain before switch. In-flight generation finishes; new work queues. Nothing gets killedmid-response.

## Architecture

```text
  Service A   Service B   Service C   Service D
      └───────────┴─────┬─────┴───────────┘
                        ▼
              ┌──────────────────┐
              │       sir        │   API layer  ─ validate, enqueue, stream back
              │      :8000       │   Scheduler  ─ which model runs next?
              └────────┬─────────┘   Registry   ─ what models exist, what they cost
                       ▼
              Backend Manager (pluggable)
                       ▼
              vLLM / Triton / Mock
                       ▼
                   DGX Spark
```

Two boundaries matter:

1. Policy vs. mechanism. The scheduler decides what should run and the backend manager decides how to run it. The scheduler never touches a vLLM process directly, so swapping in Triton or a mock changes nothing above that line.
2. Clients vs. residency. Services address model names and never learn what's loaded.

## Wire format

A service talks to `sir` exactly as it would talk to vLLM directly. Same endpoint paths, same request body, same `model` string — the tag that backend actually serves, not a nickname `sir` invented:

```bash
# straight at vLLM
curl vllm-host:8000/v1/chat/completions -d '{"model":"Qwen/Qwen3-8B", ...}'
# through sir, byte-for-byte identical
curl sir-host:8000/v1/chat/completions  -d '{"model":"Qwen/Qwen3-8B", ...}'
```

Config carries both identities. `name` is internal: it labels the model in logs and `/status`, because a swap timeline reading `chat -> translate` is legible and one reading `Qwen/Qwen3-8B -> facebook/nllb-200-distilled-600M` is not. `served_model_name` is what clients send, and matches what you would pass to vLLM's `--served-model-name`, aliases included. The internal label is not routable.

`sir` reads two fields out of a request body: `model`, to pick a queue, and `stream`, to pick a response shape. Everything else is forwarded untouched.

### Do vLLM and SGLang accept the same thing?

The OpenAI core, yes — both serve `/v1/chat/completions` with the same standard fields. The extras are where they part. Both add sampling knobs OpenAI doesn't have, and the sets overlap without matching: `top_k`, `min_p`, `repetition_penalty`, `ignore_eos` and `stop_token_ids` are common to both, while structured output diverges outright, vLLM having used `guided_json` / `guided_regex` / `guided_grammar` where SGLang uses `json_schema` / `regex` / `ebnf`. SGLang also has a native `/generate` endpoint in an entirely different shape. Both sets move between releases.

That divergence argues against adapters rather than for them. Forwarding the body as sent means whatever the configured backend accepts, its clients can send — including fields that didn't exist when this was written. An adapter layer would have to model every extra, get updated on every release of both projects, and would quietly drop whatever it hadn't learned yet. The only thing adapters would buy is translating a vLLM-shaped request into an SGLang-shaped one, and nothing here needs that: each model is addressed by the tag its own backend serves. If that changes, the pass-through is the layer an adapter would slot into.

### Not holding the socket open

A request can wait through a drain, an unload, a load, and a queue before it generates a token. Holding an HTTP connection open across all of that is how you find every idle timeout between a service and the router, so a client can ask to be given a handle instead:

```bash
curl -i sir-host:8000/v1/chat/completions -H 'prefer: respond-async' -d '{"model":"Qwen/Qwen3-8B", ...}'
# 202 Accepted
# location: /v1/jobs/req_ab12cd34
```

Then `GET /v1/jobs/{id}` until `status` is terminal, and `DELETE` to give up early.

The opt-in is a **header, not a body field** — `sir` still reads only `model` and `stream`, and the body is still forwarded byte-for-byte. It also means the same client code works against a plain vLLM, which ignores the unknown header and answers `200` with the completion. There is nothing to negotiate: `202` means you got a job, `200` means you got the answer.

While a job is queued it reports the wait, split into what the scheduler knows and what it is guessing:

```json
{"position": 12, "resident": false, "needs_swap": true, "load_seconds": 8.0,
 "dispatch_within_seconds": 94.2, "estimated_seconds": 47.5}
```

Everything but the last field is read off scheduler state and is exact. `estimated_seconds` is a projection over a running average of past request durations — good enough to pace polling, not a deadline. And `dispatch_within_seconds` is a **head-of-queue** bound: it says when the *model* is guaranteed the GPU under the starvation ceiling, not when your request finishes. Behind fifty queued requests, the model is served within that window and you are still fiftieth.

Polling doubles as the liveness signal. A held-open socket is what tells `sir` today that a client still wants its answer; a job has no socket, so a job nobody polls for is cancelled once its lease lapses, and abandoned work stops costing GPU time. The default lease is half the starvation ceiling (60s against 120s), which is what stops abandoned work from ever reaching the point where a swap becomes mandatory.

Most services shouldn't hand-roll any of this. Two clients ship from this repo, with the same surface: [`clients/python`](clients/python/README.md) and [`clients/typescript`](clients/typescript/README.md). Both route between endpoints and forward bodies untouched, exactly as the router does — the drift argued against above doesn't become an SDK's problem either, it just moves from one process into every service's pinned version. What they're actually for is making sure a caller that gives up cancels its job, so an abandoned request stops costing GPU time.

## Implementation Roadmap

[docs/ROADMAP.md](github.com/ida314/stupid-inference-router)

### MVP

Two models that can't be co-resident. Multiple services firing asynchronously at both. Requests are
accepted regardless of what's loaded, the busy model keeps running while switching is inefficient,
waiting requests eventually force a clean drain-unload-load-serve cycle, neither model starves, and
a backend crash doesn't take the router down. Clients only ever see `:8000`.

At that point it's infrastructure rather than a proxy.

## Running it

Currently phase 1 is finished, against a mock backend.

```bash
uv sync --extra dev
uv run pytest                                # the invariants: thrash, starvation, crash, cancel
uv run pytest -s tests/test_policy_sim.py    # decision timelines and the wait/swap trade-off
uv run sir serve --config config.example.yaml
```

Then, from anywhere:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"facebook/nllb-200-distilled-600M","messages":[{"role":"user","content":"hola"}]}'
curl localhost:8000/v1/models       # the tags clients may address
curl localhost:8000/status          # residency, queue depths, and the last decision's scores

# or, without holding the connection open for the wait
curl -i localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'prefer: respond-async' \
  -d '{"model":"translate","messages":[{"role":"user","content":"hola"}]}'
curl localhost:8000/v1/jobs/<id>    # position, swap, estimate — then the response
```

From a service, use a client rather than the endpoint:

```python
from sir_client import run_llm

completion = await run_llm("translate", {"messages": [{"role": "user", "content": "hola"}]})
```

```ts
import { runLlm } from "sir-client";

const completion = await runLlm("translate", { messages: [{ role: "user", content: "hola" }] });
```

Two knobs to reach for first, both under `scheduler:` in the config:
`min_residency_seconds` (how strongly to resist swapping) and `max_wait_seconds` (the
starvation ceiling). Every scheduling decision is logged with the score breakdown that
produced it, so a swap you didn't expect should be answerable from one log line.

The whole suite runs on a simulated clock — two minutes of scheduler behavior in
milliseconds — so scenarios are written in realistic seconds rather than scaled-down
constants.

## Non-goals

No distributed or multi-node inference, no Kubernetes, no cloud provider fallback, no training or
fine-tuning, no billing, no thousand-user scale, no RL-based scheduling, no exact runtime prediction.

This targets ~5 services and a handful of users. The scheduler should stay simple enough that I can
read a log line and understand why it made a decision.

## Design principles

- **Predictable beats clever.** A scheduler that explains its decisions beats an opaque optimizer.
- **Model switches are expensive.** Optimize residency on a much longer timescale than requests.
- **Separate policy from mechanism.** See above; this is what keeps backends swappable.
- **Measure before optimizing.** Collect load times and request durations before adding prediction.
- **Never allow indefinite starvation.** Non-negotiable.

## Later, maybe

Multiple resident models via real memory accounting; LoRA adapters instead of full swaps when
workloads share a base model; speculative preloading from traffic history; request deadlines and
admission control; replaying recorded workloads against alternative policies; a status dashboard.

None of these get built before there's data justifying them.

## Stack

Python 3.12+, FastAPI, asyncio, Pydantic, YAML config, vLLM, pytest. In-memory state — no database
in v1.
