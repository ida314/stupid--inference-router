# Stupid Inference Router (`sir`)

> A model-aware local inference scheduler that presents one API endpoint while dynamically
> allocating a memory-constrained GPU between multiple LLM workloads.

## Motivation

I have one DGX Spark and several homelab services that all want LLM inference — a general chat
model, translation models, an embedding model, a coding model. The box has plenty of compute, but
not enough memory to keep everything resident at once.

The naive fix is to run a vLLM server per model. That breaks down quickly:

- The models collectively don't fit in memory.
- Every service ends up owning inference-server lifecycle, which is not its job.
- Whoever swaps models last wins, and swaps are expensive (tens of seconds).
- A low-traffic model can sit blocked forever behind a busy one.
- Services have to know whether their model is currently loaded.

The real problem isn't serving inference — vLLM already does that well. It's deciding **which model
gets to be resident right now**, given asynchronous requests from independent clients. That decision
needs a single owner.

`sir` is that owner: one endpoint in front, one scheduler deciding model residency behind it.

## What it does

- Exposes a single OpenAI-compatible endpoint that every internal service talks to.
- Accepts requests for any configured model, whether or not that model is currently loaded.
- Queues requests per model, and schedules at the *model* level before the *request* level.
- Loads, drains, and unloads inference backends on demand.
- Batches work into model-serving windows so it doesn't thrash between models.
- Guarantees no model waits indefinitely.

Optimization target is **not** raw throughput. It's minimizing user-visible latency while avoiding
unnecessary model swaps and keeping worst-case wait time bounded.

## The core tension

Everything interesting in this project lives in one trade-off:

**Serving the resident model is free. Serving anything else costs a model swap.**

Lean too far toward the resident model and low-traffic models starve. Lean too far the other way and
the GPU spends its life loading weights instead of generating tokens. The scheduler's whole job is
sitting in the middle of that, and the design choices follow from it:

- **Score models, not requests.** Each model with pending work gets a score from queue age, queue
  depth, and priority, minus the cost of switching to it. Highest score wins the GPU.
- **Hysteresis.** A freshly loaded model stays resident for a minimum period, so alternating
  `A B A B` traffic doesn't produce four swaps.
- **A hard starvation ceiling.** Past a fixed maximum wait, a model becomes mandatory regardless of
  score. Fairness is a correctness property, not a tuning parameter — this is the backstop for when
  I get the weights wrong.
- **Drain before switch.** In-flight generation finishes; new work queues. Nothing gets killed
  mid-response.

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

1. **Policy vs. mechanism.** The scheduler decides *what should run*; the backend manager decides
   *how to run it*. The scheduler never touches a vLLM process directly, so swapping in Triton — or
   a mock — changes nothing above that line.
2. **Clients vs. residency.** Services address logical model names and never learn what's loaded.

## Plan of attack

**Phase 1 — Prototype against a mock backend.** API surface, model registry, per-model queues, and
the scheduler, with a simulated backend whose load and generation times are configurable. The point
is to get scheduler behavior right without ever touching the GPU, and to build the test suite that
proves it: alternating workloads that must not thrash, a starving model that must get served, bursts
that must group, priority ordering, crash recovery, client cancellation.

**Phase 2 — One real vLLM backend.** Implement the backend interface for real: start, health,
generate, stream, cancel. At the end of this phase `sir` is a transparent, boring inference endpoint
that happens to have a scheduler behind it.

**Phase 3 — Real model switching.** Two mutually exclusive models. Drain, unload, load, verify,
dispatch. Measure what swaps actually cost instead of guessing.

**Phase 4 — Scheduler refinement.** Feed real measurements back in: switch-cost estimation, request
cost estimation, minimum residency, priorities. Tune against the Phase 1 test suite plus recorded
production traffic.

**Phase 5 — Make it homelab infrastructure.** Structured logs, metrics, API keys, crash recovery,
config validation, deployment. Runs unattended.

### MVP

Two models that can't be co-resident. Multiple services firing asynchronously at both. Requests are
accepted regardless of what's loaded, the busy model keeps running while switching is inefficient,
waiting requests eventually force a clean drain-unload-load-serve cycle, neither model starves, and
a backend crash doesn't take the router down. Clients only ever see `:8000`.

At that point it's infrastructure rather than a proxy.

## Running it

Phase 1 is implemented: the full API, registry, queues, and scheduler, against a mock backend.

```bash
uv sync --extra dev
uv run pytest                                # the invariants: thrash, starvation, crash, cancel
uv run pytest -s tests/test_policy_sim.py    # decision timelines and the wait/swap trade-off
uv run sir serve --config config.example.yaml
```

Then, from anywhere:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"translate","messages":[{"role":"user","content":"hola"}]}'
curl localhost:8000/status          # residency, queue depths, and the last decision's scores
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
