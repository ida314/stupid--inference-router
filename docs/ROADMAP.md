# Plan of Attack

**Phase 1 — Prototype against a mock backend. Done.** API surface, model registry,
per-model queues, and the scheduler, with a simulated backend whose load and generation
times are configurable. The point is to get scheduler behavior right without ever touching
the GPU, and to build the test suite that proves it: alternating workloads that must not
thrash, a starving model that must get served, bursts that must group, priority ordering,
crash recovery, client cancellation.

**Phase 1b — Async submission and a client SDK. Done.** `Prefer: respond-async` returns a
job to poll instead of a held-open connection, carrying the queue position, whether a swap
is pending, and an advisory estimate. Polling is the liveness signal that a socket used to
provide, so an abandoned job is cancelled on a lease rather than competing for the GPU
forever. `clients/python` and `clients/typescript` ship the client half from this repo, so
the wire format has one home: Python's contract tests drive the real SDK against the real
app in-process, and TypeScript's spawn the real router on a free port. Both clients route
between endpoints and never translate between provider dialects, which is what keeps them
thin enough to be worth having two of.

The per-model running average of request duration added here is the first measured input
to Phase 4's cost estimation — until then it only paces polling.

**Phase 2 — One real vLLM backend. Done.** `sir/backends/vllm.py` implements the interface
against a vLLM server: health, stream, cancel, and — behind `manage_lifecycle` — start and
stop by driving the container over the Docker Engine API. Request bodies arrive in vLLM's
own shape and are forwarded unmodified, so this phase was process lifecycle, not
translation. The mock's word-count `usage` is gone: a backend may now end its own stream
with a `StreamEnd` carrying the real finish reason and token counts, and the engine forwards
it untouched. `sir` is deployed on this box as a transparent inference endpoint that happens
to have a scheduler behind it — see `deploy/`.

Triton was tried first and rejected: its OpenAI frontend accepts `response_format` and never
passes it to vLLM, which silently breaks constrained decoding for every caller here.
`deploy/SPIKE-TRITON.md` has the evidence. The measurement that made the decision easy is
that a swap costs ~5 minutes on this hardware and only ~30s of that is process startup, so
Triton's in-process load/unload was buying about 10%.

**Phase 3 — Real model switching.** Two mutually exclusive models. Drain, unload, load,
verify, dispatch. The mechanism exists — `manage_lifecycle: true` makes stopping a container
the unload — but nothing has exercised it, because the deployment runs one model and a
single model never swaps. This phase starts by adding the second: `Qwen/Qwen3.6-27B` (bf16)
is already in the HF cache and cannot co-fit with the NVFP4 copy.

Swap cost is no longer a guess. Measured from vLLM's own logs on this box: ~113s loading
weights, ~146s of engine profiling and warmup, ~30s of container and process startup —
about five minutes, cold.

**Phase 4 — Scheduler refinement.** Feed real measurements back in: switch-cost
estimation, request cost estimation, minimum residency, priorities. Tune against the Phase
1 test suite plus recorded production traffic. The three-model case is where the current
defaults visibly strain — the simulator has it spending roughly a quarter of wall clock
loading weights — so start there.

The first thing that measurement changes is the defaults themselves. `config.example.yaml`
pairs `min_residency_seconds: 30` with `max_wait_seconds: 120`, which is right for a mock
that loads in eight seconds and an order of magnitude too small for weights that take five
minutes: the starvation ceiling would fire twice over before a single load finished, making
every swap mandatory and the scores decorative. `deploy/config.example.yaml` uses 300/900
instead.

**Phase 5 — Make it homelab infrastructure.** Structured logs, metrics, API keys, crash
recovery, config validation, deployment. Runs unattended.

Deployment landed early, out of order, because Phase 2 had to be deployed to be worth
anything: `deploy/` holds the compose stack, the config, and the cutover notes. Still
outstanding here are metrics and **API keys** — `sir` binds `0.0.0.0:8000` with no
authentication and no TLS. That is the exposure vLLM already had on this box, so the
deployment is not a regression, but it is the reason the router should stay on the LAN.

## Deferred

**Provider adapters.** Accepting a vLLM-shaped request and rewriting it for an SGLang
backend, or the reverse. Not needed while every model is addressed by the tag its own
backend serves, and the pass-through in the API layer is where such a thing would go if
that ever changes. See the wire format section of the README.

**Extra endpoints.** `/v1/completions` and `/v1/embeddings`. Embeddings in particular have
a different lifecycle — no streaming, tiny requests — and probably shouldn't be
swap-scheduled the same way chat models are.
