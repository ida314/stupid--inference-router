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

**Phase 2 — One real vLLM backend.** Implement the backend interface for real: start,
health, generate, stream, cancel. Request bodies already arrive in vLLM's own shape and
are forwarded unmodified, so this phase is about process lifecycle, not translation.
Replace the mock's word-count `usage` with the backend's real token counts. At the end of
this phase `sir` is a transparent, boring inference endpoint that happens to have a
scheduler behind it.

**Phase 3 — Real model switching.** Two mutually exclusive models. Drain, unload, load,
verify, dispatch. Measure what swaps actually cost instead of guessing.

**Phase 4 — Scheduler refinement.** Feed real measurements back in: switch-cost
estimation, request cost estimation, minimum residency, priorities. Tune against the Phase
1 test suite plus recorded production traffic. The three-model case is where the current
defaults visibly strain — the simulator has it spending roughly a quarter of wall clock
loading weights — so start there.

**Phase 5 — Make it homelab infrastructure.** Structured logs, metrics, API keys, crash
recovery, config validation, deployment. Runs unattended.

## Deferred

**Provider adapters.** Accepting a vLLM-shaped request and rewriting it for an SGLang
backend, or the reverse. Not needed while every model is addressed by the tag its own
backend serves, and the pass-through in the API layer is where such a thing would go if
that ever changes. See the wire format section of the README.

**Extra endpoints.** `/v1/completions` and `/v1/embeddings`. Embeddings in particular have
a different lifecycle — no streaming, tiny requests — and probably shouldn't be
swap-scheduled the same way chat models are.
