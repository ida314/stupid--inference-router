# sir-client

The client half of [`sir`](../../README.md). One call, whether or not the model it names
happens to be loaded.

```python
from sir_client import run_llm

completion = await run_llm(
    "Qwen/Qwen3-8B",
    {"messages": [{"role": "user", "content": "hello"}]},
)
```

`sir` may have to drain, unload another model, load this one, and work through a queue
before that request generates a token. Holding an HTTP connection open across all of that
is how you find every idle timeout between a service and the router, so the router accepts
work asynchronously and hands back a job to poll. This hides that. There is no queue in
the signature.

## What it does

- **Routes by model.** `SIR_BASE_URL` for a catch-all, or `SIR_ENDPOINTS` as `model=url`
  pairs when not everything sits behind one router.
- **Polls at the server's pace.** The router knows the queue depth and how long its
  requests have been taking; its `retry_after` beats any interval hardcoded in a client.
- **Cancels what you abandon.** If your task is cancelled or your deadline passes, the job
  is cancelled too, and the GPU stops working on an answer with nowhere to go.
- **Works against a plain vLLM.** The async opt-in is a `Prefer` header. A server that
  doesn't know it ignores it and answers `200`; `sir` answers `202`. The client branches on
  the status code, so there is nothing to configure and nothing to probe.

## What it doesn't do

It does not read or rewrite your request body. `body` is forwarded exactly as written —
`top_k`, `guided_json`, `ebnf`, whatever the backend you're addressing accepts.

That's on purpose, and it's the same position the router takes. Those extras drift with
every vLLM and SGLang release, and anything that parses them has to be updated in lockstep.
Putting that job in an SDK wouldn't remove the drift — it would turn one process that needs
updating into every service that depends on this package.

## Handling failure

| Raised | Means | Usual response |
|---|---|---|
| `ModelNotRouted` | No endpoint configured for that model | Fix config; raised before anything is sent |
| `JobFailed` | Router accepted it, generation failed | Retry or surface |
| `JobLost` | Result expired, or the router restarted | Resubmit if the work is still wanted |
| `JobCancelled` | Cancelled here, elsewhere, or by lease expiry | Usually expected |
| `RequestTimeout` | Your deadline passed; job cancelled on the way out | Retry or degrade |

`JobLost` is raised rather than retried because replaying a request that may already have
run is a decision with a cost. Opt in with `resubmit_on_loss=True` and an
`idempotency_key`, which the router uses to collapse the duplicate.

## Install

Published from the `sir` repo as its own distribution; `httpx` is its only dependency.

```bash
uv add sir-client
```
