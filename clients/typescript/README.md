# sir-client (TypeScript)

The client half of [`sir`](../../README.md) for TypeScript services. One call, whether or
not the model it names happens to be loaded.

```ts
import { runLlm } from "sir-client";

const completion = await runLlm("Qwen/Qwen3-8B", {
  messages: [{ role: "user", content: "hello" }],
});
```

`sir` may have to drain, unload another model, load this one, and work through a queue
before that request generates a token. Holding an HTTP connection open across all of that
is how you find every idle timeout between a service and the router, so the router accepts
work asynchronously and hands back a job to poll. This hides that. There is no queue in the
signature.

Deliberately the same surface as [the Python client](../python/README.md) — two SDKs that
behave the same way mean a bug found in one is a bug found in both.

## What it does

- **Routes by model.** `SIR_BASE_URL` for a catch-all, or `SIR_ENDPOINTS` as `model=url`
  pairs when not everything sits behind one router.
- **Polls at the server's pace.** The router knows the queue depth and how long its
  requests have been taking; its `retry_after` beats any interval hardcoded in a client.
- **Cancels what you abandon.** Abort the signal or let the deadline pass, and the job is
  cancelled too — the GPU stops working on an answer with nowhere to go.
- **Works against a plain vLLM.** The async opt-in is a `Prefer` header. A server that
  doesn't know it ignores it and answers `200`; `sir` answers `202`. The client branches on
  the status code, so there is nothing to configure and nothing to probe.

## Cancellation

The one behaviour worth reading the code for. An `AbortSignal` reaches all the way down to
the scheduler:

```ts
const controller = new AbortController();
setTimeout(() => controller.abort(), 5_000);

try {
  await runLlm("translate", body, { signal: controller.signal });
} catch (error) {
  // The job has already been cancelled on the router.
}
```

A deadline does the same thing without the ceremony:

```ts
await runLlm("translate", body, { timeoutMs: 5_000 }); // throws RequestTimeout
```

If you never wait on a job at all, cancel it explicitly — nothing else will until its lease
lapses:

```ts
const job = await submitLlm("translate", body);
await job.cancel();
```

## What it doesn't do

It does not read or rewrite your request body. `body` is forwarded exactly as written —
`top_k`, `guided_json`, `ebnf`, whatever the backend you're addressing accepts. The type is
`{ messages: unknown[], [key: string]: unknown }` on purpose: a type that enumerated the
sampling parameters would be claiming knowledge this package has decided not to have, and
would reject a field the day a backend adds one.

That's the same position the router takes. Those extras drift with every vLLM and SGLang
release, and anything that parses them has to be updated in lockstep. Putting that job in an
SDK wouldn't remove the drift — it would turn one process that needs updating into every
service that depends on this package.

No streaming, either. Jobs return complete responses; if you need tokens as they arrive,
use the router's `stream: true` endpoint directly and accept that it holds the connection
open.

## Handling failure

| Thrown | Means | Usual response |
|---|---|---|
| `ModelNotRouted` | No endpoint configured for that model | Fix config; thrown before anything is sent |
| `JobFailed` | Router accepted it, generation failed | Retry or surface |
| `JobLost` | Result expired, or the router restarted | Resubmit if the work is still wanted |
| `JobCancelled` | Cancelled here, elsewhere, or by lease expiry | Usually expected |
| `RequestTimeout` | Your deadline passed; job cancelled on the way out | Retry or degrade |
| `TransportError` | Any other non-2xx | Inspect `.status` |

`JobLost` is thrown rather than retried because replaying a request that may already have
run is a decision with a cost. Opt in with `resubmitOnLoss: true` and an `idempotencyKey`,
which the router uses to collapse the duplicate. Without a key the option is ignored — an
unkeyed replay is exactly the double-generation it is meant to avoid.

## Install

```jsonc
// package.json
"dependencies": {
  "sir-client": "github:ida314/stupid-inference-router#v0.1.0"
}
```

npm builds it on install via the `prepare` script. No runtime dependencies: built-in
`fetch` and `AbortSignal` only, so it runs on Node 20.3+, Deno, Bun and the browser
unchanged.

For a service making more than the occasional call, build a `Client` once and reuse it
rather than going through `runLlm`:

```ts
import { Client } from "sir-client";

export const llm = new Client({ baseUrl: process.env.SIR_BASE_URL, timeoutMs: 120_000 });
```

## Tests

```bash
npm test          # units, plus a contract suite against a real spawned router
npm run check     # types only
```

The contract suite spawns the actual `sir serve` on a free port rather than mocking it —
what it asserts is the agreement between two codebases, and a mock of the router would
agree with itself forever while drifting from the thing it stands in for. It skips itself
if the Python package isn't installed.
