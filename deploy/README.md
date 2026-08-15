# Deploying `sir` on gx10

`sir` on `:8000`, vLLM behind it on the compose network. The router took over the address
vLLM used to hold, so every service on this box reaches it without changing a line of
config — which is the router's whole claim, and the thing the verification below checks.

```bash
cp config.example.yaml config.yaml          # then edit; config.yaml is gitignored
uv run sir validate -c deploy/config.yaml   # from the repo root
docker compose -f deploy/compose.yaml up -d
```

| | |
|---|---|
| `sir` | `0.0.0.0:8000` — LAN and tailnet, no auth |
| vLLM | `127.0.0.1:8001` — loopback only, for when the question is about the model |
| logs | `docker compose -f deploy/compose.yaml logs -f` (JSON on stdout, nothing to rotate) |
| state | `curl localhost:8000/status` — residency, queue depths, last decision's scores |

## What runs where

```
callers ──► sir :8000 ──► vllm :8000 (internal)
            queues,        nvidia/Qwen3.6-27B-NVFP4
            residency,     the tag is the same string end to end
            async jobs
```

`sir` reads `model` to route and `stream` to pick a rendering. Everything else in the body
is forwarded byte for byte, including fields it has never heard of. `response_format` in
particular arrives intact, which is what keeps Interlinear and job-tracker working.

## The files

| File | What it is |
|---|---|
| `compose.yaml` | Both services. The thirty vLLM flags live here, where they can be read as a unit. |
| `Dockerfile` | The router. No CUDA, no weights — the GPU is entirely behind the backend interface. |
| `config.example.yaml` | Commented template. Copy to `config.yaml`. |
| `BASELINE.md` | What `:8000` did *before* the cutover, captured while vLLM was still serving. The parity target. |
| `ROLLBACK.md` | How to put vLLM back, in two commands or from scratch. |
| `SPIKE-TRITON.md` | Why this is not Triton. Evidence, with file and line citations. |

## Adopt vs own

`config.yaml` sets `manage_lifecycle: false`, so `sir` **adopts** the vLLM container: it
health-checks and forwards, and never starts or stops anything. With one model that is the
right setting twice over — there is nothing to swap to, and it means a network-facing
service with no authentication does not hold the Docker socket, which is root-equivalent on
this box.

Setting it `true` (plus `container_name`) makes `sir` drive the container, which is what a
swap is: stop one server, start the other. That is Phase 3, and it wants a second model
first — `Qwen/Qwen3.6-27B` (bf16) is already in the HF cache and cannot co-fit with the
NVFP4 copy. The compose file would also need to stop starting both eagerly.

## Verifying a deployment

Run from the repo root, in order. Each is a stop-the-line failure.

```bash
uv run pytest                                  # 162 tests, virtual clock, no GPU needed
curl localhost:8000/healthz                    # must list nvidia/Qwen3.6-27B-NVFP4
curl localhost:8000/v1/models
curl localhost:8000/status
```

Then replay `BASELINE.md` §2–§4 through `sir` and compare. **§3 is the acceptance test**:
both `response_format` spellings must come back as JSON that parses, not as prose. A server
that accepts the field and ignores it fails silently — `job-tracker/jobtracker/llm/vllm.py:74-81`
records what that costs — so "it answered" is not a pass.

Last, the real callers, unmodified:

```bash
cd ../Interlinear && BAG_LLM_BASE_URL=http://localhost:8000/v1 uv run python -c "..."
cd ../job-tracker && uv run jobtracker resolve --llm-provider vllm --llm-url http://localhost:8000
```

Neither project may need an edit. That is what makes this a drop-in rather than a migration.

## Two things worth knowing

**A swap costs about five minutes.** Measured from vLLM's own logs: ~113s loading weights,
~146s of engine profiling and warmup, ~30s of container and process startup. This is why
`config.example.yaml` uses `min_residency_seconds: 300` and `max_wait_seconds: 900` rather
than the mock config's 30/120 — at those numbers the starvation ceiling would fire twice
over before a single load finished, and every swap would be mandatory rather than judged.

**`docker kill` does not test crash recovery.** Docker treats it as an explicit operator
stop and deliberately suppresses `restart: unless-stopped` — the container sits in `exited`
with `RestartCount=0` even on exit 137. To simulate a real crash, kill the engine process
*inside* the container:

```bash
pid=$(docker exec sir-vllm-qwen36 bash -c "ps -eo pid,args --no-headers | grep EngineCore | grep -v grep | head -1 | awk '{print \$1}'")
docker exec sir-vllm-qwen36 kill -9 "$pid"
```

Verified: the policy fires within ten seconds and the engine reloads unattended. Throughout
both the killed-container and crashed-process cases, `sir` stays up and keeps answering
`/healthz` — `api.py:86` makes it independent of backend state on purpose — requests fail
with a clean 503 rather than hanging, and it resumes serving when the engine returns without
needing a restart of its own.
