# Spike: can Triton serve this box's model? — 2026-08-14

Run against `nvcr.io/nvidia/tritonserver:26.06-vllm-python-py3` and `:26.07-...` (arm64,
the newest published tag). Nothing was torn down; the vLLM container stayed up throughout.

**Result: check 4 passes, check 1 is fine, check 2 fails on both tags.** The failure is in
Triton's OpenAI frontend, not in the hardware or the model, and it is not version-specific.

## Check 4 — explicit model control: PASS

`openai_frontend/main.py` takes `--model-control-mode {none,explicit}`,
`--enable-kserve-frontends`, `--kserve-http-port` and `--openai-port` together. One process
can serve generation on the OpenAI port while exposing the repository load/unload API on the
KServe port, which is exactly what the backend design needs. This part of the idea is sound
and `src/sir/backends/triton.py` is written against it.

One correction it forced: the OpenAI frontend advertises models under their **Triton
repository name**, built from `metadata.name` in `engine/triton_engine.py:156-176`. It
ignores vLLM's `served_model_name`. So `sir` must rewrite the outgoing `model` field, and
does.

## Check 1 — model support: fine

`vllm.model_executor.models.registry` in the 26.06 image knows
`Qwen3_5ForConditionalGeneration`, which is what `nvidia/Qwen3.6-27B-NVFP4`'s config
declares. Bundled vLLM is 0.22.1 on 26.06 and 0.24.0 on 26.07, against 0.23.1rc1 in the
working `eugr/spark-vllm` build.

Not proven: that NVFP4/modelopt kernels actually run on sm_121 under those builds. That test
was never reached, because check 2 failed first.

## Check 2 — `response_format`: FAIL, on both tags

Triton's OpenAI frontend **accepts `response_format` and never passes it to vLLM.**

In `openai_frontend/engine/utils/triton.py`, `_create_vllm_generate_request` builds the
sampling parameters and drops the field by name:

```python
    # Exclude non-sampling parameters so they aren't passed to vLLM
    excludes = {
        ...
        "response_format",
```

Structured output *is* wired up — a few lines below, the request can set
`sampling_parameters["structured_outputs"] = StructuredOutputsParams(json=...)` — but the
schema for it comes only from `_get_guided_json_from_tool()`, which reads `tools` and
`tool_choice` and returns `None` for everything else. `response_format` appears exactly
twice in the whole frontend: once in the request schema that accepts it, once in the set
that discards it. Nothing else reads it. Identical on 26.06 and 26.07.

The request schema is narrower still. `ResponseFormat` models a single field,
`type: Type6`, and `Type6` is an enum of exactly `text` and `json_object`
(`schemas/openai.py:461-471`). There is no `json_schema` member at all.

### What that does to each caller

| Caller | Sends | Under Triton |
|---|---|---|
| **Interlinear** (`translate.py:95`) | `{"type": "json_object"}` | Parses, then silently dropped → **unconstrained prose** |
| **job-tracker** (`vllm.py:90-93`) | `{"type": "json_schema", "json_schema": {...}}` | Not a valid enum value → **HTTP 422** |

The first is the failure mode `job-tracker/jobtracker/llm/vllm.py:74-81` already documents
from the `guided_json` era: a server that accepts the field, ignores it, and answers in
prose, so every verdict is rejected and a nightly pass resolves nothing while still paying
for a description fetch per posting. Nothing errors. `deploy/BASELINE.md` §3 records both
requests working correctly against the vLLM that runs today.

## Cost of making Triton work anyway

Patching the frontend inside a derived image. Two vendor files:

1. `schemas/openai.py` — add `json_schema` to `Type6` and a field to `ResponseFormat` to
   carry the schema.
2. `engine/utils/triton.py` — read `response_format` alongside the existing tool path and
   set `structured_outputs` from it (`StructuredOutputsParams` already exposes both `json`
   and `json_object`, so the plumbing into vLLM is there and working).

Perhaps 40–60 lines. The mechanism is proven; only the wiring is missing. But it is a patch
against a generated OpenAPI schema and a vendored engine file, re-applied and re-verified on
every Triton upgrade, to restore behaviour that the vLLM already on this box has natively.
And it still leaves NVFP4-on-sm_121 under Triton unproven.

## Notes

Both images are on disk (~46 GB total). `docker rmi` them if the Triton path is abandoned.
