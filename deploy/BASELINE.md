# Baseline: what `:8000` did before `sir`

Captured 2026-08-14 against the live `vllm-qwen36-27b-nvfp4` container, uncontended, before
anything was torn down. This is the contract `sir` + Triton has to keep — every caller on this
box talks to `localhost:8000` and none of them are being changed.

Raw responses are in `deploy/baseline/`. Reproduce any of them against `sir` after cutover and
compare; the shape has to match, the wording does not.

Server: `vllm-0.23.1rc1.dev701+g00eb7cefa.d20260701`, `nvidia/Qwen3.6-27B-NVFP4`,
`max_model_len` 125000.

## 1. `/v1/models` → `baseline/models.json`

One entry, `id: "nvidia/Qwen3.6-27B-NVFP4"`. `sir` must serve this exact string as a
`served_model_name` — it is what `BAG_LLM_MODEL` and job-tracker both send.

## 2. Plain completion → `baseline/plain.json`

```
"In one sentence, what is a GPU?"  max_tokens 80, temperature 0
```

Answered in prose, `finish_reason: "stop"`, `usage` present with real token counts
(21 prompt / 50 completion). Note `"reasoning": null` — **thinking is already off**, which is
the `--default-chat-template-kwargs {"enable_thinking": false}` flag doing its job. If it comes
back on under Triton, reasoning text lands in `content` and every JSON parse downstream breaks.

## 3. Constrained decoding — the acceptance test

Both callers depend on this and they use **different spellings**, so both must be checked.

### 3a. `json_schema` — job-tracker's form → `baseline/json_schema.json`

```json
"response_format": {"type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": {...}}}
```

Returned `{"level": "Senior", "years": 8}` — schema-conformant, nothing else in `content`.

This is the one with history. `job-tracker/jobtracker/llm/vllm.py:74-81` records that vLLM
0.23.1 accepts a body carrying the older `guided_json`, **silently ignores it**, and answers in
prose; `_parse_verdict` then rejects every response and the nightly pass resolves nothing while
still paying for a description fetch per posting. Nothing errors. A Triton frontend that
accepts `response_format` and ignores it reproduces that failure exactly.

### 3b. `json_object` — Interlinear's form → `baseline/json_object.json`

`Interlinear/app/pipeline/translate.py:95` sends `{"type": "json_object"}`. Returned
`{"text": "Hello, how are you?"}`.

**Verdict on both: valid JSON, not prose.** A pass is JSON that parses and conforms. Prose that
merely contains the right answer is a failure, however good it looks.

## 4. Streaming → `baseline/stream.sse`

`"stream": true` with `stream_options: {"include_usage": true}`. 18 lines: an opening chunk
carrying `delta.role`, content chunks carrying `delta.content`, a final chunk with
`choices: []` and a populated `usage`, then `data: [DONE]`.

`sir`'s Triton backend parses this shape into `Chunk`s, and the trailing usage chunk is where
the real token counts come from — `docs/ROADMAP.md:27` asks for exactly that, replacing the
mock's word-count `usage`.

## Callers held to this

| Project | How it points here | Depends on |
|---|---|---|
| Interlinear | `BAG_LLM_BASE_URL=http://localhost:8000/v1`, `BAG_LLM_MODEL=nvidia/Qwen3.6-27B-NVFP4` (`docs/DEPLOY.md:92`) | §3b |
| job-tracker | `--llm-provider vllm --llm-url http://localhost:8000` (`docs/ranking.md:173`) | §3a, temperature 0 reproducibility |

Neither may need an edit. That is the definition of a successful drop-in.
