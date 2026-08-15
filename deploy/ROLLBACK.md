# Rollback: put vLLM back on `:8000`

The cutover **stops** `vllm-qwen36-27b-nvfp4` but does not delete it. The container and the
`eugr/spark-vllm:latest` image both stay on disk, so the fast path back is two commands. The
full `docker run` below exists for the case where the container itself is gone.

Recorded 2026-08-14 from `docker inspect` of the running container (up since 2026-07-28).

## Fast path — container still exists

```bash
docker compose -f deploy/compose.yaml down          # free :8000 and the GPU first
docker update --restart=unless-stopped vllm-qwen36-27b-nvfp4
docker start vllm-qwen36-27b-nvfp4
curl -s localhost:8000/v1/models                    # answers once weights are loaded
```

Both servers want `:8000` and the whole GPU, so `sir`'s stack must be down before vLLM comes
up. Order matters more than speed here.

## Full reconstruction — container gone

```bash
docker run -d \
  --name vllm-qwen36-27b-nvfp4 \
  --restart unless-stopped \
  --gpus all \
  --ipc host \
  -p 8000:8000 \
  -v /home/dylan/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e NCCL_DEBUG=WARN \
  eugr/spark-vllm:latest \
  vllm serve nvidia/Qwen3.6-27B-NVFP4 \
    --served-model-name nvidia/Qwen3.6-27B-NVFP4 \
    --quantization modelopt \
    --enforce-eager \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 125000 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 32000 \
    --limit-mm-per-prompt.image 8 \
    --api-server-count 1 \
    --renderer-num-workers 1 \
    --mm-processor-cache-gb 4 \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --trust-remote-code
```

## The three env vars are not optional

`VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`,
`VLLM_USE_FLASHINFER_MOE_FP4=0` are what make NVFP4 work on GB10 (sm_121) in this image.
`Interlinear/docs/DEPLOY.md:96` records them as the flags that matter on this box. They carry
over to the Triton model repository for the same reason.

## Other recorded settings

- Runtime `runc`, `--ipc host`, default 64 MB shm, bridge network
- `--gpus all` (`DeviceRequests: [{Count: -1, Capabilities: [["gpu"]]}]`)
- Only mount: `/home/dylan/.cache/huggingface -> /root/.cache/huggingface`
- Image `eugr/spark-vllm:latest` — vLLM `0.23.1rc1.dev701+g00eb7cefa.d20260701`,
  torch 2.11.0+cu130, CUDA 13.0.2. A Spark-specific build; do not assume a stock
  `vllm/vllm-openai` tag substitutes for it.

## Flags with no Triton equivalent yet

`--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3` are vLLM
*frontend* flags. Nothing on this box currently sends `tools` or reads `reasoning_content` —
grepped across Interlinear and job-tracker — so losing them is not a regression today. It would
become one the moment something starts tool-calling, which is a reason to keep this file
current rather than a reason to block the cutover.
