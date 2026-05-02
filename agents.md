# agents.md — autonomous deployment runbook

Instructions for an AI agent to bring this ASR sidecar up cleanly on a
fresh host. Self-contained: you don't need to also read README.md.

## Preconditions

1. **Host kind**: Image is built for `aarch64` + `sm_120` (NVIDIA DGX Spark
   / GB10 / Blackwell consumer). On other architectures it will run, but
   flash-attn falls back to SDPA at runtime — rebuild from `Dockerfile`
   with the appropriate `FLASH_ATTN_CUDA_ARCHS` if you want native
   flash-attn there too.
2. **Docker** with the `nvidia` runtime configured.
   ```bash
   docker info | grep -i runtime    # expect: Runtimes: nvidia runc
   ```
3. **Disk + RAM**: Image is ~33 GB. Free RAM ≥ 12 GB before bring-up
   (the 0.6B model alone is ~5–10 GB resident; 1.7B variant ~18–22 GB).
4. **Network**: Outbound HTTPS to `huggingface.co` and `ghcr.io`.

## Decision points — commit BEFORE running any docker command

### 1. Which ASR variant?

Default is `Qwen/Qwen3-ASR-0.6B` (validated). Only deviate if:

- The user said "best WER / accuracy / SOTA" → `Qwen/Qwen3-ASR-1.7B`,
  AND set `GPU_MEM=0.16`, AND warn about the larger memory footprint.
- The user said "word timestamps" → also deploy `Qwen3-ForcedAligner-0.6B`
  as a *separate* container. The deploy script in this repo doesn't ship
  it; clone the [Qwen3-ForcedAligner card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
  and run it as a sibling.

If unsure: stick with the default.

### 2. Standalone or paired with vLLM main + TTS?

- **Standalone** (just transcription): run `bash deploy/deploy-asr.sh` or
  `docker compose up -d`. Done.
- **Paired with vLLM main + TTS** (full voice agent stack): bring up the
  others FIRST (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
  recommended commands), then run this one. The shared bridge is
  `aeon-stack`; create it once with `docker network create aeon-stack`.

## Bring-up — non-interactive

```bash
git clone https://github.com/AEON-7/qwen3-asr-server
cd qwen3-asr-server

# default (validated 0.6B)
docker network create aeon-stack 2>/dev/null || true
docker compose up -d
```

Or with the deploy script (interactive picker):

```bash
bash deploy/deploy-asr.sh
```

## Verification — wait for ready

vLLM compiles CUDA graphs on first boot — expect 30–90 s before the health
endpoint returns 200.

```bash
until curl -sf -m 2 http://localhost:8001/health >/dev/null 2>&1; do sleep 5; done
echo "ASR ready"
```

If polling exceeds 5 min, check logs:

```bash
docker logs --tail 50 qwen3-asr
```

### Common boot failures and recovery

**`No available memory for the cache blocks`**

KV-cache shortfall. Fix:

```bash
docker rm -f qwen3-asr
GPU_MEM=0.10 bash deploy/deploy-asr.sh   # or 0.12, 0.14, ...
```

If you've reached `GPU_MEM=0.20` and still fail, lower context:

```bash
docker rm -f qwen3-asr
MAX_LEN=4096 GPU_MEM=0.10 bash deploy/deploy-asr.sh
```

**Container boot-loops (>5 restarts in a minute)**

Almost always memory pressure — same fix.

**`Engine core init failed`** (any other reason)

Read `docker logs qwen3-asr | grep -iE '(error|traceback|raise)' | tail -20`
for the actual exception. Common culprits: missing HF_TOKEN for a gated
model variant (set `HF_TOKEN=...` and re-run the deploy script).

## Smoke test

```bash
# 1. produce a test WAV (any 1-3 sec speech sample works)
#    if you have qwen3-tts-server running on :8002, generate one:
curl -sf -X POST http://localhost:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world","response_format":"wav"}' \
  --output /tmp/agent_smoke.wav 2>/dev/null \
  || echo "(no TTS available; supply your own /tmp/agent_smoke.wav)"

# 2. transcribe
ASR_OUT=$(curl -sf -X POST http://localhost:8001/v1/audio/transcriptions \
            -F file=@/tmp/agent_smoke.wav -F model=qwen3-asr -F language=en)
echo "$ASR_OUT" | grep -q '"text"' \
  && echo "ASR ok: $ASR_OUT" \
  || { echo "ASR FAIL: $ASR_OUT"; exit 1; }
```

If the response includes a non-empty `text` field, the stack is operational.

## Common follow-up tasks

### Wire matrix-voip-agent on a separate host

See [docs/INTEGRATIONS.md → "Matrix voice calls"](docs/INTEGRATIONS.md#1-matrix-voice-calls-recommended-for-ai-on-matrix).
Don't put hardcoded IPs in checked-in code — use `SPARK_HOST` as the
substitution variable.

### Pair with the Qwen3.6-27B AEON Ultimate vLLM main + qwen3-tts-server

Run the docker commands in
[docs/ARCHITECTURE.md → "The three sidecars"](docs/ARCHITECTURE.md#the-three-sidecars).
Order matters only if memory is tight (bring up the heavy LLM first).

### Switch model variants live

```bash
docker rm -f qwen3-asr
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B GPU_MEM=0.16 bash deploy/deploy-asr.sh
```

The HF cache is bind-mounted, so a previously-downloaded variant restarts
instantly.

## Tear-down

```bash
docker rm -f qwen3-asr
docker network rm aeon-stack 2>/dev/null || true
# HF cache at ${HOME}/.cache/huggingface is preserved.
```

## Don'ts

- Don't bind `8001` to a public interface without putting an auth proxy in
  front. The endpoint accepts any `Authorization` header (no real auth).
- Don't set `--gpu-memory-utilization` above `0.88` on Spark — see
  [DGX Spark gpu-memory-utilization cap](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash#dgx-spark-gpu_memory-utilization-caps-at-088).
- Don't `pip install vllm[audio]` into a derivative image — the meta-package
  re-resolves and downgrades vLLM core deps (flashinfer-python,
  apache-tvm-ffi) which silently regresses inference latency. The Dockerfile
  installs only `av --no-deps`; soundfile is already in the base image.
- Don't use `--no-verify` on git operations against this repo unless
  explicitly asked.

## Output convention for autonomous reports

If you're an agent reporting back to a parent:

```
qwen3-asr-server bring-up: OK
- model: <model-id>
- port: 8001
- bridge: aeon-stack ready
- smoke test: <wall ms> for <sample chars>
- paired with: <none | qwen3-tts-server | full stack>
```

If FAIL, include the failing health-check output and the last 20 lines of
the container's logs.
