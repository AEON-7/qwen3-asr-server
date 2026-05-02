# Recommended Full-Stack Architecture

This page describes the deployment topology this image is designed for: a
single low-latency voice-AI host that serves an LLM main + ASR + TTS as
three OpenAI-compatible endpoints on a shared Docker bridge, with
downstream clients (Matrix server, agents, custom apps) speaking to it
over the LAN.

## Topology

```
                      ┌──────────────────────── DGX Spark (192.168.1.116) ─────────────────────────┐
                      │                                                                             │
                      │   docker bridge "aeon-stack" (172.20.0.0/16)                                │
                      │   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
                      │   │ qwen36-aeon-xs   │  │ qwen3-asr        │  │ qwen3-tts        │          │
                      │   │ vLLM main:8000   │  │ vLLM ASR:8001    │  │ FastAPI TTS:8002 │          │
                      │   │ Qwen3.6-27B      │  │ Qwen3-ASR-0.6B   │  │ Qwen3-TTS-1.7B   │          │
                      │   │ NVFP4 + DFlash   │  │ flash-attn 2     │  │ bf16+flash-attn 2│          │
                      │   └──────────┬───────┘  └─────────┬────────┘  └─────────┬────────┘          │
                      │              │                    │                     │                   │
                      │              └────────────────────┼─────────────────────┘                   │
                      │                                   │                                         │
                      └───────────────────────────────────┼─────────────────────────────────────────┘
                                                          │ host network (LAN, ~1 ms)
                                                          ▼
                      ┌─────────────────────── matrix-voip-agent host ──────────────────────────────┐
                      │                                                                             │
                      │   matrix-voip-agent (Node/TS, headless WebRTC bridge)                       │
                      │              │                                                              │
                      │              ▼                                                              │
                      │   Matrix homeserver (Synapse / Conduit / etc.)                              │
                      │              │                                                              │
                      │              ▼                                                              │
                      │   Element / nheko / any Matrix client = "dial the AI"                       │
                      └─────────────────────────────────────────────────────────────────────────────┘
```

## Why this layout

- **All three AI services on one Docker bridge.** Inter-container hops are
  loopback-fast (sub-ms). The LLM → ASR → TTS pipeline never leaves the host.
- **Orchestration on a separate host.** Voice agents only need a thin audio
  pipe from the WebRTC plane to the Spark sidecars — one LAN hop (~1-2 ms,
  negligible vs the 30-200 ms an internet hop would add).
- **No co-location of orchestration with inference.** Keeps the Spark
  dedicated to GPU work; matrix-voip-agent and Matrix homeserver can be
  independently restarted.

## The three sidecars

| sidecar                                                                                              | image                                                                          | port |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---- |
| LLM main — [Qwen3.6-27B AEON Ultimate MTP-XS](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash) | `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3`                            | 8000 |
| **ASR** (this repo)                                                                                  | `ghcr.io/aeon-7/qwen3-asr-server:latest`                                       | 8001 |
| TTS — [qwen3-tts-server](https://github.com/AEON-7/qwen3-tts-server)                                 | `ghcr.io/aeon-7/qwen3-tts-server:latest`                                       | 8002 |

Bring up all three, joined to the same `aeon-stack` bridge. Order doesn't
matter functionally, but bring up the heavy LLM main first if memory is tight.

### LLM main bring-up

```bash
docker network create aeon-stack 2>/dev/null || true

docker run -d --name qwen36-aeon-xs \
  --runtime nvidia --network aeon-stack -p 8000:8000 \
  --shm-size=4gb --restart unless-stopped \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 \
  vllm serve aeon-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MTP-XS \
    --served-model-name qwen36-ultimate-xs \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}' \
    --trust-remote-code
```

### ASR bring-up (this image)

```bash
docker run -d --name qwen3-asr \
  --runtime nvidia --network aeon-stack -p 8001:8001 \
  --shm-size=4gb --restart unless-stopped \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ghcr.io/aeon-7/qwen3-asr-server:latest
```

(Default CMD serves `Qwen3-ASR-0.6B` on `:8001`. Override with a custom
`vllm serve ...` command for other variants.)

### TTS bring-up

```bash
docker run -d --name qwen3-tts \
  --runtime nvidia --network aeon-stack -p 8002:8002 \
  --shm-size=4gb --restart unless-stopped \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ghcr.io/aeon-7/qwen3-tts-server:latest
```

## Memory budget on Spark (128 GB unified)

| service                                       | `gpu-memory-utilization` | resident   |
| --------------------------------------------- | -----------------------: | ---------: |
| qwen36-aeon-xs (27B NVFP4 + DFlash, BF16 KV)  |                    0.75  |    ~96 GB  |
| qwen3-asr (0.6B)                              |              0.06–0.08  |   ~5–10 GB |
| qwen3-tts (1.7B, transformers, bf16)          |                     n/a |  ~4 GB CUDA |
| host kernel + buffer cache + Docker overhead  |                       — |    ~10 GB  |
| free / margin                                 |                       — | **~10 GB** |

The margin is tight; **never** push `gpu-memory-utilization` past **0.88**
on unified-memory Spark — see the
[gpu-memory cap note](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash#dgx-spark-gpu_memory-utilization-caps-at-088).

## Latency budget (measured, hot path)

| stage                                       | wall      |
| ------------------------------------------- | --------- |
| inbound RTP packet → matrix-voip-agent      | ~5 ms     |
| **ASR** (1.92 s clip → text)                | **120 ms** |
| LLM (vLLM `chat/completions`, ~10 tok)      | ~480 ms   |
| TTS (text → 1.92 s WAV)                     | ~1.48 s   |
| outbound RTP → Matrix client                | ~5 ms     |
| **End-to-end voice turn**                   | **~2.1 s** |

## See also

- [docs/MODELS.md](MODELS.md) — supported ASR variants
- [docs/INTEGRATIONS.md](INTEGRATIONS.md) — wiring guides for Matrix, OpenAI
  SDK, OpenWebUI, Home Assistant
- [agents.md](../agents.md) — agent-readable runbook
