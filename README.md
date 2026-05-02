# qwen3-asr-server

OpenAI-compatible `/v1/audio/transcriptions` HTTP server backed by
[**Qwen3-ASR-0.6B**](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), served by
**vLLM** on **NVIDIA DGX Spark** (GB10, sm_121a / sm_120 wheels) and other
Blackwell consumer GPUs.

A lean image: vLLM core + flash-attention 2 (sm_120 wheel built in) +
PyAV — that's it. Drops in behind any OpenAI Whisper-compatible client.

- **30 spoken languages** + **22 zh dialects**
- **RTF ~16× real-time** on Spark (120 ms for a 2 s clip, hot path)
- **Streaming-friendly** — vLLM-native scheduling, suitable for a voice loop
- **Model-agnostic deploy** — pick from the supported [Qwen3-ASR variants](docs/MODELS.md)

For the matching TTS sidecar see
[**qwen3-tts-server**](https://github.com/AEON-7/qwen3-tts-server).

## Performance — DGX Spark, hot path

| stage             | wall    | RTF     |
| ----------------- | ------- | ------- |
| ASR transcription | 120 ms  | 16.04×  |

(input: 2 s mono 24 kHz WAV → text out)

## QuickStart

The image is published at **`ghcr.io/aeon-7/qwen3-asr-server:latest`**.

### Docker Compose

```bash
docker network create aeon-stack         # one-time
git clone https://github.com/AEON-7/qwen3-asr-server
cd qwen3-asr-server
docker compose up -d

# verify
curl http://localhost:8001/health
```

### Or the deploy script (interactive variant picker)

```bash
bash deploy/deploy-asr.sh                # pick a Qwen3-ASR variant
# or non-interactive with the validated default:
QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B bash deploy/deploy-asr.sh
```

### Transcribe a WAV

```bash
curl -X POST http://localhost:8001/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=qwen3-asr \
  -F language=en
# {"text":"The capital of France is Paris."}
```

### Or via the OpenAI SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8001/v1", api_key="ignored")
with open("speech.wav", "rb") as f:
    out = client.audio.transcriptions.create(
        model="qwen3-asr", file=f, language="en",
    )
print(out.text)
```

## Recommended pairing — full voice-AI stack

Designed to slot in next to two other sidecars on the same Docker bridge,
making a complete LLM + ASR + TTS stack on a single host:

| sidecar             | repo                                                                                                       | purpose             |
| ------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------- |
| LLM main            | [aeon-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash) | reasoning / chat    |
| **ASR** (this repo) | `ghcr.io/aeon-7/qwen3-asr-server:latest`                                                                   | speech → text       |
| TTS                 | [aeon-7/qwen3-tts-server](https://github.com/AEON-7/qwen3-tts-server)                                      | text → speech       |

Hot end-to-end voice turn (text → speech → text, both directions): **~1.6 s**
on Spark. With the LLM in the middle for a real reasoning round-trip:
**~2.6 s**. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Wire it into a Matrix voice agent

The fastest path to "I can talk to my AI in a Matrix call":

```
+-------------+   WebRTC    +--------------------+   HTTP    +------------+
| Matrix      | <---------> | matrix-voip-agent  | --------> | qwen3-asr  |
| homeserver  |             | (PipeWire bridge)  |           | (this)     |
+-------------+             |                    |           +------------+
                            |                    |     HTTP  +------------+
                            |                    | --------> | LLM main   |
                            |                    |           +------------+
                            |                    |     HTTP  +------------+
                            |                    | --------> | qwen3-tts  |
                            +--------------------+           +------------+
```

Pair this server with [**matrix-voip-agent**](https://github.com/AEON-7/matrix-voip-agent)
— a headless WebRTC bridge that auto-answers Matrix VoIP calls and pipes
audio to/from the AI sidecars. Combined with any Matrix homeserver
(stock Synapse / Conduit, or our customized matrix-voip-agent setup with
direct calling features), you get an AI you can dial directly from your
Matrix client.

QuickStart on the matrix-voip-agent host:

```bash
# .env
STT_BACKEND=qwen
ASR_ENDPOINT=http://${SPARK_HOST}:8001/v1/audio/transcriptions
ASR_MODEL=qwen3-asr
ASR_LANGUAGE=en
```

Full integration walkthrough — including PCM↔WAV adapter wiring, audio
format constraints, and TTS pairing — is in
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

## Environment variables

All optional. Sensible defaults baked into the Dockerfile CMD.

### Server (vLLM args, set via `command:` override or deploy script)

| arg                          | default               | meaning                                                      |
| ---------------------------- | --------------------- | ------------------------------------------------------------ |
| model (positional)           | `Qwen/Qwen3-ASR-0.6B` | HF repo id. See [docs/MODELS.md](docs/MODELS.md).            |
| `--served-model-name`        | `qwen3-asr`           | Public name returned in `/v1/models`.                        |
| `--host` / `--port`          | `0.0.0.0` / `8001`    | Bind address.                                                |
| `--gpu-memory-utilization`   | `0.08`                | Fraction of GPU/unified RAM. Bump to ≥0.10 for 1.7B variant. |
| `--max-model-len`            | `8192`                | ASR rarely exceeds 4 K tokens; 8 K is generous.              |
| `--max-num-seqs`             | `4`                   | Concurrent transcription jobs.                               |
| `--trust-remote-code`        | enabled               | Required by Qwen3-ASR.                                       |

### Container env (deploy-side)

| var          | default                                      | meaning                                                            |
| ------------ | -------------------------------------------- | ------------------------------------------------------------------ |
| `IMAGE`      | `ghcr.io/aeon-7/qwen3-asr-server:latest`     | Image to pull.                                                     |
| `PORT`       | `8001`                                       | Host port to bind.                                                 |
| `NETWORK`    | `aeon-stack`                                 | Docker bridge (auto-created). Join your other sidecars here.       |
| `HF_CACHE`   | `${HOME}/.cache/huggingface`                 | Bind-mounted into the container.                                   |
| `HF_TOKEN`   | (unset)                                      | Forwarded if set. Needed only for gated HF repos.                  |
| `CONTAINER`  | `qwen3-asr`                                  | Container name.                                                    |

### Client-side (set on the host calling this server)

These names aren't read by this server — they're the convention any
downstream client (matrix-voip-agent, OpenClaw, your own scripts) should use:

| var               | example                                                   |
| ----------------- | --------------------------------------------------------- |
| `SPARK_HOST`      | `192.168.1.116`                                           |
| `ASR_ENDPOINT`    | `http://${SPARK_HOST}:8001/v1/audio/transcriptions`       |
| `ASR_MODEL`       | `qwen3-asr`                                               |
| `ASR_LANGUAGE`    | `en` (or `auto`, `zh`, `ja`, ...)                         |

## Documentation index

- [docs/MODELS.md](docs/MODELS.md) — Qwen3-ASR variants and when to pick each
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — recommended full-stack
  topology with vLLM main, qwen3-tts-server, Matrix, OpenClaw
- [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) — wiring guides for Matrix
  voice calls, OpenAI SDK, OpenWebUI, Home Assistant, custom clients
- [docs/OPS.md](docs/OPS.md) — health checks, autoheal/watchdog patterns,
  triage for common failure modes
- [agents.md](agents.md) — agent-readable bring-up runbook

## Endpoints

- `GET  /health` — liveness
- `GET  /v1/models` — single served model
- `POST /v1/audio/transcriptions` — OpenAI multipart form, returns text

## License

Apache-2.0. Underlying model weights are released under the
[Qwen license](https://huggingface.co/Qwen/Qwen3-ASR-0.6B).
