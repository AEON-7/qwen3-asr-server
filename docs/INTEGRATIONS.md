# Integration Guides

How to wire `qwen3-asr-server` into popular voice / agent stacks. Each
section is self-contained: copy-pasteable config, no read-the-other-doc
required.

The convention used throughout: replace `${SPARK_HOST}` (or `${ASR_HOST}`)
with the host address where this image is running. Don't hardcode IPs in
checked-in code — keep them as env-var substitutions.

---

## 1. Matrix voice calls (recommended for AI-on-Matrix)

The fastest path to "I can dial my AI in a Matrix call" is to pair this
image with [**matrix-voip-agent**](https://github.com/AEON-7/matrix-voip-agent)
— a headless WebRTC bridge that auto-answers Matrix VoIP calls and pipes
audio between the call and your AI sidecars via PipeWire.

Combine matrix-voip-agent with **any Matrix homeserver** (stock
[Synapse](https://github.com/element-hq/synapse) /
[Conduit](https://gitlab.com/famedly/conduit), or our customized setup with
direct calling features) and you have an AI that's reachable from any
Matrix client (Element, nheko, FluffyChat, etc.) by dialing a contact.

### matrix-voip-agent `.env`

```bash
# disable the old whisper.cpp + ElevenLabs paths
WHISPER_ENABLED=false
# ELEVENLABS_API_KEY=        # leave unset

# wire to qwen3-asr-server
STT_BACKEND=qwen
ASR_ENDPOINT=http://${SPARK_HOST}:8001/v1/audio/transcriptions
ASR_MODEL=qwen3-asr
ASR_LANGUAGE=en

# pair with qwen3-tts-server for the speech reply leg
TTS_BACKEND=qwen
TTS_ENDPOINT=http://${SPARK_HOST}:8002/v1/audio/speech
TTS_MODEL=qwen3-tts
TTS_VOICE="A warm, expressive adult voice with natural cadence."

# wire LLM main
LLM_BASE_URL=http://${SPARK_HOST}:8000/v1
LLM_MODEL=qwen36-ultimate-xs
LLM_API_KEY=ignored          # any non-empty string works
```

### Audio format on the wire

- matrix-voip-agent captures **PCM s16le 16 kHz mono** from `pw-record` and
  wraps it in a WAV header in-memory before POSTing to ASR.
- vLLM's audio decoder accepts any sample rate and resamples internally.
- The TTS leg returns 24 kHz mono 16-bit PCM in a WAV; matrix-voip-agent
  strips the RIFF header and pipes raw PCM to `pw-play -r 24000 -f s16 -c 1`.

If you're rolling your own bridge, that's the contract: send WAV bytes via
multipart `file` field, get JSON `{"text": "..."}` back.

---

## 2. OpenAI SDK (Python / TS / anywhere)

This server speaks OpenAI's `/v1/audio/transcriptions`. Drop in any OpenAI
SDK and point `base_url` at it:

### Python

```python
from openai import OpenAI

client = OpenAI(base_url=f"http://{SPARK_HOST}:8001/v1", api_key="ignored")

with open("speech.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="qwen3-asr",
        file=f,
        language="en",       # or "auto"
    )
print(resp.text)
```

### TypeScript

```typescript
import OpenAI from "openai";
import fs from "node:fs";

const client = new OpenAI({
  baseURL: `http://${SPARK_HOST}:8001/v1`,
  apiKey: "ignored",
});

const resp = await client.audio.transcriptions.create({
  model: "qwen3-asr",
  file: fs.createReadStream("speech.wav"),
  language: "en",
});
console.log(resp.text);
```

---

## 3. OpenWebUI

OpenWebUI's Audio settings expect an OpenAI-compatible STT endpoint:

```
Settings → Audio → Speech-to-Text Engine: OpenAI
  STT API Base URL: http://${SPARK_HOST}:8001/v1
  STT API Key:      ignored
  STT Model:        qwen3-asr
```

Works for both the in-chat microphone button and any voice-call agents you
build on top.

---

## 4. Home Assistant (`wyoming-openai-stt` or generic OpenAI)

Two paths:

### Via the [`openai_conversation` integration's STT mode](https://www.home-assistant.io/integrations/openai_conversation/) (or any "OpenAI-compatible STT" plugin)

```yaml
# configuration.yaml
openai_conversation:
  - name: "Local ASR"
    url: !secret asr_url           # http://${SPARK_HOST}:8001/v1
    api_key: !secret asr_api_key   # any non-empty string
    stt_model: qwen3-asr
```

### Via an Assist pipeline pointing at the same endpoint

In Settings → Voice Assistants → your pipeline → STT, choose your
"OpenAI-compatible" provider and point it at the same URL. Pair with the
companion `qwen3-tts-server` for end-to-end Assist voice.

---

## 5. Custom client — raw HTTP

Dead simple, no SDK needed:

```bash
curl -sf -X POST http://${SPARK_HOST}:8001/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=qwen3-asr \
  -F language=en
# {"text": "the capital of france is paris"}
```

The endpoint accepts any `Authorization` header (no real auth — put a proxy
in front for any non-trusted network).

Multipart form fields:

| field      | required | notes                                                       |
| ---------- | :------: | ----------------------------------------------------------- |
| `file`     |    ✓     | Audio bytes. WAV / FLAC / MP3 / M4A / OGG / WebM / raw PCM. |
| `model`    |    ✓     | Use `qwen3-asr` (matches `--served-model-name`).            |
| `language` |          | Two-letter code or `auto`. Defaults to auto-detect.         |
| `prompt`   |          | (vLLM forwards to the model; useful for vocabulary biasing.)|
| `temperature` |       | Sampling temperature (default 0).                           |

---

## 6. Other LLM-orchestrated agents (LangChain, LlamaIndex, custom)

Anything that consumes an OpenAI Whisper-compatible base URL works
unchanged. The deciding fact is the `/v1/audio/transcriptions` shape, not
the underlying engine. If your framework's STT abstraction lets you set a
custom `base_url` + `api_key`, it'll work here.

---

## A note on running this image elsewhere

Nothing in the image is Spark-specific *except* the flash-attn 2 wheel,
which is built for `sm_120`. On other Blackwell / consumer / datacenter
GPUs the image will boot — flash-attn falls back to SDPA at runtime if the
kernel can't load. Re-build with the right `FLASH_ATTN_CUDA_ARCHS` if you
want native flash-attn there too.
