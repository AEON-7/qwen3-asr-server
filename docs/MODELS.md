# Supported ASR Models

The deploy script (`deploy/deploy-asr.sh`) accepts any of the variants below.
Pick interactively, set the `QWEN_ASR_MODEL` env var, or override the
docker-compose `command:` block to skip the prompt.

The image is **officially validated** with:

- **`Qwen/Qwen3-ASR-0.6B`** — the deploy-script default

Other variants are supported but unbenchmarked on this image — they should
"just work" because they share architecture with the validated one, but you
may need to tune `--gpu-memory-utilization` for the 1.7B variant.

## Qwen3-ASR family

There is no separate ASR tokenizer repo — the audio encoder is in-model.

| Repo ID                          | Params | When to pick                                                                         | Status              |
| -------------------------------- | -----: | ------------------------------------------------------------------------------------ | ------------------- |
| `Qwen/Qwen3-ASR-0.6B`            |   0.6B | 30 langs + 22 zh dialects, very high throughput (~2000× at concurrency 128).         | ✅ validated default |
| `Qwen/Qwen3-ASR-1.7B`            |   1.7B | Same coverage, SOTA WER among open ASR. Pick when accuracy > throughput.             | user-selectable     |
| `Qwen/Qwen3-ForcedAligner-0.6B`  |   0.6B | Optional companion for word/phoneme timestamps (≤ 5 min audio, 11 langs).            | optional companion  |

### Picking the right variant

- **Real-time / interactive ASR** (voice agents, streaming captions) → **0.6B** (default; RTF 16-20× on Spark).
- **Best WER, throughput is secondary** (offline batch transcription, high-stakes meeting notes) → **1.7B** (bump `GPU_MEM` to ~0.16).
- **Need word-level timestamps** (subtitling, audio search) → run **ForcedAligner-0.6B** as a separate endpoint. Out of scope for this image's deploy script — pull and serve it as a second container.

## Memory tuning

Numbers are for **DGX Spark** (128 GB unified). Adjust proportionally on
other GPUs.

| variant   | `--gpu-memory-utilization` | resident   | notes                                            |
| --------- | -------------------------- | ---------- | ------------------------------------------------ |
| 0.6B      | `0.06`–`0.08`              | ~5–10 GB   | Plenty of room for KV cache @ 8 K context.       |
| 1.7B      | `0.14`–`0.18`              | ~18–22 GB  | Bump if KV cache OOM at boot.                    |

If the container exits at boot with `No available memory for the cache
blocks`: **raise `GPU_MEM` by 0.02** and retry. If you've reached `0.20`
and still fail, **lower `MAX_LEN` to 4096** (still generous for ASR — most
inputs are < 30 s of audio = a few hundred tokens).

If the container boot-loops (`docker ps` shows >5 restarts in a minute):
that's almost always a memory shortfall — same fix as above.

## Languages supported

All variants speak the same 30 languages:

| family       | codes                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------- |
| Chinese      | `zh` (Mandarin) + 22 regional dialects (Cantonese, Wu, Min, Hakka, Xiang, Gan, etc.)        |
| Indo-European | `en`, `de`, `fr`, `es`, `it`, `pt`, `ru`, `pl`, `nl`, `ro`, `cs`, `sv`, `hu`, `tr`, `el`    |
| East Asian   | `ja`, `ko`, `vi`, `th`, `id`, `ms`                                                          |
| Other        | `ar`, `he`, `hi`, `bn`, `ta`, `ur`, `fi`, `da`, `no`                                        |

Pass `language=auto` (or omit the field) to let the model detect.

## Audio format

vLLM's audio decoder (soundfile + PyAV fallback) handles:

- WAV (any sample rate; the model resamples internally to 16 kHz)
- FLAC
- MP3, M4A, OGG, WebM (via PyAV)
- Raw PCM 16-bit LE — wrap in a WAV header before POST'ing

Mono is preferred. Stereo is accepted but downmixed.

## Throughput characteristics

Benchmarked on Spark (single GPU, single ASR container at `--max-num-seqs 4`):

| concurrency | wall (2 s clip) | aggregate RTF |
| -----------:| ---------------:| -------------:|
| 1           | ~120 ms         | 16.04×        |
| 4           | ~140 ms total   | 56×           |
| 8 (saturated)| ~250 ms        | 64×           |

For higher concurrency, raise `--max-num-seqs` and `--gpu-memory-utilization`
together.
