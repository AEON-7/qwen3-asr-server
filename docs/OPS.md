# Operations — health checks, watchdog, troubleshooting

What "healthy" means for this service, how to monitor it, and how to
make it self-heal.

## Health endpoints

The vLLM API server exposes two probes. Use the right one for the right
purpose:

| endpoint | returns 200 when | use for |
|---|---|---|
| `GET /health` | The HTTP server is up and accepting requests | Liveness probe (k8s `livenessProbe`, basic uptime monitoring) |
| `GET /v1/models` | The HTTP server is up **and** the model has finished loading | Readiness probe (k8s `readinessProbe`, real "ready to transcribe" check) |

`/health` returns 200 within seconds of container start — long before the
model is actually ready. **Don't use it as a readiness signal.** vLLM
takes ~30-90 s on Spark to load weights and compile CUDA graphs; during
that window `/health` will lie. Use `/v1/models` instead — it returns
200 only after the model is fully loaded and the engine can serve.

## Built-in container healthcheck

The Dockerfile ships with a `HEALTHCHECK` that probes `/v1/models` and
parses the JSON to confirm the model list isn't empty:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; \
    r=urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=4); \
    d=json.loads(r.read()); sys.exit(0 if d.get('data') else 1)" || exit 1
```

`docker ps` shows the result inline:

```bash
docker ps --filter name=qwen3-asr --format 'table {{.Names}}\t{{.Status}}'
# qwen3-asr   Up 5 minutes (healthy)
```

## Auto-restart on unhealthy — the autoheal pattern

**Important caveat:** Docker's `--restart=unless-stopped` only restarts
on container *crash* (exit code != 0). It does **not** restart on
`HEALTHCHECK` failures. A container that becomes wedged — port still
listening, model unloaded, GPU stalled, vLLM engine deadlocked — will
sit there reporting `unhealthy` forever unless something kicks it.

There are three good ways to wire that up. Pick one.

### Option A — `willfarrell/autoheal` (recommended for compose users)

Tiny sidecar that watches every container labeled `autoheal=true` and
`docker restart`s any reporting `unhealthy` for more than a configurable
threshold. Already wired into the bundled `docker-compose.yml`:

```yaml
services:
  qwen3-asr:
    labels:
      - "autoheal=true"
    # ...

  autoheal:
    image: willfarrell/autoheal:latest
    container_name: aeon-autoheal
    restart: unless-stopped
    environment:
      - AUTOHEAL_CONTAINER_LABEL=autoheal
      - AUTOHEAL_INTERVAL=30
      - AUTOHEAL_START_PERIOD=180
      - AUTOHEAL_DEFAULT_STOP_TIMEOUT=20
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

`docker compose up -d` brings both up. To verify:

```bash
docker logs aeon-autoheal --tail 5
# expect: "Monitoring containers for unhealthy status..."
```

To trigger-test (intentionally hose the container):

```bash
docker exec qwen3-asr kill -STOP 1   # SIGSTOP the vLLM process
# autoheal will detect unhealthy within ~90 s and restart the container
```

If you also run the `qwen3-tts-server` and the `aeon-7/vllm-aeon-ultimate-dflash`
LLM main, give them the same `autoheal=true` label and a single autoheal
sidecar covers all three.

### Option B — systemd watchdog (recommended for non-compose hosts)

If you run plain `docker run` from a systemd unit, add a `Type=notify`
watchdog or use `Restart=always` + `WatchdogSec=` with an external probe.
Sketch:

```ini
# /etc/systemd/system/qwen3-asr.service
[Unit]
Description=Qwen3-ASR sidecar
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStart=/usr/bin/docker run --rm --name qwen3-asr ... aeon-7/qwen3-asr-server:latest
ExecStop=/usr/bin/docker stop qwen3-asr

[Install]
WantedBy=multi-user.target
```

Pair with a separate systemd `.timer` that calls `curl -sf http://localhost:8001/v1/models`
every minute and `systemctl restart qwen3-asr` on failure.

### Option C — Kubernetes liveness + readiness probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 30
  periodSeconds: 30
  failureThreshold: 3
readinessProbe:
  httpGet:
    path: /v1/models
    port: 8001
  initialDelaySeconds: 60
  periodSeconds: 15
  failureThreshold: 2
startupProbe:
  httpGet:
    path: /v1/models
    port: 8001
  failureThreshold: 30          # 30 * 10 s = 5 min cold-start budget
  periodSeconds: 10
```

The `startupProbe` covers the cold-start window (model load + CUDA graph
compile) so the liveness probe doesn't kill the pod before it's ready.
Once startup passes, liveness/readiness take over.

## Common failure modes — diagnose first, restart second

Restart-everything is the wrong answer for most problems. Quick triage:

### Symptom: `unhealthy` in `docker ps`, container stays unhealthy after restart

Likely cause: KV-cache OOM at boot. vLLM logs `No available memory for
the cache blocks`. Restart won't fix it — you need more `--gpu-memory-utilization`
or less `--max-model-len`. See [MODELS.md → Memory tuning](MODELS.md#memory-tuning).

### Symptom: `/health` 200, `/v1/models` 200, but every transcription returns 200 with empty `text`

Likely the audio decode failed silently. Check container logs for
`Failed to load audio via soundfile` (means the soundfile bake-in
regressed) or `Invalid or unsupported audio file` (PyAV couldn't parse
the bytes). Fix:

```bash
docker logs qwen3-asr | grep -E '(soundfile|pyav|audio)' | tail -20
```

If you see `Please install vllm[audio]`, you're somehow on an old image
where the soundfile bake-in is missing — pull `:latest` again.

### Symptom: transcriptions return correct text but take 30+ s

PyAV is decoding instead of soundfile (10x slowdown). Same diagnosis +
fix as above — check the soundfile import succeeded inside the container:

```bash
docker exec qwen3-asr python3 -c "import soundfile; print(soundfile.__version__)"
# expect: 0.13.1 (or newer) — NOT an ImportError
```

### Symptom: `/health` and `/v1/models` both 200, but `/v1/audio/transcriptions` returns 5xx

Could be:
- KV cache filled up (request body > `--max-model-len`). Bump `MAX_LEN` or
  send shorter audio.
- Concurrent request limit hit (`--max-num-seqs`). Bump it.
- vLLM engine actually wedged. `docker logs --tail 100 qwen3-asr` will
  show the actual exception. Restart fixes engine wedge.

### Symptom: Container restarts in a loop (`Restarting (...)` in `docker ps`)

Almost always memory or config. Read the last 50 lines of logs:

```bash
docker logs --tail 50 qwen3-asr
```

Top three culprits:
1. **`No available memory for the cache blocks`** → bump `GPU_MEM`, or lower `MAX_LEN`.
2. **`HF_TOKEN` missing for a gated model** → set `HF_TOKEN=...` and re-run.
3. **Wrong served-model-name vs model arg** → these need to align with how
   your client calls it (`-F model=qwen3-asr`).

## Monitoring metrics

vLLM emits its own metrics on the same port at `/metrics` (Prometheus
format). Useful counters:

- `vllm:num_requests_running` — concurrent transcriptions
- `vllm:num_requests_waiting` — queue depth
- `vllm:gpu_cache_usage_perc` — KV cache utilization
- `vllm:e2e_request_latency_seconds_bucket` — request latency histogram
- `vllm:tokens_total` — generated tokens (transcript chars approximately)

Scrape with any Prometheus, or eyeball with:

```bash
curl -s http://localhost:8001/metrics | grep -E '^vllm:(num_requests|gpu_cache_usage|e2e)'
```

The agent log has the same info every 10 s in human-readable form:
`Avg prompt throughput: ... GPU KV cache usage: ...`.

## Tear-down + clean restart

When you genuinely need to wipe state and start fresh:

```bash
docker stop qwen3-asr aeon-autoheal 2>/dev/null
docker rm   qwen3-asr aeon-autoheal 2>/dev/null
# Optional: clear vLLM's torch.compile cache on the host
rm -rf ${HOME}/.cache/vllm/torch_compile_cache 2>/dev/null
# HF model cache at ${HOME}/.cache/huggingface is safe to keep — it's
# read-only at runtime.
docker compose up -d
```

The `torch.compile` cache wipe is rarely needed; only do it if you've
changed `--max-model-len` or `--gpu-memory-utilization` and the new
config refuses to compile against the cached artifacts.
