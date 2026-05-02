# Lean ASR sidecar — vLLM-native Qwen3-ASR serve on aarch64 / sm_121a (Spark).
# Reuses the v3 vLLM image (already has Qwen3ASRForConditionalGeneration registered)
# and adds the three pieces it's missing for vLLM's audio decode path:
#   - flash-attn 2 (sm_120 wheel) for fast attention
#   - soundfile (primary WAV/FLAC decoder, ~10x faster than PyAV for WAVs)
#   - PyAV (container-format fallback for MP4/M4A/WebM)
#
# Why not `pip install vllm[audio]`?  The meta-package re-resolves vLLM core
# deps and silently downgrades flashinfer-python (0.6.9 → 0.6.8.post1) and
# apache-tvm-ffi (0.1.10 → 0.1.9), regressing inference latency. We install
# soundfile + av directly so we get the audio path without touching anything
# else.
FROM ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3

# flash-attn for max throughput. Cap MAX_JOBS so the native build doesn't
# blow up RAM on Spark's unified pool. Single-arch (sm_120) keeps build
# time at ~10-15 min instead of 40+ for multi-arch.
ENV MAX_JOBS=4 \
    FLASH_ATTN_CUDA_ARCHS=120
RUN pip install --no-cache-dir --break-system-packages --no-build-isolation \
      "flash-attn>=2.7" \
 || (echo "flash-attn install failed; will fall back to sdpa at runtime" && true)

# Audio decoder pair for vLLM's load_audio path:
#   - soundfile (libsndfile binding) — primary WAV/FLAC decoder
#   - PyAV — container-format fallback (M4A / MP4 / WebM / odd WAVs)
# Without soundfile installed, vLLM's load_audio_soundfile() raises an
# ImportError on every call and silently falls through to PyAV, which works
# but is ~10x slower for plain WAVs and floods the logs with errors.
# cffi is soundfile's runtime dep; install it explicitly so --no-deps
# doesn't leave a broken extension.
RUN pip install --no-cache-dir --break-system-packages \
      "soundfile>=0.12" "cffi>=1.16" \
 && pip install --no-cache-dir --break-system-packages --no-deps "av>=12"

EXPOSE 8001
ENV PYTHONUNBUFFERED=1

# Container-level healthcheck: probes /v1/models, which only returns 200
# once the LLM engine has finished loading the model AND the API server
# is accepting requests. /health alone returns 200 before the model is
# ready, which would lie to autoheal during the ~30-90 s cold-start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python3 -c "import urllib.request,sys,json; \
    r=urllib.request.urlopen('http://127.0.0.1:8001/v1/models',timeout=4); \
    d=json.loads(r.read()); sys.exit(0 if d.get('data') else 1)" || exit 1

# Default: serve the validated 0.6B variant on :8001 with conservative memory.
# Override the CMD at `docker run` to pick a different model / tune memory.
CMD ["vllm", "serve", "Qwen/Qwen3-ASR-0.6B", \
     "--served-model-name", "qwen3-asr", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--gpu-memory-utilization", "0.08", \
     "--max-model-len", "8192", \
     "--max-num-seqs", "4", \
     "--trust-remote-code"]
