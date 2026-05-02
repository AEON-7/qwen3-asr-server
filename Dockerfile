# Lean ASR sidecar — vLLM-native Qwen3-ASR serve on aarch64 / sm_121a (Spark).
# Reuses the v3 vLLM image (already has Qwen3ASRForConditionalGeneration registered)
# and adds only the two pieces it's actually missing: flash-attn 2 (sm_120 wheel)
# and PyAV for vLLM's audio-decode container fallback.
#
# Why not `pip install vllm[audio]`?  The meta-package re-resolves vLLM core
# deps and silently downgrades flashinfer-python (0.6.9 → 0.6.8.post1) and
# apache-tvm-ffi (0.1.10 → 0.1.9), regressing inference latency. soundfile
# is already pulled in transitively by the base image; av is the only piece
# truly missing for vLLM's load_audio path. So we install av --no-deps.
FROM ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3

# flash-attn for max throughput. Cap MAX_JOBS so the native build doesn't
# blow up RAM on Spark's unified pool. Single-arch (sm_120) keeps build
# time at ~10-15 min instead of 40+ for multi-arch.
ENV MAX_JOBS=4 \
    FLASH_ATTN_CUDA_ARCHS=120
RUN pip install --no-cache-dir --break-system-packages --no-build-isolation \
      "flash-attn>=2.7" \
 || (echo "flash-attn install failed; will fall back to sdpa at runtime" && true)

# PyAV (audio-container fallback used by vLLM's load_audio path)
RUN pip install --no-cache-dir --break-system-packages --no-deps "av>=12"

EXPOSE 8001
ENV PYTHONUNBUFFERED=1

# Default: serve the validated 0.6B variant on :8001 with conservative memory.
# Override the CMD at `docker run` to pick a different model / tune memory.
CMD ["vllm", "serve", "Qwen/Qwen3-ASR-0.6B", \
     "--served-model-name", "qwen3-asr", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--gpu-memory-utilization", "0.08", \
     "--max-model-len", "8192", \
     "--max-num-seqs", "4", \
     "--trust-remote-code"]
