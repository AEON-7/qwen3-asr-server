#!/usr/bin/env bash
# deploy-asr.sh — pull the qwen3-asr-server image from ghcr.io and serve a
# Qwen3-ASR variant of the user's choice on port 8001.
#
# Usage:
#   bash deploy/deploy-asr.sh                    # interactive picker
#   QWEN_ASR_MODEL=Qwen/... bash deploy/deploy-asr.sh   # non-interactive
#
# Defaults:
#   IMAGE     ghcr.io/aeon-7/qwen3-asr-server:latest
#   MODEL     Qwen/Qwen3-ASR-0.6B  (validated default)
#   PORT      8001
#   NETWORK   aeon-stack  (created if missing)
#   GPU_MEM   0.08    (~10 GB on Spark)
#   MAX_LEN   8192
#   MAX_SEQS  4
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/aeon-7/qwen3-asr-server:latest}"
PORT="${PORT:-8001}"
NETWORK="${NETWORK:-aeon-stack}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
CONTAINER="${CONTAINER:-qwen3-asr}"
GPU_MEM="${GPU_MEM:-0.08}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_SEQS="${MAX_SEQS:-4}"

# ── model picker ─────────────────────────────────────────────────────────────
declare -a ASR_VARIANTS=(
  "Qwen/Qwen3-ASR-0.6B|0.6B|ASR  — 30 langs + 22 zh dialects, very high throughput (✅ validated default)"
  "Qwen/Qwen3-ASR-1.7B|1.7B|ASR  — same coverage, SOTA WER, prefer accuracy over throughput"
)

pick_model() {
  if [[ -n "${QWEN_ASR_MODEL:-}" ]]; then
    echo "[deploy-asr] Using QWEN_ASR_MODEL from env: ${QWEN_ASR_MODEL}" >&2
    return
  fi
  echo "Pick a Qwen3-ASR variant:" >&2
  local i=1
  for v in "${ASR_VARIANTS[@]}"; do
    printf "  %d) %s\n" "$i" "$(echo "$v" | awk -F'|' '{printf "%-22s  %-4s  %s", $1, $2, $3}')" >&2
    i=$((i+1))
  done
  echo >&2
  read -rp "Choice [1, default=1]: " choice
  choice="${choice:-1}"
  QWEN_ASR_MODEL="$(echo "${ASR_VARIANTS[$((choice-1))]}" | cut -d'|' -f1 | xargs)"
  echo "[deploy-asr] Selected: ${QWEN_ASR_MODEL}" >&2
}

auto_tune() {
  if [[ "$QWEN_ASR_MODEL" == *"1.7B"* && "$GPU_MEM" == "0.08" ]]; then
    GPU_MEM="0.16"
    echo "[deploy-asr] Bumping GPU_MEM to ${GPU_MEM} for 1.7B variant" >&2
  fi
}

# ── prep ─────────────────────────────────────────────────────────────────────
pick_model
auto_tune

docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK"

docker pull "$IMAGE"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# ── run ──────────────────────────────────────────────────────────────────────
docker run -d --name "$CONTAINER" \
  --runtime nvidia \
  --network "$NETWORK" \
  -p "${PORT}:8001" \
  --shm-size=4gb \
  --restart unless-stopped \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$IMAGE" \
  vllm serve "$QWEN_ASR_MODEL" \
    --served-model-name qwen3-asr \
    --host 0.0.0.0 --port 8001 \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --trust-remote-code

cat <<EOF

[deploy-asr] Container '${CONTAINER}' started on port ${PORT}.
[deploy-asr] Model:     ${QWEN_ASR_MODEL}
[deploy-asr] GPU mem:   ${GPU_MEM}    max_model_len=${MAX_LEN}    max_num_seqs=${MAX_SEQS}
[deploy-asr] Network:   ${NETWORK}

First boot loads weights + compiles CUDA graphs (~30-90 s on Spark). Health:

  curl http://localhost:${PORT}/health

Transcribe a WAV:

  curl -X POST http://localhost:${PORT}/v1/audio/transcriptions \\
    -F file=@speech.wav -F model=qwen3-asr -F language=en

Tail logs:  docker logs -f ${CONTAINER}

If you see "No available memory for the cache blocks":
  GPU_MEM=0.10 bash deploy/deploy-asr.sh
EOF
