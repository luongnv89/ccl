#!/usr/bin/env bash
#
# start-llama-server.sh — launch a tuned local llama-server for a coding agent.
#
# Every value can be overridden via an environment variable, e.g.
#   PORT=8002 CTX_SIZE=131072 MODEL=/path/to/model.gguf ./start-llama-server.sh
# Machine-specific values (model path, binary location, thread count) are picked up
# from the environment — on this host they are supplied by the systemd unit's drop-in.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- configurable knobs (env overrides win) --------------------------------
# llama-server binary: a name resolved on PATH, or an absolute path.
LLAMA_BIN="${LLAMA_BIN:-llama-server}"
# Model GGUF. Defaults to the repo-local models/ dir; override with MODEL=... to switch models.
MODEL="${MODEL:-${SCRIPT_DIR}/models/Qwopus3.6-Q4_K_M/Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf}"
# Stable public model name. Harnesses/tools must connect using THIS alias, not the underlying
# GGUF name — it stays constant even when MODEL is swapped, so client configs never change.
ALIAS="${ALIAS:-montimage-dgx-spark}"
# Default thinking OFF (low-latency agent mode). Clients can still override per request with
# chat_template_kwargs:{enable_thinking:true}. Set ENABLE_THINKING=true to flip the default.
ENABLE_THINKING="${ENABLE_THINKING:-false}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CTX_SIZE="${CTX_SIZE:-400000}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
PARALLEL="${PARALLEL:-2}"
CACHE_RAM="${CACHE_RAM:-32768}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
# Default to the CPU count; on big.LITTLE pin to perf cores via the env / systemd drop-in
# (e.g. THREADS=10 plus a CPUAffinity= in the unit). Decode here is GPU-bound, so this is minor.
THREADS="${THREADS:-$(nproc)}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"

# --- preflight -------------------------------------------------------------
if ! command -v "$LLAMA_BIN" >/dev/null 2>&1 && [[ ! -x "$LLAMA_BIN" ]]; then
  echo "error: llama-server not found on PATH or at: $LLAMA_BIN" >&2
  echo "       set LLAMA_BIN=/path/to/llama-server" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found: $MODEL" >&2
  echo "       set MODEL=/path/to/model.gguf" >&2
  exit 1
fi

# Bail out early if something is already listening on the port.
if curl -s -m 2 "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; then
  echo "llama-server already running and healthy on port ${PORT} — nothing to do."
  exit 0
fi

# --- launch ----------------------------------------------------------------
echo "Starting llama-server on ${HOST}:${PORT}"
echo "  model: ${MODEL}"
echo "  alias: ${ALIAS}  (clients connect by this name)"
echo "  ctx-size=${CTX_SIZE} n-gpu-layers=${N_GPU_LAYERS} parallel=${PARALLEL} thinking=${ENABLE_THINKING}"

exec "$LLAMA_BIN" \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --parallel "$PARALLEL" \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram "$CACHE_RAM" \
  --ubatch-size "$UBATCH_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --threads "$THREADS" \
  --jinja \
  --chat-template-kwargs "{\"enable_thinking\": ${ENABLE_THINKING}}" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --presence-penalty "$PRESENCE_PENALTY"
