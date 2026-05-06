#!/usr/bin/env bash
# End-to-end smoke test for the vLLM backend.
#
# Drives the same path users hit:
#   1. boots `vllm serve` with a tiny HF model on a non-default port,
#   2. waits for /v1/models to come up,
#   3. runs `pytest -m real_vllm tests/test_e2e_vllm_real.py`,
#   4. tears the server down.
#
# Auto-skips (exit 0 with a message) when `vllm` is not on PATH or
# `vllm --version` fails. To force a non-zero exit when vllm is missing,
# set CCL_VLLM_REQUIRED=1.
#
# Knobs (env vars):
#   CCL_VLLM_MODEL         HF model id           (default: Qwen/Qwen2.5-0.5B-Instruct)
#   CCL_VLLM_PORT          test server port      (default: 18002)
#   CCL_VLLM_BOOT_TIMEOUT  seconds for boot poll (default: 600)
#   CCL_VLLM_EXTRA_ARGS    extra `vllm serve` args
#   CCL_VLLM_LOG           path to capture vllm logs (default: $TMP/vllm.log)
#   CCL_VLLM_REQUIRED      1 → fail when vllm missing instead of skipping
#   CCL_VLLM_VENV          path to a venv whose `bin/vllm` should be used in
#                          place of whatever is on PATH — handy when the
#                          system vllm is broken (CUDA mismatch, dep
#                          conflicts) and you want to redirect to a fixed
#                          installation without polluting other shells.
#   CCL_VLLM_LD_LIBRARY_PATH  prepended to LD_LIBRARY_PATH when running tests,
#                          so vllm can find e.g. libcudart.so.12 when CUDA is
#                          installed in a non-standard location.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if [[ -n "${CCL_VLLM_VENV:-}" ]]; then
  if [[ ! -x "$CCL_VLLM_VENV/bin/vllm" ]]; then
    echo "FAIL: CCL_VLLM_VENV=$CCL_VLLM_VENV has no bin/vllm" >&2
    exit 1
  fi
  export PATH="$CCL_VLLM_VENV/bin:$PATH"
fi

if [[ -n "${CCL_VLLM_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${CCL_VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! command -v vllm >/dev/null 2>&1; then
  msg="vllm CLI not found on PATH"
  if [[ "${CCL_VLLM_REQUIRED:-0}" == "1" ]]; then
    echo "FAIL: $msg" >&2
    exit 1
  fi
  echo "skip: $msg"
  exit 0
fi

if ! vllm --version >/dev/null 2>&1; then
  msg="vllm --version failed (broken install / missing CUDA)"
  if [[ "${CCL_VLLM_REQUIRED:-0}" == "1" ]]; then
    echo "FAIL: $msg" >&2
    vllm --version 2>&1 | tail -5 >&2 || true
    exit 1
  fi
  echo "skip: $msg"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export CCL_VLLM_LOG="${CCL_VLLM_LOG:-$TMP/vllm.log}"

echo "=== E2E vLLM ==="
echo "  python : $PY"
echo "  vllm   : $(command -v vllm)  $(vllm --version 2>/dev/null | head -1)"
echo "  model  : ${CCL_VLLM_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
echo "  port   : ${CCL_VLLM_PORT:-18002}"
echo "  log    : $CCL_VLLM_LOG"

"$PY" -m pytest -m real_vllm tests/test_e2e_vllm_real.py -v "$@"
