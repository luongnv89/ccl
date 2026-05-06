#!/usr/bin/env bash
# Bootstrap an isolated venv with a working vLLM, intended for use by
# scripts/e2e_smoke_vllm.sh on machines whose system vllm install is
# broken (CUDA path mismatch, transformers/huggingface_hub conflict,
# etc).
#
# The venv inherits site-packages from a base interpreter that already
# has vllm + torch installed, and overrides only the packages that
# conflict.  No system-wide changes are made.
#
# Usage:
#   scripts/bootstrap-vllm-venv.sh
#
# Knobs:
#   CCL_VLLM_VENV         target venv path
#                         (default: $HOME/.cache/ccl/vllm-venv)
#   CCL_VLLM_BASE_PYTHON  base interpreter that already has vllm
#                         (auto-detected: prefers /home/montimage/miniconda3,
#                          then any python3 with `import vllm`)
#
# After the venv is built the script prints the env vars that
# scripts/e2e_smoke_vllm.sh needs to pick it up.
set -euo pipefail

VENV="${CCL_VLLM_VENV:-$HOME/.cache/ccl/vllm-venv}"
BASE_PY="${CCL_VLLM_BASE_PYTHON:-}"

if [[ -z "$BASE_PY" ]]; then
  for cand in \
    /home/montimage/miniconda3/bin/python \
    /opt/miniconda3/bin/python \
    "$(command -v python3 || true)"; do
    if [[ -x "$cand" ]] && "$cand" -c "import vllm" >/dev/null 2>&1; then
      BASE_PY="$cand"
      break
    fi
    # Even if vllm import fails (e.g. libcudart issue), the package may
    # still be installed — accept it as the base interpreter.
    if [[ -x "$cand" ]] && "$cand" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('vllm') else 1)" >/dev/null 2>&1; then
      BASE_PY="$cand"
      break
    fi
  done
fi

if [[ -z "$BASE_PY" ]]; then
  echo "FAIL: could not find a python3 with vllm installed" >&2
  echo "Install vllm first: pip install vllm" >&2
  exit 1
fi

echo "Base interpreter : $BASE_PY"
echo "Target venv      : $VENV"

if [[ ! -d "$VENV" ]]; then
  mkdir -p "$(dirname "$VENV")"
  "$BASE_PY" -m venv --system-site-packages "$VENV"
  # Override only the package(s) that are known to conflict with vllm's
  # transformers pin in current builds.  This pin doesn't propagate to the
  # base conda env — only the venv's site-packages.
  "$VENV/bin/pip" install --quiet --upgrade pip >/dev/null
  "$VENV/bin/pip" install --quiet 'huggingface_hub<1.0'
fi

# Write a `bin/vllm` shim that runs the CLI through the venv interpreter.
# venv with --system-site-packages inherits packages but not console-script
# shims, so we synthesize one that uses the venv's python while picking up
# vllm from the inherited site-packages.
SHIM="$VENV/bin/vllm"
cat > "$SHIM" <<'PYEOF'
#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m vllm.entrypoints.cli.main "$@"
PYEOF
chmod +x "$SHIM"

# Discover libcudart.so.12 (vllm wheels are still built against CUDA 12 on
# many platforms; the host may only ship CUDA 13 in standard search paths).
LIBCUDART_DIR=""
for d in \
  /usr/local/lib/ollama/cuda_v12 \
  /usr/local/cuda-12.0/lib64 \
  /usr/local/cuda-12.1/lib64 \
  /usr/local/cuda-12.4/lib64 \
  /usr/local/cuda-12.8/lib64; do
  if [[ -e "$d/libcudart.so.12" ]]; then
    LIBCUDART_DIR="$d"
    break
  fi
done

# Quick smoke: can we import vllm._C in the venv?
if ! "$VENV/bin/python" -c "import vllm._C" >/dev/null 2>&1; then
  if [[ -n "$LIBCUDART_DIR" ]]; then
    if ! LD_LIBRARY_PATH="$LIBCUDART_DIR" "$VENV/bin/python" -c "import vllm._C" >/dev/null 2>&1; then
      echo "WARN: vllm._C import still fails even with LD_LIBRARY_PATH=$LIBCUDART_DIR" >&2
    fi
  else
    echo "WARN: vllm._C import fails and no libcudart.so.12 found in standard paths" >&2
  fi
fi

cat <<EOF

Bootstrap complete.  Run the E2E smoke test with:

  CCL_VLLM_VENV='$VENV' \\
  CCL_VLLM_LD_LIBRARY_PATH='${LIBCUDART_DIR}' \\
  scripts/e2e_smoke_vllm.sh

EOF
