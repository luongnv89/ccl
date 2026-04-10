#!/usr/bin/env bash
# CLI end-to-end smoke test — exercises every subcommand of both entrypoints.
# Uses a temp HOME/STATE_DIR so it never touches the real user's config.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Prefer the repo's venv python if present; fall back to whatever is on PATH.
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export HOME="$TMP/home"
export CLAUDE_CODEX_LOCAL_STATE_DIR="$TMP/state"
mkdir -p "$HOME" "$CLAUDE_CODEX_LOCAL_STATE_DIR"

PASS=0
FAIL=0
run() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  ok  $desc"
    PASS=$((PASS+1))
  else
    echo "  FAIL $desc  (cmd: $*)"
    FAIL=$((FAIL+1))
  fi
}

echo "=== E2E: ccl-bridge ==="
run "ccl-bridge --help"                "$PY" -m claude_codex_local.bridge --help
run "ccl-bridge profile --help"        "$PY" -m claude_codex_local.bridge profile --help
run "ccl-bridge recommend --help"      "$PY" -m claude_codex_local.bridge recommend --help
run "ccl-bridge doctor --help"         "$PY" -m claude_codex_local.bridge doctor --help
run "ccl-bridge adapters --help"       "$PY" -m claude_codex_local.bridge adapters --help
run "ccl-bridge adapters"              "$PY" -m claude_codex_local.bridge adapters

echo "=== E2E: wizard ==="
run "wizard --help"                    "$PY" -m claude_codex_local.wizard --help
run "wizard setup --help"              "$PY" -m claude_codex_local.wizard setup --help
run "wizard find-model --help"         "$PY" -m claude_codex_local.wizard find-model --help
run "wizard doctor --help"             "$PY" -m claude_codex_local.wizard doctor --help
# wizard doctor exits 1 when no state exists — expected on a fresh HOME.
if out="$("$PY" -m claude_codex_local.wizard doctor 2>&1)"; then :; fi
if grep -q "No wizard state" <<<"$out"; then
  echo "  ok  wizard doctor (no-state path)"
  PASS=$((PASS+1))
else
  echo "  FAIL wizard doctor unexpected output: $out"
  FAIL=$((FAIL+1))
fi

echo "=== E2E: error paths ==="
if "$PY" -m claude_codex_local.bridge bogus-command >/dev/null 2>&1; then
  echo "  FAIL ccl-bridge should reject unknown command"
  FAIL=$((FAIL+1))
else
  echo "  ok  ccl-bridge rejects unknown command"
  PASS=$((PASS+1))
fi

echo "--------------------------------"
echo "E2E: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
