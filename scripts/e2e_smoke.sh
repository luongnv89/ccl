#!/usr/bin/env bash
# CLI end-to-end smoke test — exercises every subcommand of ccl + the core
# debug CLI. Uses a temp HOME/STATE_DIR so it never touches the real config.
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

echo "=== E2E: ccl ==="
run "ccl --help"                       "$PY" -m claude_codex_local.wizard --help
run "ccl --version"                    "$PY" -m claude_codex_local.wizard --version
run "ccl setup --help"                 "$PY" -m claude_codex_local.wizard setup --help
run "ccl find-model --help"            "$PY" -m claude_codex_local.wizard find-model --help
run "ccl doctor --help"                "$PY" -m claude_codex_local.wizard doctor --help
# ccl doctor exits 1 when no state exists — expected on a fresh HOME.
if out="$("$PY" -m claude_codex_local.wizard doctor 2>&1)"; then :; fi
if grep -q "No wizard state" <<<"$out"; then
  echo "  ok  ccl doctor (no-state path)"
  PASS=$((PASS+1))
else
  echo "  FAIL ccl doctor unexpected output: $out"
  FAIL=$((FAIL+1))
fi

echo "=== E2E: core debug CLI ==="
run "core --help"                      "$PY" -m claude_codex_local.core --help
run "core profile --help"              "$PY" -m claude_codex_local.core profile --help
run "core recommend --help"            "$PY" -m claude_codex_local.core recommend --help
run "core doctor --help"               "$PY" -m claude_codex_local.core doctor --help
run "core adapters --help"             "$PY" -m claude_codex_local.core adapters --help
run "core adapters"                    "$PY" -m claude_codex_local.core adapters

echo "=== E2E: error paths ==="
if "$PY" -m claude_codex_local.core bogus-command >/dev/null 2>&1; then
  echo "  FAIL core should reject unknown command"
  FAIL=$((FAIL+1))
else
  echo "  ok  core rejects unknown command"
  PASS=$((PASS+1))
fi

echo "--------------------------------"
echo "E2E: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
