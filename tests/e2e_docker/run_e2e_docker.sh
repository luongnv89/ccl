#!/usr/bin/env bash
# Docker-based end-to-end test runner for claude-codex-local installation scenarios.
#
# Each scenario starts from a minimal Python 3.11-slim Docker image and exercises
# a different installation method:
#   1. pip install   — install from pre-built wheel
#   2. uv install    — install via uv from pre-built wheel
#   3. from source   — install with `pip install .`
#   4. with extras   — install with `pip install .[dev]`
#
# Usage:
#   bash tests/e2e_docker/run_e2e_docker.sh [--scenario <name>]
#
# Options:
#   --scenario <name>   Run only the named scenario (pip|uv|source|extras).
#                       Omit to run all scenarios.
#   --no-cleanup        Skip removing Docker images after the run.
#
# Exit codes:
#   0  All scenarios passed (or selected scenario passed)
#   1  One or more scenarios failed
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DOCKER_DIR="$REPO/tests/e2e_docker"
DIST_DIR="$REPO/dist"
IMAGE_PREFIX="ccl-e2e"

# ---------- argument parsing ----------
ONLY_SCENARIO=""
NO_CLEANUP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) ONLY_SCENARIO="$2"; shift 2 ;;
    --no-cleanup) NO_CLEANUP=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------- helpers ----------
PASS=0
FAIL=0
FAILED_SCENARIOS=()

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { printf "  ${GREEN}PASS${NC}  %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "  ${RED}FAIL${NC}  %s\n" "$*"; FAIL=$((FAIL+1)); FAILED_SCENARIOS+=("$*"); }
info() { printf "  ${YELLOW}....${NC}  %s\n" "$*"; }

run_scenario() {
  local name="$1"
  local dockerfile="$2"
  local build_context="$3"
  shift 3
  local extra_build_args=("$@")

  local image="${IMAGE_PREFIX}-${name}"

  info "Building scenario: ${name}"

  if docker build \
      -f "$dockerfile" \
      ${extra_build_args[@]+"${extra_build_args[@]}"} \
      -t "$image" \
      "$build_context" \
      > /tmp/ccl_e2e_${name}_build.log 2>&1; then
    info "Running scenario: ${name}"
    if docker run --rm "$image" \
        > /tmp/ccl_e2e_${name}_run.log 2>&1; then
      ok "${name}"
    else
      fail "${name} (run failed — see /tmp/ccl_e2e_${name}_run.log)"
      cat /tmp/ccl_e2e_${name}_run.log >&2
    fi
  else
    fail "${name} (build failed — see /tmp/ccl_e2e_${name}_build.log)"
    cat /tmp/ccl_e2e_${name}_build.log >&2
  fi

  if [[ $NO_CLEANUP -eq 0 ]]; then
    docker rmi --force "$image" > /dev/null 2>&1 || true
  fi
}

# ---------- pre-flight: build wheel ----------
build_wheel() {
  info "Building wheel in $DIST_DIR"
  rm -rf "$DIST_DIR"
  if ! python -m build --wheel --outdir "$DIST_DIR" "$REPO" \
      > /tmp/ccl_e2e_build_wheel.log 2>&1; then
    echo "ERROR: wheel build failed — see /tmp/ccl_e2e_build_wheel.log" >&2
    cat /tmp/ccl_e2e_build_wheel.log >&2
    exit 1
  fi
  WHEEL_FILE="$(ls "$DIST_DIR"/*.whl | head -1 | xargs basename)"
  if [[ -z "$WHEEL_FILE" ]]; then
    echo "ERROR: no .whl found in $DIST_DIR" >&2
    exit 1
  fi
  info "Wheel: $WHEEL_FILE"
}

# ---------- scenario definitions ----------
run_all() {
  echo "=== Docker E2E: claude-codex-local installation scenarios ==="
  echo ""

  build_wheel

  if [[ -z "$ONLY_SCENARIO" || "$ONLY_SCENARIO" == "pip" ]]; then
    run_scenario "pip" \
      "$DOCKER_DIR/Dockerfile.pip" \
      "$REPO" \
      --build-arg "WHEEL_FILE=$WHEEL_FILE"
  fi

  if [[ -z "$ONLY_SCENARIO" || "$ONLY_SCENARIO" == "uv" ]]; then
    run_scenario "uv" \
      "$DOCKER_DIR/Dockerfile.uv" \
      "$REPO" \
      --build-arg "WHEEL_FILE=$WHEEL_FILE"
  fi

  if [[ -z "$ONLY_SCENARIO" || "$ONLY_SCENARIO" == "source" ]]; then
    run_scenario "source" \
      "$DOCKER_DIR/Dockerfile.source" \
      "$REPO"
  fi

  if [[ -z "$ONLY_SCENARIO" || "$ONLY_SCENARIO" == "extras" ]]; then
    run_scenario "extras" \
      "$DOCKER_DIR/Dockerfile.extras" \
      "$REPO"
  fi

  echo ""
  echo "------------------------------"
  echo "Docker E2E: ${PASS} passed, ${FAIL} failed"

  if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo "Failed scenarios:"
    for s in "${FAILED_SCENARIOS[@]}"; do
      printf "  - %s\n" "$s"
    done
    exit 1
  fi
}

run_all
