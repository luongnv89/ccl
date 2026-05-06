"""
Real end-to-end tests for the vLLM backend — boots a real `vllm serve`
process and exercises the VLLMAdapter against it.

These tests are marked `@pytest.mark.real_vllm` and skip automatically
when:
  * the `vllm` CLI is not on PATH, OR
  * `vllm --version` refuses to start (broken install / missing CUDA), OR
  * the test server fails to come up within the boot timeout.

They are NOT CI-safe — they download a model on first run and require a
GPU (or a CPU build of vLLM) on the host.

Run explicitly with:

    pytest -m real_vllm tests/test_e2e_vllm_real.py -v

What is tested:
  1. VLLMAdapter.detect()       — talks to /v1/models on the live server
  2. VLLMAdapter.healthcheck()  — same endpoint, asserts ok=True
  3. VLLMAdapter.list_models()  — server reports the model we loaded
  4. VLLMAdapter.run_test()     — sends a real /v1/chat/completions
  5. smoke_test_vllm_model()    — direct helper, asserts timing + tokens

Model used: Qwen/Qwen2.5-0.5B-Instruct
  Tiny enough to fit on most GPUs and to load in <30 s, real enough to
  exercise the full stack (tokenizer + model + generation).

Knobs (env vars):
  CCL_VLLM_MODEL         override the model id      (default above)
  CCL_VLLM_PORT          override the test port     (default: 18002)
  CCL_VLLM_BOOT_TIMEOUT  seconds to wait for /v1/models  (default: 600)
  CCL_VLLM_EXTRA_ARGS    extra args appended to `vllm serve`
                         (e.g. "--max-model-len 4096 --enforce-eager")
  CCL_VLLM_PYTHON        python interpreter that imports vllm
                         (default: same one running pytest, falls back to
                          the `vllm` shim on PATH)
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants / knobs
# ---------------------------------------------------------------------------

REAL_TEST_PORT = int(os.environ.get("CCL_VLLM_PORT", "18002"))
REAL_TEST_HOST = "127.0.0.1"
TEST_MODEL = os.environ.get("CCL_VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BOOT_TIMEOUT = float(os.environ.get("CCL_VLLM_BOOT_TIMEOUT", "600"))
EXTRA_ARGS = os.environ.get("CCL_VLLM_EXTRA_ARGS", "").split()

# vLLM reserves a fraction of *total* GPU memory at startup. We default to
# 0.15 because the test model is tiny (~1 GB) and the host may already
# have other workloads (Ollama, training jobs) holding most of the GPU.
# Override with CCL_VLLM_GPU_MEM_UTIL on a dedicated test GPU.
GPU_MEM_UTIL = os.environ.get("CCL_VLLM_GPU_MEM_UTIL", "0.15")
MAX_MODEL_LEN = os.environ.get("CCL_VLLM_MAX_MODEL_LEN", "4096")

# Default vLLM serve args: keep memory + context small so the test is
# friendly to shared GPUs and laptops.
DEFAULT_SERVE_ARGS = [
    "--host", REAL_TEST_HOST,
    "--port", str(REAL_TEST_PORT),
    "--max-model-len", MAX_MODEL_LEN,
    "--gpu-memory-utilization", GPU_MEM_UTIL,
    "--enforce-eager",
]


pytestmark = pytest.mark.real_vllm


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _vllm_cli() -> str | None:
    """Return path to a usable `vllm` CLI, or None if it can't even --help."""
    cli = shutil.which("vllm")
    if not cli:
        return None
    try:
        subprocess.run(
            [cli, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return cli


def _wait_for_server(port: int, timeout: float) -> bool:
    """Poll /v1/models until the server responds 2xx or the timeout expires."""
    url = f"http://{REAL_TEST_HOST}:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    return False


# ---------------------------------------------------------------------------
# Module-level skip — bail before fixture setup if vllm isn't usable.
# ---------------------------------------------------------------------------

_VLLM_CLI = _vllm_cli()
if _VLLM_CLI is None:
    pytest.skip(
        "vllm CLI not found or broken — install with `pip install vllm` "
        "and ensure `vllm --version` succeeds before running these tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_server():
    """
    Start a real `vllm serve` on REAL_TEST_PORT with TEST_MODEL.

    Yields the port; terminates the process on teardown.  Skips the test
    module if the server fails to come up within BOOT_TIMEOUT.
    """
    cmd = [_VLLM_CLI, "serve", TEST_MODEL, *DEFAULT_SERVE_ARGS, *EXTRA_ARGS]

    # Surface vllm logs to a temp file so failures are debuggable but the
    # test output stays clean on the happy path.
    log_path = Path(os.environ.get("CCL_VLLM_LOG", "")) or None
    log_file = log_path.open("w") if log_path else subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )

    try:
        ready = _wait_for_server(REAL_TEST_PORT, timeout=BOOT_TIMEOUT)
        if not ready:
            # Drain a snippet of the log if we have one so the skip message
            # tells the user *why* the boot failed.
            tail = ""
            if log_path and log_path.exists():
                tail = log_path.read_text(errors="replace")[-2000:]
            proc.kill()
            proc.wait(timeout=10)
            pytest.skip(
                f"vllm serve did not become ready on port {REAL_TEST_PORT} "
                f"within {BOOT_TIMEOUT:.0f}s. "
                f"Last log lines:\n{tail}"
            )
        yield REAL_TEST_PORT
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if hasattr(log_file, "close"):
            log_file.close()


@pytest.fixture(scope="module")
def core(live_server):
    """
    Reload core with VLLM_BASE_URL pointing at our test server, so the
    adapter and helper functions hit the live process — no mocks.
    """
    base_url = f"http://{REAL_TEST_HOST}:{live_server}"
    os.environ["VLLM_BASE_URL"] = base_url
    # Generous timeout: the first /chat/completions warms the kernel.
    os.environ.setdefault("VLLM_TIMEOUT", "120")
    os.environ.setdefault("VLLM_MAX_TOKENS", "32")

    import claude_codex_local.core as pb

    pb = importlib.reload(pb)
    yield pb

    # Don't unset — other tests may want them, and the process is exiting.


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealVLLMAdapter:
    """VLLMAdapter against a live vllm serve."""

    def test_name(self, core):
        adapter = core.VLLMAdapter()
        assert adapter.name == "vllm"

    def test_detect_finds_running_server(self, core):
        adapter = core.VLLMAdapter()
        result = adapter.detect()
        assert result["present"] is True, f"detect returned: {result}"
        assert result.get("base_url", "").endswith(str(REAL_TEST_PORT))

    def test_healthcheck_ok(self, core):
        adapter = core.VLLMAdapter()
        result = adapter.healthcheck()
        assert result.get("ok") is True, f"healthcheck failed: {result.get('detail')}"
        assert "model(s) available" in result.get("detail", "")

    def test_list_models_returns_loaded_model(self, core):
        adapter = core.VLLMAdapter()
        models = adapter.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1, f"expected at least one model, got {models}"
        names = [m["name"] for m in models]
        # vLLM reports the model id we passed on the command line.
        assert TEST_MODEL in names, f"{TEST_MODEL} not in {names}"

    def test_run_test_real_inference(self, core):
        adapter = core.VLLMAdapter()
        result = adapter.run_test(TEST_MODEL)
        assert result.get("ok") is True, f"run_test failed: {result}"
        assert isinstance(result.get("response"), str)
        assert len(result["response"]) > 0
        # vLLM always reports usage; assert we propagated it.
        assert result.get("completion_tokens") is not None
        assert result.get("duration_seconds") is not None
        assert result["duration_seconds"] > 0


class TestRealSmokeTestHelper:
    """smoke_test_vllm_model() — direct helper, no adapter wrapper."""

    def test_smoke_test_returns_timing(self, core, live_server):
        result = core.smoke_test_vllm_model(
            model=TEST_MODEL,
            base_url=f"http://{REAL_TEST_HOST}:{live_server}",
            api_key="",
            timeout=120,
            max_tokens=32,
        )
        assert result.get("ok") is True, f"smoke_test failed: {result}"
        assert result.get("tokens_per_second") is None or result["tokens_per_second"] > 0
        assert result.get("duration_seconds") is not None
        assert result["duration_seconds"] > 0


class TestRealVLLMConfiguration:
    """Adapter env-var wiring against the live server."""

    def test_env_var_base_url_drives_adapter(self, core):
        adapter = core.VLLMAdapter()
        assert adapter._base_url == f"http://{REAL_TEST_HOST}:{REAL_TEST_PORT}"
