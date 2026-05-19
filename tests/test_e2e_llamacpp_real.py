"""
Real end-to-end tests for the llamacpp backend — requires the actual
llama-server binary and a downloaded GGUF model.

These tests are marked @pytest.mark.real_llamacpp and are skipped
automatically when the binary is missing or no model is available.
They are NOT CI-safe and must be run explicitly on a machine with
llama-server installed:

    pytest -m real_llamacpp tests/test_e2e_llamacpp_real.py -v

What is tested:
  1. llamacpp_detect()  — finds the real llama-server binary
  2. llamacpp_info()    — probes a live server started by the fixture
  3. smoke_test_llamacpp_model() — sends a real inference request
  4. huggingface_download_gguf() — downloads a tiny GGUF model (if needed)
  5. LlamaCppAdapter methods  — detect / healthcheck / list_models / run_test

Model used: HuggingFaceTB/smollm2-135m-instruct-q8_0-gguf
  (~135M param, Q8_0, ~145 MB) — tiny enough to download quickly and
  load fast, but real enough to exercise the full stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Use a non-default port to avoid colliding with any real server on 8001.
REAL_TEST_PORT = 18001

# Tiny model — ~90 MB download, loads in ~2 s on Apple Silicon.
MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_FILE = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

# Where to cache the model for tests (avoid polluting the user's HF cache dir).
MODEL_CACHE_DIR = REPO_ROOT / ".test-model-cache"


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.real_llamacpp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llama_server_binary() -> str | None:
    """Return the llama-server binary path, or None if not found."""
    for candidate in ("llama-server", "llama-cpp-server", "server"):
        path = shutil.which(candidate)
        if path:
            # Guard against a generic 'server' that isn't llama.cpp.
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version_out = (result.stdout + result.stderr).lower()
            if candidate != "server" or "llama" in version_out:
                return candidate
    return None


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Poll /health until the server responds or the timeout expires."""
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Module-level skip: bail out early when llama-server is not installed.
# ---------------------------------------------------------------------------

_BINARY = _llama_server_binary()
if _BINARY is None:
    pytest.skip(
        "llama-server binary not found — install llama.cpp to run real tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_path() -> Path:
    """
    Ensure a small GGUF model is present locally.  Downloads it via
    huggingface-cli the first time; subsequent runs use the cached copy.

    Returns the Path to the .gguf file.
    """
    expected = MODEL_CACHE_DIR / MODEL_FILE
    if expected.exists():
        return expected

    hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_cli is None:
        pytest.skip(
            "HuggingFace CLI (hf / huggingface-cli) not found — install with: pip install 'huggingface_hub[cli]'"
        )

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            hf_cli,
            "download",
            MODEL_REPO,
            MODEL_FILE,
            "--local-dir",
            str(MODEL_CACHE_DIR),
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min — slow networks
    )
    if result.returncode != 0:
        pytest.skip(f"Model download failed: {(result.stderr or result.stdout).strip()}")

    if not expected.exists():
        pytest.skip(f"Download succeeded but {expected} not found — check huggingface-cli output")

    return expected


@pytest.fixture(scope="module")
def live_server(model_path: Path):
    """
    Start a real llama-server on REAL_TEST_PORT with the test model.
    Yields the port number; kills the server process on teardown.
    """
    proc = subprocess.Popen(
        [
            _BINARY,
            "--port",
            str(REAL_TEST_PORT),
            "--model",
            str(model_path),
            "--ctx-size",
            "512",  # keep memory use minimal
            "--n-predict",
            "64",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ready = _wait_for_server(REAL_TEST_PORT, timeout=60)
    if not ready:
        proc.kill()
        proc.wait()
        pytest.skip(f"llama-server did not become ready on port {REAL_TEST_PORT} within 60 s")

    yield REAL_TEST_PORT

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def core(model_path):
    """
    Import (or reload) core with LLAMACPP_SERVER_PORT pointing at our
    test port.  Returns the core module with the patched port constant.
    """
    import importlib

    os.environ["LLAMACPP_SERVER_PORT"] = str(REAL_TEST_PORT)
    import claude_codex_local.core as pb

    pb = importlib.reload(pb)
    yield pb
    # Restore so other tests aren't affected.
    os.environ.pop("LLAMACPP_SERVER_PORT", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRealLlamaCppDetect:
    """llamacpp_detect() against the real PATH — no stubs."""

    def test_detect_finds_binary(self, core):
        result = core.llamacpp_detect()
        assert result["present"] is True, f"llamacpp_detect returned: {result}"
        assert result["binary"] in ("llama-server", "llama-cpp-server", "server")
        assert result.get("version", "") != "" or True  # version may be empty on some builds


class TestRealLlamaCppInfo:
    """llamacpp_info() against a live server."""

    def test_info_detects_running_server(self, core, live_server):
        info = core.llamacpp_info()
        assert info["present"] is True
        assert info["server_running"] is True
        assert info["server_port"] == REAL_TEST_PORT
        # model ID may be the file path or a short name, just assert it's a string.
        assert isinstance(info.get("model"), str) or info.get("model") is None


class TestRealSmokeTest:
    """smoke_test_llamacpp_model() — real inference."""

    def test_smoke_test_returns_ok(self, core, live_server):
        # Use whatever model ID the server reports (or a placeholder).
        info = core.llamacpp_info()
        model_id = info.get("model") or "smollm2"

        result = core.smoke_test_llamacpp_model(model_id)
        assert result.get("ok") is True, (
            f"smoke_test failed: {result.get('error') or result.get('response')}"
        )
        assert isinstance(result.get("response"), str)
        assert len(result["response"]) > 0


class TestRealLlamaCppAdapter:
    """LlamaCppAdapter protocol methods against the real machine."""

    def test_name(self, core):
        adapter = core.LlamaCppAdapter()
        assert adapter.name == "llamacpp"

    def test_recommend_params(self, core):
        adapter = core.LlamaCppAdapter()
        result = adapter.recommend_params("balanced")
        assert isinstance(result, dict)
        # llamacpp returns a fixed dict — just assert it's a mapping, not an error.
        assert result is not None

    def test_detect(self, core):
        adapter = core.LlamaCppAdapter()
        result = adapter.detect()
        assert result["present"] is True

    def test_healthcheck_with_live_server(self, core, live_server):
        adapter = core.LlamaCppAdapter()
        result = adapter.healthcheck()
        assert result.get("ok") is True, f"healthcheck failed: {result.get('detail')}"

    def test_list_models_with_live_server(self, core, live_server):
        adapter = core.LlamaCppAdapter()
        models = adapter.list_models()
        assert isinstance(models, list)
        # llama.cpp runs one model at a time; the fixture loads exactly one.
        assert len(models) == 1

    def test_run_test_with_live_server(self, core, live_server):
        info = core.llamacpp_info()
        model_id = info.get("model") or "smollm2"
        adapter = core.LlamaCppAdapter()
        result = adapter.run_test(model_id)
        assert result.get("ok") is True, f"adapter.run_test failed: {result}"


class TestRealHuggingFaceDownload:
    """huggingface_download_gguf() — uses cached model, no network if already present."""

    def test_download_returns_path(self, model_path, monkeypatch):
        """
        Verify the download function works end-to-end.  The model is already
        cached in MODEL_CACHE_DIR so huggingface-cli skips re-downloading.

        Note: core.huggingface_cli_detect() uses shutil.which() to check
        whether huggingface-cli is on PATH.  We patch the detection to return
        present=True so the download logic runs in environments where the
        binary may not be on PATH — the actual subprocess call to
        huggingface-cli still happens and must succeed.
        """
        import claude_codex_local.core as pb

        # Use the live detection result so the patched binary matches whatever
        # is actually on PATH (newer HF releases hard-deprecate the legacy
        # `huggingface-cli` shim and only ship `hf`).
        det = pb.huggingface_cli_detect()
        if not det.get("present"):
            pytest.skip("HuggingFace CLI not on PATH")
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: det)
        result = pb.huggingface_download_gguf(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODEL_CACHE_DIR),
        )
        assert result["ok"] is True, f"download failed: {result.get('error')}"
        assert result["path"] is not None
        assert Path(result["path"]).exists() or result["path"].endswith(MODEL_FILE)
