"""
Security-gate tests for issue #201.

Covers:
- bounded HTTP response reads at the five user-controlled remote endpoints,
- the repo-local bash -c grep gate (scripts/check_bash_c.py).

All network access is faked via urllib.request.urlopen monkeypatching.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import claude_codex_local._config as cfg_mod
import claude_codex_local._openrouter as _openrouter_mod
import claude_codex_local._vllm as _vllm_mod
import claude_codex_local.core as pb

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_bash_c.py"

_OVERCAP = pb.MAX_RESPONSE_BYTES + 1024


class _FakeResp:
    """Fake urllib response: optional Content-Length, amt-aware read()."""

    def __init__(self, payload: bytes, headers: dict | None = None):
        self._payload = payload
        self.headers = headers or {}

    def read(self, amt: int = -1):
        if amt < 0:
            amt = len(self._payload)
        data = self._payload[:amt]
        self._payload = self._payload[len(data) :]
        return data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _oversized():
    return b"x" * _OVERCAP


# ---------------------------------------------------------------------------
# Bounded reads at the five remote endpoints.
# ---------------------------------------------------------------------------


class TestProbeOpenaiModelsEndpointBounded:
    def test_oversized_response_reported_not_buffered(self, monkeypatch):
        import urllib.request

        def fake_urlopen(req, timeout=None):
            return _FakeResp(_oversized())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = cfg_mod._probe_openai_models_endpoint(
            "http://gpu-box.local:8000/v1", service_name="Test"
        )
        assert result["ok"] is False
        assert result["error_type"] == "response_too_large"

    def test_declared_content_length_over_cap_rejected(self, monkeypatch):
        import urllib.request

        headers = {"Content-Length": str(_OVERCAP)}

        def fake_urlopen(req, timeout=None):
            return _FakeResp(b"tiny", headers=headers)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = cfg_mod._probe_openai_models_endpoint(
            "http://gpu-box.local:8000/v1", service_name="Test"
        )
        assert result["ok"] is False
        assert result["error_type"] == "response_too_large"

    def test_normal_response_still_parses(self, monkeypatch):
        import urllib.request

        body = json.dumps({"data": [{"id": "m1"}]}).encode()

        def fake_urlopen(req, timeout=None):
            return _FakeResp(body)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = cfg_mod._probe_openai_models_endpoint(
            "http://gpu-box.local:8000/v1", service_name="Test"
        )
        assert result["ok"] is True
        assert result["models"] == ["m1"]

    def test_non_http_scheme_refused_before_request(self):
        with pytest.raises(ValueError, match="Refusing non-http"):
            cfg_mod._probe_openai_models_endpoint("file:///etc/passwd", service_name="Test")


class TestOpenRouterBoundedReads:
    def test_fetch_free_models_caps_response(self, monkeypatch):
        import urllib.request

        def fake_urlopen(req, timeout=None):
            return _FakeResp(_oversized())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = _openrouter_mod.fetch_openrouter_free_models()
        assert result["ok"] is False
        assert result["models"] == []
        assert "cap" in str(result.get("error", "")).lower()

    def test_smoke_test_model_caps_response(self, monkeypatch):
        import urllib.request

        def fake_urlopen(req, timeout=None):
            return _FakeResp(_oversized())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = _openrouter_mod.smoke_test_openrouter_model("test-model")
        assert result["ok"] is False
        assert "cap" in str(result.get("error", "")).lower()

    def test_smoke_test_model_refuses_file_base_url(self, monkeypatch):
        import urllib.request

        def fail(*a, **kw):
            raise AssertionError("urlopen must not be called for file:// URLs")

        monkeypatch.setattr(urllib.request, "urlopen", fail)
        result = _openrouter_mod.smoke_test_openrouter_model(
            "test-model", base_url="file:///etc/passwd"
        )
        assert result["ok"] is False
        assert "Refusing non-http" in str(result.get("error", ""))

    def test_fetch_free_models_refuses_file_base_url(self, monkeypatch):
        import urllib.request

        def fail(*a, **kw):
            raise AssertionError("urlopen must not be called for file:// URLs")

        monkeypatch.setattr(urllib.request, "urlopen", fail)
        result = _openrouter_mod.fetch_openrouter_free_models(base_url="file:///etc/passwd")
        assert result["ok"] is False
        assert "Refusing non-http" in str(result.get("error", ""))


class TestVLLMBoundedReads:
    def test_smoke_test_vllm_model_caps_response(self, monkeypatch):
        import urllib.request

        def fake_urlopen(req, timeout=None):
            return _FakeResp(_oversized())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = _vllm_mod.smoke_test_vllm_model("qwen")
        assert result["ok"] is False
        assert "cap" in str(result.get("error", "")).lower()

    def test_adapter_healthcheck_caps_response(self, monkeypatch):
        import urllib.request

        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            # detect() probes /models first; healthcheck's own GET follows.
            if calls["n"] == 0:
                calls["n"] += 1
                return _FakeResp(json.dumps({"data": []}).encode())
            return _FakeResp(_oversized())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        adapter = pb.VLLMAdapter(_base_url="http://127.0.0.1:8000")
        result = adapter.healthcheck()
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Repo-local bash -c grep gate.
# ---------------------------------------------------------------------------


def _run_gate(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestBashCGate:
    def test_current_package_tree_passes(self):
        proc = _run_gate()
        assert proc.returncode == 0, proc.stderr
        assert "wizard_cli.py" in proc.stdout
        assert "wizard_discovery.py" in proc.stdout

    def test_flags_new_bash_c_construction(self, tmp_path):
        bad = tmp_path / "new_module.py"
        bad.write_text('run(["bash", "-c", user_cmd])\n')
        proc = _run_gate(str(tmp_path))
        assert proc.returncode == 1
        assert "new_module.py" in proc.stderr
        assert "bash -c shell-out" in proc.stderr

    def test_flags_single_quote_variant(self, tmp_path):
        bad = tmp_path / "other.py"
        bad.write_text("subprocess.run(['bash', '-c', cmd], check=True)\n")
        proc = _run_gate(str(tmp_path))
        assert proc.returncode == 1

    def test_clean_directory_passes(self, tmp_path):
        ok = tmp_path / "clean.py"
        ok.write_text('run(["bash", "-lc", login_cmd])\n')  # -lc is not -c
        plain = tmp_path / "plain.py"
        plain.write_text("run(['ollama', 'serve'])\n")
        proc = _run_gate(str(tmp_path))
        assert proc.returncode == 0, proc.stderr

    def test_allowlist_file_is_respected(self, tmp_path):
        copied = tmp_path / "wizard_cli.py"
        source = REPO_ROOT / "claude_codex_local" / "wizard_cli.py"
        text = source.read_text()
        line = next(
            ln
            for ln in text.splitlines()
            if '["bash", "-c"' in ln.replace("'bash', '-c'", '["bash", "-c"')
        )
        copied.write_text(line + "\n")
        # Same construction under a non-allowlisted name still fails.
        other = tmp_path / "not_allowlisted.py"
        other.write_text(line + "\n")
        proc = _run_gate(str(tmp_path))
        assert proc.returncode == 1
        assert "not_allowlisted.py" in proc.stderr
