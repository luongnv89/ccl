from __future__ import annotations

import importlib
import json
import os
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

import claude_codex_local.core as core
import claude_codex_local.wizard as wizard


class _StubAsk:
    """Minimal questionary stub returning a pre-programmed answer."""

    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class FakeResponse:
    def __init__(self, body: dict, status: int = 200, headers: dict | None = None):
        self.body = json.dumps(body).encode()
        self.status = status
        self._headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self):
        return self.body

    @property
    def headers(self):
        class Headers:
            def __init__(self, data: dict):
                self._data = data

            def get(self, key: str, default: str = ""):
                return self._data.get(key, default)

        return Headers(self._headers)


def reload_modules():
    pb = importlib.reload(core)
    wz = importlib.reload(wizard)
    return pb, wz


@pytest.fixture(autouse=True)
def restore_reloaded_modules(tmp_path):
    env_keys = [
        "CLAUDE_CODEX_LOCAL_STATE_DIR",
        "OLLAMA_HOST",
        "OLLAMA_API_KEY",
        "LLAMACPP_BASE_URL",
        "LLAMACPP_API_KEY",
        "VLLM_BASE_URL",
        "VLLM_API_KEY",
    ]
    original = {key: os.environ.get(key) for key in env_keys}
    os.environ["CLAUDE_CODEX_LOCAL_STATE_DIR"] = str(tmp_path / "state")
    yield
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reload_modules()


def test_remote_ollama_uses_http_without_local_cli(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    pb, _ = reload_modules()
    monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False, "version": ""})

    def fake_urlopen(req, timeout):
        assert req.full_url == "http://gpu-box.local:11434/api/tags"
        return FakeResponse({"models": [{"name": "qwen:7b", "digest": "sha256:abc"}]})

    with patch("urllib.request.urlopen", fake_urlopen):
        adapter = pb.OllamaAdapter()
        assert adapter.detect()["present"] is True
        assert adapter.list_models() == [
            {
                "name": "qwen:7b",
                "id": "sha256:abc",
                "size": "",
                "modified": "",
                "local": False,
            }
        ]
        assert adapter.healthcheck()["ok"] is True


def test_remote_llamacpp_does_not_require_local_binary(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()
    monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})

    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/health"):
            return FakeResponse({}, status=200)
        if req.full_url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "remote-gguf"}]})
        raise urllib.error.URLError("unexpected URL")

    with patch("urllib.request.urlopen", fake_urlopen):
        info = pb.llamacpp_info()

    assert info["present"] is True
    assert info["server_running"] is True
    assert info["base_url"] == "http://gpu-box.local:8001"
    assert info["model"] == "remote-gguf"
    assert info["remote"] is True


def test_remote_vllm_does_not_require_local_cli(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.local:8000")
    pb, _ = reload_modules()
    monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False, "version": ""})

    def fake_urlopen(req, timeout):
        assert req.full_url == "http://gpu-box.local:8000/v1/models"
        return FakeResponse(
            {"data": [{"id": "remote-vllm", "object": "model"}]},
            headers={"X-VLLM-Version": "0.7.0"},
        )

    with patch("urllib.request.urlopen", fake_urlopen):
        info = pb.vllm_info()

    assert info["present"] is True
    assert info["server_reachable"] is True
    assert info["version"] == "0.7.0"
    assert info["models"] == [{"name": "remote-vllm", "format": "unknown", "local": False}]


def test_remote_wiring_uses_configured_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-box.local:8001")
    _, wz = reload_modules()

    codex_ollama = wz._wire_codex("ollama", "qwen:7b")
    assert codex_ollama is not None
    assert codex_ollama.argv == ["codex", "-m", "qwen:7b"]
    assert codex_ollama.env["OPENAI_BASE_URL"] == "http://gpu-box.local:11434/v1"

    claude_ollama = wz._wire_claude("ollama", "qwen:7b")
    assert claude_ollama is not None
    assert claude_ollama.argv == ["claude", "--model", "qwen:7b"]
    assert claude_ollama.env["ANTHROPIC_BASE_URL"] == "http://gpu-box.local:11434/v1"

    codex_llamacpp = wz._wire_codex("llamacpp", "remote-gguf")
    assert codex_llamacpp is not None
    assert codex_llamacpp.env["OPENAI_BASE_URL"] == "http://llama-box.local:8001/v1"


def test_refresh_selected_engine_accepts_remote_without_local_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-box.local:8001")
    pb, wz = reload_modules()
    monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False, "version": ""})
    monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})

    def fake_urlopen(req, timeout):
        if req.full_url == "http://gpu-box.local:11434/api/tags":
            return FakeResponse({"models": [{"name": "qwen:7b"}]})
        if req.full_url == "http://llama-box.local:8001/health":
            return FakeResponse({}, status=200)
        if req.full_url == "http://llama-box.local:8001/v1/models":
            return FakeResponse({"data": [{"id": "remote-gguf"}]})
        raise urllib.error.URLError("unexpected URL")

    profile = {"tools": {}, "presence": {}}
    with patch("urllib.request.urlopen", fake_urlopen):
        assert wz._refresh_selected_engine(profile, "ollama") is True
        assert wz._refresh_selected_engine(profile, "llamacpp") is True

    assert profile["tools"]["ollama"]["base_url"] == "http://gpu-box.local:11434"
    assert profile["ollama"]["models"][0]["name"] == "qwen:7b"
    assert profile["tools"]["llamacpp"]["base_url"] == "http://llama-box.local:8001"
    assert profile["llamacpp"]["model"] == "remote-gguf"


def test_vllm_env_api_key_is_materialized_to_keyfile(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.local:8000")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-env-key")
    _, wz = reload_modules()
    assert not wz.pb.VLLM_KEY_FILE.exists()

    claude = wz._wire_claude("vllm", "remote-vllm")
    assert claude is not None
    # Key now lives on disk, never inline in env.
    assert "ANTHROPIC_API_KEY" not in claude.env
    assert "ANTHROPIC_AUTH_TOKEN" not in claude.env
    assert wz.pb.VLLM_KEY_FILE.exists()
    assert wz.pb.VLLM_KEY_FILE.read_text().strip() == "vllm-env-key"
    assert wz.pb.VLLM_KEY_FILE.stat().st_mode & 0o777 == 0o600
    expected_expr = f'"$(cat {wz.pb.VLLM_KEY_FILE!s})"'
    assert claude.raw_env["ANTHROPIC_API_KEY"] == expected_expr
    assert claude.raw_env["ANTHROPIC_AUTH_TOKEN"] == expected_expr

    codex = wz._wire_codex("vllm", "remote-vllm")
    assert codex is not None
    assert "OPENAI_API_KEY" not in codex.env
    assert codex.raw_env["OPENAI_API_KEY"] == expected_expr

    assert wz._pi_api_key_for_engine("vllm") == f"!cat {wz.pb.VLLM_KEY_FILE!s}"


def test_ollama_env_api_key_is_materialized_to_keyfile(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-env-key")
    _, wz = reload_modules()
    assert not wz.pb.OLLAMA_KEY_FILE.exists()

    claude = wz._wire_claude("ollama", "qwen:7b")
    assert claude is not None
    assert "ANTHROPIC_API_KEY" not in claude.env
    assert "ANTHROPIC_AUTH_TOKEN" not in claude.env
    assert wz.pb.OLLAMA_KEY_FILE.exists()
    assert wz.pb.OLLAMA_KEY_FILE.read_text().strip() == "ollama-env-key"
    assert wz.pb.OLLAMA_KEY_FILE.stat().st_mode & 0o777 == 0o600
    expected_expr = f'"$(cat {wz.pb.OLLAMA_KEY_FILE!s})"'
    assert claude.raw_env["ANTHROPIC_API_KEY"] == expected_expr
    assert claude.raw_env["ANTHROPIC_AUTH_TOKEN"] == expected_expr

    codex = wz._wire_codex("ollama", "qwen:7b")
    assert codex is not None
    assert "OPENAI_API_KEY" not in codex.env
    assert codex.raw_env["OPENAI_API_KEY"] == expected_expr

    assert wz._pi_api_key_for_engine("ollama") == f"!cat {wz.pb.OLLAMA_KEY_FILE!s}"


def test_ollama_without_api_key_keeps_placeholder_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    _, wz = reload_modules()

    claude = wz._wire_claude("ollama", "qwen:7b")
    assert claude is not None
    assert claude.env["ANTHROPIC_API_KEY"] == "ollama"
    assert not wz.pb.OLLAMA_KEY_FILE.exists()
    assert claude.raw_env == {}

    codex = wz._wire_codex("ollama", "qwen:7b")
    assert codex is not None
    assert codex.env["OPENAI_API_KEY"] == "ollama"
    assert codex.raw_env == {}

    assert wz._pi_api_key_for_engine("ollama") == "ollama"


def test_llamacpp_env_api_key_is_materialized_to_keyfile(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-box.local:8001")
    monkeypatch.setenv("LLAMACPP_API_KEY", "llamacpp-env-key")
    _, wz = reload_modules()
    assert not wz.pb.LLAMACPP_KEY_FILE.exists()

    claude = wz._wire_claude("llamacpp", "remote-gguf")
    assert claude is not None
    assert "ANTHROPIC_API_KEY" not in claude.env
    assert "ANTHROPIC_AUTH_TOKEN" not in claude.env
    assert wz.pb.LLAMACPP_KEY_FILE.exists()
    assert wz.pb.LLAMACPP_KEY_FILE.read_text().strip() == "llamacpp-env-key"
    assert wz.pb.LLAMACPP_KEY_FILE.stat().st_mode & 0o777 == 0o600
    expected_expr = f'"$(cat {wz.pb.LLAMACPP_KEY_FILE!s})"'
    assert claude.raw_env["ANTHROPIC_API_KEY"] == expected_expr
    assert claude.raw_env["ANTHROPIC_AUTH_TOKEN"] == expected_expr

    codex = wz._wire_codex("llamacpp", "remote-gguf")
    assert codex is not None
    assert "OPENAI_API_KEY" not in codex.env
    assert codex.raw_env["OPENAI_API_KEY"] == expected_expr

    assert wz._pi_api_key_for_engine("llamacpp") == f"!cat {wz.pb.LLAMACPP_KEY_FILE!s}"


def test_vllm_keyfile_wins_over_env_when_both_present(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.local:8000")
    monkeypatch.setenv("VLLM_API_KEY", "env-key-should-be-ignored")
    _, wz = reload_modules()
    # Simulate a pre-existing user-managed VLLM_KEY_FILE.
    wz.pb.VLLM_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    wz.pb.VLLM_KEY_FILE.write_text("user-managed-key\n")
    wz.pb.VLLM_KEY_FILE.chmod(0o600)

    wz._wire_claude("vllm", "remote-vllm")
    # File contents must not be overwritten by env-driven materialization.
    assert wz.pb.VLLM_KEY_FILE.read_text().strip() == "user-managed-key"


def test_emitted_helper_script_does_not_contain_literal_key(monkeypatch, tmp_path):
    # End-to-end shell-script emission must not leak the literal key bytes.
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "sensitive-ollama-key-do-not-leak")
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-box.local:8001")
    monkeypatch.setenv("LLAMACPP_API_KEY", "sensitive-llamacpp-key-do-not-leak")
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box.local:8000")
    monkeypatch.setenv("VLLM_API_KEY", "sensitive-vllm-key-do-not-leak")
    _, wz = reload_modules()
    for engine in ("ollama", "llamacpp", "vllm"):
        for wire_fn, harness in ((wz._wire_claude, "claude"), (wz._wire_codex, "codex")):
            result = wire_fn(engine, "model-tag")
            assert result is not None
            path = wz._write_helper_script(harness, result, engine=engine)
            body = path.read_text()
            assert "sensitive-ollama-key-do-not-leak" not in body
            assert "sensitive-llamacpp-key-do-not-leak" not in body
            assert "sensitive-vllm-key-do-not-leak" not in body


def test_verify_materializes_remote_keyfile_raw_env(monkeypatch, tmp_path):
    # The Step 7 verify path must still be able to resolve $(cat …)
    # raw_env entries produced by the new ollama/llamacpp materializations.
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-env-key")
    _, wz = reload_modules()
    claude = wz._wire_claude("ollama", "qwen:7b")
    assert claude is not None
    resolved = wz._materialize_raw_env(dict(claude.raw_env))
    assert resolved["ANTHROPIC_API_KEY"] == "ollama-env-key"
    assert resolved["ANTHROPIC_AUTH_TOKEN"] == "ollama-env-key"


def test_machine_profile_discovers_remote_engines_without_local_binaries(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://llama-box.local:8001")
    monkeypatch.setenv("VLLM_BASE_URL", "gpu-box.local:8000")
    pb, _ = reload_modules()
    monkeypatch.setattr(pb, "llmfit_system", lambda: None)
    monkeypatch.setattr(pb, "lms_info", lambda: {"present": False, "models": []})
    monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False, "version": ""})
    monkeypatch.setattr(pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""})
    monkeypatch.setattr(
        pb.OpenRouterAdapter, "detect", lambda self: {"present": False, "version": ""}
    )

    def fake_urlopen(req, timeout):
        if req.full_url == "http://gpu-box.local:11434/api/tags":
            return FakeResponse({"models": [{"name": "qwen:7b"}]})
        if req.full_url == "http://llama-box.local:8001/health":
            return FakeResponse({}, status=200)
        if req.full_url == "http://llama-box.local:8001/v1/models":
            return FakeResponse({"data": [{"id": "remote-gguf"}]})
        if req.full_url == "http://gpu-box.local:8000/v1/models":
            return FakeResponse({"data": [{"id": "remote-vllm"}]})
        raise urllib.error.URLError("unexpected URL")

    with patch("urllib.request.urlopen", fake_urlopen):
        profile = pb.machine_profile(run_llmfit=False)

    assert "ollama" in profile["presence"]["engines"]
    assert "llamacpp" in profile["presence"]["engines"]
    assert "vllm" in profile["presence"]["engines"]
    assert profile["tools"]["vllm"]["base_url"] == "http://gpu-box.local:8000"


def test_remote_ollama_verify_uses_wired_command_not_ollama_launch(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box.local:11434")
    _, wz = reload_modules()
    state = wz.WizardState(
        primary_harness="claude",
        primary_engine="ollama",
        engine_model_tag="qwen:7b",
        wire_result={
            "argv": ["claude", "--model", "qwen:7b"],
            "env": {"ANTHROPIC_BASE_URL": "http://gpu-box.local:11434/v1"},
            "effective_tag": "qwen:7b",
        },
    )
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd

        class Proc:
            returncode = 0
            stdout = "READY"
            stderr = ""

        return Proc()

    monkeypatch.setattr(wz.subprocess, "run", fake_run)
    assert wz.step_2_7_verify(state, non_interactive=True) is True
    assert captured["cmd"][:2] == ["claude", "--model"]
    assert "ollama" not in captured["cmd"]


def test_machine_profile_cache_refreshes_when_remote_endpoint_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_HOST", "http://first.local:11434")
    pb, _ = reload_modules()
    monkeypatch.setattr(pb, "llmfit_system", lambda: None)
    monkeypatch.setattr(pb, "lms_info", lambda: {"present": False, "models": []})
    monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False, "version": ""})
    monkeypatch.setattr(pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""})
    monkeypatch.setattr(
        pb.OpenRouterAdapter, "detect", lambda self: {"present": False, "version": ""}
    )

    seen: list[str] = []

    def fake_urlopen(req, timeout):
        seen.append(req.full_url)
        if req.full_url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": req.full_url.split("/")[2]}]})
        raise urllib.error.URLError("offline")

    with patch("urllib.request.urlopen", fake_urlopen):
        first = pb.machine_profile(run_llmfit=False)
        assert first["tools"]["ollama"]["base_url"] == "http://first.local:11434"

        monkeypatch.setenv("OLLAMA_HOST", "http://second.local:11434")
        pb, _ = reload_modules()
        monkeypatch.setattr(pb, "llmfit_system", lambda: None)
        monkeypatch.setattr(pb, "lms_info", lambda: {"present": False, "models": []})
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False, "version": ""})
        monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": False, "version": ""}
        )
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )
        monkeypatch.setattr(
            pb.OpenRouterAdapter, "detect", lambda self: {"present": False, "version": ""}
        )
        second = pb.machine_profile(run_llmfit=False)

    assert second["tools"]["ollama"]["base_url"] == "http://second.local:11434"
    assert any("second.local" in url for url in seen)


def test_verify_materializes_vllm_keyfile_raw_env(monkeypatch, tmp_path):
    key_file = tmp_path / "vllm-api-key"
    key_file.write_text("from-file\n")
    _, wz = reload_modules()
    state = wz.WizardState(
        primary_harness="claude",
        primary_engine="vllm",
        engine_model_tag="remote-vllm",
        wire_result={
            "argv": ["claude", "--model", "remote-vllm"],
            "env": {"ANTHROPIC_BASE_URL": "http://gpu-box.local:8000"},
            "raw_env": {
                "ANTHROPIC_API_KEY": f'"$(cat {key_file})"',
                "ANTHROPIC_AUTH_TOKEN": f'"$(cat {key_file})"',
            },
            "effective_tag": "remote-vllm",
        },
    )
    captured: dict[str, dict[str, str]] = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]

        class Proc:
            returncode = 0
            stdout = "READY"
            stderr = ""

        return Proc()

    monkeypatch.setattr(wz.subprocess, "run", fake_run)
    assert wz.step_2_7_verify(state, non_interactive=True) is True
    assert captured["env"]["ANTHROPIC_API_KEY"] == "from-file"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "from-file"


# ---------------------------------------------------------------------------
# Issue #123 — branch llamacpp downstream code paths on remote vs. local.
# Each test exercises one of the acceptance-criteria touch points: the helper
# script must not embed a local spawn, llamacpp_info must report `present`
# purely from the remote /health probe, and the start/ensure paths must
# refuse to spawn a local llama-server when the base URL is remote.
# ---------------------------------------------------------------------------


def test_helper_script_omits_llama_server_block_for_remote_llamacpp(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    _, wz = reload_modules()
    result = wz._wire_claude("llamacpp", "remote-gguf")
    assert result is not None
    path = wz._write_helper_script("claude", result, engine="llamacpp")
    body = path.read_text()
    # AC #2: no local llama-server spawn / binary check / ccl serve fallback.
    assert "llama-server" not in body
    assert "ccl serve" not in body
    assert "__CCL_HEALTH_URL" not in body
    # The remote base URL must still be wired into the harness env.
    assert "http://gpu-box.local:8001" in body


def test_helper_script_keeps_llama_server_block_for_local_llamacpp(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("LLAMACPP_BASE_URL", raising=False)
    _, wz = reload_modules()
    result = wz._wire_claude("llamacpp", "local-gguf")
    assert result is not None
    path = wz._write_helper_script("claude", result, engine="llamacpp")
    body = path.read_text()
    # AC #4: existing local behaviour preserved (pre-flight stanza still
    # emitted for loopback base URLs).
    assert "ccl serve" in body
    assert "llama-server" in body


def test_llamacpp_info_remote_present_purely_from_health_probe(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()

    detect_calls: list[bool] = []

    def fake_detect():
        detect_calls.append(True)
        return {"present": True, "binary": "llama-server", "version": "x"}

    # Even if llamacpp_detect *would* return present=True, the remote branch
    # must not call it — `present` for a remote URL comes from /health only.
    monkeypatch.setattr(pb, "llamacpp_detect", fake_detect)

    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/health"):
            return FakeResponse({}, status=200)
        if req.full_url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "remote-gguf"}]})
        raise urllib.error.URLError("unexpected URL")

    with patch("urllib.request.urlopen", fake_urlopen):
        info = pb.llamacpp_info()

    assert info["remote"] is True
    assert info["server_running"] is True
    assert info["present"] is True
    # Local detect must NOT be consulted for remote endpoints.
    assert detect_calls == []
    assert info["binary"] == ""


def test_llamacpp_info_remote_health_failure_marks_not_present(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()
    # Even if a local binary is on PATH, a remote URL whose /health is down
    # must report present=False — there is nothing for the user here.
    monkeypatch.setattr(
        pb, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server", "version": "x"}
    )

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("unreachable")

    with patch("urllib.request.urlopen", fake_urlopen):
        info = pb.llamacpp_info()

    assert info["remote"] is True
    assert info["server_running"] is False
    assert info["present"] is False


def test_llamacpp_start_server_refuses_to_spawn_when_remote(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()

    popen_calls: list[tuple] = []

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        raise AssertionError("Popen must not be invoked for remote llamacpp")

    monkeypatch.setattr(pb.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        pb,
        "llamacpp_detect",
        lambda: {"present": True, "binary": "llama-server", "version": "x"},
    )

    out = pb.llamacpp_start_server(model_path="/tmp/does-not-matter.gguf")
    # AC #1: the wizard must not start a local llama-server when remote.
    assert out["ok"] is False
    assert out.get("remote") is True
    assert "remote" in (out.get("error") or "").lower()
    assert popen_calls == []


def test_ensure_llamacpp_server_running_remote_ok_when_health_passes(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, wz = reload_modules()

    monkeypatch.setattr(
        pb,
        "llamacpp_info",
        lambda: {
            "remote": True,
            "server_running": True,
            "base_url": "http://gpu-box.local:8001",
            "server_port": 8001,
            "model": "remote-gguf",
            "present": True,
            "binary": "",
        },
    )

    def boom(**kwargs):
        raise AssertionError("llamacpp_start_server must not be called for remote")

    monkeypatch.setattr(pb, "llamacpp_start_server", boom)

    state = wz.WizardState(
        primary_harness="claude",
        primary_engine="llamacpp",
        engine_model_tag="remote-gguf",
    )
    result = wz._ensure_llamacpp_server_running(state)
    assert result["ok"] is True
    assert result.get("remote") is True


def test_ensure_llamacpp_server_running_remote_error_when_health_fails(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, wz = reload_modules()

    monkeypatch.setattr(
        pb,
        "llamacpp_info",
        lambda: {
            "remote": True,
            "server_running": False,
            "base_url": "http://gpu-box.local:8001",
            "server_port": 8001,
            "model": None,
            "present": False,
            "binary": "",
        },
    )

    def boom(**kwargs):
        raise AssertionError("must not try to spawn locally for a remote endpoint")

    monkeypatch.setattr(pb, "llamacpp_start_server", boom)

    state = wz.WizardState(
        primary_harness="claude",
        primary_engine="llamacpp",
        engine_model_tag="remote-gguf",
    )
    result = wz._ensure_llamacpp_server_running(state)
    assert result["ok"] is False
    assert result.get("remote") is True
    assert "remote" in result["error"].lower()


def test_llamacpp_adapter_healthcheck_remote_mentions_remote_not_path(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()
    monkeypatch.setattr(
        pb, "llamacpp_detect", lambda: {"present": False, "binary": "", "version": ""}
    )

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("unreachable")

    with patch("urllib.request.urlopen", fake_urlopen):
        result = pb.LlamaCppAdapter().healthcheck()

    assert result["ok"] is False
    detail = result["detail"].lower()
    assert "remote" in detail
    assert "path" not in detail


def test_llamacpp_adapter_healthcheck_remote_ok_when_health_passes(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://gpu-box.local:8001")
    pb, _ = reload_modules()
    monkeypatch.setattr(
        pb, "llamacpp_detect", lambda: {"present": False, "binary": "", "version": ""}
    )

    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/health"):
            return FakeResponse({}, status=200)
        if req.full_url.endswith("/v1/models"):
            return FakeResponse({"data": [{"id": "remote-gguf"}]})
        raise urllib.error.URLError("unexpected URL")

    with patch("urllib.request.urlopen", fake_urlopen):
        result = pb.LlamaCppAdapter().healthcheck()

    assert result["ok"] is True
    assert "http://gpu-box.local:8001" in result["detail"]


# ---------------------------------------------------------------------------
# Issue #125 — extra coverage for the interactive local-vs-remote wizard.
#
# PRs #122/#123/#124 already covered most of the seven coverage bullets in
# issue #125. The tests below close the remaining gaps:
#
#   * Bullet 1 — the local-vs-remote prompt is shown for `vllm` (the ollama
#     and llamacpp cases were covered by PR #122) AND is NOT shown for the
#     three non-local-capable engines (`9router`/`openrouter`/`lmstudio`).
#   * Bullet 3 — the optional API-key prompt fires ONLY for `llamacpp` and
#     `vllm`. PR #122 exercised the llamacpp API-key path implicitly via the
#     rc-persist test but never asserted that `questionary.password` is NOT
#     called for `ollama`, nor that it IS called exactly once for `vllm`.
# ---------------------------------------------------------------------------


def _engine_picker_profile(*, engines: list[str] | None = None) -> dict:
    """Minimal profile with the picked engine already in presence.engines."""
    return {
        "presence": {
            "harnesses": ["claude"],
            "engines": engines or ["ollama"],
        },
        "host": {"system": "Darwin", "machine": "arm64"},
        "ollama": {"models": [{"name": "qwen2.5-coder:7b"}]},
        "lmstudio": {"server_running": False, "models": []},
    }


class TestStep3LocalVsRemoteCoverageGaps:
    """Issue #125 — close the coverage gaps PRs #122/#123/#124 left open."""

    @pytest.fixture(autouse=True)
    def _sandbox_home_and_env(self, tmp_path, monkeypatch):
        """
        Tests in this class need three guarantees that aren't already
        satisfied by the module-level autouse fixture:

          1. `Path.home()` must point under tmp_path so any rc write a stray
             code path performs cannot touch the real user's shell rc.
          2. `SHELL` is deterministic so `_detect_shell_rc` is reproducible.
          3. The wizard module is reloaded after `Path.home` is patched so
             any module-level cached home resolves to the sandbox.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        # Reload so module-level reads of env/HOME pick up the sandbox.
        reload_modules()
        yield

    # -- Bullet 1: prompt IS shown for vllm ---------------------------------

    def test_local_or_remote_prompt_shown_for_vllm(self, monkeypatch):
        """vllm picker must surface the local-vs-remote question."""
        pb, wz = reload_modules()
        state = wz.WizardState(primary_harness="claude")
        state.profile = _engine_picker_profile(engines=["vllm"])

        prompts: list[str] = []

        def fake_select(message, choices, default=None):
            prompts.append(message)
            if "primary" in message:
                return _StubAsk("vllm")
            # The local-vs-remote prompt — pick Local to keep the test small.
            if "Local" in str(choices):
                return _StubAsk("Local")
            raise AssertionError(f"unexpected select prompt: {message}")

        monkeypatch.setattr(wz.questionary, "select", fake_select)
        monkeypatch.setattr(wz, "_refresh_selected_engine", lambda *a, **kw: True)
        monkeypatch.setattr(wz, "_ensure_tool", lambda name: True)

        assert wz.step_3_select_engine(state, non_interactive=False) is True
        assert state.primary_engine == "vllm"
        assert any("Run `vllm` locally, or use a remote endpoint" in p for p in prompts)

    # -- Bullet 1: prompt is NOT shown for non-local-capable engines --------

    @pytest.mark.parametrize("engine", ["9router", "openrouter", "lmstudio"])
    def test_local_or_remote_prompt_skipped_for_non_local_engines(self, monkeypatch, engine):
        """
        Engines outside `_LOCAL_OR_REMOTE_ENGINES` must never reach
        `_prompt_local_or_remote`. We assert this directly by replacing it
        with a boom — the cleanest possible signal.
        """
        pb, wz = reload_modules()
        state = wz.WizardState(primary_harness="claude")
        state.profile = _engine_picker_profile(engines=[engine])

        prompts: list[str] = []

        def fake_select(message, choices, default=None):
            prompts.append(message)
            if "primary" in message:
                return _StubAsk(engine)
            raise AssertionError(f"unexpected select prompt for {engine!r}: {message!r}")

        def boom_local_or_remote(*args, **kwargs):
            raise AssertionError(f"_prompt_local_or_remote must not be called for {engine!r}")

        monkeypatch.setattr(wz.questionary, "select", fake_select)
        monkeypatch.setattr(wz, "_prompt_local_or_remote", boom_local_or_remote)
        monkeypatch.setattr(wz, "_refresh_selected_engine", lambda *a, **kw: True)
        monkeypatch.setattr(wz, "_ensure_tool", lambda name: True)

        assert wz.step_3_select_engine(state, non_interactive=False) is True
        assert state.primary_engine == engine
        # Belt-and-braces: the prompt message text must never appear.
        assert all("locally, or use a remote endpoint" not in p for p in prompts)
        assert engine in {"9router", "openrouter", "lmstudio"}
        assert engine not in wz._LOCAL_OR_REMOTE_ENGINES

    # -- Bullet 3: ollama remote DOES NOT prompt for API key ----------------

    def test_remote_ollama_does_not_prompt_for_api_key(self, monkeypatch):
        """
        AC #3 of issue #125: the API-key prompt is for `llamacpp`/`vllm`
        only. Selecting Remote for `ollama` must drive the URL prompt but
        must NOT invoke `questionary.password`.
        """
        pb, wz = reload_modules()
        state = wz.WizardState(primary_harness="claude")
        state.profile = _engine_picker_profile(engines=["ollama"])

        def fake_select(message, choices, default=None):
            if "primary" in message:
                return _StubAsk("ollama")
            if "Local" in str(choices):
                return _StubAsk("Remote")
            raise AssertionError(f"unexpected select prompt: {message}")

        text_calls: list[str] = []

        def fake_text(message, default=""):
            text_calls.append(message)
            return _StubAsk("http://gpu-box.local:11434")

        def boom_password(*args, **kwargs):
            raise AssertionError(
                "questionary.password must NOT be called for the ollama "
                "remote branch (ollama has no per-request API key)"
            )

        monkeypatch.setattr(wz.questionary, "select", fake_select)
        monkeypatch.setattr(wz.questionary, "text", fake_text)
        monkeypatch.setattr(wz.questionary, "password", boom_password)
        monkeypatch.setattr(wz.questionary, "confirm", lambda *a, **kw: _StubAsk(False))
        monkeypatch.setattr(wz, "_refresh_selected_engine", lambda *a, **kw: True)
        monkeypatch.setattr(wz, "_ensure_tool", lambda name: True)

        assert wz.step_3_select_engine(state, non_interactive=False) is True
        # URL prompt did fire — make sure the test would actually catch a
        # regression where the whole remote path is short-circuited.
        assert any("base URL" in m for m in text_calls)
        assert pb.OLLAMA_BASE_URL == "http://gpu-box.local:11434"

    # -- Bullet 3: llamacpp remote DOES prompt for API key ------------------

    def test_remote_llamacpp_prompts_for_api_key(self, monkeypatch):
        """Llamacpp remote must invoke `questionary.password` exactly once."""
        pb, wz = reload_modules()
        state = wz.WizardState(primary_harness="claude")
        state.profile = _engine_picker_profile(engines=["llamacpp"])

        def fake_select(message, choices, default=None):
            if "primary" in message:
                return _StubAsk("llamacpp")
            if "Local" in str(choices):
                return _StubAsk("Remote")
            raise AssertionError(f"unexpected select prompt: {message}")

        password_calls: list[str] = []

        def fake_password(message, default=""):
            password_calls.append(message)
            return _StubAsk("a-llamacpp-key")

        monkeypatch.setattr(wz.questionary, "select", fake_select)
        monkeypatch.setattr(
            wz.questionary, "text", lambda *a, **kw: _StubAsk("http://llama-box.local:8001")
        )
        monkeypatch.setattr(wz.questionary, "password", fake_password)
        monkeypatch.setattr(wz.questionary, "confirm", lambda *a, **kw: _StubAsk(False))
        monkeypatch.setattr(wz, "_refresh_selected_engine", lambda *a, **kw: True)
        monkeypatch.setattr(wz, "_ensure_tool", lambda name: True)

        assert wz.step_3_select_engine(state, non_interactive=False) is True
        assert len(password_calls) == 1
        assert "llamacpp" in password_calls[0]
        assert pb.LLAMACPP_BASE_URL == "http://llama-box.local:8001"
        assert pb.LLAMACPP_API_KEY == "a-llamacpp-key"

    # -- Bullet 3: vllm remote DOES prompt for API key ----------------------

    def test_remote_vllm_prompts_for_api_key(self, monkeypatch):
        """vLLM remote must invoke `questionary.password` exactly once."""
        pb, wz = reload_modules()
        state = wz.WizardState(primary_harness="claude")
        state.profile = _engine_picker_profile(engines=["vllm"])

        def fake_select(message, choices, default=None):
            if "primary" in message:
                return _StubAsk("vllm")
            if "Local" in str(choices):
                return _StubAsk("Remote")
            raise AssertionError(f"unexpected select prompt: {message}")

        password_calls: list[str] = []

        def fake_password(message, default=""):
            password_calls.append(message)
            return _StubAsk("a-vllm-key")

        monkeypatch.setattr(wz.questionary, "select", fake_select)
        monkeypatch.setattr(
            wz.questionary, "text", lambda *a, **kw: _StubAsk("http://gpu-box.local:8000")
        )
        monkeypatch.setattr(wz.questionary, "password", fake_password)
        monkeypatch.setattr(wz.questionary, "confirm", lambda *a, **kw: _StubAsk(False))
        monkeypatch.setattr(wz, "_refresh_selected_engine", lambda *a, **kw: True)
        monkeypatch.setattr(wz, "_ensure_tool", lambda name: True)

        assert wz.step_3_select_engine(state, non_interactive=False) is True
        assert len(password_calls) == 1
        assert "vllm" in password_calls[0]
        assert pb.VLLM_BASE_URL == "http://gpu-box.local:8000"
        assert pb.VLLM_API_KEY == "a-vllm-key"
