"""
Supplementary tests that push coverage into the remaining easy branches:
  * llamacpp_detect
  * smoke_test_ollama_model timeout / failure paths
  * machine_profile aggregation when everything is stubbed
  * wizard.run_find_model_standalone
  * wizard.main CLI dispatcher
  * huggingface_cli_detect / huggingface_download_gguf
  * wizard._download_gguf_via_hf_cli / _download_model llamacpp branch
"""

from __future__ import annotations

import subprocess
import sys

# ---------------------------------------------------------------------------
# core.llamacpp_detect — all-missing and one-present branches.
# ---------------------------------------------------------------------------


class TestLlamacppDetect:
    def test_all_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        assert pb.llamacpp_detect() == {"present": False, "version": ""}

    def test_first_candidate_present(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        calls = []

        def fake(name, args=None):
            calls.append(name)
            if name == "llama-server":
                return {"present": True, "version": "llama.cpp b2000"}
            return {"present": False}

        monkeypatch.setattr(pb, "command_version", fake)
        out = pb.llamacpp_detect()
        assert out == {"present": True, "binary": "llama-server", "version": "llama.cpp b2000"}

    def test_second_candidate_present(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state

        def fake(name, args=None):
            if name == "llama-cpp-server":
                return {"present": True, "version": "1.0"}
            return {"present": False}

        monkeypatch.setattr(pb, "command_version", fake)
        out = pb.llamacpp_detect()
        assert out["binary"] == "llama-cpp-server"


# ---------------------------------------------------------------------------
# smoke_test_ollama_model — HTTP path (with eval_count/eval_duration timing)
# and CLI fallback path covering timeout + exception + mismatch branches.
# ---------------------------------------------------------------------------


class _FakeHttpResp:
    """Minimal fake urllib response for monkeypatching urllib.request.urlopen."""

    def __init__(self, body: dict):
        import json as _json

        self._data = _json.dumps(body).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _force_urlopen_fail(pb_module):
    """Make all urllib.request.urlopen calls fail with URLError (forces CLI fallback)."""
    import urllib.error
    import urllib.request

    def _fail(*a, **kw):
        raise urllib.error.URLError("connection refused")

    # Monkeypatch must target the same urllib.request module claude_codex_local.core imports.
    return urllib.request, "urlopen", _fail


class TestSmokeTestOllamaHTTP:
    """Exercises the primary HTTP path with real timing fields from Ollama."""

    def test_success_with_timing_fields(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        import urllib.request

        # eval_count=20 tokens in 1 second → 20 tok/s
        body = {
            "response": "READY",
            "eval_count": 20,
            "eval_duration": 1_000_000_000,  # 1s in nanoseconds
        }
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeHttpResp(body))
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is True
        assert result["response"] == "READY"
        assert result["completion_tokens"] == 20
        assert result["duration_seconds"] == 1.0
        assert result["tokens_per_second"] == 20.0

    def test_response_mismatch_on_http(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeHttpResp(
                {"response": "nope", "eval_count": 2, "eval_duration": 1_000_000_000}
            ),
        )
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert result["response"] == "nope"
        assert result["tokens_per_second"] == 2.0

    def test_missing_timing_fields_returns_none(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        import urllib.request

        # Body without eval_count/eval_duration — tokens_per_second should be None.
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeHttpResp({"response": "READY"}),
        )
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is True
        assert result["tokens_per_second"] is None
        assert result["completion_tokens"] is None


class TestSmokeTestOllama:
    """Legacy CLI-fallback path — reached when the HTTP daemon is unreachable."""

    def _fail_urlopen(self, monkeypatch):
        import urllib.error
        import urllib.request

        def _raise(*a, **kw):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)

    def test_success(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        self._fail_urlopen(monkeypatch)
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "READY\n", "")
        )
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is True
        # CLI fallback has no timing info.
        assert result["tokens_per_second"] is None

    def test_response_mismatch(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        self._fail_urlopen(monkeypatch)
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "nope\n", "")
        )
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "nope" in result["response"]

    def test_timeout(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        self._fail_urlopen(monkeypatch)

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=180)

        monkeypatch.setattr(pb, "run", boom)
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "timeout" in result["error"]

    def test_generic_exception(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        self._fail_urlopen(monkeypatch)
        monkeypatch.setattr(pb, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# machine_profile aggregation with every sub-call stubbed.
# ---------------------------------------------------------------------------


class TestMachineProfileAggregation:
    def test_aggregates_subcalls_into_full_dict(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "llmfit_system", lambda: {"system": {"ram_gb": 64}})
        monkeypatch.setattr(
            pb,
            "lms_info",
            lambda: {"present": True, "server_running": True, "server_port": 1234, "models": []},
        )
        monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
        monkeypatch.setattr(pb, "parse_ollama_list", lambda: [{"name": "x", "local": True}])
        monkeypatch.setattr(
            pb,
            "command_version",
            lambda name, args=None: {"present": True, "version": f"{name} v1"},
        )

        profile = pb.machine_profile()
        assert profile["tools"]["ollama"]["present"] is True
        assert profile["tools"]["lmstudio"]["present"] is True
        assert profile["presence"]["has_minimum"] is True
        assert "ollama" in profile["presence"]["engines"]
        assert "lmstudio" in profile["presence"]["engines"]
        assert set(profile["presence"]["harnesses"]) == {"claude", "codex"}
        assert profile["llmfit_system"] == {"ram_gb": 64}


# ---------------------------------------------------------------------------
# wizard.main CLI dispatcher — all three subcommands.
# ---------------------------------------------------------------------------


class TestWizardMain:
    def test_setup_delegates_to_run_wizard(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        called = {}

        def fake(**kw):
            called["kw"] = kw
            return 0

        monkeypatch.setattr(wiz, "run_wizard", fake)
        monkeypatch.setattr(
            sys, "argv", ["wizard", "setup", "--non-interactive", "--harness", "codex"]
        )
        assert wiz.main() == 0
        assert called["kw"]["non_interactive"] is True
        assert called["kw"]["force_harness"] == "codex"

    def test_no_subcommand_defaults_to_setup(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        called = {}

        def fake(**kw):
            called["hit"] = True
            return 0

        monkeypatch.setattr(wiz, "run_wizard", fake)
        monkeypatch.setattr(sys, "argv", ["wizard"])
        assert wiz.main() == 0
        assert called["hit"] is True

    def test_doctor_subcommand_delegates(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setattr(wiz, "run_doctor", lambda: 7)
        monkeypatch.setattr(sys, "argv", ["wizard", "doctor"])
        assert wiz.main() == 7

    def test_find_model_subcommand_delegates(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setattr(wiz, "run_find_model_standalone", lambda: 0)
        monkeypatch.setattr(sys, "argv", ["wizard", "find-model"])
        assert wiz.main() == 0


# ---------------------------------------------------------------------------
# wizard.run_find_model_standalone — llmfit-missing and success branches.
# ---------------------------------------------------------------------------


class TestFindModelStandalone:
    def test_fails_when_llmfit_missing(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda: {
                "presence": {"llmfit": False, "engines": ["ollama"]},
            },
        )
        # Make llmfit appear absent so _ensure_llmfit triggers the prompt.
        monkeypatch.setattr(pb, "command_version", lambda cmd, **kw: {"present": False})
        # User declines install offer → should still return 1.
        import questionary as _q

        monkeypatch.setattr(
            _q, "confirm", lambda *a, **kw: type("Q", (), {"ask": lambda self: False})()
        )
        assert wiz.run_find_model_standalone() == 1

    def test_calls_interactive_picker_and_returns_0_on_success(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda: {
                "presence": {"llmfit": True, "engines": ["ollama"]},
            },
        )
        monkeypatch.setattr(
            wiz,
            "_find_model_interactive",
            lambda engine: {"display": "Qwen3", "tag": "qwen3-coder:30b"},
        )
        assert wiz.run_find_model_standalone() == 0

    def test_returns_1_when_picker_cancelled(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda: {
                "presence": {"llmfit": True, "engines": ["ollama"]},
            },
        )
        monkeypatch.setattr(wiz, "_find_model_interactive", lambda engine: None)
        assert wiz.run_find_model_standalone() == 1


# ---------------------------------------------------------------------------
# core.smoke_test_codex — the one big side-effect-heavy path still uncovered.
# ---------------------------------------------------------------------------


class TestSmokeTestCodex:
    def test_ok_when_ready_in_output(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "READY", "")
        )
        assert pb.smoke_test_codex("qwen3-coder:30b", "ollama")["ok"] is True

    def test_flags_auth_noise(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, "READY", "failed to refresh available models"
            ),
        )
        result = pb.smoke_test_codex("qwen3-coder:30b", "ollama")
        assert result["ok"] is True
        assert result["auth_noise"] is True

    def test_timeout_branch(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], 240))
        )
        result = pb.smoke_test_codex("qwen3-coder:30b", "ollama")
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ---------------------------------------------------------------------------
# core.huggingface_cli_detect
# ---------------------------------------------------------------------------


class TestHuggingfaceCliDetect:
    def test_present(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb.shutil, "which", lambda name: "/usr/local/bin/huggingface-cli")
        result = pb.huggingface_cli_detect()
        assert result["present"] is True
        assert result["version"] == ""

    def test_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb.shutil, "which", lambda name: None)
        result = pb.huggingface_cli_detect()
        assert result["present"] is False


# ---------------------------------------------------------------------------
# core.huggingface_download_gguf
# ---------------------------------------------------------------------------


class TestHuggingfaceDownloadGguf:
    def test_returns_error_when_cli_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False})
        result = pb.huggingface_download_gguf("org/repo")
        assert result["ok"] is False
        assert "huggingface-cli not found" in result["error"]
        assert result["path"] is None

    def test_success_returns_path(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, "/home/user/.cache/huggingface/hub/model.gguf\n", ""
            ),
        )
        result = pb.huggingface_download_gguf("org/repo", filename="model.gguf")
        assert result["ok"] is True
        assert result["path"] == "/home/user/.cache/huggingface/hub/model.gguf"
        assert result["error"] is None

    def test_download_failure(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "Repository not found"),
        )
        result = pb.huggingface_download_gguf("nonexistent/repo")
        assert result["ok"] is False
        assert "Repository not found" in result["error"]

    def test_exception_is_caught(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network error"))
        )
        result = pb.huggingface_download_gguf("org/repo")
        assert result["ok"] is False
        assert "network error" in result["error"]


# ---------------------------------------------------------------------------
# wizard._download_gguf_via_hf_cli
# ---------------------------------------------------------------------------


class TestDownloadGgufViaHfCli:
    def test_warns_and_fails_when_cli_missing(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False})
        # User declines install offer → should still return {"ok": False}.
        import questionary as _q

        monkeypatch.setattr(
            _q, "confirm", lambda *a, **kw: type("Q", (), {"ask": lambda self: False})()
        )
        result = wiz._download_gguf_via_hf_cli("org/repo")
        assert result["ok"] is False

    def test_success_returns_path(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda repo, filename=None, local_dir=None: {
                "ok": True,
                "path": "/tmp/model.gguf",
                "error": None,
            },
        )
        result = wiz._download_gguf_via_hf_cli("org/repo")
        assert result["ok"] is True
        assert result["path"] == "/tmp/model.gguf"

    def test_splits_repo_and_filename(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        captured = {}
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda repo, filename=None, local_dir=None: (
                captured.update({"repo": repo, "filename": filename})
                or {"ok": True, "path": "/tmp/model.gguf", "error": None}
            ),
        )
        wiz._download_gguf_via_hf_cli("org/repo model-Q4_K_M.gguf")
        assert captured["repo"] == "org/repo"
        assert captured["filename"] == "model-Q4_K_M.gguf"

    def test_download_failure_propagates(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda *a, **kw: {"ok": False, "path": None, "error": "404"},
        )
        result = wiz._download_gguf_via_hf_cli("org/repo")
        assert result["ok"] is False
