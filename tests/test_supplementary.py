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
from pathlib import Path

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
        assert profile["llmfit_system"] == {"system": {"ram_gb": 64}}


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
    def test_present_legacy_name(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state

        def which_legacy_only(name: str) -> str | None:
            return "/usr/local/bin/huggingface-cli" if name == "huggingface-cli" else None

        monkeypatch.setattr(pb.shutil, "which", which_legacy_only)
        result = pb.huggingface_cli_detect()
        assert result["present"] is True
        assert result["binary"] == "huggingface-cli"
        assert result["version"] == ""

    def test_present_modern_hf_name(self, isolated_state, monkeypatch):
        # huggingface_hub >=0.20 installs the CLI as `hf`, not `huggingface-cli`.
        pb, _, _ = isolated_state

        def which_hf_only(name: str) -> str | None:
            return "/usr/local/bin/hf" if name == "hf" else None

        monkeypatch.setattr(pb.shutil, "which", which_hf_only)
        result = pb.huggingface_cli_detect()
        assert result["present"] is True
        assert result["binary"] == "hf"
        assert result["version"] == ""

    def test_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb.shutil, "which", lambda name: None)
        result = pb.huggingface_cli_detect()
        assert result["present"] is False
        assert result["binary"] == ""


# ---------------------------------------------------------------------------
# core.huggingface_download_gguf
# ---------------------------------------------------------------------------


class TestHuggingfaceDownloadGguf:
    def test_returns_error_when_cli_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False})
        result = pb.huggingface_download_gguf("org/repo")
        assert result["ok"] is False
        assert "not found" in result["error"]
        assert result["path"] is None
        # Schema contract (#38/#39): every result exposes these keys.
        assert result["bytes_downloaded"] is None
        assert result["elapsed_seconds"] is None
        assert result["not_found"] is False

    def test_success_returns_path_non_streaming(self, isolated_state, monkeypatch):
        """stream=False preserves the original capture-based path extraction."""
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        monkeypatch.setattr(
            pb,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, "/home/user/.cache/huggingface/hub/model.gguf\n", ""
            ),
        )
        result = pb.huggingface_download_gguf("org/repo", filename="model.gguf", stream=False)
        assert result["ok"] is True
        assert result["path"] == "/home/user/.cache/huggingface/hub/model.gguf"
        assert result["error"] is None
        assert isinstance(result["elapsed_seconds"], float)

    def test_uses_detected_binary_name(self, isolated_state, monkeypatch):
        # The download command must use the binary name returned by detect,
        # not the hardcoded string "huggingface-cli".
        pb, _, _ = isolated_state
        captured: list[list[str]] = []
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        monkeypatch.setattr(
            pb,
            "run",
            lambda cmd, **kw: (
                captured.append(cmd) or subprocess.CompletedProcess(cmd, 0, "/tmp/model.gguf\n", "")
            ),
        )
        pb.huggingface_download_gguf("org/repo", stream=False)
        assert captured[0][0] == "hf"

    def test_download_failure_non_streaming_flags_not_found(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "huggingface-cli", "version": ""},
        )
        monkeypatch.setattr(
            pb,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(a[0], 1, "", "Repository Not Found"),
        )
        result = pb.huggingface_download_gguf("nonexistent/repo", stream=False)
        assert result["ok"] is False
        assert "Repository Not Found" in result["error"]
        assert result["not_found"] is True

    def test_exception_is_caught(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network error"))
        )
        result = pb.huggingface_download_gguf("org/repo", stream=False)
        assert result["ok"] is False
        assert "network error" in result["error"]

    def test_streaming_success_uses_popen(self, isolated_state, monkeypatch, tmp_path):
        """stream=True delegates to subprocess.Popen so progress is visible."""
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )

        class _FakeProc:
            def __init__(self, cmd, env=None):
                self.cmd = cmd
                self.env = env

            def wait(self, timeout=None):
                # Simulate the CLI writing a file into local_dir.
                dest = Path(local_dir) / "m.gguf"
                dest.write_bytes(b"hello-gguf")
                return 0

        local_dir = str(tmp_path / "hf-dl")
        Path(local_dir).mkdir(parents=True)
        calls: list[list[str]] = []

        def fake_popen(cmd, env=None):
            calls.append(cmd)
            return _FakeProc(cmd, env=env)

        monkeypatch.setattr(pb.subprocess, "Popen", fake_popen)
        result = pb.huggingface_download_gguf(
            "org/repo", filename="m.gguf", local_dir=local_dir, stream=True
        )
        assert result["ok"] is True
        assert result["path"] == str(Path(local_dir) / "m.gguf")
        assert result["bytes_downloaded"] == len(b"hello-gguf")
        assert result["not_found"] is False
        assert isinstance(result["elapsed_seconds"], float)
        # sanity: Popen was invoked with the detected binary
        assert calls and calls[0][0] == "hf"

    def test_streaming_failure_returns_rc_message(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )

        class _FailProc:
            def wait(self, timeout=None):
                return 42

        monkeypatch.setattr(pb.subprocess, "Popen", lambda cmd, env=None: _FailProc())
        result = pb.huggingface_download_gguf("org/repo", stream=True)
        assert result["ok"] is False
        assert "42" in result["error"]
        # Streaming path can't scrape stderr, so we don't claim 404.
        assert result["not_found"] is False

    def test_looks_like_not_found_predicate(self, isolated_state):
        pb, _, _ = isolated_state
        assert pb._looks_like_not_found("HTTPError: 404 Client Error")
        assert pb._looks_like_not_found("RepositoryNotFoundError: ...")
        assert pb._looks_like_not_found("Repository not found")
        assert not pb._looks_like_not_found("Permission denied")
        assert not pb._looks_like_not_found("")


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
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda repo, filename=None, local_dir=None, stream=True: {
                "ok": True,
                "path": "/tmp/model.gguf",
                "error": None,
                "bytes_downloaded": 1234,
                "elapsed_seconds": 0.5,
                "not_found": False,
            },
        )
        result = wiz._download_gguf_via_hf_cli("org/repo")
        assert result["ok"] is True
        assert result["path"] == "/tmp/model.gguf"
        # The wrapper must echo the successful repo_id back so _download_model
        # can detect whether a fuzzy-search pick changed the selection (#38).
        assert result["repo_id"] == "org/repo"

    def test_splits_repo_and_filename(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        captured = {}
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda repo, filename=None, local_dir=None, stream=True: (
                captured.update({"repo": repo, "filename": filename})
                or {
                    "ok": True,
                    "path": "/tmp/model.gguf",
                    "error": None,
                    "bytes_downloaded": 100,
                    "elapsed_seconds": 0.1,
                    "not_found": False,
                }
            ),
        )
        wiz._download_gguf_via_hf_cli("org/repo model-Q4_K_M.gguf")
        assert captured["repo"] == "org/repo"
        assert captured["filename"] == "model-Q4_K_M.gguf"

    def test_download_failure_propagates(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
        )
        # A failure that doesn't look like not-found should propagate as-is
        # without invoking the fuzzy-search loop.
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda *a, **kw: {
                "ok": False,
                "path": None,
                "error": "Permission denied",
                "not_found": False,
                "bytes_downloaded": None,
                "elapsed_seconds": 0.1,
            },
        )
        # Ensure the fuzzy-search probe is never exercised here.
        monkeypatch.setattr(pb, "huggingface_search_models", lambda *a, **kw: ["org/repo"])
        result = wiz._download_gguf_via_hf_cli("org/repo")
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Hugging Face fuzzy search (#38) — huggingface_search_models and
# huggingface_fuzzy_find plus the wizard's fuzzy re-prompt loop.
# ---------------------------------------------------------------------------


class _StubAsk:
    def __init__(self, value):
        self._value = value

    def ask(self):
        return self._value


class TestHuggingfaceSearchModels:
    def test_returns_ids_on_success(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        import json as _json

        payload = _json.dumps(
            [{"id": "org/alpha-gguf"}, {"modelId": "org/beta-gguf"}, {"no_id": True}]
        ).encode("utf-8")

        class _Resp:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(pb, "huggingface_search_models", pb.huggingface_search_models)  # sanity
        # Patch urllib.request.urlopen inside core.
        import urllib.request as _ur

        monkeypatch.setattr(_ur, "urlopen", lambda req, timeout=10.0: _Resp())
        out = pb.huggingface_search_models("qwen2.5-coder")
        assert "org/alpha-gguf" in out
        assert "org/beta-gguf" in out

    def test_returns_empty_on_blank_query(self, isolated_state):
        pb, _, _ = isolated_state
        assert pb.huggingface_search_models("") == []
        assert pb.huggingface_search_models("   ") == []

    def test_returns_empty_on_network_error(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        import urllib.error as _ue
        import urllib.request as _ur

        def boom(req, timeout=10.0):
            raise _ue.URLError("DNS failure")

        monkeypatch.setattr(_ur, "urlopen", boom)
        assert pb.huggingface_search_models("qwen") == []

    def test_handles_unexpected_payload(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state

        class _Resp:
            def read(self):
                return b"not-json"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import urllib.request as _ur

        monkeypatch.setattr(_ur, "urlopen", lambda req, timeout=10.0: _Resp())
        assert pb.huggingface_search_models("qwen") == []


class TestHuggingfaceFuzzyFind:
    def test_clamps_to_three_matches(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_search_models",
            lambda q, limit=10: [
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
                "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
                "some/other-unrelated-model",
            ],
        )
        out = pb.huggingface_fuzzy_find("qwen2.5-coder-7b-instruct-gguf", max_results=3)
        assert len(out) == 3
        assert all("Qwen2.5-Coder" in m for m in out)

    def test_returns_empty_when_no_candidates(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_search_models", lambda q, limit=10: [])
        assert pb.huggingface_fuzzy_find("does-not-exist") == []

    def test_falls_back_to_api_order_when_difflib_returns_nothing(
        self, isolated_state, monkeypatch
    ):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_search_models",
            lambda q, limit=10: ["totally/unrelated-name-a", "another/unrelated-b"],
        )
        out = pb.huggingface_fuzzy_find("qwen-cod", max_results=3)
        # Query has ~nothing in common with candidates → difflib yields [],
        # but we still surface *some* suggestion.
        assert len(out) == 2
        assert out[0] == "totally/unrelated-name-a"


class TestFuzzySearchReprompt:
    """Tests for _prompt_fuzzy_hf_match (#38) — numbered picker + re-entry."""

    def test_picker_returns_selected_match(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_fuzzy_find",
            lambda q, max_results=3: [
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
                "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            ],
        )
        monkeypatch.setattr(
            wiz.questionary,
            "select",
            lambda *a, **kw: _StubAsk("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF"),
        )
        out = wiz._prompt_fuzzy_hf_match("qwen2.5-coder")
        assert out == "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF"

    def test_picker_reprompt_lets_user_type_new_name(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_fuzzy_find", lambda q, max_results=3: ["some/match"])
        monkeypatch.setattr(wiz.questionary, "select", lambda *a, **kw: _StubAsk("__reenter__"))
        monkeypatch.setattr(wiz.questionary, "text", lambda *a, **kw: _StubAsk("user/custom-typed"))
        out = wiz._prompt_fuzzy_hf_match("qwen2.5-coder")
        assert out == "user/custom-typed"

    def test_picker_cancel_returns_none(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_fuzzy_find", lambda q, max_results=3: ["some/match"])
        monkeypatch.setattr(wiz.questionary, "select", lambda *a, **kw: _StubAsk("__cancel__"))
        out = wiz._prompt_fuzzy_hf_match("qwen2.5-coder")
        assert out is None

    def test_zero_matches_reprompts_for_name(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        # No fuzzy hits — the picker should be skipped and we drop directly
        # into the text re-entry prompt.
        monkeypatch.setattr(pb, "huggingface_fuzzy_find", lambda q, max_results=3: [])
        monkeypatch.setattr(wiz.questionary, "text", lambda *a, **kw: _StubAsk("user/backup-typed"))
        out = wiz._prompt_fuzzy_hf_match("gibberish")
        assert out == "user/backup-typed"

    def test_zero_matches_blank_reentry_returns_none(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "huggingface_fuzzy_find", lambda q, max_results=3: [])
        monkeypatch.setattr(wiz.questionary, "text", lambda *a, **kw: _StubAsk(""))
        assert wiz._prompt_fuzzy_hf_match("gibberish") is None


class TestFuzzySearchDownloadFlow:
    """End-to-end wizard flow: failure → fuzzy search → retry with picked repo."""

    def test_not_found_triggers_fuzzy_and_retries_with_new_repo(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )
        # Call 1: fail with 404. Call 2: success with the fuzzy-picked repo.
        calls: list[str] = []

        def fake_download(repo, filename=None, local_dir=None, stream=True):
            calls.append(repo)
            if len(calls) == 1:
                return {
                    "ok": False,
                    "path": None,
                    "error": "404 Client Error: Repository Not Found",
                    "not_found": True,
                    "bytes_downloaded": None,
                    "elapsed_seconds": 0.3,
                }
            return {
                "ok": True,
                "path": "/tmp/picked/model.gguf",
                "error": None,
                "bytes_downloaded": 1_000_000,
                "elapsed_seconds": 5.0,
                "not_found": False,
            }

        monkeypatch.setattr(pb, "huggingface_download_gguf", fake_download)
        monkeypatch.setattr(
            pb,
            "huggingface_fuzzy_find",
            lambda q, max_results=3: ["bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"],
        )
        monkeypatch.setattr(
            wiz.questionary,
            "select",
            lambda *a, **kw: _StubAsk("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"),
        )
        out = wiz._download_gguf_via_hf_cli("user/typo-here")
        assert out["ok"] is True
        # Second attempt used the fuzzy-picked repo, not the original typo.
        assert calls == ["user/typo-here", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"]
        assert out["repo_id"] == "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        assert out["path"] == "/tmp/picked/model.gguf"

    def test_streamed_failure_probed_via_search_api(self, isolated_state, monkeypatch):
        """When the CLI error is just 'exited with status N', we consult the HF
        search API to decide whether the repo is missing (#38 trigger)."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )
        attempt = {"n": 0}

        def fake_download(repo, filename=None, local_dir=None, stream=True):
            attempt["n"] += 1
            if attempt["n"] == 1:
                return {
                    "ok": False,
                    "path": None,
                    "error": "huggingface-cli exited with status 1",
                    "not_found": False,
                    "bytes_downloaded": None,
                    "elapsed_seconds": 0.3,
                }
            return {
                "ok": True,
                "path": "/tmp/good/model.gguf",
                "error": None,
                "bytes_downloaded": 42,
                "elapsed_seconds": 0.1,
                "not_found": False,
            }

        monkeypatch.setattr(pb, "huggingface_download_gguf", fake_download)
        # HF search returns NO exact match for "user/typo" → triggers fuzzy
        # search fallback even though the direct error was generic.
        monkeypatch.setattr(
            pb, "huggingface_search_models", lambda q, limit=10, **kw: ["other/unrelated"]
        )
        monkeypatch.setattr(
            pb,
            "huggingface_fuzzy_find",
            lambda q, max_results=3: ["bartowski/real-repo"],
        )
        monkeypatch.setattr(
            wiz.questionary, "select", lambda *a, **kw: _StubAsk("bartowski/real-repo")
        )
        out = wiz._download_gguf_via_hf_cli("user/typo")
        assert out["ok"] is True
        assert out["repo_id"] == "bartowski/real-repo"

    def test_user_cancels_fuzzy_picker_returns_failure(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda *a, **kw: {
                "ok": False,
                "path": None,
                "error": "Repository not found",
                "not_found": True,
                "bytes_downloaded": None,
                "elapsed_seconds": 0.3,
            },
        )
        monkeypatch.setattr(pb, "huggingface_fuzzy_find", lambda q, max_results=3: ["some/match"])
        monkeypatch.setattr(wiz.questionary, "select", lambda *a, **kw: _StubAsk("__cancel__"))
        out = wiz._download_gguf_via_hf_cli("user/typo")
        assert out["ok"] is False
        assert out["repo_id"] == "user/typo"


# ---------------------------------------------------------------------------
# Download progress & summary formatting (#39) — human formatters and the
# _download_model summary-line behaviour for ollama / lmstudio / llamacpp.
# ---------------------------------------------------------------------------


class TestHumanFormatters:
    def test_human_bytes_tiers(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._human_bytes(0) == "0 B"
        assert wiz._human_bytes(512) == "512 B"
        assert wiz._human_bytes(2048).startswith("2.0 KiB")
        assert wiz._human_bytes(5 * 1024 * 1024).endswith("MiB")
        assert wiz._human_bytes(10 * 1024**3).endswith("GiB")

    def test_human_duration_scales(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._human_duration(3.2) == "3.2s"
        assert "m" in wiz._human_duration(95)
        assert "h" in wiz._human_duration(3700)


class TestDownloadModelSummary:
    """_download_model should always print a time-bounded summary (#39)."""

    def test_ollama_success_prints_summary(self, isolated_state, monkeypatch, capsys):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: {"ollama": {"models": []}})
        # subprocess.run → no-op (simulates a silent-but-fast pull).
        calls: list[list[str]] = []
        monkeypatch.setattr(
            wiz.subprocess,
            "run",
            lambda cmd, check=True: (
                calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
            ),
        )
        # Stub hint lookup so the summary line contains "Downloaded ... in ...".
        monkeypatch.setattr(wiz, "_ollama_model_size_hint", lambda tag: "3.4 GB")
        state = wiz.WizardState(
            primary_engine="ollama", engine_model_tag="qwen3-coder:7b", model_name="qwen3-coder:7b"
        )
        state.profile = {"ollama": {"models": []}}
        assert wiz._download_model(state) is True
        out = capsys.readouterr().out
        assert "Downloaded qwen3-coder:7b" in out
        assert "3.4 GB" in out
        assert "in " in out  # elapsed time appears
        # sanity: we actually invoked the pull.
        assert calls and calls[0] == ["ollama", "pull", "qwen3-coder:7b"]

    def test_lmstudio_success_prints_summary(self, isolated_state, monkeypatch, capsys):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: {"lmstudio": {"models": []}})
        monkeypatch.setattr(pb, "lms_binary", lambda: "/fake/lms")
        monkeypatch.setattr(
            wiz.subprocess,
            "run",
            lambda cmd, check=True: subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        monkeypatch.setattr(wiz, "_lms_model_size_hint", lambda tag: "512.0 MiB")
        state = wiz.WizardState(
            primary_engine="lmstudio",
            engine_model_tag="qwen/qwen3-coder-7b",
            model_name="qwen/qwen3-coder-7b",
        )
        assert wiz._download_model(state) is True
        out = capsys.readouterr().out
        assert "Downloaded qwen/qwen3-coder-7b" in out
        assert "512.0 MiB" in out
        assert "in " in out

    def test_llamacpp_fuzzy_pick_updates_state_tag(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda: {"llamacpp": {"present": True, "server_running": False}},
        )
        monkeypatch.setattr(
            wiz,
            "_download_gguf_via_hf_cli",
            lambda repo: {
                "ok": True,
                "path": "/tmp/picked.gguf",
                "repo_id": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                "bytes_downloaded": 1234,
                "elapsed_seconds": 2.5,
            },
        )
        state = wiz.WizardState(
            primary_engine="llamacpp",
            engine_model_tag="user/typo-here",
            model_name="user/typo-here",
        )
        assert wiz._download_model(state) is True
        # The fuzzy-search picker changed the repo ID → state must be updated
        # so step 2.6 wires the correct model (#38 + #39 together).
        assert state.engine_model_tag == "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        assert state.model_name == "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        assert state.profile.get("llamacpp_model_path") == "/tmp/picked.gguf"

    def test_download_failure_returns_false(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state

        def boom(cmd, check=True):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(wiz.subprocess, "run", boom)
        state = wiz.WizardState(
            primary_engine="ollama", engine_model_tag="qwen3-coder:7b", model_name="qwen3-coder:7b"
        )
        assert wiz._download_model(state) is False

    def test_download_keyboard_interrupt_returns_false(self, isolated_state, monkeypatch):
        """Ctrl-C during a pull must stop the indicator cleanly (#39 AC)."""
        _, wiz, _ = isolated_state

        def raise_ki(cmd, check=True):
            raise KeyboardInterrupt()

        monkeypatch.setattr(wiz.subprocess, "run", raise_ki)
        state = wiz.WizardState(
            primary_engine="ollama", engine_model_tag="qwen3-coder:7b", model_name="qwen3-coder:7b"
        )
        assert wiz._download_model(state) is False

    def test_download_keyboard_interrupt_llamacpp_returns_false(self, isolated_state, monkeypatch):
        """Ctrl-C during the llamacpp / HF CLI download path must also be
        handled by the wizard and return False cleanly (PR #44 review note)."""
        pb, wiz, _ = isolated_state

        def raise_ki(tag):
            raise KeyboardInterrupt()

        monkeypatch.setattr(wiz, "_download_gguf_via_hf_cli", raise_ki)
        state = wiz.WizardState(
            primary_engine="llamacpp",
            engine_model_tag="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            model_name="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        )
        assert wiz._download_model(state) is False


# ---------------------------------------------------------------------------
# PR #44 review notes — regression tests for the three fixes.
# ---------------------------------------------------------------------------


class TestHuggingfaceDownloadGgufKI:
    """huggingface_download_gguf must terminate the streamed Popen child on
    KeyboardInterrupt so we don't leak an orphan HF CLI process."""

    def test_streamed_ki_terminates_child_and_reraises(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )

        events: list[str] = []

        class _FakePopen:
            def __init__(self, cmd, env=None):
                events.append("spawn")

            def wait(self, timeout=None):
                # First wait() — from the download logic — raises KI, as if
                # the user pressed Ctrl-C while the child was running.
                # Second wait() — from our cleanup path — completes quickly.
                if "terminate" not in events:
                    raise KeyboardInterrupt()
                events.append("wait-after-terminate")
                return 0

            def terminate(self):
                events.append("terminate")

            def kill(self):
                events.append("kill")

        monkeypatch.setattr(pb.subprocess, "Popen", _FakePopen)

        import pytest

        with pytest.raises(KeyboardInterrupt):
            pb.huggingface_download_gguf("some/repo", stream=True)

        # The critical invariant: terminate() was called before re-raising,
        # and we waited (bounded) for the child before giving up.
        assert "spawn" in events
        assert "terminate" in events
        # Terminate happened before we let the KI propagate — so the child
        # had a chance to exit. If terminate didn't hang, kill() wasn't used.
        assert events.index("terminate") > events.index("spawn")

    def test_streamed_ki_force_kills_when_terminate_hangs(self, isolated_state, monkeypatch):
        """If the child ignores SIGTERM within 3s, escalate to SIGKILL."""
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )

        events: list[str] = []

        class _StubbornPopen:
            def __init__(self, cmd, env=None):
                events.append("spawn")

            def wait(self, timeout=None):
                events.append(f"wait({timeout})")
                # Initial wait → KI. Subsequent waits after terminate →
                # always hang (TimeoutExpired) to force the kill escalation.
                if len([e for e in events if e.startswith("wait")]) == 1:
                    raise KeyboardInterrupt()
                raise pb.subprocess.TimeoutExpired(cmd="hf", timeout=timeout)

            def terminate(self):
                events.append("terminate")

            def kill(self):
                events.append("kill")

        monkeypatch.setattr(pb.subprocess, "Popen", _StubbornPopen)

        import pytest

        with pytest.raises(KeyboardInterrupt):
            pb.huggingface_download_gguf("some/repo", stream=True)

        assert "terminate" in events
        assert "kill" in events
        # Order: terminate, then wait(timeout=3), then kill.
        assert events.index("terminate") < events.index("kill")


class TestLooksLikeMissingRepoSearchApiError:
    """PR #44 review note: when the HF search API itself fails, _looks_like_missing_repo
    must NOT return True — otherwise a network outage masquerades as a missing
    repo and triggers a fuzzy fallback that finds nothing."""

    def test_search_api_network_error_does_not_trigger_fuzzy(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state

        def boom(query, limit=10, raise_on_error=False):
            if raise_on_error:
                raise OSError("network unreachable")
            return []

        monkeypatch.setattr(pb, "huggingface_search_models", boom)
        # A generic streamed-failure error — without search-API signal, we
        # can't know whether the repo is missing, so we must NOT claim it is.
        assert (
            wiz._looks_like_missing_repo("user/real-repo", "huggingface-cli exited with status 1")
            is False
        )

    def test_search_api_timeout_does_not_trigger_fuzzy(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state

        def boom(query, limit=10, raise_on_error=False):
            if raise_on_error:
                raise TimeoutError("HF API timed out")
            return []

        monkeypatch.setattr(pb, "huggingface_search_models", boom)
        assert (
            wiz._looks_like_missing_repo("user/real-repo", "huggingface-cli exited with status 1")
            is False
        )

    def test_search_api_success_with_no_hits_still_triggers(self, isolated_state, monkeypatch):
        """Positive control: when the API responds OK but reports no matches,
        the fuzzy path should still fire — that's the original #38 behaviour."""
        pb, wiz, _ = isolated_state

        def ok(query, limit=10, raise_on_error=False):
            return ["other/unrelated"]

        monkeypatch.setattr(pb, "huggingface_search_models", ok)
        assert (
            wiz._looks_like_missing_repo("user/typo-here", "huggingface-cli exited with status 1")
            is True
        )

    def test_download_flow_surfaces_error_when_search_api_down(self, isolated_state, monkeypatch):
        """End-to-end: streamed download fails, search API is down → we
        surface the original download error rather than launching a fuzzy
        picker that would find nothing."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "huggingface_cli_detect",
            lambda: {"present": True, "binary": "hf", "version": ""},
        )
        monkeypatch.setattr(
            pb,
            "huggingface_download_gguf",
            lambda *a, **kw: {
                "ok": False,
                "path": None,
                "error": "huggingface-cli exited with status 1",
                "not_found": False,
                "bytes_downloaded": None,
                "elapsed_seconds": 0.3,
            },
        )

        def search_boom(query, limit=10, raise_on_error=False):
            if raise_on_error:
                raise OSError("offline")
            return []

        monkeypatch.setattr(pb, "huggingface_search_models", search_boom)

        # If the fuzzy picker is reached, this will explode — which is the
        # bug we're guarding against.
        def _should_not_be_called(*a, **kw):
            raise AssertionError("fuzzy picker must not be invoked when the HF search API is down")

        monkeypatch.setattr(wiz, "_prompt_fuzzy_hf_match", _should_not_be_called)

        out = wiz._download_gguf_via_hf_cli("user/real-repo")
        assert out["ok"] is False
        assert "exited with status 1" in (out["error"] or "")
