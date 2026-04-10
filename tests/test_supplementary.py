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
# poc_bridge.llamacpp_detect — all-missing and one-present branches.
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
# smoke_test_ollama_model — timeout + exception + success + mismatch branches.
# ---------------------------------------------------------------------------


class TestSmokeTestOllama:
    def test_success(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "READY\n", "")
        )
        assert pb.smoke_test_ollama_model("qwen3-coder:30b")["ok"] is True

    def test_response_mismatch(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "nope\n", "")
        )
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "nope" in result["response"]

    def test_timeout(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state

        def boom(*a, **kw):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=180)

        monkeypatch.setattr(pb, "run", boom)
        result = pb.smoke_test_ollama_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "timeout" in result["error"]

    def test_generic_exception(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
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
                "presence": {"llmfit": False, "engines": []},
            },
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
# poc_bridge.smoke_test_codex — the one big side-effect-heavy path still uncovered.
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
# poc_bridge.huggingface_cli_detect
# ---------------------------------------------------------------------------


class TestHuggingfaceCliDetect:
    def test_present(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "command_version",
            lambda name, args=None: {"present": True, "version": "0.21.0"}
            if name == "huggingface-cli"
            else {"present": False},
        )
        result = pb.huggingface_cli_detect()
        assert result["present"] is True
        assert "0.21.0" in result["version"]

    def test_missing(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        result = pb.huggingface_cli_detect()
        assert result["present"] is False


# ---------------------------------------------------------------------------
# poc_bridge.huggingface_download_gguf
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
        # _show_install_hint just prints, no need to stub it
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
            lambda repo, filename=None, local_dir=None: captured.update(
                {"repo": repo, "filename": filename}
            )
            or {"ok": True, "path": "/tmp/model.gguf", "error": None},
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
