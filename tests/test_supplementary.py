"""
Supplementary tests that push coverage into the remaining easy branches:
  * llamacpp_detect
  * smoke_test_ollama_model timeout / failure paths
  * machine_profile aggregation when everything is stubbed
  * wizard.run_find_model_standalone
  * wizard.main CLI dispatcher
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
