"""
End-to-end tests for claude-codex-local — stubbed, CI-safe.

These tests verify the full wiring of the POC:
  * poc_bridge CLI subcommands invoked via main()
  * wizard.run_wizard() executing all 8 steps in non-interactive mode
  * wizard.run_doctor() re-checking presence after a successful setup
  * bin/ shims spawned as real subprocesses with a fake PATH

Everything that would normally shell out to ollama/lms/claude/codex/llmfit
is either patched at the module level or hit through the `fake_bin` fixture.
Zero network, zero real LLM calls, under 3s total.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers — install a synthetic profile + candidate list into poc_bridge.
# ---------------------------------------------------------------------------


def _installed_profile(pb_mod, harness="claude", engine="ollama"):
    """Build a machine_profile() payload for an 'everything installed' world."""
    return {
        "host": {"platform": "Darwin-x", "system": "Darwin", "release": "25", "machine": "arm64"},
        "tools": {
            "ollama": {"present": True, "version": "0.1.99"},
            "lmstudio": {"present": engine == "lmstudio", "version": "0.2.0"},
            "llamacpp": {"present": False, "version": ""},
            "claude": {"present": harness == "claude", "version": "claude 1.0.0"},
            "codex": {"present": harness == "codex", "version": "codex 0.1.0"},
            "llmfit": {"present": True, "version": "llmfit 1.2.3"},
        },
        "presence": {
            "harnesses": [harness],
            "engines": [engine],
            "llmfit": True,
            "has_minimum": True,
        },
        "ollama": {
            "models": [
                {
                    "name": "qwen3-coder:30b",
                    "id": "abc",
                    "size": "19 GB",
                    "modified": "x",
                    "local": True,
                }
            ]
        }
        if engine == "ollama"
        else {"models": []},
        "lmstudio": {
            "present": engine == "lmstudio",
            "server_running": engine == "lmstudio",
            "server_port": 1234,
            "models": [{"path": "qwen/qwen3-coder-30b", "format": "mlx"}]
            if engine == "lmstudio"
            else [],
        },
        "llamacpp": {"present": False, "version": ""},
        "disk": {
            "path": str(pb_mod.STATE_DIR),
            "total_bytes": 1 << 40,
            "used_bytes": 0,
            "free_bytes": 1 << 40,
            "free_gib": 1024.0,
            "total_gib": 1024.0,
        },
        "state_dir": str(pb_mod.STATE_DIR),
    }


def _stub_candidates():
    return [
        {
            "name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "category": "Coding",
            "score": 95,
            "fit_level": "Perfect",
            "estimated_tps": 40,
            "memory_required_gb": 18,
            "best_quant": "mlx-4bit",
            "ollama_tag": "qwen3-coder:30b",
            "lms_mlx_path": "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
            "lms_hub_name": "qwen/qwen3-coder-30b",
        }
    ]


# ---------------------------------------------------------------------------
# poc_bridge CLI subcommands — invoke main() with argv injection.
# ---------------------------------------------------------------------------


class TestPocBridgeCli:
    def test_profile_prints_json(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(sys, "argv", ["poc_bridge", "profile"])
        pb.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["presence"]["has_minimum"] is True
        assert data["tools"]["ollama"]["present"] is True

    def test_recommend_prints_selected_model(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(pb, "llmfit_coding_candidates", _stub_candidates)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(sys, "argv", ["poc_bridge", "recommend", "--mode", "balanced"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert data["selected_model"] == "qwen3-coder:30b"
        assert data["runtime"] == "ollama"
        assert data["status"] == "ready"

    def test_doctor_prints_issues_and_fixes(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(pb, "llmfit_coding_candidates", _stub_candidates)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        # ensure_config would try to shell out; stub both sides.
        monkeypatch.setattr(
            pb,
            "configure_ollama_integration",
            lambda target, model: {"target": target, "model": model},
        )
        monkeypatch.setattr(sys, "argv", ["poc_bridge", "doctor"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert "profile" in data and "recommendation" in data
        assert data["recommendation"]["selected_model"] == "qwen3-coder:30b"
        assert data["config_written"]["codex"]["model"] == "qwen3-coder:30b"

    def test_doctor_flags_missing_tools(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        bad = _installed_profile(pb)
        bad["tools"]["ollama"] = {"present": False, "error": "not found"}
        bad["ollama"]["models"] = []
        bad["presence"]["engines"] = []
        monkeypatch.setattr(pb, "machine_profile", lambda: bad)
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda: [])
        monkeypatch.setattr(
            pb,
            "configure_ollama_integration",
            lambda target, model: {"target": target, "model": model},
        )
        monkeypatch.setattr(sys, "argv", ["poc_bridge", "doctor"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert any("Missing tool: ollama" in i for i in data["issues"])
        assert any("No suitable local coding model" in i for i in data["issues"])

    def test_ensure_config_codex_ollama_writes_state(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(pb, "llmfit_coding_candidates", _stub_candidates)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(
            pb,
            "configure_ollama_integration",
            lambda target, model: {
                "target": target,
                "model": model,
                "state_dir": str(pb.STATE_DIR),
            },
        )
        monkeypatch.setattr(
            sys, "argv", ["poc_bridge", "ensure-config", "codex", "--model", "qwen3-coder:30b"]
        )
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert data["target"] == "codex"
        assert data["model"] == "qwen3-coder:30b"

    def test_adapters_subcommand_lists_both(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        # Keep healthchecks cheap and deterministic.
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "x"}
        )
        monkeypatch.setattr(pb, "parse_ollama_list", lambda: [])
        monkeypatch.setattr(
            pb,
            "lms_info",
            lambda: {"present": True, "server_running": True, "server_port": 1234, "models": []},
        )
        monkeypatch.setattr(sys, "argv", ["poc_bridge", "adapters"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        names = {a["name"] for a in data["adapters"]}
        assert names == {"ollama", "lmstudio"}


# ---------------------------------------------------------------------------
# Full wizard run in --non-interactive mode, everything stubbed.
# ---------------------------------------------------------------------------


def _stub_subprocess_success(*args, **kwargs):
    return subprocess.CompletedProcess(
        args=args[0] if args else [], returncode=0, stdout="READY\n", stderr=""
    )


class TestWizardFullFlow:
    def test_non_interactive_run_completes_all_steps(self, isolated_state, monkeypatch):
        pb, wiz, state_dir = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(
            pb,
            "ollama_ensure_nothink_variant",
            lambda tag: (tag, {"patched": False, "reason": "stubbed"}),
        )
        # step 2.7 calls subprocess.run on the verify command directly.
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        rc = wiz.run_wizard(non_interactive=True)
        assert rc == 0

        state = wiz.WizardState.load()
        assert set(state.completed_steps) >= {"2.1", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"}
        assert state.primary_harness == "claude"
        assert state.primary_engine == "ollama"
        assert state.engine_model_tag == "qwen3-coder:30b"
        assert state.verify_result["ok"] is True

        # Verifies step 2.6 wrote the isolated Claude settings.json.
        settings = pb.STATE_HOME / ".claude" / "settings.json"
        assert settings.exists()
        cfg = json.loads(settings.read_text())
        assert cfg["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:11434"

        # Verifies step 2.8 wrote a guide.md.
        assert wiz.GUIDE_PATH.exists()
        body = wiz.GUIDE_PATH.read_text()
        assert "qwen3-coder:30b" in body
        assert "claude" in body

    def test_non_interactive_fails_cleanly_on_smoke_test_failure(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": False, "error": "simulated failure"}
        )
        rc = wiz.run_wizard(non_interactive=True)
        assert rc == 1
        # Completed up through 2.4; 2.5 should NOT be in completed_steps.
        state = wiz.WizardState.load()
        assert "2.5" not in state.completed_steps
        assert "2.4" in state.completed_steps

    def test_resume_skips_completed_steps(self, isolated_state, monkeypatch):
        pb, wiz, state_dir = isolated_state
        # First run: succeed through 2.5, fail at 2.6 via a forced error.
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )

        call_count = {"nothink": 0}

        def flaky_variant(tag):
            call_count["nothink"] += 1
            if call_count["nothink"] == 1:
                raise RuntimeError("simulated wire failure")
            return tag, {"patched": False, "reason": "stubbed"}

        monkeypatch.setattr(pb, "ollama_ensure_nothink_variant", flaky_variant)
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        with pytest.raises(RuntimeError):
            wiz.run_wizard(non_interactive=True)

        state_before = wiz.WizardState.load()
        assert "2.5" in state_before.completed_steps
        assert "2.6" not in state_before.completed_steps

        # Second run with --resume: should skip 2.1-2.5 and succeed at 2.6.
        rc = wiz.run_wizard(resume=True, non_interactive=True)
        assert rc == 0
        state_after = wiz.WizardState.load()
        assert "2.6" in state_after.completed_steps
        assert "2.8" in state_after.completed_steps

    def test_doctor_reports_clean_state_after_setup(self, isolated_state, monkeypatch, capsys):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(
            pb,
            "ollama_ensure_nothink_variant",
            lambda tag: (tag, {"patched": False, "reason": "stubbed"}),
        )
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        wiz.run_wizard(non_interactive=True)
        rc = wiz.run_doctor()
        assert rc == 0

    def test_doctor_detects_missing_settings_file(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(
            pb,
            "ollama_ensure_nothink_variant",
            lambda tag: (tag, {"patched": False, "reason": "stubbed"}),
        )
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)
        wiz.run_wizard(non_interactive=True)

        # Nuke the settings file and confirm doctor notices.
        (pb.STATE_HOME / ".claude" / "settings.json").unlink()
        rc = wiz.run_doctor()
        assert rc == 1

    def test_doctor_no_state_file_returns_1(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz.run_doctor() == 1


# ---------------------------------------------------------------------------
# bin/ shims spawned as real subprocesses with a fake PATH.
# ---------------------------------------------------------------------------


class TestBinShims:
    def _spawn(self, shim_name, extra_env=None, tmp_path=None, fake_bin=None, extra_args=None):
        """Invoke a bin/ shim with isolated STATE_DIR + a fake PATH."""
        bdir, _ = fake_bin
        env = os.environ.copy()
        env["PATH"] = f"{bdir}:/usr/bin:/bin"
        env["CLAUDE_CODEX_LOCAL_STATE_DIR"] = str(tmp_path / "state")
        env["HOME"] = str(tmp_path / "home")
        (tmp_path / "home").mkdir(exist_ok=True)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(REPO_ROOT / "bin" / shim_name), *(extra_args or [])],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    def test_poc_machine_profile_shim_emits_json(self, fake_bin, tmp_path):
        result = self._spawn("poc-machine-profile", tmp_path=tmp_path, fake_bin=fake_bin)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "tools" in data
        assert "presence" in data

    def test_poc_recommend_shim_returns_fallback_when_no_candidates(self, fake_bin, tmp_path):
        # Default llmfit stub returns {"models": []}, so we should hit pass 5 fallback.
        result = self._spawn("poc-recommend", tmp_path=tmp_path, fake_bin=fake_bin)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "download-required"
        assert data["selected_model"] == "qwen2.5-coder:7b"

    def test_claude_codex_local_doctor_subcommand_no_state(self, fake_bin, tmp_path):
        result = self._spawn(
            "claude-codex-local",
            extra_args=["doctor"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Without a prior setup, doctor exits 1 and says so on stderr/stdout.
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "wizard" in combined.lower() or "setup" in combined.lower()
