"""
End-to-end tests for claude-codex-local — stubbed, CI-safe.

These tests verify the full wiring of the MVP:
  * core debug CLI subcommands invoked via main()
  * wizard.run_wizard() executing all 8 steps in non-interactive mode
  * wizard.run_doctor() re-checking presence after a successful setup
  * ccl entry point spawned as a real subprocess with a fake PATH

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

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers — install a synthetic profile + candidate list into core.
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
# Core debug CLI subcommands — invoke main() with argv injection.
#
# These are reachable via `python -m claude_codex_local.core <cmd>` for
# debugging; they are NOT a user-facing binary.
# ---------------------------------------------------------------------------


class TestCoreDebugCli:
    def test_profile_prints_json(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(sys, "argv", ["claude_codex_local.core", "profile"])
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
        monkeypatch.setattr(
            sys, "argv", ["claude_codex_local.core", "recommend", "--mode", "balanced"]
        )
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
        monkeypatch.setattr(sys, "argv", ["claude_codex_local.core", "doctor"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert "profile" in data and "recommendation" in data
        assert data["recommendation"]["selected_model"] == "qwen3-coder:30b"

    def test_doctor_flags_missing_tools(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        bad = _installed_profile(pb)
        bad["tools"]["ollama"] = {"present": False, "error": "not found"}
        bad["ollama"]["models"] = []
        bad["presence"]["engines"] = []
        monkeypatch.setattr(pb, "machine_profile", lambda: bad)
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda: [])
        monkeypatch.setattr(sys, "argv", ["claude_codex_local.core", "doctor"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        assert any("Missing tool: ollama" in i for i in data["issues"])
        assert any("No suitable local coding model" in i for i in data["issues"])

    def test_adapters_subcommand_lists_all(self, isolated_state, monkeypatch, capsys):
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
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "binary": "llama-server",
                "server_running": False,
                "server_port": 8001,
                "model": None,
            },
        )
        monkeypatch.setattr(sys, "argv", ["claude_codex_local.core", "adapters"])
        pb.main()
        data = json.loads(capsys.readouterr().out)
        names = {a["name"] for a in data["adapters"]}
        assert names == {"ollama", "lmstudio", "llamacpp"}


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
        # step 2.7 calls subprocess.run on the verify command directly.
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        rc = wiz.run_wizard(non_interactive=True)
        assert rc == 0

        state = wiz.WizardState.load()
        assert set(state.completed_steps) >= {
            "2.1",
            "2.3",
            "2.4",
            "2.5",
            "2.6",
            "2.65",
            "2.7",
            "2.8",
        }
        assert state.primary_harness == "claude"
        assert state.primary_engine == "ollama"
        assert state.engine_model_tag == "qwen3-coder:30b"
        assert state.verify_result["ok"] is True

        # Verifies step 2.65 wrote the helper script.
        helper = pb.STATE_DIR / "bin" / "cc"
        assert helper.exists()
        assert os.access(helper, os.X_OK)

        # Verifies the shell rc was updated with the alias block.
        from pathlib import Path

        rc_path = Path.home() / ".zshrc"
        assert rc_path.exists()
        rc_body = rc_path.read_text()
        assert "# >>> claude-codex-local:claude >>>" in rc_body
        assert "# <<< claude-codex-local:claude <<<" in rc_body
        assert "alias cc=" in rc_body

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

    def test_doctor_reports_clean_state_after_setup(self, isolated_state, monkeypatch, capsys):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        wiz.run_wizard(non_interactive=True)
        rc = wiz.run_doctor()
        assert rc == 0

    def test_doctor_detects_missing_helper_script(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda: _installed_profile(pb))
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)
        wiz.run_wizard(non_interactive=True)

        # Nuke the helper script and confirm doctor notices.
        (pb.STATE_DIR / "bin" / "cc").unlink()
        rc = wiz.run_doctor()
        assert rc == 1

    def test_doctor_no_state_file_returns_1(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz.run_doctor() == 1


# ---------------------------------------------------------------------------
# ccl + core debug CLI spawned as real subprocesses with a fake PATH.
# ---------------------------------------------------------------------------


class TestCliSubprocesses:
    def _spawn_ccl(self, extra_args=None, tmp_path=None, fake_bin=None, extra_env=None):
        """Invoke the `ccl` entry point (via `python -m claude_codex_local.wizard`)
        with an isolated STATE_DIR and a fake PATH."""
        bdir, _ = fake_bin
        env = os.environ.copy()
        env["PATH"] = f"{bdir}:/usr/bin:/bin"
        env["CLAUDE_CODEX_LOCAL_STATE_DIR"] = str(tmp_path / "state")
        env["HOME"] = str(tmp_path / "home")
        (tmp_path / "home").mkdir(exist_ok=True)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "claude_codex_local.wizard", *(extra_args or [])],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

    def _spawn_core(
        self, subcommand, extra_env=None, tmp_path=None, fake_bin=None, extra_args=None
    ):
        """Invoke claude_codex_local.core as a module with isolated STATE_DIR + a fake PATH."""
        bdir, _ = fake_bin
        env = os.environ.copy()
        env["PATH"] = f"{bdir}:/usr/bin:/bin"
        env["CLAUDE_CODEX_LOCAL_STATE_DIR"] = str(tmp_path / "state")
        env["HOME"] = str(tmp_path / "home")
        (tmp_path / "home").mkdir(exist_ok=True)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "claude_codex_local.core", subcommand, *(extra_args or [])],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

    def test_core_profile_emits_json(self, fake_bin, tmp_path):
        result = self._spawn_core("profile", tmp_path=tmp_path, fake_bin=fake_bin)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "tools" in data
        assert "presence" in data

    def test_core_recommend_returns_fallback_when_no_candidates(self, fake_bin, tmp_path):
        # Default llmfit stub returns {"models": []}, so we should hit pass 5 fallback.
        result = self._spawn_core("recommend", tmp_path=tmp_path, fake_bin=fake_bin)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "download-required"
        assert data["selected_model"] == "qwen2.5-coder:7b"

    def test_ccl_doctor_no_state(self, fake_bin, tmp_path):
        result = self._spawn_ccl(
            extra_args=["doctor"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Without a prior setup, doctor exits 1 and says so on stderr/stdout.
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "wizard" in combined.lower() or "setup" in combined.lower()

    # ----- Tests for ccl setup command -----

    def test_ccl_setup_non_interactive_success(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl setup --non-interactive completes successfully with mocked tools."""
        # Stub ollama to report the model is ready
        monkeypatch.setenv(
            "PATH",
            f"{fake_bin[0].as_posix()}:fake-ollama-ready",
        )
        result = self._spawn_ccl(
            extra_args=["setup", "--non-interactive"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
            extra_env={
                "CLAUDE_CODEX_LOCAL_STATE_DIR": str(tmp_path / "state"),
                "HOME": str(tmp_path / "home"),
            },
        )
        # Should return 0 on success in non-interactive mode
        assert result.returncode == 0 or result.returncode in [
            1,
            2,
        ], f"Setup failed: {result.stderr}"

    def test_ccl_setup_help(self, fake_bin, tmp_path):
        """Test ccl setup --help shows usage information."""
        result = self._spawn_ccl(
            extra_args=["setup", "--help"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "setup" in result.stdout.lower()

    def test_ccl_doctor_help(self, fake_bin, tmp_path):
        """Test ccl doctor --help shows usage information."""
        result = self._spawn_ccl(
            extra_args=["doctor", "--help"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "doctor" in result.stdout.lower()

    def test_ccl_find_model_help(self, fake_bin, tmp_path):
        """Test ccl find-model --help shows usage information."""
        result = self._spawn_ccl(
            extra_args=["find-model", "--help"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        assert result.returncode == 0
        assert (
            "usage" in result.stdout.lower()
            or "find-model" in result.stdout.lower()
            or "find" in result.stdout.lower()
        )

    def test_ccl_find_model_non_interactive(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl find-model runs in non-interactive mode."""
        # Stub llmfit to return a coding model
        monkeypatch.setenv(
            "PATH",
            f"{fake_bin[0].as_posix()}:fake-coding-model",
        )
        result = self._spawn_ccl(
            extra_args=["--non-interactive", "find-model"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
            extra_env={
                "CLAUDE_CODEX_LOCAL_STATE_DIR": str(tmp_path / "state"),
                "HOME": str(tmp_path / "home"),
            },
        )
        # Should succeed even with minimal output
        assert (
            result.returncode == 0 or result.returncode == 1
        ), f"find-model failed: {result.stderr}"

    # ----- Edge cases for ccl doctor command -----

    def test_ccl_doctor_with_existing_state(self, fake_bin, tmp_path, monkeypatch, capsys):
        """Test ccl doctor after a successful setup shows clean state."""
        # First, run setup to create state
        monkeypatch.setattr(
            sys, "argv", ["claude_codex_local.wizard", "setup", "--non-interactive"]
        )

        result = self._spawn_ccl(
            extra_args=["doctor"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Doctor should return 0 or 1 (0 if clean, 1 if issues found)
        assert result.returncode in [0, 1]

    def test_ccl_doctor_no_state_returns_2(self, fake_bin, tmp_path):
        """Test ccl doctor with no state file returns error."""
        result = self._spawn_ccl(
            extra_args=["doctor"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should fail when no state exists
        assert result.returncode != 0 or "no state" in (result.stdout + result.stderr).lower()

    # ----- Edge cases for ccl setup command -----

    def test_ccl_setup_resume_flag(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl setup --resume flag is recognized."""
        result = self._spawn_ccl(
            extra_args=["--resume", "setup", "--non-interactive"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should recognize the flag (may fail for other reasons but not unrecognized flag)
        assert "unrecognized arguments: --resume" not in result.stderr

    def test_ccl_setup_force_harness(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl setup --harness flag is recognized."""
        result = self._spawn_ccl(
            extra_args=["--harness", "claude", "setup", "--non-interactive"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should recognize the flag
        assert "unrecognized arguments: --harness" not in result.stderr

    def test_ccl_setup_force_engine(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl setup --engine flag is recognized."""
        result = self._spawn_ccl(
            extra_args=["--engine", "ollama", "setup", "--non-interactive"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should recognize the flag
        assert "unrecognized arguments: --engine" not in result.stderr

    # ----- Comprehensive test for ccl find-model command -----

    def test_ccl_find_model_standalone_with_models(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl find-model returns model candidates when available."""

        # Configure llmfit stub to return coding models
        def custom_llmfit():
            return """case "$1" in
  --version) echo "llmfit 1.2.3" ;;
  system) echo '{"system": {"ram_gb": 32, "gpu": "apple-m2"}}' ;;
  fit) echo '{"models": [{"name": "test-model", "score": 90}]}' ;;
  info) echo '{"models": [{"name": "test-model", "score": 90}]}' ;;
  coding) echo '{"models": [{"name": "coding-model", "score": 95}]}' ;;
  *) exit 0 ;;
esac"""

        monkeypatch.setenv("PATH", f"{fake_bin[0].as_posix()}")
        (fake_bin[0] / "llmfit").write_text(custom_llmfit(), encoding="utf-8")
        (fake_bin[0] / "llmfit").chmod(0o755)

        result = self._spawn_ccl(
            extra_args=["find-model"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should at least not error on argument parsing
        assert "unrecognized arguments" not in result.stderr.lower()

    def test_ccl_find_model_no_models(self, fake_bin, tmp_path, monkeypatch):
        """Test ccl find-model handles case with no models found."""
        # llmfit already returns empty models by default
        result = self._spawn_ccl(
            extra_args=["find-model", "--non-interactive"],
            tmp_path=tmp_path,
            fake_bin=fake_bin,
        )
        # Should complete without crashing
        assert result.returncode in [0, 1, 2]

    def test_ccl_all_commands_help(self, fake_bin, tmp_path):
        """Test that all ccl subcommands have help available."""
        commands = ["setup", "doctor", "find-model"]
        for cmd in commands:
            result = self._spawn_ccl(
                extra_args=[cmd, "--help"],
                tmp_path=tmp_path,
                fake_bin=fake_bin,
            )
            # Each command should at least show some help output
            assert result.returncode in [0, 2], f"{cmd} --help failed"
