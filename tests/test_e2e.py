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
import shlex
import subprocess
import sys
from pathlib import Path

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


def _stub_candidates(*a, **k):
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
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
        monkeypatch.setattr(sys, "argv", ["claude_codex_local.core", "profile"])
        pb.main()
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["presence"]["has_minimum"] is True
        assert data["tools"]["ollama"]["present"] is True

    def test_recommend_prints_selected_model(self, isolated_state, monkeypatch, capsys):
        pb, _, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
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
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
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
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: bad)
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
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
        assert names == {"ollama", "lmstudio", "llamacpp", "vllm", "9router", "openrouter"}


# ---------------------------------------------------------------------------
# Full wizard run in --non-interactive mode, everything stubbed.
# ---------------------------------------------------------------------------


def _stub_subprocess_success(*args, **kwargs):
    return subprocess.CompletedProcess(
        args=args[0] if args else [], returncode=0, stdout="READY\n", stderr=""
    )


def _stub_targeted_setup_checks(pb, monkeypatch):
    def fake_command_version(name, *args, **kwargs):
        present = name in {"claude", "ollama", "llmfit"}
        return {"present": present, "version": f"{name} 1.0" if present else ""}

    monkeypatch.setattr(pb, "command_version", fake_command_version)
    monkeypatch.setattr(
        pb,
        "parse_ollama_list",
        lambda: [{"name": "qwen3-coder:30b", "local": True, "size": "19 GB"}],
    )
    # Block the HTTP ollama path — force parse_ollama_list as the only model source.
    monkeypatch.setattr(pb, "_ollama_http_models", lambda timeout=5: None)


class TestWizardFullFlow:
    def test_non_interactive_run_completes_all_steps(self, isolated_state, monkeypatch):
        pb, wiz, state_dir = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
        _stub_targeted_setup_checks(pb, monkeypatch)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        # step 7 calls subprocess.run on the verify command directly.
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        rc = wiz.run_wizard(non_interactive=True)
        assert rc == 0

        state = wiz.WizardState.load()
        assert set(state.completed_steps) >= {
            "1",
            "3",
            "4",
            "5",
            "6",
            "6.5",
            "7",
            "8",
        }
        assert state.primary_harness == "claude"
        assert state.primary_engine == "ollama"
        assert state.engine_model_tag == "qwen3-coder:30b"
        assert state.verify_result["ok"] is True

        # Verifies step 6.5 wrote the helper script.
        helper = pb.STATE_DIR / "bin" / "cc"
        assert helper.exists()
        assert os.access(helper, os.X_OK)

        # Verifies the shell rc was updated with the alias block.
        rc_path = Path.home() / ".zshrc"
        assert rc_path.exists()
        rc_body = rc_path.read_text()
        assert "# >>> claude-codex-local:claude >>>" in rc_body
        assert "# <<< claude-codex-local:claude <<<" in rc_body
        assert "alias cc=" in rc_body

        # Verifies step 8 wrote a guide.md.
        assert wiz.GUIDE_PATH.exists()
        body = wiz.GUIDE_PATH.read_text()
        assert "qwen3-coder:30b" in body
        assert "claude" in body

    def test_non_interactive_fails_cleanly_on_smoke_test_failure(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
        _stub_targeted_setup_checks(pb, monkeypatch)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": False, "error": "simulated failure"}
        )
        rc = wiz.run_wizard(non_interactive=True)
        assert rc == 1
        # Completed up through step 4; step 5 should NOT be in completed_steps.
        state = wiz.WizardState.load()
        assert "5" not in state.completed_steps
        assert "4" in state.completed_steps

    def test_doctor_reports_clean_state_after_setup(self, isolated_state, monkeypatch, capsys):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
        _stub_targeted_setup_checks(pb, monkeypatch)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(wiz.subprocess, "run", _stub_subprocess_success)

        wiz.run_wizard(non_interactive=True)
        rc = wiz.run_doctor()
        assert rc == 0

    def test_doctor_detects_missing_helper_script(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "machine_profile", lambda **_kw: _installed_profile(pb))
        _stub_targeted_setup_checks(pb, monkeypatch)
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


class TestCoreSubprocessCommands:
    def test_core_profile_emits_json(self, cli_runner):
        result = cli_runner.spawn_core("profile")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "tools" in data
        assert "presence" in data

    def test_core_recommend_returns_fallback_when_no_candidates(self, cli_runner):
        # Default llmfit stub returns {"models": []}, so we should hit pass 5 fallback.
        result = cli_runner.spawn_core("recommend")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["status"] == "download-required"
        assert data["selected_model"] == "qwen2.5-coder:7b"

    def test_ccl_doctor_no_state(self, cli_runner):
        result = cli_runner.spawn_ccl(["doctor"])
        # Without a prior setup, doctor exits 1 and says so on stderr/stdout.
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "no wizard state" in combined
        assert "ccl setup" in combined


class TestCclSetupSubprocessCommand:
    def test_ccl_setup_non_interactive_success(self, cli_runner, fake_bin, tmp_path):
        """Test ccl setup --non-interactive completes successfully with mocked tools."""
        _, put_stub = fake_bin
        put_stub(
            "ollama",
            'case "$1" in\n'
            '  --version) echo "ollama version 0.1.99" ;;\n'
            '  list) printf "NAME  ID  SIZE  MODIFIED\\nqwen3-coder:30b  abc  19 GB  now\\n" ;;\n'
            '  *) echo "READY" ;;\n'
            "esac\n"
            "exit 0",
        )
        result = cli_runner.spawn_ccl(["setup", "--non-interactive"])
        assert result.returncode == 0, result.stderr
        state_file = tmp_path / "state" / "wizard-state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert set(data["completed_steps"]) >= {"1", "3", "4", "5", "6", "6.5", "7", "8"}

    def test_ccl_setup_help(self, cli_runner):
        """Test ccl setup --help shows usage information."""
        result = cli_runner.spawn_ccl(["setup", "--help"])
        assert result.returncode == 0
        out = result.stdout.lower()
        assert "usage:" in out and "setup" in out


class TestCclHelpSubprocessCommands:
    def test_ccl_doctor_help(self, cli_runner):
        """Test ccl doctor --help shows usage information."""
        result = cli_runner.spawn_ccl(["doctor", "--help"])
        assert result.returncode == 0
        out = result.stdout.lower()
        assert "usage:" in out and "doctor" in out

    def test_ccl_find_model_help(self, cli_runner):
        """Test ccl find-model --help shows usage information."""
        result = cli_runner.spawn_ccl(["find-model", "--help"])
        assert result.returncode == 0
        out = result.stdout.lower()
        assert "usage:" in out and "find-model" in out


class TestCclFindModelSubprocessCommand:
    def test_ccl_find_model_non_interactive_reports_no_pick(self, cli_runner):
        """Test ccl find-model returns its documented no-recommendation failure."""
        result = cli_runner.spawn_ccl(["--non-interactive", "find-model"])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "find-model" in combined and "ranking models" in combined


class TestCclDoctorSubprocessCommand:
    def test_ccl_doctor_with_existing_state_reports_missing_helper(self, cli_runner):
        """Doctor with incomplete state reports specific setup drift."""
        cli_runner.seed_state("claude", "lmstudio", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["doctor"])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "helper script" in combined and "missing" in combined

    def test_ccl_doctor_no_state_returns_1(self, cli_runner):
        """Test ccl doctor with no state file returns error."""
        result = cli_runner.spawn_ccl(["doctor"])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "no wizard state" in combined
        assert "ccl setup" in combined


class TestCclSetupFlagSubprocessCommand:
    def test_ccl_setup_resume_flag(self, cli_runner, tmp_path):
        """Test ccl setup --resume flag is recognized."""
        result = cli_runner.spawn_ccl(["--resume", "setup", "--non-interactive"])
        assert result.returncode == 1
        assert "non-interactive find-model failed" in result.stdout.lower()

    def test_ccl_setup_force_harness(self, cli_runner, tmp_path):
        """Test ccl setup --harness flag is recognized."""
        result = cli_runner.spawn_ccl(["setup", "--harness", "claude", "--non-interactive"])
        assert result.returncode == 1
        assert "non-interactive find-model failed" in result.stdout.lower()

    def test_ccl_setup_force_engine(self, cli_runner, tmp_path):
        """Test ccl setup --engine flag is recognized."""
        result = cli_runner.spawn_ccl(["setup", "--engine", "ollama", "--non-interactive"])
        assert result.returncode == 1
        assert "non-interactive find-model failed" in result.stdout.lower()


class TestCclFindModelScenarios:
    def test_ccl_find_model_standalone_with_models_requires_tty(self, cli_runner, fake_bin):
        """Interactive ccl find-model surfaces candidates before failing without a TTY."""

        # Configure llmfit stub to return coding models
        def custom_llmfit():
            return """case "$1" in
  --version) echo "llmfit 1.2.3" ;;
  system) echo '{"system": {"ram_gb": 32, "gpu": "apple-m2"}}' ;;
  --ram) echo '{"models": [{"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "category": "Coding", "score": 95, "best_quant": "mlx-4bit"}]}' ;;
  fit) echo '{"models": [{"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "category": "Coding", "score": 95, "best_quant": "mlx-4bit"}]}' ;;
  info) echo '{"models": [{"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "category": "Coding", "score": 95, "best_quant": "mlx-4bit"}]}' ;;
  coding) echo '{"models": [{"name": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "category": "Coding", "score": 95, "best_quant": "mlx-4bit"}]}' ;;
  *) exit 0 ;;
esac"""

        _, put_stub = fake_bin
        put_stub("llmfit", custom_llmfit())

        result = cli_runner.spawn_ccl(["find-model"])
        assert result.returncode == 1
        assert "Input is not a terminal" in result.stderr
        assert "Running llmfit to rank coding models" in result.stdout

    def test_ccl_find_model_no_models(self, cli_runner):
        """Test ccl find-model handles case with no models found."""
        # llmfit already returns empty models by default
        result = cli_runner.spawn_ccl(["find-model"])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "find-model" in combined and "ranking models" in combined

    def test_ccl_all_commands_help(self, cli_runner):
        """Test that all ccl subcommands have help available."""
        commands = ["setup", "doctor", "find-model", "run"]
        for cmd in commands:
            result = cli_runner.spawn_ccl([cmd, "--help"])
            assert result.returncode == 0, f"{cmd} --help failed: {result.stderr}"
            assert "usage:" in result.stdout.lower()


class TestCclRunSubprocessCommand:
    def test_ccl_run_help(self, cli_runner):
        """`ccl run --help` should describe the subcommand and -p/--prompt."""
        result = cli_runner.spawn_ccl(["run", "--help"])
        assert result.returncode == 0
        out = result.stdout.lower()
        assert "prompt" in out and "-p" in result.stdout

    def test_ccl_run_no_state_errors(self, cli_runner):
        """`ccl run -p ...` without prior setup must fail cleanly, not crash."""
        result = cli_runner.spawn_ccl(["run", "-p", "hello"])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "no wizard state" in combined
        assert "ccl setup" in combined

    def test_ccl_run_with_prompt_claude_ollama(self, cli_runner, fake_bin, tmp_path):
        """
        With prompt + Claude/Ollama state seeded, `ccl run` should dispatch to
        `ollama launch claude ... -- -p <prompt> --model <tag>` (mirrors verify
        step shape). The fake `ollama` records its argv so we can assert.
        """
        bdir, put_stub = fake_bin
        argv_log = tmp_path / "ollama-argv.log"
        put_stub(
            "ollama",
            f'echo "$@" > {shlex.quote(str(argv_log))}\n'
            'case "$1" in --version) echo "ollama version 0.1.99";; esac\n'
            "exit 0",
        )
        cli_runner.seed_state("claude", "ollama", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["run", "-p", "test prompt 1"])
        assert result.returncode == 0, result.stderr
        recorded = argv_log.read_text()
        # The `--` separator + `-p PROMPT` pattern is what makes the prompt
        # land on the claude binary instead of being eaten by `ollama launch`.
        assert "launch claude --model qwen3-coder:30b -- -p test prompt 1" in recorded

    def test_ccl_run_with_prompt_codex_ollama(self, cli_runner, fake_bin, tmp_path):
        """
        Codex+Ollama needs the special exec-after-`--` shape so `--oss` and
        `--local-provider=ollama` land AFTER the `exec` subcommand. Without
        this branch, `ccl run -p ...` would hit the documented limitation and
        fail with a ChatGPT-account error.
        """
        bdir, put_stub = fake_bin
        argv_log = tmp_path / "ollama-argv.log"
        put_stub(
            "ollama",
            f'echo "$@" > {shlex.quote(str(argv_log))}\n'
            'case "$1" in --version) echo "ollama version 0.1.99";; esac\n'
            "exit 0",
        )
        cli_runner.seed_state("codex", "ollama", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["run", "-p", "say hi"])
        assert result.returncode == 0, result.stderr
        recorded = argv_log.read_text()
        assert "exec --skip-git-repo-check --oss --local-provider=ollama say hi" in recorded

    def test_ccl_run_with_prompt_codex_lmstudio(self, cli_runner, fake_bin, tmp_path):
        """Codex+LM Studio path uses `codex exec --skip-git-repo-check -m TAG PROMPT`."""
        bdir, put_stub = fake_bin
        argv_log = tmp_path / "codex-argv.log"
        put_stub(
            "codex",
            f'echo "$@" > {shlex.quote(str(argv_log))}\nexit 0',
        )
        cli_runner.seed_state("codex", "lmstudio", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["run", "-p", "test prompt 2"])
        assert result.returncode == 0, result.stderr
        recorded = argv_log.read_text()
        assert "exec --skip-git-repo-check -m qwen3-coder:30b test prompt 2" in recorded

    def test_ccl_run_long_form_prompt(self, cli_runner, fake_bin, tmp_path):
        """Argparse aliasing — `--prompt` must work the same as `-p`."""
        bdir, put_stub = fake_bin
        argv_log = tmp_path / "codex-argv.log"
        put_stub(
            "codex",
            f'echo "$@" > {shlex.quote(str(argv_log))}\nexit 0',
        )
        cli_runner.seed_state("codex", "lmstudio", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["run", "--prompt", "long form test"])
        assert result.returncode == 0, result.stderr
        assert "long form test" in argv_log.read_text()

    def test_ccl_run_empty_prompt_rejected(self, cli_runner):
        """Empty `-p ""` is a footgun for shell-script callers — reject it."""
        cli_runner.seed_state("claude", "lmstudio", "qwen3-coder:30b")
        result = cli_runner.spawn_ccl(["run", "-p", ""])
        assert result.returncode == 1
        combined = (result.stdout + result.stderr).lower()
        assert "prompt" in combined and "empty" in combined

    def test_ccl_run_with_raw_env_keyfile(self, cli_runner, fake_bin, tmp_path):
        """
        9router/vllm key-on-disk path: `raw_env` shell expressions
        (`"$(cat /path)"`) must be resolved at exec-time without leaking the
        key into the wizard state. Asserts the harness sees the resolved
        value, not the literal expression.
        """
        bdir, put_stub = fake_bin
        keyfile = tmp_path / "router9-key"
        keyfile.write_text("sk-router9-test-secret\n")
        env_log = tmp_path / "claude-env.log"
        put_stub(
            "claude",
            f'echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > {shlex.quote(str(env_log))}\n'
            f'echo "ANTHROPIC_AUTH_TOKEN=$ANTHROPIC_AUTH_TOKEN" >> {shlex.quote(str(env_log))}\n'
            "exit 0",
        )
        key_expr = f'"$(cat {shlex.quote(str(keyfile))})"'
        cli_runner.seed_state(
            "claude",
            "9router",
            "kr/claude-sonnet-4.5",
            raw_env={"ANTHROPIC_API_KEY": key_expr, "ANTHROPIC_AUTH_TOKEN": key_expr},
        )
        result = cli_runner.spawn_ccl(["run", "-p", "ping"])
        assert result.returncode == 0, result.stderr
        leaked = env_log.read_text()
        assert "ANTHROPIC_API_KEY=sk-router9-test-secret" in leaked
        assert "ANTHROPIC_AUTH_TOKEN=sk-router9-test-secret" in leaked
        # Critical: the literal `$(cat ...)` expression must NOT reach the harness.
        assert "$(cat" not in leaked

    def test_ccl_run_with_openrouter_raw_env_keyfile(self, cli_runner, fake_bin, tmp_path):
        """
        OpenRouter key-on-disk path: same security boundary as 9router. The
        `$(cat ...)` shell expression must be resolved at exec-time so the
        harness sees the real key, never the literal expression. The key
        value must never leak into the helper script body or wizard state.
        """
        bdir, put_stub = fake_bin
        keyfile = tmp_path / "openrouter-key"
        keyfile.write_text("openrouter-test-secret\n")
        env_log = tmp_path / "claude-env.log"
        put_stub(
            "claude",
            f'echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > {shlex.quote(str(env_log))}\n'
            f'echo "ANTHROPIC_AUTH_TOKEN=$ANTHROPIC_AUTH_TOKEN" >> {shlex.quote(str(env_log))}\n'
            "exit 0",
        )
        key_expr = f'"$(cat {shlex.quote(str(keyfile))})"'
        cli_runner.seed_state(
            "claude",
            "openrouter",
            "anthropic/claude-sonnet-4.6",
            raw_env={"ANTHROPIC_API_KEY": key_expr, "ANTHROPIC_AUTH_TOKEN": key_expr},
        )
        result = cli_runner.spawn_ccl(["run", "-p", "ping"])
        assert result.returncode == 0, result.stderr
        leaked = env_log.read_text()
        assert "ANTHROPIC_API_KEY=openrouter-test-secret" in leaked
        assert "ANTHROPIC_AUTH_TOKEN=openrouter-test-secret" in leaked
        # Critical: the literal `$(cat ...)` expression must NOT reach the harness.
        assert "$(cat" not in leaked

    def test_ccl_run_no_prompt_execs_helper(self, cli_runner, tmp_path):
        """
        Without -p, `ccl run` should defer to the helper script — preserving
        the existing interactive-launch UX (AC #4 of issue #70).
        """
        state_dir = cli_runner.seed_state("claude", "lmstudio", "qwen3-coder:30b")
        helper_dir = state_dir / "bin"
        helper_dir.mkdir(exist_ok=True)
        helper = helper_dir / "cc"
        marker = tmp_path / "helper-was-called"
        helper.write_text(
            f"#!/usr/bin/env bash\necho invoked > {shlex.quote(str(marker))}\nexit 0\n"
        )
        helper.chmod(0o755)
        result = cli_runner.spawn_ccl(["run"])
        assert result.returncode == 0, result.stderr
        assert marker.exists(), "helper script should have been invoked"
