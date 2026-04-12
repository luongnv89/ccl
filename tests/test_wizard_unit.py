"""
Unit tests for wizard.py helpers — state persistence, engine/model picking
logic, presence checks, and the Claude/Codex wiring helpers.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# WizardState persistence — roundtrip + mark + resilience to a missing file.
# ---------------------------------------------------------------------------


class TestWizardState:
    def test_load_returns_empty_when_no_state_file(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState.load()
        assert state.completed_steps == []
        assert state.primary_harness == ""

    def test_save_then_load_roundtrips_fields(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_harness="claude",
            primary_engine="ollama",
            model_name="qwen3-coder:30b",
            engine_model_tag="qwen3-coder:30b",
            completed_steps=["2.1", "2.2"],
        )
        state.save()
        reloaded = wiz.WizardState.load()
        assert reloaded.primary_harness == "claude"
        assert reloaded.primary_engine == "ollama"
        assert reloaded.completed_steps == ["2.1", "2.2"]

    def test_mark_is_idempotent(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState()
        state.mark("2.1")
        state.mark("2.1")
        state.mark("2.2")
        assert state.completed_steps == ["2.1", "2.2"]

    def test_load_handles_corrupt_state_file(self, isolated_state):
        _, wiz, state_dir = isolated_state
        state_dir.mkdir(parents=True, exist_ok=True)
        wiz.STATE_FILE.write_text("this is not json {")
        state = wiz.WizardState.load()
        assert state.completed_steps == []


# ---------------------------------------------------------------------------
# _default_engine — platform-aware preference rules.
# ---------------------------------------------------------------------------


class TestDefaultEngine:
    def _profile(self, **overrides):
        base = {
            "host": {"system": "Darwin", "machine": "arm64"},
            "ollama": {"models": []},
            "lmstudio": {"server_running": False, "models": []},
        }
        base.update(overrides)
        return base

    def test_prefers_lmstudio_when_ready_on_apple_silicon(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = self._profile(lmstudio={"server_running": True, "models": [{"path": "a"}]})
        assert wiz._default_engine(["ollama", "lmstudio"], profile) == "lmstudio"

    def test_prefers_ollama_when_it_has_models(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = self._profile(ollama={"models": [{"name": "qwen3-coder:30b"}]})
        assert wiz._default_engine(["ollama", "lmstudio"], profile) == "ollama"

    def test_lmstudio_fallback_on_apple_silicon_when_nothing_ready(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._default_engine(["lmstudio", "ollama"], self._profile()) == "lmstudio"

    def test_ollama_fallback_on_linux_x86(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = self._profile(host={"system": "Linux", "machine": "x86_64"})
        assert wiz._default_engine(["ollama", "llamacpp"], profile) == "ollama"

    def test_first_engine_when_nothing_matches_rules(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = self._profile(host={"system": "Linux", "machine": "x86_64"})
        assert wiz._default_engine(["llamacpp"], profile) == "llamacpp"


# ---------------------------------------------------------------------------
# _map_to_engine — resolve a user-typed name to an engine-specific tag.
# ---------------------------------------------------------------------------


class TestMapToEngine:
    def test_ollama_passthrough_for_existing_tag(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._map_to_engine("qwen3-coder:30b", "ollama") == "qwen3-coder:30b"

    def test_ollama_resolves_hf_name(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._map_to_engine("Qwen/Qwen2.5-Coder-7B", "ollama") == "qwen2.5-coder:7b"

    def test_lmstudio_passthrough_for_hub_slash_name(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._map_to_engine("qwen/qwen3-coder-30b", "lmstudio") == "qwen/qwen3-coder-30b"

    def test_lmstudio_resolves_hf_shortname(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._map_to_engine("Qwen3-Coder-30B", "lmstudio") == "qwen/qwen3-coder-30b"

    def test_llamacpp_returns_input_unchanged(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._map_to_engine("whatever", "llamacpp") == "whatever"


# ---------------------------------------------------------------------------
# _candidate_tag — engine-specific field extraction.
# ---------------------------------------------------------------------------


class TestCandidateTag:
    def test_ollama_reads_ollama_tag(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._candidate_tag({"ollama_tag": "qwen3-coder:30b"}, "ollama") == "qwen3-coder:30b"

    def test_lmstudio_prefers_hub_over_mlx_path(self, isolated_state):
        _, wiz, _ = isolated_state
        c = {"lms_hub_name": "qwen/qwen3-coder-30b", "lms_mlx_path": "lmstudio-community/x"}
        assert wiz._candidate_tag(c, "lmstudio") == "qwen/qwen3-coder-30b"

    def test_lmstudio_falls_back_to_mlx_path(self, isolated_state):
        _, wiz, _ = isolated_state
        c = {"lms_mlx_path": "lmstudio-community/x-MLX-4bit"}
        assert wiz._candidate_tag(c, "lmstudio") == "lmstudio-community/x-MLX-4bit"

    def test_llamacpp_falls_back_to_raw_name(self, isolated_state):
        _, wiz, _ = isolated_state
        assert (
            wiz._candidate_tag({"name": "Qwen/Qwen3-Coder-30B"}, "llamacpp")
            == "Qwen/Qwen3-Coder-30B"
        )


# ---------------------------------------------------------------------------
# _model_already_installed — check profile dicts.
# ---------------------------------------------------------------------------


class TestModelAlreadyInstalled:
    def test_ollama_match(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = {"ollama": {"models": [{"name": "qwen3-coder:30b"}]}, "lmstudio": {"models": []}}
        assert wiz._model_already_installed("ollama", "qwen3-coder:30b", profile) is True

    def test_ollama_no_match(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = {"ollama": {"models": [{"name": "llama2:7b"}]}, "lmstudio": {"models": []}}
        assert wiz._model_already_installed("ollama", "qwen3-coder:30b", profile) is False

    def test_lmstudio_match_by_path(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = {
            "ollama": {"models": []},
            "lmstudio": {"models": [{"path": "qwen/qwen3-coder-30b"}]},
        }
        assert wiz._model_already_installed("lmstudio", "qwen/qwen3-coder-30b", profile) is True

    def test_llamacpp_always_false(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._model_already_installed("llamacpp", "x", {}) is False


# ---------------------------------------------------------------------------
# _model_known_incompatible_with_claude_code — tag-pattern check.
# ---------------------------------------------------------------------------


class TestModelIncompatibility:
    def test_qwen3_is_incompatible(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._model_known_incompatible_with_claude_code("qwen/qwen3-coder-30b") is True

    def test_qwen25_is_compatible(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._model_known_incompatible_with_claude_code("qwen/qwen2.5-coder-7b") is False

    def test_llama_is_compatible(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._model_known_incompatible_with_claude_code("meta-llama/llama-3-8b") is False


# ---------------------------------------------------------------------------
# _wire_claude — returns a WireResult, no settings.json written.
# ---------------------------------------------------------------------------


class TestWireClaude:
    def test_ollama_returns_ollama_launch_argv(self, isolated_state):
        _, wiz, _ = isolated_state
        result = wiz._wire_claude("ollama", "qwen3-coder:30b")
        # Trailing "--" lets the helper script forward user args to claude
        # instead of having `ollama launch` eat them.
        assert result.argv == [
            "ollama",
            "launch",
            "claude",
            "--model",
            "qwen3-coder:30b",
            "--",
        ]
        assert result.env == {}
        assert result.effective_tag == "qwen3-coder:30b"

    def test_lmstudio_returns_inline_env(self, isolated_state):
        pb, wiz, _ = isolated_state
        result = wiz._wire_claude("lmstudio", "qwen/qwen2.5-coder-7b")
        assert result.argv == ["claude", "--model", "qwen/qwen2.5-coder-7b"]
        assert result.env["ANTHROPIC_BASE_URL"] == f"http://localhost:{pb.LMS_SERVER_PORT}"
        assert result.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "qwen/qwen2.5-coder-7b"
        assert result.env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"

    def test_llamacpp_returns_inline_env(self, isolated_state):
        _, wiz, _ = isolated_state
        result = wiz._wire_claude("llamacpp", "some-gguf")
        assert result.argv == ["claude", "--model", "some-gguf"]
        assert result.env["ANTHROPIC_BASE_URL"] == "http://localhost:8001"

    def test_unknown_engine_returns_none(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._wire_claude("bogus", "tag") is None


# ---------------------------------------------------------------------------
# _wire_codex — returns WireResult with engine-specific argv/env.
# ---------------------------------------------------------------------------


class TestWireCodex:
    def test_ollama_path(self, isolated_state):
        _, wiz, _ = isolated_state
        result = wiz._wire_codex("ollama", "qwen3-coder:30b")
        assert result.argv[:5] == ["ollama", "launch", "codex", "--model", "qwen3-coder:30b"]
        assert "--oss" in result.argv
        assert result.env == {}

    def test_lmstudio_path(self, isolated_state):
        pb, wiz, _ = isolated_state
        result = wiz._wire_codex("lmstudio", "qwen/qwen3-coder-30b")
        assert result.argv == ["codex", "-m", "qwen/qwen3-coder-30b"]
        assert result.env["OPENAI_BASE_URL"] == f"http://localhost:{pb.LMS_SERVER_PORT}/v1"
        assert result.env["OPENAI_API_KEY"] == "lmstudio"

    def test_llamacpp_path(self, isolated_state):
        _, wiz, _ = isolated_state
        result = wiz._wire_codex("llamacpp", "some-gguf")
        assert result.argv == ["codex", "-m", "some-gguf"]
        assert result.env["OPENAI_BASE_URL"] == "http://localhost:8001/v1"

    def test_unknown_engine(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._wire_codex("bogus", "x") is None


# ---------------------------------------------------------------------------
# Helper script writer — bash file under .claude-codex-local/bin/.
# ---------------------------------------------------------------------------


class TestHelperScriptWriter:
    def test_ollama_script_uses_ollama_launch_no_exports(self, isolated_state):
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["ollama", "launch", "claude", "--model", "qwen2.5-coder:7b", "--"],
            env={},
            effective_tag="qwen2.5-coder:7b",
        )
        path = wiz._write_helper_script("claude", result)
        body = path.read_text()
        # The "--" must appear before "$@" so user flags reach claude, not
        # `ollama launch`.
        assert 'exec ollama launch claude --model qwen2.5-coder:7b -- "$@"' in body
        assert "export " not in body
        assert os.access(path, os.X_OK)

    def test_lmstudio_script_exports_env_and_execs_claude(self, isolated_state):
        pb, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude", "--model", "qwen/qwen2.5-coder-7b"],
            env={
                "ANTHROPIC_BASE_URL": f"http://localhost:{pb.LMS_SERVER_PORT}",
                "ANTHROPIC_API_KEY": "lmstudio",
            },
            effective_tag="qwen/qwen2.5-coder-7b",
        )
        path = wiz._write_helper_script("claude", result)
        body = path.read_text()
        assert "export ANTHROPIC_BASE_URL=" in body
        assert "exec claude --model" in body and '"$@"' in body
        assert os.access(path, os.X_OK)


# ---------------------------------------------------------------------------
# Shell alias installer — fenced block, idempotent overwrite.
# ---------------------------------------------------------------------------


class TestShellAliasInstaller:
    def _make_script(self, tmp_path, name="cc"):
        path = tmp_path / name
        path.write_text("#!/bin/sh\necho hi\n")
        path.chmod(0o755)
        return path

    def test_fresh_install_into_empty_rc(self, isolated_state, tmp_path):
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text("")
        script = self._make_script(tmp_path)
        rc_path, names = wiz._install_shell_aliases(script, "claude", non_interactive=True)
        assert rc_path == rc
        body = rc.read_text()
        assert "# >>> claude-codex-local:claude >>>" in body
        assert "# <<< claude-codex-local:claude <<<" in body
        assert "alias cc=" in body
        assert "alias claude-local=" in body
        assert "cc" in names

    def test_overwrite_replaces_existing_block(self, isolated_state, tmp_path):
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text(
            "# >>> claude-codex-local:claude >>>\n"
            "alias cc='/old/path'\n"
            "# <<< claude-codex-local:claude <<<\n"
        )
        script = self._make_script(tmp_path)
        wiz._install_shell_aliases(script, "claude", non_interactive=True)
        body = rc.read_text()
        # Should be exactly one block for claude.
        assert body.count("# >>> claude-codex-local:claude >>>") == 1
        assert "/old/path" not in body
        assert str(script) in body

    def test_preserves_surrounding_content(self, isolated_state, tmp_path):
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text(
            "export FOO=bar\n"
            "# >>> claude-codex-local:claude >>>\n"
            "alias cc='/old/path'\n"
            "# <<< claude-codex-local:claude <<<\n"
            "export BAZ=qux\n"
        )
        script = self._make_script(tmp_path)
        wiz._install_shell_aliases(script, "claude", non_interactive=True)
        body = rc.read_text()
        assert "export FOO=bar" in body
        assert "export BAZ=qux" in body
        assert body.count("# >>> claude-codex-local:claude >>>") == 1

    def test_unknown_shell_returns_none(self, isolated_state, tmp_path, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv("SHELL", "/bin/fish")
        script = self._make_script(tmp_path)
        rc_path, names = wiz._install_shell_aliases(script, "claude", non_interactive=True)
        assert rc_path is None
        assert "cc" in names

    # ---- Issue #16: cc and cx aliases must coexist ------------------------

    def test_claude_then_codex_coexist(self, isolated_state, tmp_path):
        """Installing claude, then codex, leaves both harness blocks intact."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text("")

        cc_script = self._make_script(tmp_path, "cc")
        cx_script = self._make_script(tmp_path, "cx")

        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)
        wiz._install_shell_aliases(cx_script, "codex", non_interactive=True)

        body = rc.read_text()
        # Both fenced blocks must be present.
        assert "# >>> claude-codex-local:claude >>>" in body
        assert "# <<< claude-codex-local:claude <<<" in body
        assert "# >>> claude-codex-local:codex >>>" in body
        assert "# <<< claude-codex-local:codex <<<" in body
        # Both alias sets must be present.
        assert "alias cc=" in body
        assert "alias claude-local=" in body
        assert "alias cx=" in body
        assert "alias codex-local=" in body
        # Both helper scripts referenced.
        assert str(cc_script) in body
        assert str(cx_script) in body

    def test_codex_then_claude_coexist(self, isolated_state, tmp_path):
        """Installing codex first, then claude, leaves both harness blocks intact."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text("")

        cx_script = self._make_script(tmp_path, "cx")
        cc_script = self._make_script(tmp_path, "cc")

        wiz._install_shell_aliases(cx_script, "codex", non_interactive=True)
        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)

        body = rc.read_text()
        assert "# >>> claude-codex-local:claude >>>" in body
        assert "# >>> claude-codex-local:codex >>>" in body
        assert "alias cc=" in body
        assert "alias cx=" in body
        assert str(cc_script) in body
        assert str(cx_script) in body

    def test_reinstall_same_harness_updates_only_its_block(self, isolated_state, tmp_path):
        """Re-running for the same harness updates only its own block."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text("")

        cc_script_v1 = self._make_script(tmp_path, "cc-v1")
        cx_script = self._make_script(tmp_path, "cx")
        cc_script_v2 = self._make_script(tmp_path, "cc-v2")

        wiz._install_shell_aliases(cc_script_v1, "claude", non_interactive=True)
        wiz._install_shell_aliases(cx_script, "codex", non_interactive=True)
        wiz._install_shell_aliases(cc_script_v2, "claude", non_interactive=True)

        body = rc.read_text()
        # Exactly one claude block and one codex block.
        assert body.count("# >>> claude-codex-local:claude >>>") == 1
        assert body.count("# >>> claude-codex-local:codex >>>") == 1
        # The claude block now references v2, not v1.
        assert str(cc_script_v2) in body
        assert str(cc_script_v1) not in body
        # The codex block is untouched.
        assert str(cx_script) in body

    def test_migration_from_legacy_claude_block(self, isolated_state, tmp_path):
        """A legacy unified block containing alias cc= is migrated to the claude fence."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text(
            "export FOO=bar\n"
            "# >>> claude-codex-local >>>\n"
            "alias cc='/old/cc'\n"
            "alias claude-local='/old/cc'\n"
            "# <<< claude-codex-local <<<\n"
            "export BAZ=qux\n"
        )
        cx_script = self._make_script(tmp_path, "cx")
        wiz._install_shell_aliases(cx_script, "codex", non_interactive=True)

        body = rc.read_text()
        # Legacy fence is gone — replaced by the per-harness fence.
        assert "# >>> claude-codex-local >>>" not in body
        assert "# <<< claude-codex-local <<<" not in body
        # Legacy claude block was preserved as a claude-tagged block.
        assert "# >>> claude-codex-local:claude >>>" in body
        assert "alias cc='/old/cc'" in body
        # New codex block was appended alongside.
        assert "# >>> claude-codex-local:codex >>>" in body
        assert "alias cx=" in body
        assert str(cx_script) in body
        # Surrounding content preserved.
        assert "export FOO=bar" in body
        assert "export BAZ=qux" in body

    def test_migration_from_legacy_codex_block(self, isolated_state, tmp_path):
        """A legacy unified block containing alias cx= is migrated to the codex fence."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text(
            "# >>> claude-codex-local >>>\n"
            "alias cx='/old/cx'\n"
            "alias codex-local='/old/cx'\n"
            "# <<< claude-codex-local <<<\n"
        )
        cc_script = self._make_script(tmp_path, "cc")
        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)

        body = rc.read_text()
        assert "# >>> claude-codex-local >>>" not in body
        # Legacy codex block was preserved as a codex-tagged block.
        assert "# >>> claude-codex-local:codex >>>" in body
        assert "alias cx='/old/cx'" in body
        # New claude block was appended alongside.
        assert "# >>> claude-codex-local:claude >>>" in body
        assert "alias cc=" in body
        assert str(cc_script) in body

    def test_migration_then_reinstall_same_harness_replaces_legacy(self, isolated_state, tmp_path):
        """Re-installing the harness that owned the legacy block replaces its contents."""
        _, wiz, _ = isolated_state
        from pathlib import Path

        rc = Path.home() / ".zshrc"
        rc.write_text(
            "# >>> claude-codex-local >>>\n"
            "alias cc='/old/cc'\n"
            "alias claude-local='/old/cc'\n"
            "# <<< claude-codex-local <<<\n"
        )
        cc_script = self._make_script(tmp_path, "cc")
        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)

        body = rc.read_text()
        assert "# >>> claude-codex-local >>>" not in body
        assert body.count("# >>> claude-codex-local:claude >>>") == 1
        # Old alias path gone, new script path in.
        assert "/old/cc" not in body
        assert str(cc_script) in body


# ---------------------------------------------------------------------------
# _estimate_model_size — delegates to pb.llmfit_estimate_size_bytes.
# ---------------------------------------------------------------------------


class TestEstimateModelSize:
    def test_uses_captured_candidate(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "llmfit_estimate_size_bytes", lambda x: 42 if isinstance(x, dict) else None
        )
        state = wiz.WizardState(model_candidate={"total_memory_gb": 4}, model_name="irrelevant")
        assert wiz._estimate_model_size(state) == 42

    def test_falls_back_to_name_lookup(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "llmfit_estimate_size_bytes", lambda x: 99 if isinstance(x, str) else None
        )
        state = wiz.WizardState(model_name="qwen3-coder:30b")
        assert wiz._estimate_model_size(state) == 99

    def test_returns_none_when_nothing_known(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "llmfit_estimate_size_bytes", lambda x: None)
        assert wiz._estimate_model_size(wiz.WizardState()) is None


# ---------------------------------------------------------------------------
# _find_model_auto — installed-first preference, llmfit fallback.
# ---------------------------------------------------------------------------


class TestFindModelAuto:
    def test_prefers_installed_coding_model(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = {
            "ollama": {
                "models": [
                    {"name": "llama2:7b", "local": True},
                    {"name": "qwen2.5-coder:7b", "local": True},
                ]
            },
            "lmstudio": {"models": []},
        }
        result = wiz._find_model_auto("ollama", profile)
        assert result["tag"] == "qwen2.5-coder:7b"

    def test_lmstudio_prefers_coder_model(self, isolated_state):
        _, wiz, _ = isolated_state
        profile = {
            "ollama": {"models": []},
            "lmstudio": {
                "models": [
                    {"path": "meta/llama-3-8b", "format": "mlx"},
                    {"path": "qwen/qwen3-coder-30b", "format": "mlx"},
                ]
            },
        }
        result = wiz._find_model_auto("lmstudio", profile)
        assert result["tag"] == "qwen/qwen3-coder-30b"

    def test_falls_back_to_llmfit_when_nothing_installed(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "ollama_tag": "qwen3-coder:30b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "score": 90,
                },
            ],
        )
        profile = {"ollama": {"models": []}, "lmstudio": {"models": []}}
        result = wiz._find_model_auto("ollama", profile)
        assert result["tag"] == "qwen3-coder:30b"

    def test_returns_none_when_nothing_at_all(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda: [])
        profile = {"ollama": {"models": []}, "lmstudio": {"models": []}}
        assert wiz._find_model_auto("ollama", profile) is None


# ---------------------------------------------------------------------------
# Smoke test speed reporting — throughput verdicts + slow-model prompt.
# ---------------------------------------------------------------------------


class TestSpeedVerdict:
    def test_slow_below_10(self, isolated_state):
        _, wiz, _ = isolated_state
        label, printer = wiz._speed_verdict(5.5)
        assert "slow" in label
        assert printer is wiz.warn

    def test_acceptable_between_10_and_30(self, isolated_state):
        _, wiz, _ = isolated_state
        label, printer = wiz._speed_verdict(20.0)
        assert "acceptable" in label
        assert printer is wiz.info

    def test_fast_at_or_above_30(self, isolated_state):
        _, wiz, _ = isolated_state
        label, printer = wiz._speed_verdict(42.0)
        assert "fast" in label
        assert printer is wiz.ok

    def test_format_helper(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._format_tokens_per_second(15.3) == "~15.3 tok/s"
        assert wiz._format_tokens_per_second(42.0) == "~42.0 tok/s"


class TestReportSmokeTestSpeed:
    def test_missing_tps_does_not_block(self, isolated_state):
        _, wiz, _ = isolated_state
        result = {"ok": True, "response": "READY"}
        # No tokens_per_second field — function must return True without crashing.
        assert wiz._report_smoke_test_speed(result, non_interactive=True) is True

    def test_fast_throughput_continues_without_prompt(self, isolated_state):
        _, wiz, _ = isolated_state
        result = {
            "ok": True,
            "response": "READY",
            "tokens_per_second": 45.0,
            "completion_tokens": 20,
            "duration_seconds": 0.4,
        }
        assert wiz._report_smoke_test_speed(result, non_interactive=False) is True

    def test_acceptable_throughput_continues(self, isolated_state):
        _, wiz, _ = isolated_state
        result = {
            "ok": True,
            "response": "READY",
            "tokens_per_second": 15.0,
            "completion_tokens": 30,
            "duration_seconds": 2.0,
        }
        assert wiz._report_smoke_test_speed(result, non_interactive=False) is True

    def test_slow_throughput_non_interactive_continues(self, isolated_state):
        _, wiz, _ = isolated_state
        result = {
            "ok": True,
            "response": "READY",
            "tokens_per_second": 5.0,
            "completion_tokens": 10,
            "duration_seconds": 2.0,
        }
        assert wiz._report_smoke_test_speed(result, non_interactive=True) is True

    def test_slow_throughput_interactive_keep_returns_true(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state

        class _FakeAsk:
            def ask(self):
                return True

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _FakeAsk())
        result = {
            "ok": True,
            "response": "READY",
            "tokens_per_second": 4.0,
            "completion_tokens": 8,
            "duration_seconds": 2.0,
        }
        assert wiz._report_smoke_test_speed(result, non_interactive=False) is True

    def test_slow_throughput_interactive_decline_returns_false(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state

        class _FakeAsk:
            def ask(self):
                return False

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _FakeAsk())
        result = {
            "ok": True,
            "response": "READY",
            "tokens_per_second": 4.0,
            "completion_tokens": 8,
            "duration_seconds": 2.0,
        }
        assert wiz._report_smoke_test_speed(result, non_interactive=False) is False


class TestStep2_5SmokeTest:
    """Integration of step_2_5_smoke_test with the speed reporting helper."""

    def _setup_state(self, wiz, engine="ollama"):
        state = wiz.WizardState(
            primary_engine=engine,
            engine_model_tag="qwen3-coder:30b",
        )
        return state

    def test_fast_model_passes(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "smoke_test_ollama_model",
            lambda tag: {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 42.0,
                "completion_tokens": 20,
                "duration_seconds": 0.5,
            },
        )
        state = self._setup_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert "2.5" in state.completed_steps
        assert state.smoke_test_result["tokens_per_second"] == 42.0

    def test_slow_model_non_interactive_still_passes(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "smoke_test_ollama_model",
            lambda tag: {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 3.0,
                "completion_tokens": 6,
                "duration_seconds": 2.0,
            },
        )
        state = self._setup_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert "2.5" in state.completed_steps

    def test_slow_model_interactive_decline_aborts(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "smoke_test_ollama_model",
            lambda tag: {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 3.0,
                "completion_tokens": 6,
                "duration_seconds": 2.0,
            },
        )

        class _FakeAsk:
            def ask(self):
                return False

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _FakeAsk())
        state = self._setup_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=False) is False
        assert "2.5" not in state.completed_steps

    def test_failed_smoke_test_reports_failure(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": False, "error": "boom"}
        )
        state = self._setup_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is False
        assert "2.5" not in state.completed_steps


# ---------------------------------------------------------------------------
# Welcome banner — print_welcome_banner and run_wizard startup display.
# ---------------------------------------------------------------------------


class TestWelcomeBanner:
    def test_banner_contains_ccl(self, isolated_state):
        _, wiz, _ = isolated_state
        assert "CCL" in wiz._CCL_BANNER or "██" in wiz._CCL_BANNER

    def test_tagline_text(self, isolated_state):
        _, wiz, _ = isolated_state
        assert "Hit your limit" in wiz._CCL_TAGLINE
        assert "swap the model" in wiz._CCL_TAGLINE

    def test_banner_not_shown_on_resume(self, isolated_state, monkeypatch):
        """Banner must not appear when --resume is used."""
        _, wiz, _ = isolated_state
        printed = []
        monkeypatch.setattr(wiz, "print_welcome_banner", lambda: printed.append(True))
        # Stub every step to succeed immediately so run_wizard can complete.
        monkeypatch.setattr(wiz, "STEPS", [])
        wiz.run_wizard(resume=True, non_interactive=True)
        assert printed == [], "print_welcome_banner must not be called when resuming"

    def test_banner_not_shown_in_non_interactive(self, isolated_state, monkeypatch):
        """Banner must not appear when --non-interactive is used."""
        _, wiz, _ = isolated_state
        printed = []
        monkeypatch.setattr(wiz, "print_welcome_banner", lambda: printed.append(True))
        monkeypatch.setattr(wiz, "STEPS", [])
        wiz.run_wizard(resume=False, non_interactive=True)
        assert printed == [], "print_welcome_banner must not be called in non-interactive mode"


# ---------------------------------------------------------------------------
# CLI argument parsing — --resume flag at top level.
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Tests for top-level CLI argument parsing, particularly the --resume flag."""

    def test_resume_flag_recognized_at_top_level(self, isolated_state, monkeypatch):
        """ccl --resume should be recognized without the setup subcommand."""
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--resume"])
        # cmd is None when no subcommand given; defaults to setup in main()
        assert args.resume is True

    def test_non_interactive_flag_recognized_at_top_level(self, isolated_state, monkeypatch):
        """ccl --non-interactive should be recognized without the setup subcommand."""
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--non-interactive"])
        # cmd is None when no subcommand given; defaults to setup in main()
        assert args.non_interactive is True

    def test_flags_combined_without_subcommand(self, isolated_state, monkeypatch):
        """Multiple top-level flags can be combined without a subcommand."""
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--resume", "--non-interactive"])
        assert args.resume is True
        assert args.non_interactive is True

    def test_resume_flag_recognized_with_setup_subcommand(self, isolated_state, monkeypatch):
        """ccl setup with top-level flags should work."""
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["setup", "--non-interactive"])
        assert args.cmd == "setup"
        assert args.non_interactive is True

    def test_resume_flag_allowed_without_any_subcommand(self, isolated_state, monkeypatch):
        """
        The --resume flag must be usable without any explicit subcommand.
        This tests the fix for issue #28.
        """
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        # This used to fail with: error: unrecognized arguments: --resume
        args = parser.parse_args(["--resume"])
        assert args.resume is True
        # cmd defaults to 'setup' in main() via: cmd = args.cmd or "setup"

    def test_help_shows_resume_at_top_level(self, isolated_state, monkeypatch):
        """The help output should show --resume as a top-level option."""
        import io

        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        f = io.StringIO()
        try:
            parser.print_help(f)
            help_text = f.getvalue()
        finally:
            f.close()

        assert "--resume" in help_text
        assert "Resume from the last checkpointed step" in help_text

    def test_help_shows_non_interactive_at_top_level(self, isolated_state, monkeypatch):
        """The help output should show --non-interactive as a top-level option."""
        import io

        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        f = io.StringIO()
        try:
            parser.print_help(f)
            help_text = f.getvalue()
        finally:
            f.close()

        assert "--non-interactive" in help_text
        assert "Auto-pick defaults" in help_text
