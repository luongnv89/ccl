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
            completed_steps=["1", "2"],
        )
        state.save()
        reloaded = wiz.WizardState.load()
        assert reloaded.primary_harness == "claude"
        assert reloaded.primary_engine == "ollama"
        assert reloaded.completed_steps == ["1", "2"]

    def test_mark_is_idempotent(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState()
        state.mark("1")
        state.mark("1")
        state.mark("2")
        assert state.completed_steps == ["1", "2"]

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

    def test_raw_env_is_emitted_unquoted(self, isolated_state):
        """raw_env values must be emitted verbatim so the shell expands them."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude", "--model", "kr/claude-sonnet-4.5"],
            env={"ANTHROPIC_BASE_URL": "http://localhost:20128/v1"},
            effective_tag="kr/claude-sonnet-4.5",
            raw_env={
                "ANTHROPIC_AUTH_TOKEN": '"$(cat /tmp/key)"',
                "ANTHROPIC_API_KEY": '"$(cat /tmp/key)"',
            },
        )
        path = wiz._write_helper_script("claude", result)
        body = path.read_text()
        # The literal $(cat ...) must be in the script, NOT shlex-quoted as
        # a single-quoted string literal (which would make it a literal value).
        assert 'export ANTHROPIC_AUTH_TOKEN="$(cat /tmp/key)"' in body
        assert 'export ANTHROPIC_API_KEY="$(cat /tmp/key)"' in body
        # Make sure the literal $(cat is not wrapped in single quotes (which
        # would defeat shell expansion).
        assert "'$(cat" not in body
        # Quoted env still emitted with shlex.quote.
        assert "export ANTHROPIC_BASE_URL=" in body

    def test_env_remains_quoted_when_raw_env_also_present(self, isolated_state):
        """Adding raw_env must NOT turn off shlex-quoting on the env field."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude"],
            env={"ANTHROPIC_BASE_URL": "http://example.com/with spaces"},
            effective_tag="x",
            raw_env={"FOO": '"$(cat /tmp/k)"'},
        )
        path = wiz._write_helper_script("claude", result)
        body = path.read_text()
        # Spaces in the value mean ruff will surround the value in single
        # quotes. The literal $(cat is still un-quoted.
        assert "export ANTHROPIC_BASE_URL='http://example.com/with spaces'" in body
        assert 'export FOO="$(cat /tmp/k)"' in body

    def test_claude9_dispatches_to_cc9_filename(self, isolated_state):
        """The 9router fence tag claude9 maps to a `cc9` helper script."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude", "--model", "kr/claude-sonnet-4.5"],
            env={"ANTHROPIC_BASE_URL": "http://localhost:20128/v1"},
            effective_tag="kr/claude-sonnet-4.5",
        )
        path = wiz._write_helper_script("claude9", result)
        assert path.name == "cc9"
        assert path.exists()

    def test_codex9_dispatches_to_cx9_filename(self, isolated_state):
        """The 9router fence tag codex9 maps to a `cx9` helper script."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["codex", "-m", "kr/claude-sonnet-4.5"],
            env={"OPENAI_BASE_URL": "http://localhost:20128/v1"},
            effective_tag="kr/claude-sonnet-4.5",
        )
        path = wiz._write_helper_script("codex9", result)
        assert path.name == "cx9"

    def test_unknown_fence_tag_raises_value_error(self, isolated_state):
        """Defensive: unknown fence tags must fail loudly, not silently fall back."""
        import pytest

        _, wiz, _ = isolated_state
        result = wiz.WireResult(argv=["claude"], env={}, effective_tag="x")
        with pytest.raises(ValueError):
            wiz._write_helper_script("bogus", result)

    def test_alias_block_claude9_short_form_only(self, isolated_state, tmp_path):
        """claude9 emits ONLY the short `cc9` alias, not a long form."""
        _, wiz, _ = isolated_state
        script = tmp_path / "cc9"
        script.write_text("#!/bin/sh\n")
        block, names = wiz._alias_block(script, "claude9")
        assert names == ["cc9"]
        assert "alias cc9=" in block
        assert "alias claude-local=" not in block
        assert "# >>> claude-codex-local:claude9 >>>" in block

    def test_alias_block_codex9_short_form_only(self, isolated_state, tmp_path):
        """codex9 emits ONLY the short `cx9` alias, not a long form."""
        _, wiz, _ = isolated_state
        script = tmp_path / "cx9"
        script.write_text("#!/bin/sh\n")
        block, names = wiz._alias_block(script, "codex9")
        assert names == ["cx9"]
        assert "alias cx9=" in block
        assert "alias codex-local=" not in block
        assert "# >>> claude-codex-local:codex9 >>>" in block


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
            lambda *a, **k: [
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
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
        profile = {"ollama": {"models": []}, "lmstudio": {"models": []}}
        assert wiz._find_model_auto("ollama", profile) is None


# ---------------------------------------------------------------------------
# _build_profile_recommendations — Speed/Balanced/Quality pre-fill (issue #35).
# ---------------------------------------------------------------------------


class TestBuildProfileRecommendations:
    def _candidates(self, *a, **k):
        return [
            {
                "name": "Qwen/Qwen3-Coder-30B",
                "score": 95,
                "estimated_tps": 12,
                "ollama_tag": "qwen3-coder:30b",
                "lms_hub_name": "qwen/qwen3-coder-30b",
                "fit_level": "Good",
            },
            {
                "name": "Qwen/Qwen2.5-Coder-7B",
                "score": 70,
                "estimated_tps": 80,
                "ollama_tag": "qwen2.5-coder:7b",
                "lms_hub_name": "qwen/qwen2.5-coder-7b",
                "fit_level": "Perfect",
            },
        ]

    def test_returns_empty_map_when_llmfit_missing(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        out = wiz._build_profile_recommendations("ollama", {})
        # All three keys present, all values None (graceful no-op).
        assert set(out.keys()) == set(pb.RECOMMENDATION_MODES)
        assert all(v is None for v in out.values())

    def test_returns_per_mode_tags_for_ollama(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        out = wiz._build_profile_recommendations("ollama", {})
        assert out["quality"]["engine_tag"] == "qwen3-coder:30b"
        assert out["fast"]["engine_tag"] == "qwen2.5-coder:7b"
        assert out["balanced"]["engine_tag"] == "qwen3-coder:30b"

    def test_does_not_crash_when_llmfit_raises(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )

        def boom(*_a, **_kw):
            raise RuntimeError("llmfit exploded")

        monkeypatch.setattr(pb, "recommend_for_mode", boom)
        out = wiz._build_profile_recommendations("ollama", {})
        assert all(v is None for v in out.values())

    def test_respects_engine_argument_for_lmstudio(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        out = wiz._build_profile_recommendations("lmstudio", {})
        # Picks lms_hub_name, not ollama_tag — ensures engine is honored.
        assert out["quality"]["engine_tag"] == "qwen/qwen3-coder-30b"


class TestProfileChoiceLabel:
    def test_label_contains_tag_and_metrics(self, isolated_state):
        _, wiz, _ = isolated_state
        rec = {
            "engine_tag": "qwen3-coder:30b",
            "score": 95,
            "estimated_tps": 12,
            "fit_level": "Good",
        }
        label = wiz._profile_choice_label("balanced", rec)
        assert "Balanced" in label
        assert "qwen3-coder:30b" in label
        assert "score=95" in label
        assert "12 tok/s" in label

    def test_label_handles_missing_metrics(self, isolated_state):
        _, wiz, _ = isolated_state
        label = wiz._profile_choice_label("fast", {"engine_tag": "x:y"})
        assert "Speed" in label
        assert "x:y" in label


# ---------------------------------------------------------------------------
# step_2_4_pick_model — mixed picker (issue #35 + #36 integration).
# ---------------------------------------------------------------------------


class _StubAsk:
    """Minimal questionary stub returning a pre-programmed answer."""

    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class TestStep24PickerIntegration:
    """
    Integration tests for the refactored step_2_4_pick_model picker that must
    surface both Speed/Balanced/Quality profiles and installed local models
    alongside the existing manual-entry / llmfit fallback choices.
    """

    def _profile_with_installed_ollama(self):
        return {
            "ollama": {
                "models": [
                    {"name": "llama2:7b", "local": True},
                    {"name": "qwen2.5-coder:7b", "local": True, "size": "4.1 GB"},
                ]
            },
            "lmstudio": {"present": False, "models": []},
            "llamacpp": {"present": False, "server_running": False},
            "disk": {"free_bytes": 1 << 40},
        }

    def _candidates(self, *a, **k):
        return [
            {
                "name": "Qwen/Qwen3-Coder-30B",
                "score": 95,
                "estimated_tps": 12,
                "ollama_tag": "qwen3-coder:30b",
                "lms_hub_name": "qwen/qwen3-coder-30b",
                "fit_level": "Good",
            },
            {
                "name": "Qwen/Qwen2.5-Coder-7B",
                "score": 70,
                "estimated_tps": 80,
                "ollama_tag": "qwen2.5-coder:7b",
                "lms_hub_name": "qwen/qwen2.5-coder-7b",
                "fit_level": "Perfect",
            },
        ]

    def test_installed_model_choice_pre_populated(self, isolated_state, monkeypatch):
        """Picking an installed model skips download and marks step complete (#36)."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        # Capture the choices the picker renders, then return the first installed
        # model choice so the test flow mimics a user picking it.
        captured_choices: list = []

        def fake_select(msg, choices):
            captured_choices.extend(choices)
            for c in choices:
                if isinstance(c.value, str) and c.value.startswith("installed:"):
                    return _StubAsk(c.value)
            return _StubAsk(None)

        monkeypatch.setattr(wiz.questionary, "select", fake_select)
        # _handle_model_presence → model must be already installed so no confirm dialog.
        state = wiz.WizardState(primary_engine="ollama")
        state.profile = self._profile_with_installed_ollama()
        assert wiz.step_2_4_pick_model(state, non_interactive=False) is True
        # Picked tag corresponds to the installed ollama model.
        assert state.engine_model_tag == "qwen2.5-coder:7b"
        assert state.model_source == "installed"
        assert "4" in state.completed_steps
        # Picker surfaced at least the three profile choices + 2 installed models.
        profile_choices = [
            c
            for c in captured_choices
            if isinstance(c.value, str) and c.value.startswith("profile:")
        ]
        installed_choices = [
            c
            for c in captured_choices
            if isinstance(c.value, str) and c.value.startswith("installed:")
        ]
        assert len(profile_choices) >= 1
        assert len(installed_choices) >= 1

    def test_profile_choice_picks_recommended_tag(self, isolated_state, monkeypatch):
        """Picking a Speed/Quality/Balanced profile fills the state with the llmfit tag (#35)."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)

        def fake_select(msg, choices):
            for c in choices:
                if c.value == "profile:quality":
                    return _StubAsk(c.value)
            return _StubAsk(None)

        monkeypatch.setattr(wiz.questionary, "select", fake_select)
        # Model is not installed — _handle_model_presence will ask for download.
        # Stub the confirm prompts to accept defaults and the actual download.
        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _StubAsk(True))

        # Stub the download step so we don't shell out.
        monkeypatch.setattr(wiz, "_download_model", lambda _state: True)

        state = wiz.WizardState(primary_engine="ollama")
        # Profile with no installed ollama coders so the quality pick is not
        # trivially already installed.
        state.profile = {
            "ollama": {"models": []},
            "lmstudio": {"present": False, "models": []},
            "llamacpp": {"present": False, "server_running": False},
            "disk": {"free_bytes": 1 << 40},
        }
        assert wiz.step_2_4_pick_model(state, non_interactive=False) is True
        assert state.engine_model_tag == "qwen3-coder:30b"
        assert state.model_source == "profile:quality"
        assert "4" in state.completed_steps

    def test_direct_entry_still_works(self, isolated_state, monkeypatch):
        """Manual model entry path is unchanged when the user chooses 'I'll type a name'."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": False, "version": ""}
        )

        def fake_select(msg, choices):
            for c in choices:
                if c.value == "direct":
                    return _StubAsk(c.value)
            return _StubAsk(None)

        monkeypatch.setattr(wiz.questionary, "select", fake_select)
        monkeypatch.setattr(wiz.questionary, "text", lambda *a, **kw: _StubAsk("qwen3-coder:30b"))
        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _StubAsk(True))

        state = wiz.WizardState(primary_engine="ollama")
        state.profile = {
            "ollama": {"models": [{"name": "qwen3-coder:30b", "local": True, "size": "19 GB"}]},
            "lmstudio": {"present": False, "models": []},
            "llamacpp": {"present": False, "server_running": False},
            "disk": {"free_bytes": 1 << 40},
        }
        assert wiz.step_2_4_pick_model(state, non_interactive=False) is True
        assert state.engine_model_tag == "qwen3-coder:30b"
        assert state.model_source == "direct"
        assert "4" in state.completed_steps

    def test_cancel_returns_false(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": False, "version": ""}
        )
        monkeypatch.setattr(wiz.questionary, "select", lambda *a, **kw: _StubAsk("cancel"))
        state = wiz.WizardState(primary_engine="ollama")
        state.profile = {
            "ollama": {"models": []},
            "lmstudio": {"present": False, "models": []},
            "llamacpp": {"present": False, "server_running": False},
        }
        assert wiz.step_2_4_pick_model(state, non_interactive=False) is False
        assert "4" not in state.completed_steps

    def test_profile_choices_filtered_to_chosen_engine(self, isolated_state, monkeypatch):
        """
        Profile recommendations must target state.primary_engine. When the engine
        is lmstudio the profile picks must surface lms_hub_name, not ollama_tag.
        """
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)

        captured_choices: list = []

        def fake_select(msg, choices):
            captured_choices.extend(choices)
            # Cancel immediately — we just want to inspect the choice list.
            for c in choices:
                if c.value == "cancel":
                    return _StubAsk(c.value)
            return _StubAsk(None)

        monkeypatch.setattr(wiz.questionary, "select", fake_select)

        state = wiz.WizardState(primary_engine="lmstudio")
        state.profile = {
            "ollama": {"models": []},
            "lmstudio": {"present": True, "server_running": True, "models": []},
            "llamacpp": {"present": False, "server_running": False},
            "disk": {"free_bytes": 1 << 40},
        }
        # Also verify the underlying recommendations dict that the picker
        # consumes — a more direct assertion that doesn't rely on label parsing.
        recs = wiz._build_profile_recommendations("lmstudio", state.profile)
        assert recs["quality"]["engine_tag"].startswith("qwen/")
        assert recs["balanced"]["engine_tag"].startswith("qwen/")
        # An ollama-style tag would contain a colon after the model family.
        for _mode, rec in recs.items():
            if rec is None:
                continue
            assert (
                ":" not in rec["engine_tag"]
            ), f"{_mode} leaked an ollama-style tag: {rec['engine_tag']}"

        wiz.step_2_4_pick_model(state, non_interactive=False)

        # Extract the choice labels for profile entries and ensure at least
        # one shows the lmstudio hub name rather than the ollama tag.
        labels = [c.title if hasattr(c, "title") else str(c) for c in captured_choices]
        joined = "\n".join(labels)
        assert "qwen/qwen3-coder-30b" in joined or "qwen/qwen2.5-coder-7b" in joined

    def test_non_interactive_prefers_installed_model(self, isolated_state):
        """--non-interactive path still hits _find_model_auto (installed-first)."""
        _, wiz, _ = isolated_state
        state = wiz.WizardState(primary_engine="ollama")
        state.profile = self._profile_with_installed_ollama()
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "qwen2.5-coder:7b"


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
        assert "5" in state.completed_steps
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
        assert "5" in state.completed_steps

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
        assert "5" not in state.completed_steps

    def test_failed_smoke_test_reports_failure(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": False, "error": "boom"}
        )
        state = self._setup_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is False
        assert "5" not in state.completed_steps


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

    def test_repo_url_is_github(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._CCL_REPO_URL.startswith("https://github.com/")
        assert "claude-codex-local" in wiz._CCL_REPO_URL

    def test_banner_output_contains_version(self, isolated_state):
        """print_welcome_banner must render the current package version."""
        _, wiz, _ = isolated_state
        with wiz.console.capture() as cap:
            wiz.print_welcome_banner()
        output = cap.get()
        assert wiz.__version__ in output
        assert f"v{wiz.__version__}" in output

    def test_banner_output_contains_github_url(self, isolated_state):
        """print_welcome_banner must render the GitHub repository URL."""
        _, wiz, _ = isolated_state
        with wiz.console.capture() as cap:
            wiz.print_welcome_banner()
        output = cap.get()
        assert wiz._CCL_REPO_URL in output

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


class TestEnginesList9Router:
    """Issue #51 — `9router` is a 4th supported engine alongside the local trio."""

    def test_all_engines_constant_includes_9router(self, isolated_state):
        _, wiz, _ = isolated_state
        assert "9router" in wiz._ALL_ENGINES

    def test_argparse_engine_choice_accepts_9router(self, isolated_state):
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["setup", "--engine", "9router"])
        assert args.engine == "9router"

    def test_argparse_rejects_unknown_engine(self, isolated_state):
        import pytest

        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["setup", "--engine", "totally-bogus"])


class TestEnsureTool9Router:
    """Issue #51 — _ensure_tool must NOT auto-install 9router; it lives on user's machine."""

    def test_returns_true_when_router9_endpoint_reachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": True, "version": ""}
        )
        assert wiz._ensure_tool("9router") is True

    def test_returns_false_when_router9_unreachable_no_install_attempted(
        self, isolated_state, monkeypatch
    ):
        """When 9router is not reachable, _ensure_tool prints help and returns False
        WITHOUT trying to subprocess.run an install command. This is critical:
        9router is a long-running server the user must start manually."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )

        called: dict[str, bool] = {"subprocess_run": False}

        def fake_run(*a, **kw):
            called["subprocess_run"] = True
            raise AssertionError("must not subprocess.run for 9router")

        import subprocess as sp

        monkeypatch.setattr(sp, "run", fake_run)
        # Also block questionary so we'd notice an interactive confirm.
        import questionary

        monkeypatch.setattr(
            questionary,
            "confirm",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not prompt confirm for 9router")
            ),
        )
        assert wiz._ensure_tool("9router") is False
        assert called["subprocess_run"] is False
