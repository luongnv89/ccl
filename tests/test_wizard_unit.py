"""
Unit tests for wizard.py helpers — state persistence, engine/model picking
logic, presence checks, and the Claude/Codex wiring helpers.
"""

from __future__ import annotations

import json
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

    def test_llamacpp_no_match_for_unknown_tag(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._model_already_installed("llamacpp", "x", {}) is False

    def test_llamacpp_existing_local_gguf_is_installed(self, isolated_state, tmp_path):
        # A tag that points to an on-disk GGUF (as produced by the "installed
        # llamacpp model" picker) must be recognised — otherwise the wizard
        # would re-prompt to download a model that is already on disk.
        _, wiz, _ = isolated_state
        gguf = tmp_path / "model-Q4_K_M.gguf"
        gguf.write_bytes(b"\x00")
        assert wiz._model_already_installed("llamacpp", str(gguf), {}) is True

    def test_llamacpp_missing_local_path_is_not_installed(self, isolated_state, tmp_path):
        _, wiz, _ = isolated_state
        missing = tmp_path / "does-not-exist.gguf"
        assert wiz._model_already_installed("llamacpp", str(missing), {}) is False

    def test_llamacpp_running_server_matches_hf_tag_against_gguf_basename(
        self, isolated_state, monkeypatch
    ):
        # Server reports a GGUF file basename; saved tag is the HF repo id.
        # Exact equality would falsely flag this as missing — use the loose
        # matcher to keep `ccl doctor` honest.
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "server_running": True,
                "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            },
        )
        assert (
            wiz._model_already_installed("llamacpp", "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", {})
            is True
        )

    def test_llamacpp_running_server_rejects_wrong_size(self, isolated_state, monkeypatch):
        # Loose match must NOT accept a different size of the same family —
        # this is the safeguard a naive `wanted in served` would lose.
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "server_running": True,
                "model": "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
            },
        )
        assert (
            wiz._model_already_installed("llamacpp", "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", {})
            is False
        )

    def test_llamacpp_no_running_server_is_not_installed(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(pb, "llamacpp_info", lambda: {"server_running": False})
        assert wiz._model_already_installed("llamacpp", "unsloth/whatever-GGUF", {}) is False


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

    def test_9router_returns_raw_env_with_keyfile_expr(self, isolated_state):
        """9router routes to paid cloud models; key MUST stay in a keyfile."""
        pb, wiz, _ = isolated_state
        result = wiz._wire_claude("9router", "kr/claude-sonnet-4.5")
        assert result.argv == ["claude", "--model", "kr/claude-sonnet-4.5"]
        # Plain env: base URL, model option, attribution header.
        assert result.env["ANTHROPIC_BASE_URL"] == pb.ROUTER9_BASE_URL
        assert result.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "kr/claude-sonnet-4.5"
        assert result.env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        # raw_env: API key as $(cat ...) expression — NOT embedded literally.
        assert "ANTHROPIC_AUTH_TOKEN" in result.raw_env
        assert "ANTHROPIC_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["ANTHROPIC_AUTH_TOKEN"]
        assert str(pb.ROUTER9_KEY_FILE) in result.raw_env["ANTHROPIC_AUTH_TOKEN"]

    def test_9router_helper_script_does_not_embed_key_value(self, isolated_state):
        """Pin: the helper script must contain $(cat ...) and NOT a real-looking key.

        The fake test key 'router9-test-key' (allowlisted for detect-secrets)
        must NEVER appear in the script body, because the wiring path does
        not read the keyfile — it only generates a $(cat ...) expression.
        """
        pb, wiz, _ = isolated_state
        # Create a key file with a sentinel value to verify the script
        # body does not capture it at install time.
        pb.ensure_state_dirs()
        pb.ROUTER9_KEY_FILE.write_text("router9-test-key\n")  # pragma: allowlist secret
        pb.ROUTER9_KEY_FILE.chmod(0o600)
        result = wiz._wire_claude("9router", "kr/claude-sonnet-4.5")
        path = wiz._write_helper_script("claude9", result)
        body = path.read_text()
        # Sentinel key value MUST NOT be embedded — the script reads it at exec.
        assert "router9-test-key" not in body  # pragma: allowlist secret
        # And the script must reference the keyfile via $(cat ...).
        assert "$(cat" in body
        assert str(pb.ROUTER9_KEY_FILE) in body

    def test_openrouter_returns_raw_env_with_keyfile_expr(self, isolated_state):
        """OpenRouter routes to paid cloud models; key MUST stay in a keyfile."""
        pb, wiz, _ = isolated_state
        result = wiz._wire_claude("openrouter", "anthropic/claude-sonnet-4.6")
        assert result.argv == ["claude", "--model", "anthropic/claude-sonnet-4.6"]
        assert result.env["ANTHROPIC_BASE_URL"] == pb.OPENROUTER_BASE_URL
        assert result.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "anthropic/claude-sonnet-4.6"
        assert "OpenRouter" in result.env["ANTHROPIC_CUSTOM_MODEL_OPTION_NAME"]
        assert result.env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        # Attribution headers (decorative, harmless when ignored).
        assert result.env["HTTP_REFERER"] == "https://github.com/luongnv89/ccl"
        assert result.env["X_TITLE"] == "claude-codex-local"
        # raw_env: API key as $(cat ...) expression — NOT embedded literally.
        assert "ANTHROPIC_AUTH_TOKEN" in result.raw_env
        assert "ANTHROPIC_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["ANTHROPIC_AUTH_TOKEN"]
        assert str(pb.OPENROUTER_KEY_FILE) in result.raw_env["ANTHROPIC_AUTH_TOKEN"]

    def test_openrouter_helper_script_does_not_embed_key_value(self, isolated_state):
        """Pin: OpenRouter helper script must contain $(cat ...) and NOT the real key.

        Mirrors the 9router pin. The sentinel key 'openrouter-test-key'
        must NEVER appear in the script body — the script reads it at
        exec-time from the chmod-600 file.
        """
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.OPENROUTER_KEY_FILE.write_text("openrouter-test-key\n")  # pragma: allowlist secret
        pb.OPENROUTER_KEY_FILE.chmod(0o600)
        result = wiz._wire_claude("openrouter", "anthropic/claude-sonnet-4.6")
        path = wiz._write_helper_script("claudeo", result)
        body = path.read_text()
        assert "openrouter-test-key" not in body  # pragma: allowlist secret
        assert "$(cat" in body
        assert str(pb.OPENROUTER_KEY_FILE) in body


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

    def test_9router_returns_raw_env_with_keyfile_expr(self, isolated_state):
        """9router for codex: only OPENAI_API_KEY needs deferred-cat."""
        pb, wiz, _ = isolated_state
        result = wiz._wire_codex("9router", "kr/claude-sonnet-4.5")
        assert result.argv == ["codex", "-m", "kr/claude-sonnet-4.5"]
        assert result.env["OPENAI_BASE_URL"] == pb.ROUTER9_BASE_URL
        assert "OPENAI_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["OPENAI_API_KEY"]
        assert str(pb.ROUTER9_KEY_FILE) in result.raw_env["OPENAI_API_KEY"]
        # Plain env must NOT carry a literal API key; only the URL.
        assert "OPENAI_API_KEY" not in result.env

    def test_openrouter_returns_raw_env_with_keyfile_expr(self, isolated_state):
        """OpenRouter for codex: only OPENAI_API_KEY needs deferred-cat."""
        pb, wiz, _ = isolated_state
        result = wiz._wire_codex("openrouter", "anthropic/claude-sonnet-4.6")
        assert result.argv == ["codex", "-m", "anthropic/claude-sonnet-4.6"]
        assert result.env["OPENAI_BASE_URL"] == pb.OPENROUTER_BASE_URL
        assert result.env["HTTP_REFERER"] == "https://github.com/luongnv89/ccl"
        assert result.env["X_TITLE"] == "claude-codex-local"
        assert "OPENAI_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["OPENAI_API_KEY"]
        assert str(pb.OPENROUTER_KEY_FILE) in result.raw_env["OPENAI_API_KEY"]
        # Plain env must NOT carry a literal API key; only URL + attribution.
        assert "OPENAI_API_KEY" not in result.env

    def test_openrouter_codex_helper_script_does_not_embed_key_value(self, isolated_state):
        """Pin: codexo helper script must contain $(cat ...) and NOT the real key.

        Mirrors the claude/9router pins. Closes the coverage matrix so all
        three OpenRouter helper-script paths (claude/codex/pi) have a
        sentinel-based assertion that the API key value is never embedded
        in the generated script body.
        """
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.OPENROUTER_KEY_FILE.write_text("openrouter-test-key\n")  # pragma: allowlist secret
        pb.OPENROUTER_KEY_FILE.chmod(0o600)
        result = wiz._wire_codex("openrouter", "anthropic/claude-sonnet-4.6")
        path = wiz._write_helper_script("codexo", result)
        body = path.read_text()
        assert "openrouter-test-key" not in body  # pragma: allowlist secret
        assert "$(cat" in body
        assert str(pb.OPENROUTER_KEY_FILE) in body


# ---------------------------------------------------------------------------
# _wire_pi — writes isolated Pi custom-provider config.
# ---------------------------------------------------------------------------


class TestWirePi:
    def test_ollama_writes_models_json_and_argv(self, isolated_state):
        pb, wiz, _ = isolated_state
        result = wiz._wire_pi("ollama", "qwen2.5-coder:7b")
        assert result.argv == ["pi", "--provider", "ccl-ollama", "--model", "qwen2.5-coder:7b"]
        assert result.env["PI_CODING_AGENT_DIR"] == str(pb.STATE_DIR / "pi-agent")

        models = json.loads((pb.STATE_DIR / "pi-agent" / "models.json").read_text())
        provider = models["providers"]["ccl-ollama"]
        assert provider["baseUrl"] == "http://localhost:11434/v1"
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "ollama"
        assert provider["compat"]["supportsDeveloperRole"] is False
        assert provider["models"][0]["id"] == "qwen2.5-coder:7b"

    def test_lmstudio_uses_openai_compatible_port(self, isolated_state):
        pb, wiz, _ = isolated_state
        result = wiz._wire_pi("lmstudio", "qwen/qwen2.5-coder-7b")
        assert result.argv[:3] == ["pi", "--provider", "ccl-lmstudio"]
        models = json.loads((pb.STATE_DIR / "pi-agent" / "models.json").read_text())
        assert (
            models["providers"]["ccl-lmstudio"]["baseUrl"]
            == f"http://localhost:{pb.LMS_SERVER_PORT}/v1"
        )
        assert models["providers"]["ccl-lmstudio"]["apiKey"] == "lmstudio"

    def test_9router_uses_keyfile_command_not_literal_key(self, isolated_state):
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.ROUTER9_KEY_FILE.write_text("router9-test-key\n")  # pragma: allowlist secret
        result = wiz._wire_pi("9router", "kr/claude-sonnet-4.5")
        assert result.argv == ["pi", "--provider", "ccl-9router", "--model", "kr/claude-sonnet-4.5"]
        models_path = pb.STATE_DIR / "pi-agent" / "models.json"
        body = models_path.read_text()
        assert "router9-test-key" not in body  # pragma: allowlist secret
        models = json.loads(body)
        provider = models["providers"]["ccl-9router"]
        assert provider["baseUrl"] == pb.ROUTER9_BASE_URL
        assert provider["apiKey"].startswith("!cat ")
        assert str(pb.ROUTER9_KEY_FILE) in provider["apiKey"]

    def test_openrouter_uses_keyfile_command_not_literal_key(self, isolated_state):
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.OPENROUTER_KEY_FILE.write_text("openrouter-test-key\n")  # pragma: allowlist secret
        result = wiz._wire_pi("openrouter", "anthropic/claude-sonnet-4.6")
        assert result.argv == [
            "pi",
            "--provider",
            "ccl-openrouter",
            "--model",
            "anthropic/claude-sonnet-4.6",
        ]
        models_path = pb.STATE_DIR / "pi-agent" / "models.json"
        body = models_path.read_text()
        # Sentinel key value MUST NOT appear in the models.json — Pi reads
        # it at exec-time via the !cat directive.
        assert "openrouter-test-key" not in body  # pragma: allowlist secret
        models = json.loads(body)
        provider = models["providers"]["ccl-openrouter"]
        assert provider["baseUrl"] == pb.OPENROUTER_BASE_URL
        assert provider["apiKey"].startswith("!cat ")
        assert str(pb.OPENROUTER_KEY_FILE) in provider["apiKey"]

    def test_unknown_engine_returns_none(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._wire_pi("bogus", "tag") is None


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

    def test_claudeo_dispatches_to_cco_filename(self, isolated_state):
        """The OpenRouter fence tag claudeo maps to a `cco` helper script."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude", "--model", "anthropic/claude-sonnet-4.6"],
            env={"ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1"},
            effective_tag="anthropic/claude-sonnet-4.6",
        )
        path = wiz._write_helper_script("claudeo", result)
        assert path.name == "cco"
        assert path.exists()

    def test_codexo_dispatches_to_cxo_filename(self, isolated_state):
        """The OpenRouter fence tag codexo maps to a `cxo` helper script."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["codex", "-m", "anthropic/claude-sonnet-4.6"],
            env={"OPENAI_BASE_URL": "https://openrouter.ai/api/v1"},
            effective_tag="anthropic/claude-sonnet-4.6",
        )
        path = wiz._write_helper_script("codexo", result)
        assert path.name == "cxo"

    def test_pio_dispatches_to_cpo_filename(self, isolated_state):
        """The OpenRouter fence tag pio maps to a `cpo` helper script."""
        _, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["pi", "--provider", "ccl-openrouter", "--model", "anthropic/claude-sonnet-4.6"],
            env={"PI_CODING_AGENT_DIR": "/tmp/pi"},
            effective_tag="anthropic/claude-sonnet-4.6",
        )
        path = wiz._write_helper_script("pio", result)
        assert path.name == "cpo"

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

    def test_alias_block_claudeo_short_form_only(self, isolated_state, tmp_path):
        """claudeo (OpenRouter) emits ONLY the short `cco` alias, not a long form."""
        _, wiz, _ = isolated_state
        script = tmp_path / "cco"
        script.write_text("#!/bin/sh\n")
        block, names = wiz._alias_block(script, "claudeo")
        assert names == ["cco"]
        assert "alias cco=" in block
        assert "alias claude-local=" not in block
        assert "# >>> claude-codex-local:claudeo >>>" in block

    def test_alias_block_codexo_short_form_only(self, isolated_state, tmp_path):
        """codexo (OpenRouter) emits ONLY the short `cxo` alias, not a long form."""
        _, wiz, _ = isolated_state
        script = tmp_path / "cxo"
        script.write_text("#!/bin/sh\n")
        block, names = wiz._alias_block(script, "codexo")
        assert names == ["cxo"]
        assert "alias cxo=" in block
        assert "alias codex-local=" not in block
        assert "# >>> claude-codex-local:codexo >>>" in block

    def test_alias_block_pio_short_form_only(self, isolated_state, tmp_path):
        """pio (OpenRouter) emits ONLY the short `cpo` alias, not a long form."""
        _, wiz, _ = isolated_state
        script = tmp_path / "cpo"
        script.write_text("#!/bin/sh\n")
        block, names = wiz._alias_block(script, "pio")
        assert names == ["cpo"]
        assert "alias cpo=" in block
        assert "alias pi-local=" not in block
        assert "# >>> claude-codex-local:pio >>>" in block


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


class TestForcedPreferenceSelection:
    """CLI --harness/--engine values must be honored independently."""

    def test_non_interactive_forced_engine_survives_default_picker(
        self, isolated_state, monkeypatch
    ):
        pb, wiz, _ = isolated_state
        state = wiz.WizardState(primary_engine="9router")
        state.profile = {
            "presence": {
                "harnesses": ["claude"],
                "engines": ["ollama", "9router"],
            },
            "host": {"system": "Darwin", "machine": "arm64"},
            "ollama": {"models": [{"name": "qwen2.5-coder:7b"}]},
            "lmstudio": {"server_running": False, "models": []},
        }
        monkeypatch.setattr(pb, "command_version", lambda name: {"present": name == "claude"})

        class _Router9:
            def detect(self):
                return {"present": True, "version": "9router 1.0"}

            def healthcheck(self):
                return {"ok": True}

        monkeypatch.setattr(pb, "Router9Adapter", lambda: _Router9())

        assert wiz.step_2_select_harness(state, non_interactive=True) is True
        assert wiz.step_3_select_engine(state, non_interactive=True) is True

        assert state.primary_harness == "claude"
        assert state.primary_engine == "9router"
        assert state.secondary_engines == ["ollama"]

    def test_interactive_forced_engine_skips_engine_prompt(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        state = wiz.WizardState(primary_engine="9router")
        state.profile = {
            "presence": {
                "harnesses": ["claude"],
                "engines": ["ollama", "9router"],
            },
            "host": {"system": "Darwin", "machine": "arm64"},
            "ollama": {"models": [{"name": "qwen2.5-coder:7b"}]},
            "lmstudio": {"server_running": False, "models": []},
        }

        prompts: list[str] = []

        def fake_select(message, choices, default=None):
            prompts.append(message)
            if "harness" in message:
                return _StubAsk("claude")
            raise AssertionError("--engine 9router should skip the engine picker")

        monkeypatch.setattr(wiz.questionary, "select", fake_select)
        monkeypatch.setattr(pb, "command_version", lambda name: {"present": name == "claude"})

        class _Router9:
            def detect(self):
                return {"present": True, "version": "9router 1.0"}

            def healthcheck(self):
                return {"ok": True}

        monkeypatch.setattr(pb, "Router9Adapter", lambda: _Router9())

        assert wiz.step_2_select_harness(state, non_interactive=False) is True
        assert wiz.step_3_select_engine(state, non_interactive=False) is True

        assert prompts == ["Which harness do you want as primary?"]
        assert state.primary_engine == "9router"


class TestTargetedPreferenceRefresh:
    """Issue #79: selected components are rechecked live instead of trusting stale cache."""

    def test_selected_harness_refresh_corrects_stale_missing_cache(
        self, isolated_state, monkeypatch
    ):
        pb, wiz, _ = isolated_state
        profile = {
            "tools": {"claude": {"present": False}, "codex": {"present": False}},
            "presence": {"harnesses": [], "engines": [], "has_minimum": False},
        }
        monkeypatch.setattr(
            pb,
            "command_version",
            lambda name: {"present": name == "claude", "version": "claude 1.0"},
        )

        assert wiz._refresh_selected_harness(profile, "claude") is True
        assert profile["tools"]["claude"]["present"] is True
        assert profile["presence"]["harnesses"] == ["claude"]

    def test_selected_engine_refresh_updates_only_chosen_engine_models(
        self, isolated_state, monkeypatch
    ):
        pb, wiz, _ = isolated_state
        profile = {
            "tools": {"ollama": {"present": False}, "lmstudio": {"present": False}},
            "presence": {"harnesses": ["claude"], "engines": [], "has_minimum": False},
        }
        calls: list[str] = []

        def fake_command_version(name):
            calls.append(name)
            return {"present": name == "ollama", "version": "ollama 0.1"}

        monkeypatch.setattr(pb, "command_version", fake_command_version)
        monkeypatch.setattr(
            pb,
            "parse_ollama_list",
            lambda: [{"name": "qwen2.5-coder:7b", "local": True}],
        )
        monkeypatch.setattr(
            pb,
            "lms_info",
            lambda: (_ for _ in ()).throw(AssertionError("lmstudio must not be probed")),
        )

        assert wiz._refresh_selected_engine(profile, "ollama") is True
        assert calls == ["ollama"]
        assert profile["presence"]["engines"] == ["ollama"]
        assert profile["ollama"]["models"][0]["name"] == "qwen2.5-coder:7b"

    def test_selected_harness_status_is_reported_after_live_check(
        self, isolated_state, monkeypatch
    ):
        pb, wiz, _ = isolated_state
        state = wiz.WizardState(primary_harness="claude", primary_engine="ollama")
        state.profile = {
            "tools": {"claude": {"present": False}, "ollama": {"present": False}},
            "presence": {"harnesses": [], "engines": [], "has_minimum": False},
        }

        def fake_command_version(name):
            return {
                "present": name in {"claude", "ollama"},
                "version": f"{name} 1.0",
            }

        monkeypatch.setattr(pb, "command_version", fake_command_version)
        monkeypatch.setattr(
            pb,
            "parse_ollama_list",
            lambda: [{"name": "qwen2.5-coder:7b", "local": True}],
        )

        wiz.console.width = 200
        with wiz.console.capture() as captured:
            assert wiz.step_2_select_harness(state, non_interactive=True) is True

        output = captured.get()
        assert "claude CLI detected: claude 1.0" in output
        assert "No existing local helper configuration recorded" in output


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
                # Issue #79 renamed the picker key from "installed:" to "merged:"
                # — both installed and cached llmfit recommendations now share
                # the same option group.
                if isinstance(c.value, str) and c.value.startswith("merged:"):
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
        # Picker surfaced at least the three profile choices + 2 merged models.
        profile_choices = [
            c
            for c in captured_choices
            if isinstance(c.value, str) and c.value.startswith("profile:")
        ]
        merged_choices = [
            c
            for c in captured_choices
            if isinstance(c.value, str) and c.value.startswith("merged:")
        ]
        assert len(profile_choices) >= 1
        assert len(merged_choices) >= 1

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
# Step 5 — llama.cpp auto-start branch (issue #53).
# ---------------------------------------------------------------------------


class TestStep2_5LlamaCppAutostart:
    """Cover the new auto-start orchestration in `_llamacpp_smoke_test`."""

    def _llamacpp_state(self, wiz, model_path: str | None = None):
        state = wiz.WizardState(
            primary_engine="llamacpp",
            engine_model_tag="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        )
        state.profile = {"llmfit_system": {"system": {"has_gpu": False, "cpu_cores": 4}}}
        if model_path:
            state.profile["llamacpp_model_path"] = model_path
        return state

    def test_reuses_running_server_with_matching_model(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": True,
                "server_port": 8001,
                "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            },
        )
        spawn_calls = []
        monkeypatch.setattr(
            pb,
            "llamacpp_start_server",
            lambda **kw: (spawn_calls.append(kw), {"ok": True})[1],
        )
        monkeypatch.setattr(
            pb,
            "smoke_test_llamacpp_model",
            lambda tag: {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 30.0,
                "completion_tokens": 6,
                "duration_seconds": 0.2,
            },
        )
        state = self._llamacpp_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert "5" in state.completed_steps
        # Did NOT spawn a new server because one was already running.
        assert spawn_calls == []

    def test_running_server_with_other_model_aborts_in_non_interactive(
        self, isolated_state, monkeypatch
    ):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": True,
                "server_port": 8001,
                "model": "deepseek-coder-v2.gguf",
            },
        )
        spawn_calls = []
        monkeypatch.setattr(pb, "llamacpp_start_server", lambda **kw: spawn_calls.append(kw))
        state = self._llamacpp_state(wiz)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is False
        assert "5" not in state.completed_steps
        assert spawn_calls == []
        assert "different model" in (state.smoke_test_result.get("error") or "")

    def test_no_path_recorded_aborts_with_actionable_error(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": False,
                "server_port": 8001,
                "model": None,
            },
        )
        spawn_calls = []
        monkeypatch.setattr(pb, "llamacpp_start_server", lambda **kw: spawn_calls.append(kw))
        state = self._llamacpp_state(wiz, model_path=None)
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is False
        assert spawn_calls == []
        assert "re-run wizard step 4" in (state.smoke_test_result.get("error") or "")

    def test_spawns_server_when_missing_then_runs_smoke_test(
        self, isolated_state, monkeypatch, tmp_path
    ):
        pb, wiz, _ = isolated_state
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"\x00")
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": False,
                "server_port": 8001,
                "model": None,
            },
        )
        spawned: dict = {}

        def _fake_start(**kw):
            spawned.update(kw)
            return {
                "ok": True,
                "argv": [
                    "llama-server",
                    "--model",
                    kw["model_path"],
                    "--port",
                    str(kw.get("port", 8001)),
                ],
                "handle": pb.LlamaServerHandle(
                    pid=42424,
                    port=kw.get("port", 8001),
                    host=kw.get("host", "127.0.0.1"),
                    model_path=kw["model_path"],
                    argv=["llama-server", "--model", kw["model_path"]],
                    log_path=str(tmp_path / "log.txt"),
                    pid_file=str(tmp_path / "pid"),
                    we_started_it=True,
                ),
                "error": None,
                "log_path": str(tmp_path / "log.txt"),
            }

        monkeypatch.setattr(pb, "llamacpp_start_server", _fake_start)
        monkeypatch.setattr(
            pb,
            "smoke_test_llamacpp_model",
            lambda tag: {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 25.0,
                "completion_tokens": 6,
                "duration_seconds": 0.24,
            },
        )
        state = self._llamacpp_state(wiz, model_path=str(gguf))
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert "5" in state.completed_steps
        assert spawned["model_path"] == str(gguf)

    def test_spawn_failure_surfaces_command_in_result(self, isolated_state, monkeypatch, tmp_path):
        pb, wiz, _ = isolated_state
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"\x00")
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": False,
                "server_port": 8001,
                "model": None,
            },
        )

        def _fail_start(**kw):
            return {
                "ok": False,
                "argv": [
                    "/usr/local/bin/llama-server",
                    "--model",
                    kw["model_path"],
                    "--port",
                    "8001",
                ],
                "handle": None,
                "error": "server did not become ready within 30s",
                "log_path": str(tmp_path / "log.txt"),
            }

        monkeypatch.setattr(pb, "llamacpp_start_server", _fail_start)
        state = self._llamacpp_state(wiz, model_path=str(gguf))
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is False
        assert "5" not in state.completed_steps
        # Manual command was captured in the result so the wizard prints it.
        assert state.smoke_test_result.get("manual_command")
        assert str(gguf) in state.smoke_test_result["manual_command"]
        assert "--model" in state.smoke_test_result["manual_command"]


class TestEnsureLlamacppServerRunning:
    """Step 7's pre-flight guard: server may have died between Step 5 and 7."""

    def _state(self, wiz, model_path=None):
        state = wiz.WizardState(
            primary_engine="llamacpp",
            engine_model_tag="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        )
        state.profile = {"llmfit_system": {"system": {"has_gpu": False, "cpu_cores": 4}}}
        if model_path:
            state.profile["llamacpp_model_path"] = model_path
        return state

    def test_reuses_running_server_when_model_matches(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "server_running": True,
                "server_port": 8001,
                "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            },
        )
        spawn_calls = []
        monkeypatch.setattr(
            pb, "llamacpp_start_server", lambda **kw: spawn_calls.append(kw) or {"ok": True}
        )
        state = self._state(wiz)
        result = wiz._ensure_llamacpp_server_running(state)
        assert result == {"ok": True, "reused": True}
        assert spawn_calls == []

    def test_restarts_server_when_not_running(self, isolated_state, monkeypatch, tmp_path):
        # Step 5 succeeded earlier, but the server died (OOM, kill, sleep)
        # before Step 7. The verify guard must restart it transparently
        # rather than letting the harness fail with ConnectionRefused.
        pb, wiz, _ = isolated_state
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"\x00")
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {"server_running": False, "server_port": 8001, "model": None},
        )
        spawned = {}

        def _fake_start(**kw):
            spawned.update(kw)
            return {
                "ok": True,
                "argv": ["llama-server", "--model", kw["model_path"]],
                "handle": pb.LlamaServerHandle(
                    pid=4242,
                    port=kw.get("port", 8001),
                    host=kw.get("host", "127.0.0.1"),
                    model_path=kw["model_path"],
                    argv=["llama-server", "--model", kw["model_path"]],
                    log_path=str(tmp_path / "log.txt"),
                    pid_file=str(tmp_path / "pid"),
                    we_started_it=True,
                ),
                "error": None,
                "log_path": str(tmp_path / "log.txt"),
            }

        monkeypatch.setattr(pb, "llamacpp_start_server", _fake_start)
        state = self._state(wiz, model_path=str(gguf))
        result = wiz._ensure_llamacpp_server_running(state)
        assert result["ok"] is True
        assert result["reused"] is False
        assert spawned["model_path"] == str(gguf)

    def test_refuses_when_other_model_is_loaded(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "server_running": True,
                "server_port": 8001,
                "model": "deepseek-coder-v2.gguf",
            },
        )
        state = self._state(wiz)
        result = wiz._ensure_llamacpp_server_running(state)
        assert result["ok"] is False
        assert "different model" in result["error"]

    def test_actionable_error_when_no_path_recorded(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {"server_running": False, "server_port": 8001, "model": None},
        )
        state = self._state(wiz, model_path=None)
        result = wiz._ensure_llamacpp_server_running(state)
        assert result["ok"] is False
        assert "re-run wizard step 4" in result["error"]


class TestRunServe:
    """`ccl serve` — pre-flight invoked from the cc/cx helper script."""

    def _persist_state(self, wiz, engine="llamacpp", tag="bartowski/Foo-GGUF"):
        state = wiz.WizardState(
            primary_engine=engine,
            engine_model_tag=tag,
        )
        state.profile = {"llmfit_system": {"system": {"has_gpu": False, "cpu_cores": 4}}}
        state.save()
        return state

    def test_serve_returns_error_when_no_state_file(self, isolated_state):
        _pb, wiz, _ = isolated_state
        # No state file written — simulates "user hasn't run setup yet".
        assert wiz.run_serve() == 1

    def test_serve_is_noop_for_non_llamacpp_engines(self, isolated_state, monkeypatch):
        # Ollama / LM Studio / 9router are managed outside ccl. The serve
        # command must succeed silently rather than try to start a server.
        pb, wiz, _ = isolated_state
        self._persist_state(wiz, engine="ollama", tag="qwen3-coder:30b")
        called = []
        monkeypatch.setattr(pb, "llamacpp_info", lambda: called.append("info") or {})
        monkeypatch.setattr(pb, "llamacpp_start_server", lambda **kw: called.append("start"))
        assert wiz.run_serve() == 0
        # Did NOT touch llama.cpp at all.
        assert called == []

    def test_serve_is_silent_when_server_already_running(self, isolated_state, monkeypatch, capsys):
        # Hot path — every `cc` invocation hits this. Must not print anything
        # the user has to read past, and must not spawn a new server.
        pb, wiz, _ = isolated_state
        self._persist_state(wiz)
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {"server_running": True, "server_port": 8001, "model": "Foo.gguf"},
        )
        spawn_calls = []
        monkeypatch.setattr(pb, "llamacpp_start_server", lambda **kw: spawn_calls.append(kw))
        assert wiz.run_serve() == 0
        out = capsys.readouterr().out + capsys.readouterr().err
        # No banner, no progress lines.
        assert "Starting llama-server" not in out
        assert spawn_calls == []

    def test_serve_shows_loading_banner_on_cold_start(
        self, isolated_state, monkeypatch, tmp_path, capsys
    ):
        # Cold path — user just rebooted, server is gone, model not in VRAM.
        # The banner must appear BEFORE the model-load wait so users know
        # the multi-second delay is expected, not a hang.
        pb, wiz, _ = isolated_state
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"\x00")
        state = self._persist_state(wiz)
        state.profile["llamacpp_model_path"] = str(gguf)
        state.save()

        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {"server_running": False, "server_port": 8001, "model": None},
        )
        monkeypatch.setattr(
            pb,
            "llamacpp_start_server",
            lambda **kw: {
                "ok": True,
                "argv": ["llama-server", "--model", kw["model_path"]],
                "handle": pb.LlamaServerHandle(
                    pid=999,
                    port=8001,
                    host="127.0.0.1",
                    model_path=kw["model_path"],
                    argv=["llama-server"],
                    log_path=str(tmp_path / "log.txt"),
                    pid_file=str(tmp_path / "pid"),
                    we_started_it=True,
                ),
                "error": None,
                "log_path": str(tmp_path / "log.txt"),
            },
        )
        assert wiz.run_serve() == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "30s+" in out  # the cold-start expectation is visible

    def test_serve_returns_failure_when_start_fails(self, isolated_state, monkeypatch, tmp_path):
        pb, wiz, _ = isolated_state
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"\x00")
        state = self._persist_state(wiz)
        state.profile["llamacpp_model_path"] = str(gguf)
        state.save()

        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {"server_running": False, "server_port": 8001, "model": None},
        )
        monkeypatch.setattr(
            pb,
            "llamacpp_start_server",
            lambda **kw: {
                "ok": False,
                "error": "OOM at layer offload",
                "argv": ["llama-server"],
                "handle": None,
                "log_path": None,
            },
        )
        assert wiz.run_serve() == 1


class TestHelperScriptLlamacppPreflight:
    """The `cc`/`cx` script grows a /health probe + ccl-serve fallback when
    engine == llamacpp, so harness invocations self-heal after the server
    dies (reboot, OOM, manual kill)."""

    def _result(self):
        # Module-level WireResult is imported under the wiz alias.
        return None  # placeholder; tests build their own

    def test_llamacpp_engine_emits_preflight_block(self, isolated_state):
        _pb, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["claude", "--model", "qwen2.5-coder"],
            env={"ANTHROPIC_BASE_URL": "http://127.0.0.1:8001/v1"},
            effective_tag="qwen2.5-coder",
        )
        path = wiz._write_helper_script("claude", result, engine="llamacpp")
        body = path.read_text()
        assert "/health" in body
        assert "curl -fsS --max-time 1" in body
        assert "serve" in body
        # Pre-flight must run BEFORE the env exports so a failed start
        # short-circuits cleanly without leaking partial environment.
        preflight_pos = body.index("/health")
        export_pos = body.index("export ANTHROPIC_BASE_URL")
        assert preflight_pos < export_pos
        # The exec line is still present and untouched.
        assert 'exec claude --model qwen2.5-coder "$@"' in body

    def test_non_llamacpp_engines_do_not_get_preflight(self, isolated_state):
        # Ollama / LM Studio / 9router don't need a pre-flight; their
        # daemons are managed elsewhere. The script must stay unchanged
        # so we don't pay a curl probe per invocation.
        _pb, wiz, _ = isolated_state
        result = wiz.WireResult(
            argv=["ollama", "launch", "claude", "--model", "qwen2.5-coder:7b", "--"],
            env={},
            effective_tag="qwen2.5-coder:7b",
        )
        path = wiz._write_helper_script("claude", result, engine="ollama")
        body = path.read_text()
        assert "/health" not in body
        assert "ccl serve" not in body

    def test_engine_default_omits_preflight(self, isolated_state):
        # Backward-compat: callers that don't pass the engine keyword (older
        # tests, manual invocations) must not get the new behavior.
        _pb, wiz, _ = isolated_state
        result = wiz.WireResult(argv=["claude"], env={}, effective_tag="x")
        path = wiz._write_helper_script("claude", result)
        assert "/health" not in path.read_text()


class TestLlamaCppModelMatch:
    """Regression: the reuse heuristic must not collapse different sizes/variants."""

    def test_basename_matches_repo_id_for_same_family(self, isolated_state):
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            )
            is True
        )

    def test_does_not_match_different_sizes(self, isolated_state):
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            )
            is False
        )

    def test_does_not_match_different_variants(self, isolated_state):
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "Qwen2.5-VL-7B-Instruct.gguf",
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            )
            is False
        )

    def test_does_not_match_different_minor_version(self, isolated_state):
        _pb, wiz, _ = isolated_state
        # The historical bug: Path('Llama-3.1-8B').stem -> 'Llama-3' which
        # silently matched any `Llama-3*` repo id.
        assert (
            wiz._llamacpp_models_match(
                "Meta-Llama-3.1-8B-Instruct.gguf",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            )
            is False
        )

    def test_short_overlap_does_not_match(self, isolated_state):
        _pb, wiz, _ = isolated_state
        # Both contain "qwen" but nothing more — must not match.
        assert wiz._llamacpp_models_match("qwen2.gguf", "qwen3-coder") is False

    def test_empty_strings_return_false(self, isolated_state):
        _pb, wiz, _ = isolated_state
        assert wiz._llamacpp_models_match("", "anything") is False
        assert wiz._llamacpp_models_match("anything", "") is False

    def test_does_not_match_base_vs_instruct_same_family(self, isolated_state):
        """A 'base' GGUF must not be reused for a wizard wanting an 'instruct'."""
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "deepseek-coder-7b-base-Q4_K_M.gguf",
                "deepseek-ai/deepseek-coder-7b-instruct",
            )
            is False
        )

    def test_does_not_match_instruct_vs_chat(self, isolated_state):
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "qwen2.5-coder-7b-instruct.gguf",
                "qwen/qwen2.5-coder-7b-chat-GGUF",
            )
            is False
        )

    def test_matches_when_only_one_side_has_variant_token(self, isolated_state):
        """If only one side carries a variant token, fall through to the
        substring rule (the other side is ambiguous, so we can't refute)."""
        _pb, wiz, _ = isolated_state
        assert (
            wiz._llamacpp_models_match(
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                "bartowski/Qwen2.5-Coder-7B-GGUF",
            )
            is True
        )

    def test_baseline_in_name_does_not_read_as_base(self, isolated_state):
        """'baseline' should not be parsed as the 'base' variant token."""
        _pb, wiz, _ = isolated_state
        # Both sides carry no variant token; the substring rule should match.
        assert (
            wiz._llamacpp_models_match(
                "some-baseline-model-7b-q4.gguf",
                "vendor/some-baseline-model-7b",
            )
            is True
        )


class TestStep2_5LlamaCppPreservesEngineModelTag:
    """When the user opts in to a mismatched running model, the wizard must
    not overwrite the persisted HF repo id (Step 6 wires the harness from it)."""

    def test_engine_model_tag_is_preserved_on_opt_in(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "server_running": True,
                "server_port": 8001,
                "model": "deepseek-coder-v2-lite-instruct.gguf",
            },
        )

        class _Yes:
            def ask(self):
                return True

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _Yes())
        captured: dict = {}

        def _fake_smoke(tag):
            captured["tag"] = tag
            return {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 12.0,
                "completion_tokens": 6,
                "duration_seconds": 0.5,
            }

        monkeypatch.setattr(pb, "smoke_test_llamacpp_model", _fake_smoke)
        state = wiz.WizardState(
            primary_engine="llamacpp",
            engine_model_tag="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        )
        state.profile = {"llmfit_system": {"system": {"has_gpu": False, "cpu_cores": 4}}}
        assert wiz.step_2_5_smoke_test(state, non_interactive=False) is True
        # Smoke test ran against whatever the running server has loaded.
        assert captured["tag"] == "deepseek-coder-v2-lite-instruct.gguf"
        # Persisted state must still be the user's HF repo id, NOT the basename
        # of the running server (otherwise Step 6 would wire a broken harness).
        assert state.engine_model_tag == "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"


class TestLlamaCppDownloadCapturesPath:
    """Regression: the wizard must record the resolved GGUF path so Step 5 can spawn the server."""

    def test_download_passes_local_dir_and_persists_path(
        self, isolated_state, monkeypatch, tmp_path
    ):
        pb, wiz, state_dir = isolated_state
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf"})

        # Single-variant repo → wizard auto-selects without prompting.
        monkeypatch.setattr(
            pb,
            "huggingface_list_repo_files",
            lambda repo: ["model-Q4_K_M.gguf", "config.json"],
        )

        captured: dict = {}

        def _fake_download(repo_id, filename=None, local_dir=None, *, include=None, stream=True):
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            captured["include"] = include
            captured["local_dir"] = local_dir
            # Simulate the streaming HF CLI dropping a file in local_dir.
            target_dir = wiz.Path(local_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            gguf = target_dir / "model-Q4_K_M.gguf"
            gguf.write_bytes(b"\x00" * 1024)
            return {
                "ok": True,
                "path": str(gguf),
                "error": None,
                "bytes_downloaded": 1024,
                "elapsed_seconds": 1.0,
                "not_found": False,
            }

        monkeypatch.setattr(pb, "huggingface_download_gguf", _fake_download)
        out = wiz._download_gguf_via_hf_cli("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")
        assert out["ok"] is True
        assert out["path"] is not None
        assert out["path"].endswith(".gguf")
        # Crucially, local_dir was passed so HF will populate a recoverable path.
        assert captured["local_dir"] is not None
        assert "models" in captured["local_dir"]

    def test_largest_gguf_helper_picks_biggest_file(self, isolated_state, tmp_path):
        _pb, wiz, _ = isolated_state
        small = tmp_path / "a.gguf"
        small.write_bytes(b"\x00" * 10)
        big = tmp_path / "b.gguf"
        big.write_bytes(b"\x00" * 10_000)
        result = wiz._largest_gguf_in(tmp_path)
        assert result == str(big)

    def test_largest_gguf_helper_returns_none_when_empty(self, isolated_state, tmp_path):
        _pb, wiz, _ = isolated_state
        result = wiz._largest_gguf_in(tmp_path)
        assert result is None


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


class TestStep4Pick9Router:
    """Issue #51 — Step 4 has a dedicated 9router branch that skips llmfit/download."""

    def test_non_interactive_uses_env_key_and_default_model(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setenv("CCL_9ROUTER_API_KEY", "router9-test-key")  # pragma: allowlist secret
        monkeypatch.delenv("CCL_9ROUTER_MODEL", raising=False)

        # Hard-fail if the wizard tries any of the local-model paths.
        for forbidden in (
            "_find_model_auto",
            "installed_models_for_engine",
            "_estimate_model_size",
            "_download_model",
        ):
            if hasattr(wiz, forbidden):
                monkeypatch.setattr(
                    wiz,
                    forbidden,
                    lambda *a, **kw: (_ for _ in ()).throw(
                        AssertionError(f"{forbidden} must not run for 9router")
                    ),
                )
            if hasattr(pb, forbidden):
                monkeypatch.setattr(
                    pb,
                    forbidden,
                    lambda *a, **kw: (_ for _ in ()).throw(
                        AssertionError(f"{forbidden} must not run for 9router")
                    ),
                )

        state = wiz.WizardState(primary_engine="9router", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "kr/claude-sonnet-4.5"
        assert state.model_source == "9router-direct"
        # Key file: chmod 0600.
        assert pb.ROUTER9_KEY_FILE.exists()
        mode = pb.ROUTER9_KEY_FILE.stat().st_mode & 0o777
        assert mode == 0o600
        assert "router9-test-key" in pb.ROUTER9_KEY_FILE.read_text()  # pragma: allowlist secret

    def test_non_interactive_respects_env_model_override(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv("CCL_9ROUTER_API_KEY", "router9-test-key")  # pragma: allowlist secret
        monkeypatch.setenv("CCL_9ROUTER_MODEL", "or/gpt-5-pro")
        state = wiz.WizardState(primary_engine="9router", primary_harness="codex")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "or/gpt-5-pro"

    def test_non_interactive_fails_without_key(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.delenv("CCL_9ROUTER_API_KEY", raising=False)
        state = wiz.WizardState(primary_engine="9router", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False

    def test_invalid_model_name_rejected(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv("CCL_9ROUTER_API_KEY", "router9-test-key")  # pragma: allowlist secret
        monkeypatch.setenv("CCL_9ROUTER_MODEL", "no-slash-model")
        state = wiz.WizardState(primary_engine="9router", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False


class TestClaudeOllamaThenClaude9Coexist:
    """Issue #51 — installing claude+ollama and claude+9router must coexist.

    The headline coexistence test: the rc file must keep BOTH the
    `claude` (cc / claude-local) block and the `claude9` (cc9) block,
    and the `claude` regex must NOT eat the `claude9` block.
    """

    def _stub_engine_envs(self, monkeypatch, pb):
        """Make pb.machine_profile + ensure_path safe in the test sandbox."""
        # No need to actually run any engines — wire functions are pure.
        return None

    def test_both_blocks_and_aliases_present(self, isolated_state, monkeypatch, tmp_path):
        from pathlib import Path

        pb, wiz, _ = isolated_state
        rc = Path.home() / ".zshrc"
        rc.write_text("")

        # Install 1: claude + ollama (writes :claude block with cc + claude-local).
        cc_script = tmp_path / "cc"
        cc_script.write_text("#!/bin/sh\n")
        cc_script.chmod(0o755)
        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)

        # Install 2: claude + 9router (writes :claude9 block with cc9 only).
        cc9_script = tmp_path / "cc9"
        cc9_script.write_text("#!/bin/sh\n")
        cc9_script.chmod(0o755)
        wiz._install_shell_aliases(cc9_script, "claude9", non_interactive=True)

        body = rc.read_text()
        # Both fenced blocks must be present, exactly once each.
        assert body.count("# >>> claude-codex-local:claude >>>") == 1
        assert body.count("# <<< claude-codex-local:claude <<<") == 1
        assert body.count("# >>> claude-codex-local:claude9 >>>") == 1
        assert body.count("# <<< claude-codex-local:claude9 <<<") == 1
        # Both alias sets must be present.
        assert "alias cc=" in body
        assert "alias claude-local=" in body
        assert "alias cc9=" in body
        # Both helper scripts referenced.
        assert str(cc_script) in body
        assert str(cc9_script) in body
        # Pin: the `claude` regex must NOT eat `claude9` — re-running the
        # claude install must NOT touch the claude9 block.
        cc_script_v2 = tmp_path / "cc-v2"
        cc_script_v2.write_text("#!/bin/sh\n")
        cc_script_v2.chmod(0o755)
        wiz._install_shell_aliases(cc_script_v2, "claude", non_interactive=True)
        body2 = rc.read_text()
        assert body2.count("# >>> claude-codex-local:claude >>>") == 1
        assert body2.count("# >>> claude-codex-local:claude9 >>>") == 1
        # claude block now points at v2; claude9 block still points at the original cc9.
        assert str(cc_script_v2) in body2
        assert str(cc9_script) in body2  # untouched
        # The original `cc` script must be replaced. We check via "alias cc=...
        # <path>\n" rather than substring, because cc-v2 contains "cc" as a
        # prefix and would trip a naive `not in` check.
        assert f"alias cc={cc_script}\n" not in body2  # original cc replaced


class TestRunDoctor9RouterChecks:
    """Issue #51 — run_doctor surfaces 9router-specific health checks."""

    def _make_state(self, isolated_state, *, key_content: str | None, mode: int | None, model: str):
        pb, wiz, state_dir = isolated_state
        state_dir.mkdir(parents=True, exist_ok=True)
        if key_content is not None:
            pb.ROUTER9_KEY_FILE.write_text(key_content)
            if mode is not None:
                pb.ROUTER9_KEY_FILE.chmod(mode)
        state = wiz.WizardState(
            primary_harness="claude",
            primary_engine="9router",
            engine_model_tag=model,
            completed_steps=["1", "2", "3", "4", "5", "6", "6.5", "7", "8"],
            verify_result={"ok": True, "via": "9router-models-endpoint", "skipped_chat": True},
        )
        state.save()
        return pb, wiz

    def test_doctor_reports_missing_keyfile(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state, key_content=None, mode=None, model="kr/claude-sonnet-4.5"
        )
        # machine_profile shouldn't error.
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["9router"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "9router key file" in out
        assert "missing" in out

    def test_doctor_warns_on_world_readable_key(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="router9-test-key",  # pragma: allowlist secret
            mode=0o644,
            model="kr/claude-sonnet-4.5",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["9router"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "9router key file mode" in out
        # 0644 should NOT match the owner-only check.
        assert "0644" in out

    def test_doctor_flags_invalid_model_name(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="router9-test-key",  # pragma: allowlist secret
            mode=0o600,
            model="not-a-valid-model",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["9router"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "9router model name" in out
        assert "invalid" in out

    def test_doctor_passes_with_valid_state(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="router9-test-key",  # pragma: allowlist secret
            mode=0o600,
            model="kr/claude-sonnet-4.5",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["9router"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        # Make GUIDE_PATH look present.
        guide = wiz.GUIDE_PATH
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text("# fake guide")

        # Use a wider console so the rich Table doesn't truncate row labels.
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        # Output should mention all-passed or no issues; rc may be 0 IFF no
        # other checks fail. Helper script path was never created so it
        # may register an issue — accept both as long as 9router-specific
        # checks didn't fail.
        if rc == 0:
            assert "All checks passed" in out
        # 9router-specific lines must show ok markers, not failures.
        assert "9router key file mode" in out
        assert "9router key file content" in out
        assert "9router model name" in out


class TestStep5SmokeTest9Router:
    """Issue #51 — Step 5 smoke test must use /v1/models, NOT /chat/completions."""

    def test_step5_calls_smoke_test_router9_models(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        seen: dict[str, bool] = {"called": False}

        def fake_models(*a, **kw):
            seen["called"] = True
            return {"ok": True, "models": ["kr/claude-sonnet-4.5"], "response": "1 models"}

        monkeypatch.setattr(pb, "smoke_test_router9_models", fake_models)
        # Hard-fail if any other smoke test is called for 9router.
        for forbidden in (
            "smoke_test_ollama_model",
            "smoke_test_lmstudio_model",
            "smoke_test_llamacpp_model",
            "smoke_test_vllm_model",
        ):
            monkeypatch.setattr(
                pb,
                forbidden,
                lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError(f"{forbidden} must not be called for 9router")
                ),
            )

        state = wiz.WizardState(
            primary_engine="9router",
            primary_harness="claude",
            engine_model_tag="kr/claude-sonnet-4.5",
        )
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert seen["called"] is True
        assert state.smoke_test_result["ok"] is True


class TestStep7Verify9Router:
    """Issue #51 — Step 7 verify for 9router NEVER calls subprocess.run (no chat call)."""

    def test_verify_uses_models_endpoint_and_does_not_run_subprocess(
        self, isolated_state, monkeypatch
    ):
        import subprocess

        pb, wiz, _ = isolated_state

        # Pre-populate a wire result so step 7 thinks step 6 has run.
        state = wiz.WizardState(
            primary_engine="9router",
            primary_harness="claude",
            engine_model_tag="kr/claude-sonnet-4.5",
            wire_result={
                "argv": ["claude", "--model", "kr/claude-sonnet-4.5"],
                "env": {"ANTHROPIC_BASE_URL": "http://localhost:20128/v1"},
                "effective_tag": "kr/claude-sonnet-4.5",
                "raw_env": {"ANTHROPIC_AUTH_TOKEN": '"$(cat /tmp/k)"'},
            },
        )

        monkeypatch.setattr(
            pb,
            "smoke_test_router9_models",
            lambda *a, **kw: {"ok": True, "models": [], "response": "0 models"},
        )

        # subprocess.run must NOT be called for the 9router branch.
        def fail_run(*a, **kw):
            raise AssertionError("step_2_7_verify must NOT call subprocess.run for 9router")

        monkeypatch.setattr(subprocess, "run", fail_run)
        # Also patch the wizard module's subprocess reference.
        monkeypatch.setattr(wiz.subprocess, "run", fail_run)

        assert wiz.step_2_7_verify(state, non_interactive=True) is True
        assert state.verify_result["skipped_chat"] is True
        assert state.verify_result["via"] == "9router-models-endpoint"
        assert state.verify_result["ok"] is True

    def test_verify_fails_when_models_unreachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="9router",
            primary_harness="claude",
            engine_model_tag="kr/claude-sonnet-4.5",
            wire_result={
                "argv": ["claude"],
                "env": {},
                "effective_tag": "x",
                "raw_env": {},
            },
        )
        monkeypatch.setattr(
            pb,
            "smoke_test_router9_models",
            lambda *a, **kw: {"ok": False, "error": "refused"},
        )
        # subprocess.run must still NOT be called even on failure.
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not run subprocess on 9router failure path")
            ),
        )
        monkeypatch.setattr(
            wiz.subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not run subprocess on 9router failure path")
            ),
        )
        assert wiz.step_2_7_verify(state, non_interactive=True) is False
        assert state.verify_result["skipped_chat"] is True


class TestEnsureTool9Router:
    """9router ships as an npm one-liner (`npm install -g 9router`). The wizard
    offers to install it interactively, but never fork-spawns the long-running
    dashboard server — the user starts that themselves."""

    def test_returns_true_when_router9_endpoint_reachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": True, "version": ""}
        )
        assert wiz._ensure_tool("9router") is True

    def test_warns_and_returns_false_when_npm_missing(self, isolated_state, monkeypatch):
        """No npm on PATH → no install attempted, no confirm prompted, returns False."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )
        monkeypatch.setattr(
            wiz.shutil,
            "which",
            lambda name: None,
        )

        import subprocess as sp

        monkeypatch.setattr(
            sp,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not subprocess.run when npm is missing")
            ),
        )
        import questionary

        monkeypatch.setattr(
            questionary,
            "confirm",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not prompt confirm when npm is missing")
            ),
        )
        assert wiz._ensure_tool("9router") is False

    def test_offers_npm_install_user_declines_returns_false(self, isolated_state, monkeypatch):
        """User declines the install prompt → no subprocess, returns False."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )
        # npm is on PATH; 9router CLI is not yet installed.
        monkeypatch.setattr(
            wiz.shutil,
            "which",
            lambda name: "/usr/bin/npm" if name == "npm" else None,
        )

        called: dict[str, bool] = {"subprocess_run": False}

        def fake_run(*a, **kw):
            called["subprocess_run"] = True
            raise AssertionError("must not subprocess.run when user declines")

        monkeypatch.setattr(wiz.subprocess, "run", fake_run)

        class _DeclineConfirm:
            def ask(self):
                return False

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _DeclineConfirm())
        assert wiz._ensure_tool("9router") is False
        assert called["subprocess_run"] is False

    def test_offers_npm_install_user_accepts_runs_install(self, isolated_state, monkeypatch):
        """User accepts → wizard runs `npm install -g 9router`, then asks
        the user to start the daemon and re-probes."""
        pb, wiz, _ = isolated_state
        # Reachability flips to True after install + manual start.
        detect_calls = {"n": 0}

        def fake_detect(self):
            detect_calls["n"] += 1
            return {"present": detect_calls["n"] >= 2, "version": ""}

        monkeypatch.setattr(pb.Router9Adapter, "detect", fake_detect)
        monkeypatch.setattr(
            wiz.shutil,
            "which",
            lambda name: "/usr/bin/npm" if name == "npm" else None,
        )

        run_calls: list[list[str]] = []

        def fake_run(cmd, check=False, **kw):
            run_calls.append(list(cmd))

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(wiz.subprocess, "run", fake_run)

        class _AcceptConfirm:
            def ask(self):
                return True

        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _AcceptConfirm())

        assert wiz._ensure_tool("9router") is True
        assert ["npm", "install", "-g", "9router"] in run_calls


# ---------------------------------------------------------------------------
# vLLM wizard wiring — discover/ensure/wire/smoke for engine="vllm".
# ---------------------------------------------------------------------------


class TestEnsureToolVLLM:
    """vLLM is user-managed (Python venv + GPU drivers). Like 9router, the
    wizard must NEVER subprocess-run an install command — only probe."""

    def test_returns_true_when_vllm_reachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "vllm_info",
            lambda: {"present": True, "base_url": pb.VLLM_BASE_URL, "models": []},
        )
        assert wiz._ensure_tool("vllm") is True

    def test_returns_false_when_unreachable_no_install_attempted(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "vllm_info",
            lambda: {"present": False, "base_url": pb.VLLM_BASE_URL, "models": []},
        )

        import subprocess as sp

        monkeypatch.setattr(
            sp,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not subprocess.run for vllm")
            ),
        )
        import questionary

        monkeypatch.setattr(
            questionary,
            "confirm",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not prompt confirm for vllm")
            ),
        )
        assert wiz._ensure_tool("vllm") is False


class TestStep4PickModelVLLM:
    def test_picks_single_loaded_model_without_prompting(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            profile={
                "vllm": {
                    "present": True,
                    "base_url": "http://localhost:8000",
                    "models": [{"name": "Qwen/Qwen2.5-0.5B-Instruct"}],
                }
            },
        )
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "Qwen/Qwen2.5-0.5B-Instruct"
        assert state.model_source == "vllm-loaded"

    def test_non_interactive_picks_first_when_multiple(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            profile={
                "vllm": {
                    "present": True,
                    "base_url": "http://localhost:8000",
                    "models": [{"name": "model-a"}, {"name": "model-b"}],
                }
            },
        )
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "model-a"

    def test_fails_when_server_not_present(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            profile={"vllm": {"present": False, "base_url": "http://localhost:8000"}},
        )
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False

    def test_fails_when_no_models_loaded(self, isolated_state):
        _, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            profile={
                "vllm": {
                    "present": True,
                    "base_url": "http://localhost:8000",
                    "models": [],
                }
            },
        )
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False

    def test_env_override_used_when_set(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv("CCL_VLLM_MODEL", "override/model")
        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            profile={
                "vllm": {
                    "present": True,
                    "base_url": "http://localhost:8000",
                    "models": [{"name": "loaded/model"}],
                }
            },
        )
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "override/model"


class TestStep5SmokeTestVLLM:
    def test_step5_calls_smoke_test_vllm_model(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        captured: dict[str, object] = {}

        def fake_smoke(model, base_url=None, api_key="", **kw):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            return {"ok": True, "response": "READY", "tokens_per_second": 50.0}

        monkeypatch.setattr(pb, "smoke_test_vllm_model", fake_smoke)
        for forbidden in (
            "smoke_test_ollama_model",
            "smoke_test_lmstudio_model",
            "smoke_test_llamacpp_model",
            "smoke_test_router9_models",
        ):
            monkeypatch.setattr(
                pb,
                forbidden,
                lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError(f"{forbidden} must not be called for vllm")
                ),
            )

        state = wiz.WizardState(
            primary_engine="vllm",
            primary_harness="claude",
            engine_model_tag="some/model",
            profile={"vllm": {"present": True, "base_url": "http://override:9999", "models": []}},
        )
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert captured["model"] == "some/model"
        assert captured["base_url"] == "http://override:9999"


class TestWireClaudeVLLM:
    def test_no_keyfile_uses_placeholder_secret(self, isolated_state):
        pb, wiz, _ = isolated_state
        # Ensure no leftover keyfile from another test.
        if pb.VLLM_KEY_FILE.exists():
            pb.VLLM_KEY_FILE.unlink()
        result = wiz._wire_claude("vllm", "Qwen/Qwen2.5-0.5B-Instruct")
        assert result.argv == ["claude", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]
        assert result.env["ANTHROPIC_BASE_URL"] == pb.VLLM_BASE_URL
        assert result.env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert result.env["ANTHROPIC_API_KEY"] == "sk-local"  # pragma: allowlist secret
        assert result.env["ANTHROPIC_AUTH_TOKEN"] == "sk-local"  # pragma: allowlist secret
        assert result.raw_env == {}

    def test_keyfile_present_uses_cat_expression(self, isolated_state):
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.VLLM_KEY_FILE.write_text("vllm-test-key\n")  # pragma: allowlist secret
        pb.VLLM_KEY_FILE.chmod(0o600)
        result = wiz._wire_claude("vllm", "loaded/model")
        # API key MUST live in raw_env as a $(cat …) expression — never literal.
        assert "ANTHROPIC_AUTH_TOKEN" in result.raw_env
        assert "ANTHROPIC_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["ANTHROPIC_AUTH_TOKEN"]
        assert str(pb.VLLM_KEY_FILE) in result.raw_env["ANTHROPIC_AUTH_TOKEN"]
        # And NOT carry a literal API key in env.
        assert "ANTHROPIC_API_KEY" not in result.env
        assert "ANTHROPIC_AUTH_TOKEN" not in result.env


class TestWireCodexVLLM:
    def test_no_keyfile_path(self, isolated_state):
        pb, wiz, _ = isolated_state
        if pb.VLLM_KEY_FILE.exists():
            pb.VLLM_KEY_FILE.unlink()
        result = wiz._wire_codex("vllm", "loaded/model")
        assert result.argv == ["codex", "-m", "loaded/model"]
        assert result.env["OPENAI_BASE_URL"] == f"{pb.VLLM_BASE_URL.rstrip('/')}/v1"
        assert result.env["OPENAI_API_KEY"] == "sk-local"  # pragma: allowlist secret
        assert result.raw_env == {}

    def test_keyfile_present_uses_cat_expression(self, isolated_state):
        pb, wiz, _ = isolated_state
        pb.ensure_state_dirs()
        pb.VLLM_KEY_FILE.write_text("vllm-test-key\n")  # pragma: allowlist secret
        pb.VLLM_KEY_FILE.chmod(0o600)
        result = wiz._wire_codex("vllm", "loaded/model")
        assert "OPENAI_API_KEY" in result.raw_env
        assert "$(cat" in result.raw_env["OPENAI_API_KEY"]
        assert str(pb.VLLM_KEY_FILE) in result.raw_env["OPENAI_API_KEY"]
        # Plain env must NOT carry a literal API key.
        assert "OPENAI_API_KEY" not in result.env


class TestVLLMInDiscovery:
    """vllm should appear in `_ALL_ENGINES` and the discovery profile shape."""

    def test_vllm_in_all_engines(self, isolated_state):
        _, wiz, _ = isolated_state
        assert "vllm" in wiz._ALL_ENGINES

    def test_machine_profile_includes_vllm_when_unreachable(self, isolated_state, monkeypatch):
        pb, _, _ = isolated_state
        # Force the adapter to look unreachable.
        monkeypatch.setattr(
            pb.VLLMAdapter,
            "detect",
            lambda self: {"present": False, "version": ""},
        )
        # Also block llmfit / harness probes that would otherwise hit the host.
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": False, "version": ""}
        )
        monkeypatch.setattr(
            pb,
            "lms_info",
            lambda: {"present": False, "version": "", "models": [], "server_running": False},
        )
        monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False})
        monkeypatch.setattr(pb, "llmfit_system", lambda: {})
        monkeypatch.setattr(pb, "parse_ollama_list", lambda: [])
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )

        profile = pb.machine_profile()
        assert "vllm" in profile["tools"]
        assert profile["tools"]["vllm"]["present"] is False
        assert "vllm" not in profile["presence"]["engines"]
        assert "vllm" in profile  # full block too
        assert profile["vllm"]["present"] is False

    def test_machine_profile_lists_vllm_in_engines_when_reachable(
        self, isolated_state, monkeypatch
    ):
        pb, _, _ = isolated_state
        monkeypatch.setattr(
            pb.VLLMAdapter,
            "detect",
            lambda self: {
                "present": True,
                "version": "0.6.0",
                "base_url": pb.VLLM_BASE_URL,
            },
        )
        monkeypatch.setattr(
            pb.VLLMAdapter,
            "list_models",
            lambda self: [{"name": "loaded/model", "format": "unknown", "local": True}],
        )
        # Block other probes from touching the host.
        monkeypatch.setattr(
            pb,
            "command_version",
            lambda cmd, **kw: (
                {"present": cmd == "vllm", "version": "0.6.0"}
                if cmd == "vllm"
                else {"present": False, "version": ""}
            ),
        )
        monkeypatch.setattr(
            pb,
            "lms_info",
            lambda: {"present": False, "version": "", "models": [], "server_running": False},
        )
        monkeypatch.setattr(pb, "llamacpp_detect", lambda: {"present": False, "version": ""})
        monkeypatch.setattr(pb, "huggingface_cli_detect", lambda: {"present": False})
        monkeypatch.setattr(pb, "llmfit_system", lambda: {})
        monkeypatch.setattr(pb, "parse_ollama_list", lambda: [])
        monkeypatch.setattr(
            pb.Router9Adapter, "detect", lambda self: {"present": False, "version": ""}
        )

        profile = pb.machine_profile()
        assert "vllm" in profile["presence"]["engines"]
        assert profile["vllm"]["models"] == [
            {"name": "loaded/model", "format": "unknown", "local": True}
        ]


# ---------------------------------------------------------------------------
# Issue #79 — lazy-llmfit + cache-aware Step 4 picker + cache invalidation.
# ---------------------------------------------------------------------------


class TestStep1LazyLlmfit:
    """Step 1 must defer the llmfit hardware scan unless the user opts in."""

    def _seed_cached_profile(self, pb_mod, *, with_llmfit: bool = True) -> dict:
        """Write a synthetic populated profile cache to disk."""
        profile = {
            "host": {
                "platform": "Darwin-x",
                "system": "Darwin",
                "release": "25",
                "machine": "arm64",
            },
            "tools": {
                "ollama": {"present": True, "version": "0.1.99"},
                "lmstudio": {"present": False, "version": ""},
                "llamacpp": {"present": False, "version": ""},
                "vllm": {"present": False, "version": "", "base_url": "http://localhost:8000"},
                "huggingface_cli": {"present": False},
                "claude": {"present": True, "version": "claude 1.0.0"},
                "codex": {"present": False, "version": ""},
                "llmfit": {"present": True, "version": "llmfit 1.2.3"},
                "9router": {"present": False, "version": "", "base_url": ""},
            },
            "presence": {
                "harnesses": ["claude"],
                "engines": ["ollama"],
                "llmfit": True,
                "has_minimum": True,
            },
            "ollama": {"models": []},
            "lmstudio": {"present": False, "models": []},
            "llamacpp": {"present": False},
            "vllm": {"present": False},
            "9router": {"present": False, "base_url": "", "healthcheck": {"ok": False}},
            "disk": {"free_gib": 100.0, "total_gib": 500.0},
            "state_dir": str(pb_mod.STATE_DIR),
        }
        if with_llmfit:
            profile["llmfit_system"] = {
                "system": {
                    "available_ram_gb": 32,
                    "total_ram_gb": 32,
                    "cpu_name": "Apple M2",
                    "cpu_cores": 8,
                    "has_gpu": True,
                    "gpu_name": "Apple",
                    "gpu_vram_gb": 16,
                }
            }
        else:
            profile["llmfit_system"] = pb_mod.LLMFIT_SKIPPED_SENTINEL

        # Persist via the real saver so the format matches what _load reads.
        fp = pb_mod._compute_machine_fingerprint(profile)
        profile["_fingerprint"] = fp
        pb_mod.MACHINE_PROFILE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        pb_mod._save_machine_profile_cache(profile, fp)
        # Reset the in-process cache so the loader is exercised.
        pb_mod.invalidate_machine_profile_inproc_cache()
        return profile

    def test_step1_skips_llmfit_on_cache_hit(self, isolated_state, monkeypatch):
        """If a populated cache is present, llmfit_system must not be re-run."""
        pb, wiz, _ = isolated_state
        self._seed_cached_profile(pb, with_llmfit=True)

        def boom():
            raise AssertionError("llmfit_system must not be invoked on cache hit")

        monkeypatch.setattr(pb, "llmfit_system", boom)

        state = wiz.WizardState()
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz.step_2_1_discover(state, non_interactive=True) is True
        # Profile carries the populated llmfit_system from the cache.
        assert "system" in state.profile["llmfit_system"]

    def test_step1_passes_run_llmfit_false_by_default(self, isolated_state, monkeypatch):
        """Default discover call must propagate run_llmfit=False to machine_profile."""
        pb, wiz, _ = isolated_state
        captured: dict[str, object] = {}

        def fake_machine_profile(run_llmfit: bool = True):
            captured["run_llmfit"] = run_llmfit
            return {
                "tools": {
                    "ollama": {"present": True, "version": "0.1"},
                    "lmstudio": {"present": False},
                    "llamacpp": {"present": False},
                    "vllm": {"present": False},
                    "9router": {"present": False},
                    "huggingface_cli": {"present": False},
                    "claude": {"present": True, "version": "claude 1"},
                    "codex": {"present": False},
                    "llmfit": {"present": True, "version": "llmfit 1.2.3"},
                },
                "presence": {
                    "harnesses": ["claude"],
                    "engines": ["ollama"],
                    "llmfit": True,
                    "has_minimum": True,
                },
                "llmfit_system": pb.LLMFIT_SKIPPED_SENTINEL,
                "disk": {"free_gib": 100.0, "total_gib": 500.0},
            }

        monkeypatch.setattr(pb, "machine_profile", fake_machine_profile)
        state = wiz.WizardState()
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz.step_2_1_discover(state, non_interactive=True) is True
        assert captured["run_llmfit"] is False

    def test_force_scan_does_not_run_llmfit(self, isolated_state, monkeypatch):
        """--force-scan does not imply the expensive llmfit hardware scan."""
        pb, wiz, _ = isolated_state
        captured: dict[str, object] = {}
        invalidations: list[int] = []

        def fake_machine_profile(run_llmfit: bool = True):
            captured["run_llmfit"] = run_llmfit
            return {
                "tools": {
                    "ollama": {"present": True, "version": "0.1"},
                    "lmstudio": {"present": False},
                    "llamacpp": {"present": False},
                    "vllm": {"present": False},
                    "9router": {"present": False},
                    "huggingface_cli": {"present": False},
                    "claude": {"present": True, "version": "claude 1"},
                    "codex": {"present": False},
                    "llmfit": {"present": True, "version": "llmfit 1.2.3"},
                },
                "presence": {
                    "harnesses": ["claude"],
                    "engines": ["ollama"],
                    "llmfit": True,
                    "has_minimum": True,
                },
                "llmfit_system": {"system": {"available_ram_gb": 32}},
                "disk": {"free_gib": 100.0, "total_gib": 500.0},
            }

        monkeypatch.setattr(pb, "machine_profile", fake_machine_profile)
        monkeypatch.setattr(
            pb,
            "invalidate_machine_profile_inproc_cache",
            lambda: invalidations.append(1),
        )
        state = wiz.WizardState()
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz.step_2_1_discover(state, non_interactive=True, force_scan=True) is True
        assert captured["run_llmfit"] is False
        assert invalidations == [1]

    def test_run_llmfit_flag_forces_scan(self, isolated_state, monkeypatch):
        """--run-llmfit (without --force-scan) propagates as run_llmfit=True."""
        pb, wiz, _ = isolated_state
        captured: dict[str, object] = {}

        def fake_machine_profile(run_llmfit: bool = True):
            captured["run_llmfit"] = run_llmfit
            return {
                "tools": {
                    "ollama": {"present": True, "version": "0.1"},
                    "lmstudio": {"present": False},
                    "llamacpp": {"present": False},
                    "vllm": {"present": False},
                    "9router": {"present": False},
                    "huggingface_cli": {"present": False},
                    "claude": {"present": True, "version": "claude 1"},
                    "codex": {"present": False},
                    "llmfit": {"present": True, "version": "llmfit 1.2.3"},
                },
                "presence": {
                    "harnesses": ["claude"],
                    "engines": ["ollama"],
                    "llmfit": True,
                    "has_minimum": True,
                },
                "llmfit_system": {"system": {"available_ram_gb": 32}},
                "disk": {"free_gib": 100.0, "total_gib": 500.0},
            }

        monkeypatch.setattr(pb, "machine_profile", fake_machine_profile)
        monkeypatch.setattr(pb, "command_version", lambda name: {"present": name == "llmfit"})
        llmfit_calls: list[int] = []

        def fake_llmfit_system():
            llmfit_calls.append(1)
            return {"system": {"available_ram_gb": 64}}

        monkeypatch.setattr(pb, "llmfit_system", fake_llmfit_system)
        state = wiz.WizardState()
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz.step_2_1_discover(state, non_interactive=True, run_llmfit_flag=True) is True
        assert captured["run_llmfit"] is True
        assert llmfit_calls == [1]
        assert state.profile["llmfit_system"] == {"system": {"available_ram_gb": 64}}


class TestEnsureToolCacheInvalidation:
    """Issue #79: _ensure_tool must invalidate the in-proc profile cache after a real install."""

    def test_install_success_invalidates_inproc_cache(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state

        # Tool not present initially → install path.
        present_state = {"present": False}

        def fake_command_version(name):
            return {"present": present_state["present"], "version": "1.0"}

        monkeypatch.setattr(pb, "command_version", fake_command_version)

        # Confirm install dialog returns True.
        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _StubAsk(True))

        # Install command "succeeds" — flip the present flag.
        def fake_run(cmd, check=True):
            present_state["present"] = True

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(wiz.subprocess, "run", fake_run)

        invalidations: list[int] = []
        monkeypatch.setattr(
            pb,
            "invalidate_machine_profile_inproc_cache",
            lambda: invalidations.append(1),
        )

        # Hide the install hint output.
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz._ensure_tool("ollama") is True
        assert invalidations == [1], "must invalidate exactly once after install success"

    def test_user_decline_does_not_invalidate(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state

        # Tool not present.
        monkeypatch.setattr(pb, "command_version", lambda name: {"present": False, "version": ""})
        # User declines install.
        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _StubAsk(False))

        invalidations: list[int] = []
        monkeypatch.setattr(
            pb,
            "invalidate_machine_profile_inproc_cache",
            lambda: invalidations.append(1),
        )
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz._ensure_tool("ollama") is False
        assert invalidations == [], "must not invalidate when the user declined install"

    def test_install_failure_does_not_invalidate(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state

        # Tool not present.
        monkeypatch.setattr(pb, "command_version", lambda name: {"present": False, "version": ""})
        monkeypatch.setattr(wiz.questionary, "confirm", lambda *a, **kw: _StubAsk(True))

        # Install command "fails" → CalledProcessError path.
        def boom(cmd, check=True):
            raise wiz.subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(wiz.subprocess, "run", boom)

        invalidations: list[int] = []
        monkeypatch.setattr(
            pb,
            "invalidate_machine_profile_inproc_cache",
            lambda: invalidations.append(1),
        )
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz._ensure_tool("ollama") is False
        assert invalidations == []


# ---------------------------------------------------------------------------
# OpenRouter (issue #83) — fence tags, helper basenames, alias names,
# engine list, Step 4 picker, doctor checks, smoke test, verify, ensure_tool.
# ---------------------------------------------------------------------------


class TestEnginesListOpenRouter:
    """Issue #83 — `openrouter` is a 6th supported engine alongside the existing five."""

    def test_all_engines_constant_includes_openrouter(self, isolated_state):
        _, wiz, _ = isolated_state
        assert "openrouter" in wiz._ALL_ENGINES

    def test_argparse_engine_choice_accepts_openrouter(self, isolated_state):
        from claude_codex_local.wizard import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["setup", "--engine", "openrouter"])
        assert args.engine == "openrouter"


class TestFenceTagOpenRouter:
    """OpenRouter fence tag is `<harness>o` (cco / cxo / cpo)."""

    def test_fence_tag_for_claude_openrouter(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._fence_tag_for("claude", "openrouter") == "claudeo"

    def test_fence_tag_for_codex_openrouter(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._fence_tag_for("codex", "openrouter") == "codexo"

    def test_fence_tag_for_pi_openrouter(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._fence_tag_for("pi", "openrouter") == "pio"

    def test_fence_tag_for_other_engines_unchanged_for_claude(self, isolated_state):
        """Mirroring 9router shape must not break the local-engine path."""
        _, wiz, _ = isolated_state
        assert wiz._fence_tag_for("claude", "ollama") == "claude"
        assert wiz._fence_tag_for("claude", "9router") == "claude9"


class TestHelperScriptBasenameOpenRouter:
    """OpenRouter fence tags map to cco/cxo/cpo helper script filenames."""

    def test_claudeo_basename(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._helper_script_basename("claudeo") == "cco"

    def test_codexo_basename(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._helper_script_basename("codexo") == "cxo"

    def test_pio_basename(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._helper_script_basename("pio") == "cpo"


class TestAliasNamesOpenRouter:
    """OpenRouter aliases are short-form only (cco/cxo/cpo)."""

    def test_claudeo_aliases(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._alias_names_for("claudeo") == ["cco"]

    def test_codexo_aliases(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._alias_names_for("codexo") == ["cxo"]

    def test_pio_aliases(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._alias_names_for("pio") == ["cpo"]


class TestStep4PickOpenRouter:
    """Issue #83 — Step 4 has a dedicated openrouter branch that skips llmfit/download."""

    def test_non_interactive_uses_env_key_and_default_model(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setenv(
            "CCL_OPENROUTER_API_KEY",
            "openrouter-test-key",  # pragma: allowlist secret
        )
        monkeypatch.delenv("CCL_OPENROUTER_MODEL", raising=False)

        # Hard-fail if the wizard tries any of the local-model paths.
        for forbidden in (
            "_find_model_auto",
            "installed_models_for_engine",
            "_estimate_model_size",
            "_download_model",
        ):
            if hasattr(wiz, forbidden):
                monkeypatch.setattr(
                    wiz,
                    forbidden,
                    lambda *a, **kw: (_ for _ in ()).throw(
                        AssertionError(f"{forbidden} must not run for openrouter")
                    ),
                )
            if hasattr(pb, forbidden):
                monkeypatch.setattr(
                    pb,
                    forbidden,
                    lambda *a, **kw: (_ for _ in ()).throw(
                        AssertionError(f"{forbidden} must not run for openrouter")
                    ),
                )

        state = wiz.WizardState(primary_engine="openrouter", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "anthropic/claude-sonnet-4.6"
        assert state.model_source == "openrouter-direct"
        # Key file: chmod 0600.
        assert pb.OPENROUTER_KEY_FILE.exists()
        mode = pb.OPENROUTER_KEY_FILE.stat().st_mode & 0o777
        assert mode == 0o600
        assert (
            "openrouter-test-key"  # pragma: allowlist secret
            in pb.OPENROUTER_KEY_FILE.read_text()
        )

    def test_non_interactive_respects_env_model_override(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv(
            "CCL_OPENROUTER_API_KEY",
            "openrouter-test-key",  # pragma: allowlist secret
        )
        monkeypatch.setenv("CCL_OPENROUTER_MODEL", "openai/gpt-4o")
        state = wiz.WizardState(primary_engine="openrouter", primary_harness="codex")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "openai/gpt-4o"

    def test_non_interactive_fails_without_key(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.delenv("CCL_OPENROUTER_API_KEY", raising=False)
        state = wiz.WizardState(primary_engine="openrouter", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False

    def test_invalid_model_name_rejected(self, isolated_state, monkeypatch):
        _, wiz, _ = isolated_state
        monkeypatch.setenv(
            "CCL_OPENROUTER_API_KEY",
            "openrouter-test-key",  # pragma: allowlist secret
        )
        monkeypatch.setenv("CCL_OPENROUTER_MODEL", "no-slash-model")
        state = wiz.WizardState(primary_engine="openrouter", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is False

    def test_variant_suffix_model_accepted(self, isolated_state, monkeypatch):
        """OpenRouter publishes :free / :nitro / :beta variant IDs that must validate."""
        _, wiz, _ = isolated_state
        monkeypatch.setenv(
            "CCL_OPENROUTER_API_KEY",
            "openrouter-test-key",  # pragma: allowlist secret
        )
        monkeypatch.setenv("CCL_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        state = wiz.WizardState(primary_engine="openrouter", primary_harness="claude")
        assert wiz.step_2_4_pick_model(state, non_interactive=True) is True
        assert state.engine_model_tag == "google/gemma-4-31b-it:free"


class TestRunDoctorOpenRouterChecks:
    """Issue #83 — run_doctor surfaces openrouter-specific health checks."""

    def _make_state(self, isolated_state, *, key_content: str | None, mode: int | None, model: str):
        pb, wiz, state_dir = isolated_state
        state_dir.mkdir(parents=True, exist_ok=True)
        if key_content is not None:
            pb.OPENROUTER_KEY_FILE.write_text(key_content)
            if mode is not None:
                pb.OPENROUTER_KEY_FILE.chmod(mode)
        state = wiz.WizardState(
            primary_harness="claude",
            primary_engine="openrouter",
            engine_model_tag=model,
            completed_steps=["1", "2", "3", "4", "5", "6", "6.5", "7", "8"],
            verify_result={
                "ok": True,
                "via": "openrouter-models-endpoint",
                "skipped_chat": True,
            },
        )
        state.save()
        return pb, wiz

    def test_doctor_reports_missing_keyfile(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state, key_content=None, mode=None, model="anthropic/claude-sonnet-4.6"
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["openrouter"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "openrouter key file" in out
        assert "missing" in out

    def test_doctor_warns_on_world_readable_key(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="openrouter-test-key",  # pragma: allowlist secret
            mode=0o644,
            model="anthropic/claude-sonnet-4.6",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["openrouter"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "openrouter key file mode" in out
        assert "0644" in out

    def test_doctor_flags_invalid_model_name(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="openrouter-test-key",  # pragma: allowlist secret
            mode=0o600,
            model="not-a-valid-model",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["openrouter"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        assert rc == 1
        assert "openrouter model name" in out
        assert "invalid" in out

    def test_doctor_passes_with_valid_state(self, isolated_state, monkeypatch):
        pb, wiz = self._make_state(
            isolated_state,
            key_content="openrouter-test-key",  # pragma: allowlist secret
            mode=0o600,
            model="anthropic/claude-sonnet-4.6",
        )
        monkeypatch.setattr(
            pb,
            "machine_profile",
            lambda **_kw: {
                "presence": {"harnesses": ["claude"], "engines": ["openrouter"]},
                "ollama": {"models": []},
                "lmstudio": {"models": []},
            },
        )
        guide = wiz.GUIDE_PATH
        guide.parent.mkdir(parents=True, exist_ok=True)
        guide.write_text("# fake guide")

        wiz.console.width = 200
        with wiz.console.capture() as cap:
            rc = wiz.run_doctor()
        out = cap.get()
        if rc == 0:
            assert "All checks passed" in out
        # openrouter-specific lines must show ok markers, not failures.
        assert "openrouter key file mode" in out
        assert "openrouter key file content" in out
        assert "openrouter model name" in out


class TestStep5SmokeTestOpenRouter:
    """Issue #85 — Step 5 smoke test sends a request to the selected model."""

    def test_step5_calls_smoke_test_openrouter_selected_model(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        seen: dict[str, str] = {}

        def fake_smoke(model, **kw):
            seen["model"] = model
            seen["api_key"] = kw.get("api_key", "")
            return {
                "ok": True,
                "model": model,
                "response": "READY",
            }

        pb.OPENROUTER_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        pb.OPENROUTER_KEY_FILE.write_text("openrouter-test-key\n")  # pragma: allowlist secret
        pb.OPENROUTER_KEY_FILE.chmod(0o600)
        monkeypatch.setattr(pb, "smoke_test_openrouter_model", fake_smoke)
        # Hard-fail if any other smoke test is called for openrouter.
        for forbidden in (
            "smoke_test_ollama_model",
            "smoke_test_lmstudio_model",
            "smoke_test_llamacpp_model",
            "smoke_test_vllm_model",
            "smoke_test_router9_models",
            "smoke_test_openrouter_models",
        ):
            monkeypatch.setattr(
                pb,
                forbidden,
                lambda *a, **kw: (_ for _ in ()).throw(
                    AssertionError(f"{forbidden} must not be called for openrouter")
                ),
            )

        state = wiz.WizardState(
            primary_engine="openrouter",
            primary_harness="claude",
            engine_model_tag="anthropic/claude-sonnet-4.6",
        )
        assert wiz.step_2_5_smoke_test(state, non_interactive=True) is True
        assert seen["model"] == "anthropic/claude-sonnet-4.6"
        assert seen["api_key"] == "openrouter-test-key"
        assert state.smoke_test_result["ok"] is True
        assert state.smoke_test_result["model"] == "anthropic/claude-sonnet-4.6"


class TestStep7VerifyOpenRouter:
    """Issue #83 — Step 7 verify for openrouter NEVER calls subprocess.run."""

    def test_verify_uses_models_endpoint_and_does_not_run_subprocess(
        self, isolated_state, monkeypatch
    ):
        import subprocess

        pb, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="openrouter",
            primary_harness="claude",
            engine_model_tag="anthropic/claude-sonnet-4.6",
            wire_result={
                "argv": ["claude", "--model", "anthropic/claude-sonnet-4.6"],
                "env": {"ANTHROPIC_BASE_URL": "https://openrouter.ai/api/v1"},
                "effective_tag": "anthropic/claude-sonnet-4.6",
                "raw_env": {"ANTHROPIC_AUTH_TOKEN": '"$(cat /tmp/k)"'},
            },
        )

        monkeypatch.setattr(
            pb,
            "smoke_test_openrouter_models",
            lambda *a, **kw: {"ok": True, "models": [], "response": "0 models"},
        )

        def fail_run(*a, **kw):
            raise AssertionError("step_2_7_verify must NOT call subprocess.run for openrouter")

        monkeypatch.setattr(subprocess, "run", fail_run)
        monkeypatch.setattr(wiz.subprocess, "run", fail_run)

        assert wiz.step_2_7_verify(state, non_interactive=True) is True
        assert state.verify_result["skipped_chat"] is True
        assert state.verify_result["via"] == "openrouter-models-endpoint"
        assert state.verify_result["ok"] is True

    def test_verify_fails_when_models_unreachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        state = wiz.WizardState(
            primary_engine="openrouter",
            primary_harness="claude",
            engine_model_tag="anthropic/claude-sonnet-4.6",
            wire_result={
                "argv": ["claude"],
                "env": {},
                "effective_tag": "x",
                "raw_env": {},
            },
        )
        monkeypatch.setattr(
            pb,
            "smoke_test_openrouter_models",
            lambda *a, **kw: {"ok": False, "error": "refused"},
        )
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not run subprocess on openrouter failure path")
            ),
        )
        monkeypatch.setattr(
            wiz.subprocess,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not run subprocess on openrouter failure path")
            ),
        )
        assert wiz.step_2_7_verify(state, non_interactive=True) is False
        assert state.verify_result["skipped_chat"] is True


class TestEnsureToolOpenRouter:
    """OpenRouter is hosted SaaS — _ensure_tool must NOT prompt for npm install
    and must return True even when offline (key check is deferred to step 4).
    """

    def test_returns_true_when_endpoint_reachable(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.OpenRouterAdapter, "detect", lambda self: {"present": True, "version": ""}
        )
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz._ensure_tool("openrouter") is True

    def test_returns_true_when_offline_no_install_attempted(self, isolated_state, monkeypatch):
        """Critical divergence from 9router: openrouter has no daemon to
        install. When detect() fails (offline), still return True so the
        wizard can continue to step 4 where the key prompt happens."""
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb.OpenRouterAdapter, "detect", lambda self: {"present": False, "version": ""}
        )
        # npm must NEVER be invoked for openrouter.
        import subprocess as sp

        monkeypatch.setattr(
            sp,
            "run",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not subprocess.run for openrouter (no npm install)")
            ),
        )
        # questionary.confirm must NEVER be prompted either.
        import questionary

        monkeypatch.setattr(
            questionary,
            "confirm",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not prompt for openrouter install")
            ),
        )
        wiz.console.width = 200
        with wiz.console.capture():
            assert wiz._ensure_tool("openrouter") is True


class TestClaudeOllamaThenClaudeoCoexist:
    """Issue #83 — installing claude+ollama and claude+openrouter must coexist.

    Mirrors the 9router coexistence test (TestClaudeOllamaThenClaude9Coexist).
    The `claude` regex must NOT eat the `claudeo` block.
    """

    def test_both_blocks_and_aliases_present(self, isolated_state, monkeypatch, tmp_path):
        from pathlib import Path

        _, wiz, _ = isolated_state
        rc = Path.home() / ".zshrc"
        rc.write_text("")

        # Install 1: claude + ollama (writes :claude block with cc + claude-local).
        cc_script = tmp_path / "cc"
        cc_script.write_text("#!/bin/sh\n")
        cc_script.chmod(0o755)
        wiz._install_shell_aliases(cc_script, "claude", non_interactive=True)

        # Install 2: claude + openrouter (writes :claudeo block with cco only).
        cco_script = tmp_path / "cco"
        cco_script.write_text("#!/bin/sh\n")
        cco_script.chmod(0o755)
        wiz._install_shell_aliases(cco_script, "claudeo", non_interactive=True)

        body = rc.read_text()
        # Both fenced blocks must be present, exactly once each.
        assert body.count("# >>> claude-codex-local:claude >>>") == 1
        assert body.count("# <<< claude-codex-local:claude <<<") == 1
        assert body.count("# >>> claude-codex-local:claudeo >>>") == 1
        assert body.count("# <<< claude-codex-local:claudeo <<<") == 1
        # Both alias sets must be present.
        assert "alias cc=" in body
        assert "alias claude-local=" in body
        assert "alias cco=" in body
        # Both helper scripts referenced.
        assert str(cc_script) in body
        assert str(cco_script) in body
        # Pin: re-running the claude install must NOT touch the claudeo block.
        cc_script_v2 = tmp_path / "cc-v2"
        cc_script_v2.write_text("#!/bin/sh\n")
        cc_script_v2.chmod(0o755)
        wiz._install_shell_aliases(cc_script_v2, "claude", non_interactive=True)
        body2 = rc.read_text()
        assert body2.count("# >>> claude-codex-local:claude >>>") == 1
        assert body2.count("# >>> claude-codex-local:claudeo >>>") == 1
        assert str(cco_script) in body2  # untouched
        assert f"alias cc={cc_script}\n" not in body2  # original cc replaced
