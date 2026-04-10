"""
Unit tests for wizard.py helpers — state persistence, engine/model picking
logic, presence checks, and the Claude/Codex wiring helpers.
"""

from __future__ import annotations

import json

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
# _lmstudio_needs_nothink — tag-pattern check.
# ---------------------------------------------------------------------------


class TestLmstudioNeedsNothink:
    def test_qwen3_needs_nothink(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._lmstudio_needs_nothink("qwen/qwen3-coder-30b") is True

    def test_qwen25_does_not_need_nothink(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._lmstudio_needs_nothink("qwen/qwen2.5-coder-7b") is False

    def test_llama_does_not_need_nothink(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._lmstudio_needs_nothink("meta-llama/llama-3-8b") is False


# ---------------------------------------------------------------------------
# _wire_claude — writes isolated settings.json, returns the launch command.
# ---------------------------------------------------------------------------


class TestWireClaude:
    def test_ollama_writes_settings_and_builds_cmd(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        # Short-circuit the no-think variant builder (no real ollama).
        monkeypatch.setattr(
            pb,
            "ollama_ensure_nothink_variant",
            lambda tag: (tag, {"patched": False, "reason": "unit test"}),
        )
        cmd, tag = wiz._wire_claude("ollama", "qwen3-coder:30b")
        assert cmd == [
            f"HOME={pb.STATE_HOME}",
            "claude",
            "--model",
            "qwen3-coder:30b",
        ]
        assert tag == "qwen3-coder:30b"

        settings_file = pb.STATE_HOME / ".claude" / "settings.json"
        assert settings_file.exists()
        body = json.loads(settings_file.read_text())
        assert body["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:11434"
        assert body["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert body["env"]["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "qwen3-coder:30b"
        assert "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME" in body["env"]

    def test_ollama_picks_patched_variant_tag(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb,
            "ollama_ensure_nothink_variant",
            lambda tag: ("qwen3-coder-cclocal:30b", {"patched": True, "reused": False}),
        )
        cmd, tag = wiz._wire_claude("ollama", "qwen3-coder:30b")
        assert tag == "qwen3-coder-cclocal:30b"
        assert cmd[-1] == "qwen3-coder-cclocal:30b"

    def test_lmstudio_uses_lms_port(self, isolated_state):
        pb, wiz, _ = isolated_state
        cmd, tag = wiz._wire_claude("lmstudio", "qwen/qwen3-coder-30b")
        settings_file = pb.STATE_HOME / ".claude" / "settings.json"
        body = json.loads(settings_file.read_text())
        assert f"{pb.LMS_SERVER_PORT}" in body["env"]["ANTHROPIC_BASE_URL"]
        assert body["env"]["ANTHROPIC_AUTH_TOKEN"] == "lmstudio"

    def test_unknown_engine_returns_none(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._wire_claude("bogus", "tag") is None


# ---------------------------------------------------------------------------
# _wire_codex — dispatch by engine, with the config-writing side-effects mocked.
# ---------------------------------------------------------------------------


class TestWireCodex:
    def test_ollama_path(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        called = {}
        monkeypatch.setattr(
            pb,
            "configure_ollama_integration",
            lambda target, model: called.setdefault("args", (target, model)) or {"ok": True},
        )
        cmd, tag = wiz._wire_codex("ollama", "qwen3-coder:30b")
        assert called["args"] == ("codex", "qwen3-coder:30b")
        assert cmd == ["codex", "--oss", "-m", "qwen3-coder:30b"]

    def test_lmstudio_path(self, isolated_state, monkeypatch):
        pb, wiz, _ = isolated_state
        monkeypatch.setattr(
            pb, "configure_lmstudio_integration", lambda target, model: {"ok": True}
        )
        cmd, tag = wiz._wire_codex("lmstudio", "qwen/qwen3-coder-30b")
        assert cmd == ["codex", "-m", "qwen/qwen3-coder-30b"]

    def test_llamacpp_path(self, isolated_state):
        _, wiz, _ = isolated_state
        cmd, tag = wiz._wire_codex("llamacpp", "some-gguf")
        assert cmd == ["codex", "-m", "some-gguf"]

    def test_unknown_engine(self, isolated_state):
        _, wiz, _ = isolated_state
        assert wiz._wire_codex("bogus", "x") is None


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

    def test_skips_cclocal_variants(self, isolated_state):
        pb, wiz, _ = isolated_state
        profile = {
            "ollama": {
                "models": [
                    {"name": f"qwen3-coder{pb.NOTHINK_VARIANT_SUFFIX}:30b", "local": True},
                    {"name": "qwen3-coder:30b", "local": True},
                ]
            },
            "lmstudio": {"models": []},
        }
        result = wiz._find_model_auto("ollama", profile)
        assert result["tag"] == "qwen3-coder:30b"

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
