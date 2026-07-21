"""
Tests for issue #157 — Separate model-selection decisions from side effects.

Acceptance criteria:
  AC1  Model ranking and selection can be tested with supplied input data only.
  AC2  Runtime checks and smoke tests are performed outside the pure decision path.
  AC3  The returned recommendation remains equivalent for existing common scenarios.
  AC4  Tests cover both decision-only and side-effect orchestration behavior.
"""

from __future__ import annotations

import pytest

from claude_codex_local import core as pb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _empty_profile() -> dict:
    return {
        "ollama": {"models": []},
        "lmstudio": {"present": False, "server_running": False, "models": []},
        "presence": {"engines": [], "harnesses": [], "llmfit": False, "has_minimum": False},
        "host": {"system": "Darwin", "machine": "arm64"},
        "disk": {"free_bytes": 1 << 40, "free_gib": 1024.0},
    }


def _make_candidate(
    name: str = "Qwen/Qwen3-Coder-30B",
    score: int = 95,
    estimated_tps: int = 12,
    fit_level: str = "Perfect",
    ollama_tag: str | None = "qwen3-coder:30b",
    lms_mlx_path: str | None = None,
    lms_hub_name: str | None = None,
    memory_required_gb: float = 20.0,
) -> dict:
    return {
        "name": name,
        "score": score,
        "estimated_tps": estimated_tps,
        "fit_level": fit_level,
        "ollama_tag": ollama_tag,
        "lms_mlx_path": lms_mlx_path,
        "lms_hub_name": lms_hub_name,
        "memory_required_gb": memory_required_gb,
    }


# ---------------------------------------------------------------------------
# AC1 — Pure decision can be tested with supplied input data only
# ---------------------------------------------------------------------------


class TestSelectModelDecisionPureInput:
    """select_model_decision must work with pre-supplied candidates — no llmfit subprocess."""

    def test_accepts_supplied_candidates(self):
        """Given explicit candidates, the decision is made without calling llmfit."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision["selected_model"] == "qwen3-coder:30b"
        assert decision["runtime"] == "ollama"
        assert decision["model_source"] == "llmfit"

    def test_empty_candidates_falls_back(self):
        """No candidates → hardcoded fallback."""
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=[])
        assert decision["selected_model"] == "qwen2.5-coder:7b"
        assert decision["status"] == "download-required"

    def test_mode_fast_picks_highest_tps(self):
        """Fast mode: highest estimated_tps wins."""
        cands = [
            _make_candidate(name="A", score=95, estimated_tps=10, ollama_tag="a:1b"),
            _make_candidate(name="B", score=70, estimated_tps=90, ollama_tag="b:1b"),
        ]
        decision = pb.select_model_decision(_empty_profile(), "fast", candidates=cands)
        assert decision["selected_model"] == "b:1b"

    def test_mode_quality_picks_highest_score(self):
        """Quality mode: highest score wins."""
        cands = [
            _make_candidate(name="A", score=95, estimated_tps=10, ollama_tag="a:1b"),
            _make_candidate(name="B", score=70, estimated_tps=90, ollama_tag="b:1b"),
        ]
        decision = pb.select_model_decision(_empty_profile(), "quality", candidates=cands)
        assert decision["selected_model"] == "a:1b"

    def test_balanced_preserves_llmfit_order(self):
        """Balanced mode: first candidate (already sorted by llmfit) wins."""
        cands = [
            _make_candidate(name="A", score=95, ollama_tag="a:1b"),
            _make_candidate(name="B", score=70, ollama_tag="b:1b"),
        ]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision["selected_model"] == "a:1b"

    def test_invalid_mode_coerced_to_balanced(self):
        """Invalid mode string is coerced to 'balanced'."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "bogus", candidates=cands)
        assert decision["selected_model"] == "qwen3-coder:30b"
        assert decision["mode"] == "balanced"  # coerced to balanced

    def test_skips_candidates_without_ollama_tag(self):
        """Candidates with no ollama_tag are skipped; next matching wins in Pass 2."""
        # In Pass 2 (installed Ollama match), candidates without ollama_tag
        # are skipped. But in Pass 4 (download), the first candidate is used
        # regardless of tag. So with no installed models, Pass 4 takes
        # candidates[0] which has ollama_tag=None → falls to Pass 5.
        profile = _empty_profile()
        profile["ollama"]["models"] = [
            {"name": "b:1b", "local": True, "size": "5 GB"},
        ]
        cands = [
            _make_candidate(name="A", score=99, ollama_tag=None),
            _make_candidate(name="B", score=70, ollama_tag="b:1b"),
        ]
        decision = pb.select_model_decision(profile, "balanced", candidates=cands)
        assert decision["selected_model"] == "b:1b"

    def test_installed_ollama_model_preferred(self):
        """When an llmfit candidate matches an installed Ollama model, it is selected."""
        profile = _empty_profile()
        profile["ollama"]["models"] = [{"name": "qwen3-coder:30b", "local": True, "size": "19 GB"}]
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(profile, "balanced", candidates=cands)
        assert decision["selected_model"] == "qwen3-coder:30b"
        assert decision["runtime"] == "ollama"

    def test_installed_lmstudio_model_selected(self):
        """LM Studio MLX match is preferred over Ollama."""
        profile = _empty_profile()
        profile["lmstudio"]["present"] = True
        profile["lmstudio"]["server_running"] = True
        profile["lmstudio"]["models"] = [
            {"path": "qwen/qwen3-coder-30b", "local": True}
        ]
        cands = [
            _make_candidate(
                score=90,
                ollama_tag="qwen3-coder:30b",
                lms_hub_name="qwen/qwen3-coder-30b",
            )
        ]
        decision = pb.select_model_decision(
            profile, "balanced", candidates=cands, _lms_responses_api_ok=lambda _: True
        )
        assert decision["selected_model"] == "qwen/qwen3-coder-30b"
        assert decision["runtime"] == "lmstudio"

    def test_lmstudio_server_down_falls_back(self):
        """When LM Studio server is not running, decision falls through to Ollama."""
        profile = _empty_profile()
        profile["lmstudio"]["present"] = True
        profile["lmstudio"]["server_running"] = False
        profile["lmstudio"]["models"] = [{"path": "qwen/qwen3-coder-30b", "local": True}]
        cands = [
            _make_candidate(
                score=90,
                ollama_tag="qwen3-coder:30b",
                lms_hub_name="qwen/qwen3-coder-30b",
            )
        ]
        decision = pb.select_model_decision(
            profile, "balanced", candidates=cands, _lms_responses_api_ok=lambda _: True
        )
        assert decision["selected_model"] == "qwen3-coder:30b"
        assert decision["runtime"] == "ollama"

    def test_lmstudio_api_fail_falls_back(self):
        """When LM Studio Responses API check fails, decision falls through to Ollama."""
        profile = _empty_profile()
        profile["lmstudio"]["present"] = True
        profile["lmstudio"]["server_running"] = True
        profile["lmstudio"]["models"] = [{"path": "qwen/qwen3-coder-30b", "local": True}]
        cands = [
            _make_candidate(
                score=90,
                ollama_tag="qwen3-coder:30b",
                lms_hub_name="qwen/qwen3-coder-30b",
            )
        ]
        decision = pb.select_model_decision(
            profile, "balanced", candidates=cands, _lms_responses_api_ok=lambda _: False
        )
        assert decision["selected_model"] == "qwen3-coder:30b"
        assert decision["runtime"] == "ollama"

    def test_largest_installed_ollama_fallback(self):
        """When no llmfit candidate matches, largest installed Ollama model is used."""
        profile = _empty_profile()
        profile["ollama"]["models"] = [
            {"name": "llama2:7b", "local": True},
            {"name": "custom:13b", "local": True},
        ]
        cands = [_make_candidate(ollama_tag="qwen3-coder:30b")]
        decision = pb.select_model_decision(profile, "balanced", candidates=cands)
        assert decision["selected_model"] == "custom:13b"
        assert decision["model_source"] == "installed-fallback"

    def test_download_required_when_no_installed_match(self):
        """When candidates exist but none are installed, status is download-required."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision["status"] == "download-required"
        assert "ollama pull" in decision["next_steps"][0]

    def test_rationale_contains_decision_reason(self):
        """Rationale list documents why the model was selected."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert any("llmfit" in r.lower() for r in decision["rationale"])

    def test_caveats_empty_when_no_issues(self):
        """When an installed model is selected, caveats should be empty."""
        profile = _empty_profile()
        profile["ollama"]["models"] = [
            {"name": "qwen3-coder:30b", "local": True, "size": "19 GB"},
        ]
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(profile, "balanced", candidates=cands)
        assert decision["caveats"] == []

    def test_modes_dict_included(self):
        """Modes dict contains balanced/fast/quality entries."""
        cands = [_make_candidate(score=90, fit_level="Perfect")]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert "balanced" in decision["modes"]
        assert "fast" in decision["modes"]
        assert "quality" in decision["modes"]

    def test_model_source_reflected(self):
        """model_source reflects the origin of the decision."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision["model_source"] == "llmfit"

    def test_candidates_evaluated_count(self):
        """candidates_evaluated reflects the number of candidates processed."""
        cands = [_make_candidate(score=i) for i in range(5)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision["candidates_evaluated"] == 5

    def test_no_side_effects_no_smoke_key(self):
        """Pure decision does NOT include smoke_test — that's the orchestrator's job."""
        cands = [_make_candidate(score=90)]
        decision = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert "smoke_test" not in decision
        assert "llmfit" not in decision  # no llmfit metadata block (orchestrator adds it)


# ---------------------------------------------------------------------------
# AC2 — Runtime checks and smoke tests outside the pure decision path
# ---------------------------------------------------------------------------


class TestSideEffectsInOrchestrator:
    """select_best_model must add smoke tests and model loads on top of the decision."""

    def test_select_best_model_includes_smoke_test_key(self):
        """The orchestrator result includes smoke_test (the pure decision does not)."""
        # Use empty candidates to trigger the hardcoded fallback, which has
        # status="download-required" — smoke tests are only run for ready models.
        decision = pb.select_model_decision(
            _empty_profile(), "balanced", candidates=[]
        )
        assert "smoke_test" not in decision

        # The orchestrator also won't smoke-test a download-required model,
        # but the key will be present (as None).
        profile = _empty_profile()
        result = pb.select_best_model(profile)
        assert "smoke_test" in result

    def test_select_best_model_delegates_to_decision(self, monkeypatch):
        """select_best_model must call select_model_decision and build on its result."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "test-model",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "test-model", "fast": "test-model", "quality": "test-model"},
            "rationale": ["test rationale"],
            "caveats": [],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        result = pb.select_best_model({})
        assert result["selected_model"] == "test-model"
        assert result["runtime"] == "ollama"
        assert result["smoke_test"] is not None
        assert "Live ollama smoke test passed." in result["rationale"]

    def test_select_best_model_no_smoke_for_download_required(self, monkeypatch):
        """When decision says download-required, orchestrator does not run smoke tests."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "download-required",
            "selected_model": "qwen3-coder:30b",
            "selected_candidate": {"score": 90, "fit_level": "Good", "estimated_tps": 30, "name": "Test"},
            "modes": {"balanced": "qwen3-coder:30b", "fast": "qwen3-coder:30b", "quality": None},
            "rationale": ["recommend download"],
            "caveats": [],
            "next_steps": ["ollama pull qwen3-coder:30b"],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        smoke_called = []
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: (smoke_called.append(tag), {"ok": False})[1]
        )
        result = pb.select_best_model({})
        assert result["status"] == "download-required"
        assert smoke_called == [], "Smoke test must NOT be called for download-required models"

    def test_select_best_model_propagates_rationale(self, monkeypatch):
        """Rationale from the decision is carried through the orchestrator."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "test-model",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "test-model", "fast": "test-model", "quality": "test-model"},
            "rationale": ["custom rationale from decision"],
            "caveats": [],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        result = pb.select_best_model({})
        assert "custom rationale from decision" in result["rationale"]

    def test_select_best_model_propagates_caveats(self, monkeypatch):
        """Caveats from the decision are carried through the orchestrator."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "test-model",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "test-model", "fast": "test-model", "quality": "test-model"},
            "rationale": [],
            "caveats": ["pre-existing caveat"],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        result = pb.select_best_model({})
        assert "pre-existing caveat" in result["caveats"]

    def test_select_best_model_lmstudio_side_effects(self, monkeypatch):
        """When runtime is lmstudio, orchestrator calls lms_load_model + smoke test."""
        mock_decision = {
            "runtime": "lmstudio",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "qwen/qwen3-coder-30b",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "qwen/qwen3-coder-30b", "fast": "qwen/qwen3-coder-30b", "quality": "qwen/qwen3-coder-30b"},
            "rationale": ["lmstudio rationale"],
            "caveats": [],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        load_called = []
        smoke_called = []
        monkeypatch.setattr(
            pb, "lms_load_model", lambda tag: (load_called.append(tag), {"ok": True})[1]
        )
        monkeypatch.setattr(
            pb, "smoke_test_lmstudio_model", lambda tag: (smoke_called.append(tag), {"ok": True})[1]
        )
        result = pb.select_best_model({})
        assert load_called == ["qwen/qwen3-coder-30b"]
        assert smoke_called == ["qwen/qwen3-coder-30b"]


# ---------------------------------------------------------------------------
# AC3 — Returned recommendation remains equivalent for existing scenarios
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """The refactored select_best_model must produce equivalent results."""

    def test_hardcoded_fallback_same_result(self, monkeypatch):
        """Fallback when no candidates must still be qwen2.5-coder:7b via ollama."""
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        result = pb.select_best_model(_empty_profile())
        assert result["selected_model"] == "qwen2.5-coder:7b"
        assert result["runtime"] == "ollama"
        assert result["status"] == "download-required"

    def test_installed_ollama_match_same_result(self, monkeypatch):
        """Installed Ollama model matching llmfit candidate is selected."""
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": True, "response": "READY"}
        )
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda *a, **k: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "score": 90,
                    "ollama_tag": "qwen3-coder:30b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "fit_level": "Perfect",
                    "estimated_tps": 30,
                },
            ],
        )
        profile = _empty_profile()
        profile["ollama"]["models"] = [{"name": "qwen3-coder:30b", "local": True}]
        result = pb.select_best_model(profile, mode="balanced")
        assert result["selected_model"] == "qwen3-coder:30b"
        assert result["runtime"] == "ollama"
        assert result["status"] == "ready"

    def test_mode_fast_picks_fastest(self, monkeypatch):
        """Fast mode still picks the highest-tps model."""
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda *a, **k: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "score": 95,
                    "ollama_tag": "qwen3-coder:30b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "estimated_tps": 10,
                    "fit_level": "Good",
                },
                {
                    "name": "Qwen/Qwen2.5-Coder-7B",
                    "score": 70,
                    "ollama_tag": "qwen2.5-coder:7b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "estimated_tps": 90,
                    "fit_level": "Perfect",
                },
            ],
        )
        result = pb.select_best_model(_empty_profile(), mode="fast")
        assert result["selected_model"] == "qwen2.5-coder:7b"
        assert result["mode"] == "fast"

    def test_download_recommended_when_no_match(self, monkeypatch):
        """When candidates exist but none are installed, download is recommended."""
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda *a, **k: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "score": 90,
                    "ollama_tag": "qwen3-coder:30b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "fit_level": "Good",
                    "memory_required_gb": 20,
                    "estimated_tps": 25,
                },
            ],
        )
        result = pb.select_best_model(_empty_profile(), mode="balanced")
        assert result["status"] == "download-required"
        assert result["selected_model"] == "qwen3-coder:30b"
        assert any("ollama pull" in step for step in result["next_steps"])

    def test_result_has_all_expected_keys(self, monkeypatch):
        """select_best_model result must have all legacy keys."""
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        result = pb.select_best_model(_empty_profile())
        expected_keys = {
            "runtime", "mode", "status", "selected_model", "modes",
            "rationale", "caveats", "next_steps", "smoke_test", "llmfit", "state_dir",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_llmfit_metadata_present(self, monkeypatch):
        """select_best_model must include llmfit metadata block."""
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda *a, **k: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "score": 90,
                    "ollama_tag": "qwen3-coder:30b",
                    "lms_mlx_path": None,
                    "lms_hub_name": None,
                    "fit_level": "Perfect",
                    "estimated_tps": 30,
                    "memory_required_gb": 20,
                    "best_quant": "Q4_K_M",
                },
            ],
        )
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        profile = _empty_profile()
        profile["ollama"]["models"] = [{"name": "qwen3-coder:30b", "local": True}]
        result = pb.select_best_model(profile)
        llmfit = result["llmfit"]
        assert llmfit["score"] == 90
        assert llmfit["fit_level"] == "Perfect"
        assert llmfit["estimated_tps"] == 30
        assert llmfit["memory_required_gb"] == 20
        assert llmfit["hf_name"] == "Qwen/Qwen3-Coder-30B"
        assert llmfit["best_quant"] == "Q4_K_M"
        assert llmfit["candidates_evaluated"] == 1


# ---------------------------------------------------------------------------
# AC4 — Tests cover both decision-only and side-effect orchestration
# ---------------------------------------------------------------------------


class TestCoverageBothPaths:
    """Ensure both the pure decision path and the orchestrator path are tested."""

    def test_decision_path_has_no_network_calls(self):
        """select_model_decision must not call any network-dependent functions."""
        # The function accepts a _lms_responses_api_ok mock — if it called
        # the real function, the mock wouldn't matter.
        profile = _empty_profile()
        profile["lmstudio"]["present"] = True
        profile["lmstudio"]["server_running"] = True
        profile["lmstudio"]["models"] = [{"path": "test/model", "local": True}]
        call_log = []

        def mock_api_ok(key: str) -> bool:
            call_log.append(key)
            return True

        cands = [_make_candidate(lms_hub_name="test/model", ollama_tag="fallback:1b")]
        pb.select_model_decision(profile, "balanced", candidates=cands, _lms_responses_api_ok=mock_api_ok)
        assert call_log == ["test/model"], "Only the mock should be called"

    def test_orchestrator_path_runs_smoke_tests(self, monkeypatch):
        """select_best_model must actually run smoke tests for ready models."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "test-model",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "test-model", "fast": "test-model", "quality": "test-model"},
            "rationale": [],
            "caveats": [],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        smoke_results = []
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: smoke_results.append(tag) or {"ok": True}
        )
        pb.select_best_model({})
        assert smoke_results == ["test-model"], "Orchestrator must run smoke tests"

    def test_decision_independent_of_smoke_test_outcomes(self):
        """The decision must be the same regardless of smoke test results."""
        cands = [_make_candidate(score=90)]
        decision1 = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        decision2 = pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert decision1["selected_model"] == decision2["selected_model"]
        assert decision1["runtime"] == decision2["runtime"]

    def test_orchestrator_caveat_on_smoke_failure(self, monkeypatch):
        """When smoke test fails, orchestrator adds a caveat but keeps the decision."""
        mock_decision = {
            "runtime": "ollama",
            "mode": "balanced",
            "status": "ready",
            "selected_model": "test-model",
            "selected_candidate": {"score": 80, "fit_level": "Good", "estimated_tps": 50, "name": "Test"},
            "modes": {"balanced": "test-model", "fast": "test-model", "quality": "test-model"},
            "rationale": ["selected"],
            "caveats": [],
            "next_steps": [],
            "model_source": "llmfit",
            "candidates_evaluated": 1,
        }
        monkeypatch.setattr(pb, "select_model_decision", lambda *a, **k: mock_decision)
        monkeypatch.setattr(
            pb, "smoke_test_ollama_model", lambda tag: {"ok": False, "error": "test smoke failure"}
        )
        result = pb.select_best_model({})
        assert result["selected_model"] == "test-model"  # decision unchanged
        assert any("smoke test failed" in c.lower() for c in result["caveats"])

    def test_decision_does_not_mutate_input_candidates(self):
        """select_model_decision must not mutate the input candidates list."""
        cands = [
            _make_candidate(score=90, ollama_tag="a:1b"),
            _make_candidate(score=70, ollama_tag="b:1b"),
        ]
        original = [dict(c) for c in cands]
        pb.select_model_decision(_empty_profile(), "balanced", candidates=cands)
        assert cands == original, "Input candidates must not be mutated"
