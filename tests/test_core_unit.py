"""
Unit tests for core — pure functions + subprocess-mockable helpers.

These tests never touch real ollama / lms / claude / codex binaries. Anything
that would shell out is either patched or routed through the `fake_bin`
fixture defined in conftest.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import claude_codex_local.core as pb

# ---------------------------------------------------------------------------
# HF → Ollama / LM Studio tag mapping (pure regex lookups).
# ---------------------------------------------------------------------------


class TestHfToOllamaTag:
    def test_maps_qwen3_coder_30b(self):
        assert pb.hf_name_to_ollama_tag("Qwen/Qwen3-Coder-30B-A3B-Instruct") == "qwen3-coder:30b"

    def test_maps_qwen25_coder_7b_case_insensitive(self):
        assert pb.hf_name_to_ollama_tag("qwen2.5-coder-7B") == "qwen2.5-coder:7b"

    def test_maps_deepseek_coder_v2_lite(self):
        assert (
            pb.hf_name_to_ollama_tag("deepseek-ai/DeepSeek-Coder-V2-Lite")
            == "deepseek-coder-v2:16b"
        )

    def test_unknown_returns_none(self):
        assert pb.hf_name_to_ollama_tag("totally-unknown-model") is None

    def test_empty_string_returns_none(self):
        assert pb.hf_name_to_ollama_tag("") is None


class TestHfToLmsHub:
    def test_maps_qwen3_coder_30b(self):
        assert pb.hf_name_to_lms_hub("Qwen/Qwen3-Coder-30B") == "qwen/qwen3-coder-30b"

    def test_maps_codellama_13b(self):
        assert (
            pb.hf_name_to_lms_hub("meta-llama/CodeLlama-13b-Python") == "meta-llama/codellama-13b"
        )

    def test_unknown_returns_none(self):
        assert pb.hf_name_to_lms_hub("random/model") is None


# ---------------------------------------------------------------------------
# parse_ollama_list — patches run() to feed synthetic `ollama list` output.
# ---------------------------------------------------------------------------


class _FakeCP:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class TestParseOllamaList:
    def test_parses_multiple_rows(self, monkeypatch):
        sample = (
            "NAME                  ID              SIZE      MODIFIED\n"
            "qwen3-coder:30b       abc123          19 GB     2 days ago\n"
            "qwen2.5-coder:7b      def456          4.1 GB    1 week ago\n"
        )
        monkeypatch.setattr(pb, "run", lambda *a, **kw: _FakeCP(stdout=sample))
        models = pb.parse_ollama_list()
        assert len(models) == 2
        assert models[0]["name"] == "qwen3-coder:30b"
        assert models[0]["id"] == "abc123"
        assert models[0]["size"] == "19 GB"
        assert models[0]["local"] is True
        assert models[1]["name"] == "qwen2.5-coder:7b"

    def test_only_header_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            pb, "run", lambda *a, **kw: _FakeCP(stdout="NAME  ID  SIZE  MODIFIED\n")
        )
        assert pb.parse_ollama_list() == []

    def test_subprocess_failure_returns_empty(self, monkeypatch):
        def boom(*a, **kw):
            raise FileNotFoundError("ollama")

        monkeypatch.setattr(pb, "run", boom)
        assert pb.parse_ollama_list() == []

    def test_marks_unsized_rows_nonlocal(self, monkeypatch):
        sample = "NAME  ID  SIZE  MODIFIED\nphantom:latest  xxx  -  never\n"
        monkeypatch.setattr(pb, "run", lambda *a, **kw: _FakeCP(stdout=sample))
        models = pb.parse_ollama_list()
        assert models[0]["local"] is False


# ---------------------------------------------------------------------------
# disk_usage_for — walks to the nearest existing parent.
# ---------------------------------------------------------------------------


class TestDiskUsageFor:
    def test_returns_usage_for_tmp_path(self, tmp_path):
        usage = pb.disk_usage_for(tmp_path)
        assert "free_bytes" in usage
        assert usage["total_gib"] > 0
        assert usage["free_gib"] >= 0

    def test_walks_up_to_existing_parent(self, tmp_path):
        nonexistent = tmp_path / "a" / "b" / "c"
        usage = pb.disk_usage_for(nonexistent)
        # probe should have walked back to tmp_path (which exists)
        assert "total_bytes" in usage


# ---------------------------------------------------------------------------
# ensure_path + state_env — no subprocess, just env dict manipulation.
# ---------------------------------------------------------------------------


class TestEnsurePath:
    def test_keeps_path_when_extras_missing(self, monkeypatch, tmp_path):
        # Point ORIG_HOME at a dir with no .lmstudio/.local — no extras to prepend.
        monkeypatch.setattr(pb, "ORIG_HOME", tmp_path)
        env = pb.ensure_path({"PATH": "/usr/bin"})
        assert env["PATH"] == "/usr/bin"

    def test_prepends_lmstudio_bin_when_present(self, monkeypatch, tmp_path):
        lms_bin = tmp_path / ".lmstudio" / "bin"
        lms_bin.mkdir(parents=True)
        monkeypatch.setattr(pb, "ORIG_HOME", tmp_path)
        env = pb.ensure_path({"PATH": "/usr/bin"})
        assert env["PATH"].startswith(str(lms_bin))
        assert "/usr/bin" in env["PATH"]

    def test_does_not_duplicate_existing_entry(self, monkeypatch, tmp_path):
        lms_bin = tmp_path / ".lmstudio" / "bin"
        lms_bin.mkdir(parents=True)
        monkeypatch.setattr(pb, "ORIG_HOME", tmp_path)
        env = pb.ensure_path({"PATH": f"{lms_bin}:/usr/bin"})
        # No duplicate
        assert env["PATH"].count(str(lms_bin)) == 1


class TestStateEnv:
    def test_returns_path_env_without_home_override(self, isolated_state):
        pb_mod, _, state_dir = isolated_state
        env = pb_mod.state_env()
        assert "PATH" in env
        # state_env() no longer rewrites HOME / XDG_*
        assert env.get("HOME") != str(state_dir / "home")


class TestEnsureStateDirs:
    def test_creates_state_dir_and_bin(self, isolated_state):
        pb_mod, _, state_dir = isolated_state
        pb_mod.ensure_state_dirs()
        assert state_dir.exists()
        assert (state_dir / "bin").exists()


# ---------------------------------------------------------------------------
# llmfit helpers — mock subprocess.
# ---------------------------------------------------------------------------


def _fake_cp_json(payload):
    return _FakeCP(stdout=json.dumps(payload))


class TestLlmfitCodingCandidates:
    def test_returns_empty_when_llmfit_absent(self, monkeypatch):
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        assert pb.llmfit_coding_candidates() == []

    def test_filters_and_sorts_by_score(self, monkeypatch):
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        payload = {
            "models": [
                {
                    "name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    "category": "Coding",
                    "score": 95,
                    "best_quant": "mlx-4bit",
                    "fit_level": "Perfect",
                    "estimated_tps": 40,
                },
                {"name": "meta-llama/Llama-3-8B", "category": "General", "score": 80},
                {
                    "name": "Qwen/Qwen2.5-Coder-7B",
                    "category": "code",
                    "score": 70,
                    "best_quant": "q4_k_m",
                },
            ]
        }
        monkeypatch.setattr(pb, "run", lambda *a, **kw: _fake_cp_json(payload))
        cands = pb.llmfit_coding_candidates()
        names = [c["name"] for c in cands]
        assert "meta-llama/Llama-3-8B" not in names
        assert cands[0]["score"] == 95
        assert cands[0]["ollama_tag"] == "qwen3-coder:30b"
        assert cands[0]["lms_hub_name"] == "qwen/qwen3-coder-30b"

    def test_dedupes_by_canonical_key_keeps_higher_score(self, monkeypatch):
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "1.0"}
        )
        payload = {
            "models": [
                {
                    "name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    "category": "coding",
                    "score": 90,
                    "best_quant": "mlx-4bit",
                },
                {
                    "name": "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-8bit",
                    "category": "coding",
                    "score": 92,
                    "best_quant": "mlx-8bit",
                },
            ]
        }
        monkeypatch.setattr(pb, "run", lambda *a, **kw: _fake_cp_json(payload))
        cands = pb.llmfit_coding_candidates()
        assert len(cands) == 1
        assert cands[0]["score"] == 92


class TestLlmfitEstimateSizeBytes:
    def test_uses_total_memory_gb(self):
        b = pb.llmfit_estimate_size_bytes({"total_memory_gb": 4})
        assert b == 4 * (1024**3)

    def test_falls_back_to_params_times_bits(self):
        b = pb.llmfit_estimate_size_bytes({"params_b": 7, "best_quant": "mlx-4bit"})
        assert b == int(7 * 4 / 8 * (1024**3))

    def test_returns_none_when_insufficient_data(self):
        assert pb.llmfit_estimate_size_bytes({"params_b": 7}) is None


# ---------------------------------------------------------------------------
# select_best_model — the heart of the recommendation engine.
# ---------------------------------------------------------------------------


def _empty_profile():
    return {
        "ollama": {"models": []},
        "lmstudio": {"present": False, "server_running": False, "models": []},
        "presence": {"engines": [], "harnesses": [], "llmfit": False, "has_minimum": False},
        "host": {"system": "Darwin", "machine": "arm64"},
        "disk": {"free_bytes": 1 << 40, "free_gib": 1024.0},
    }


class TestSelectBestModel:
    def test_hardcoded_fallback_when_no_candidates(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
        rec = pb.select_best_model(_empty_profile(), mode="balanced")
        assert rec["selected_model"] == "qwen2.5-coder:7b"
        assert rec["status"] == "download-required"
        assert rec["runtime"] == "ollama"

    def test_picks_installed_ollama_model_matching_candidate(self, monkeypatch):
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
        rec = pb.select_best_model(profile, mode="balanced")
        assert rec["selected_model"] == "qwen3-coder:30b"
        assert rec["runtime"] == "ollama"
        assert rec["status"] == "ready"

    def test_ollama_fallback_to_largest_installed_when_no_candidate_match(self, monkeypatch):
        monkeypatch.setattr(pb, "smoke_test_ollama_model", lambda tag: {"ok": True})
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
                },
            ],
        )
        profile = _empty_profile()
        profile["ollama"]["models"] = [
            {"name": "llama2:7b", "local": True},
            {"name": "custom:13b", "local": True},
        ]
        rec = pb.select_best_model(profile, mode="balanced")
        assert rec["selected_model"] == "custom:13b"  # picks larger B

    def test_recommends_download_when_candidates_but_none_installed(self, monkeypatch):
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
        rec = pb.select_best_model(_empty_profile(), mode="balanced")
        assert rec["status"] == "download-required"
        assert rec["selected_model"] == "qwen3-coder:30b"
        assert any("ollama pull" in step for step in rec["next_steps"])

    def test_mode_fast_sorts_by_tps(self, monkeypatch):
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
        rec = pb.select_best_model(_empty_profile(), mode="fast")
        assert rec["selected_model"] == "qwen2.5-coder:7b"
        assert rec["mode"] == "fast"

    def test_invalid_mode_coerced_to_balanced(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
        rec = pb.select_best_model(_empty_profile(), mode="bogus")
        assert rec["mode"] == "balanced"


# ---------------------------------------------------------------------------
# rank_candidates_for_mode — pure helper used by the wizard profile picker.
# ---------------------------------------------------------------------------


class TestRankCandidatesForMode:
    def _candidates(self, *a, **k):
        return [
            {
                "name": "Qwen/Qwen3-Coder-30B",
                "score": 95,
                "estimated_tps": 12,
                "ollama_tag": "qwen3-coder:30b",
            },
            {
                "name": "Qwen/Qwen2.5-Coder-7B",
                "score": 70,
                "estimated_tps": 80,
                "ollama_tag": "qwen2.5-coder:7b",
            },
            {
                "name": "Qwen/Qwen2.5-Coder-3B",
                "score": 55,
                "estimated_tps": 120,
                "ollama_tag": "qwen2.5-coder:3b",
            },
        ]

    def test_balanced_preserves_input_order(self):
        c = self._candidates()
        out = pb.rank_candidates_for_mode(c, "balanced")
        assert [m["name"] for m in out] == [m["name"] for m in c]

    def test_fast_sorts_by_tps_descending(self):
        out = pb.rank_candidates_for_mode(self._candidates(), "fast")
        assert out[0]["name"].endswith("Qwen2.5-Coder-3B")
        assert out[-1]["name"].endswith("Qwen3-Coder-30B")

    def test_quality_sorts_by_score_descending(self):
        out = pb.rank_candidates_for_mode(self._candidates(), "quality")
        assert out[0]["name"].endswith("Qwen3-Coder-30B")
        assert out[-1]["name"].endswith("Qwen2.5-Coder-3B")

    def test_invalid_mode_coerced_to_balanced(self):
        c = self._candidates()
        out = pb.rank_candidates_for_mode(c, "bogus")
        assert [m["name"] for m in out] == [m["name"] for m in c]

    def test_empty_input_returns_empty(self):
        assert pb.rank_candidates_for_mode([], "fast") == []

    def test_does_not_mutate_input(self):
        c = self._candidates()
        snapshot = [m["name"] for m in c]
        pb.rank_candidates_for_mode(c, "fast")
        assert [m["name"] for m in c] == snapshot


# ---------------------------------------------------------------------------
# recommend_for_mode — engine-aware top pick for a given mode.
# ---------------------------------------------------------------------------


class TestRecommendForMode:
    def _candidates(self, *a, **k):
        return [
            {
                "name": "Qwen/Qwen3-Coder-30B",
                "score": 95,
                "estimated_tps": 12,
                "ollama_tag": "qwen3-coder:30b",
                "lms_hub_name": "qwen/qwen3-coder-30b",
            },
            {
                "name": "Qwen/Qwen2.5-Coder-7B",
                "score": 70,
                "estimated_tps": 80,
                "ollama_tag": "qwen2.5-coder:7b",
                "lms_hub_name": "qwen/qwen2.5-coder-7b",
            },
        ]

    def test_fast_picks_fastest_for_ollama(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        rec = pb.recommend_for_mode(_empty_profile(), "fast", "ollama")
        assert rec is not None
        assert rec["engine_tag"] == "qwen2.5-coder:7b"
        assert rec["mode"] == "fast"

    def test_quality_picks_highest_score_for_lmstudio(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        rec = pb.recommend_for_mode(_empty_profile(), "quality", "lmstudio")
        assert rec is not None
        assert rec["engine_tag"] == "qwen/qwen3-coder-30b"
        assert rec["mode"] == "quality"

    def test_llamacpp_resolves_gguf_mirror(self, monkeypatch):
        # llmfit's catalog is MLX-centric; for llama.cpp we must resolve to a
        # repo that actually contains GGUF files (#58). The picker silently
        # drops candidates with no GGUF mirror — this test verifies the top
        # candidate's resolved tag wins when a mirror exists.
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        # Clear the mirror cache so the test sees fresh resolution.
        pb._GGUF_MIRROR_CACHE.clear()
        monkeypatch.setattr(
            pb,
            "resolve_gguf_mirror",
            lambda name: "bartowski/Qwen3-Coder-30B-GGUF" if name else None,
        )
        rec = pb.recommend_for_mode(_empty_profile(), "balanced", "llamacpp")
        assert rec is not None
        assert rec["engine_tag"] == "bartowski/Qwen3-Coder-30B-GGUF"

    def test_llamacpp_skips_when_no_gguf_mirror(self, monkeypatch):
        # When no GGUF mirror is found for *any* candidate, recommend_for_mode
        # must return None instead of falsely surfacing an MLX-only repo (#58).
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        pb._GGUF_MIRROR_CACHE.clear()
        monkeypatch.setattr(pb, "resolve_gguf_mirror", lambda name: None)
        assert pb.recommend_for_mode(_empty_profile(), "balanced", "llamacpp") is None

    def test_returns_none_when_no_candidates(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda *a, **k: [])
        assert pb.recommend_for_mode(_empty_profile(), "balanced", "ollama") is None

    def test_returns_none_for_unknown_engine(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        assert pb.recommend_for_mode(_empty_profile(), "balanced", "vllm") is None

    def test_skips_candidates_without_engine_tag(self, monkeypatch):
        # A candidate with no ollama_tag must be skipped; the next matching
        # candidate must be picked instead.
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda *a, **k: [
                {
                    "name": "Foo/NoOllamaTag",
                    "score": 99,
                    "estimated_tps": 50,
                    "ollama_tag": None,
                },
                {
                    "name": "Qwen/Qwen2.5-Coder-7B",
                    "score": 70,
                    "estimated_tps": 80,
                    "ollama_tag": "qwen2.5-coder:7b",
                },
            ],
        )
        rec = pb.recommend_for_mode(_empty_profile(), "balanced", "ollama")
        assert rec is not None
        assert rec["engine_tag"] == "qwen2.5-coder:7b"


# ---------------------------------------------------------------------------
# installed_models_for_engine — discovery helper used by the wizard.
# ---------------------------------------------------------------------------


class TestInstalledModelsForEngine:
    def test_ollama_lists_local_models_only(self):
        profile = {
            "ollama": {
                "models": [
                    {"name": "qwen3-coder:30b", "local": True, "size": "19 GB"},
                    {"name": "phantom:latest", "local": False, "size": "-"},
                ]
            }
        }
        out = pb.installed_models_for_engine(profile, "ollama")
        assert [e["tag"] for e in out] == ["qwen3-coder:30b"]
        assert out[0]["source"] == "ollama"

    def test_ollama_orders_coder_models_first(self):
        profile = {
            "ollama": {
                "models": [
                    {"name": "llama2:7b", "local": True},
                    {"name": "qwen2.5-coder:7b", "local": True},
                    {"name": "deepseek-coder:6.7b", "local": True},
                ]
            }
        }
        out = pb.installed_models_for_engine(profile, "ollama")
        tags = [e["tag"] for e in out]
        # Coder models come before the non-coder one.
        assert tags.index("qwen2.5-coder:7b") < tags.index("llama2:7b")
        assert tags.index("deepseek-coder:6.7b") < tags.index("llama2:7b")

    def test_lmstudio_lists_model_paths(self):
        profile = {
            "lmstudio": {
                "models": [
                    {"path": "qwen/qwen3-coder-30b", "format": "mlx"},
                    {"path": "meta/llama-3-8b", "format": "mlx"},
                ]
            }
        }
        out = pb.installed_models_for_engine(profile, "lmstudio")
        tags = [e["tag"] for e in out]
        # qwen3-coder surfaces first because of the coder-first ordering rule.
        assert tags[0] == "qwen/qwen3-coder-30b"

    def test_llamacpp_surfaces_running_server_model(self):
        with patch("claude_codex_local.core.scan_huggingface_gguf_cache", return_value=[]):
            profile = {
                "llamacpp": {
                    "present": True,
                    "server_running": True,
                    "server_port": 8001,
                    "model": "local/qwen3-coder-30b.gguf",
                }
            }
            out = pb.installed_models_for_engine(profile, "llamacpp")
            assert len(out) == 1
            assert out[0]["tag"] == "local/qwen3-coder-30b.gguf"
            assert out[0]["running"] is True

    def test_llamacpp_returns_empty_when_server_not_running(self):
        with patch("claude_codex_local.core.scan_huggingface_gguf_cache", return_value=[]):
            profile = {
                "llamacpp": {
                    "present": True,
                    "server_running": False,
                    "server_port": 8001,
                    "model": None,
                }
            }
            assert pb.installed_models_for_engine(profile, "llamacpp") == []

    def test_empty_engine_section_returns_empty(self):
        assert pb.installed_models_for_engine({"ollama": {"models": []}}, "ollama") == []

    def test_unknown_engine_returns_empty(self):
        assert pb.installed_models_for_engine({}, "vllm") == []


# ---------------------------------------------------------------------------
# Runtime adapters — verify Protocol implementations return normalised dicts.
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_ollama_adapter_name_and_recommend_params(self):
        adapter = pb.OllamaAdapter()
        assert adapter.name == "ollama"
        assert adapter.recommend_params("balanced") == {"provider": "ollama", "extra_flags": []}

    def test_lmstudio_adapter_name_and_recommend_params(self):
        adapter = pb.LMStudioAdapter()
        assert adapter.name == "lmstudio"
        assert adapter.recommend_params("fast") == {"provider": "lmstudio", "extra_flags": []}

    def test_ollama_adapter_healthcheck_when_missing(self, monkeypatch):
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        adapter = pb.OllamaAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is False

    def test_ollama_adapter_healthcheck_reports_model_count(self, monkeypatch):
        monkeypatch.setattr(
            pb, "command_version", lambda *a, **kw: {"present": True, "version": "0.1"}
        )
        monkeypatch.setattr(pb, "parse_ollama_list", lambda: [{"name": "a"}, {"name": "b"}])
        adapter = pb.OllamaAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is True
        assert "2" in result["detail"]

    def test_all_adapters_registry_contains_all_six(self):
        names = {a.name for a in pb.ALL_ADAPTERS}
        assert names == {"ollama", "lmstudio", "llamacpp", "vllm", "9router", "openrouter"}


# ---------------------------------------------------------------------------
# llamacpp helpers — mock HTTP and subprocess.
# ---------------------------------------------------------------------------


class TestLlamaCppDetect:
    def test_returns_present_for_llama_server(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "command_version",
            lambda name, *a, **kw: (
                {"present": True, "version": "b1234"}
                if name == "llama-server"
                else {"present": False}
            ),
        )
        result = pb.llamacpp_detect()
        assert result["present"] is True
        assert result["binary"] == "llama-server"
        assert result["version"] == "b1234"

    def test_falls_back_to_llama_cpp_server(self, monkeypatch):
        def fake_version(name, *a, **kw):
            if name == "llama-cpp-server":
                return {"present": True, "version": "b5678"}
            return {"present": False}

        monkeypatch.setattr(pb, "command_version", fake_version)
        result = pb.llamacpp_detect()
        assert result["present"] is True
        assert result["binary"] == "llama-cpp-server"

    def test_returns_not_present_when_all_missing(self, monkeypatch):
        monkeypatch.setattr(pb, "command_version", lambda *a, **kw: {"present": False})
        result = pb.llamacpp_detect()
        assert result["present"] is False

    def test_server_candidate_rejected_when_not_llama(self, monkeypatch):
        # A generic binary named "server" (e.g., Apache helper) must not be accepted.
        def fake_version(name, *a, **kw):
            if name == "server":
                return {"present": True, "version": "Apache/2.4.57"}
            return {"present": False}

        monkeypatch.setattr(pb, "command_version", fake_version)
        result = pb.llamacpp_detect()
        assert result["present"] is False

    def test_server_candidate_accepted_when_version_contains_llama(self, monkeypatch):
        def fake_version(name, *a, **kw):
            if name == "server":
                return {"present": True, "version": "llama.cpp b3447"}
            return {"present": False}

        monkeypatch.setattr(pb, "command_version", fake_version)
        result = pb.llamacpp_detect()
        assert result["present"] is True
        assert result["binary"] == "server"


class TestLlamaCppInfo:
    def test_returns_not_present_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(
            pb, "llamacpp_detect", lambda: {"present": False, "binary": "", "version": ""}
        )
        result = pb.llamacpp_info()
        assert result["present"] is False
        assert result["server_running"] is False

    def test_server_running_when_health_and_models_respond(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_detect",
            lambda: {"present": True, "binary": "llama-server", "version": "b1234"},
        )

        class _Health:
            status = 200

            def read(self):
                return b'{"status":"ok"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        class _Models:
            status = 200

            def read(self):
                return json.dumps({"data": [{"id": "my-model.gguf"}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        import urllib.request

        def _fake_urlopen(req, *a, **kw):
            url = req if isinstance(req, str) else req.full_url
            return _Health() if url.endswith("/health") else _Models()

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        result = pb.llamacpp_info()
        assert result["server_running"] is True
        assert result["model"] == "my-model.gguf"

    def test_server_running_during_model_load_via_health_503(self, monkeypatch):
        # Regression: large models (35B+) can take 30s+ to load. During that
        # window, /v1/models doesn't respond yet — but /health returns 503
        # with "loading model". The probe must treat this as "server up"
        # so a second `ccl serve` doesn't try to spawn a duplicate on the
        # already-bound port.
        monkeypatch.setattr(
            pb,
            "llamacpp_detect",
            lambda: {"present": True, "binary": "llama-server", "version": "b1234"},
        )

        class _Health503:
            status = 503

            def read(self):
                return b'{"status":"loading model"}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        import urllib.error
        import urllib.request

        def _fake_urlopen(req, *a, **kw):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/health"):
                return _Health503()
            # /v1/models still 503ing during model load — raise on it.
            raise urllib.error.URLError("models endpoint not ready yet")

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
        result = pb.llamacpp_info()
        assert result["server_running"] is True  # the whole point
        assert result["model"] is None  # not yet known — fine

    def test_server_not_running_when_connection_refused(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_detect",
            lambda: {"present": True, "binary": "llama-server", "version": "b1234"},
        )
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        result = pb.llamacpp_info()
        assert result["server_running"] is False
        assert result["model"] is None


class _FakeChatResp:
    """Minimal fake OpenAI-compatible chat response for urllib mocking."""

    def __init__(
        self,
        content: str,
        usage: dict | None = None,
        reasoning_content: str | None = None,
        finish_reason: str | None = None,
    ):
        message = {"content": content}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        choice = {"message": message}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        body = {"choices": [choice]}
        if usage is not None:
            body["usage"] = usage
        self._data = json.dumps(body).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class TestSmokeTestLlamaCppModel:
    def test_returns_ok_true_when_ready_in_response(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("READY", usage={"completion_tokens": 4}),
        )
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is True
        assert result["response"] == "READY"
        assert result["completion_tokens"] == 4
        # duration_seconds should be positive and tokens_per_second computed.
        assert result["duration_seconds"] > 0
        assert isinstance(result["tokens_per_second"], float)
        assert result["tokens_per_second"] > 0

    def test_returns_ok_false_when_response_not_ready(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("Hello!", usage={"completion_tokens": 2}),
        )
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is False

    def test_missing_usage_leaves_tokens_per_second_none(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("READY"),  # no usage block
        )
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is True
        assert result["tokens_per_second"] is None
        assert result["completion_tokens"] is None

    def test_returns_ok_false_on_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is False
        assert "error" in result

    def test_accepts_ready_in_reasoning_content(self, monkeypatch):
        # Reasoning models (Qwen3+, etc.) may emit READY inside the
        # `reasoning_content` trace and leave `content` empty when the
        # token budget is reached mid-thought. The smoke test must still
        # treat that as success — engine + chat template are wired up.
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp(
                content="",
                reasoning_content="The user wants me to reply READY.",
                usage={"completion_tokens": 12},
                finish_reason="length",
            ),
        )
        result = pb.smoke_test_llamacpp_model("qwen3.6-thinker.gguf")
        assert result["ok"] is True
        assert result["finish_reason"] == "length"

    def test_failure_surfaces_finish_reason_and_snippet(self, monkeypatch):
        # When neither field contains READY, the wizard prints
        # `error or response`. Empty content alone yielded a useless
        # blank line; the failure result must carry a diagnostic.
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp(
                content="",
                reasoning_content="Here's a thinking process: 1. Analyze input.",
                usage={"completion_tokens": 16},
                finish_reason="length",
            ),
        )
        result = pb.smoke_test_llamacpp_model("qwen3.6-thinker.gguf")
        assert result["ok"] is False
        assert "finish_reason=length" in result["error"]
        assert "reasoning but no final answer" in result["error"]
        assert "Analyze input" in result["error"]

    def test_request_disables_thinking_and_lifts_token_budget(self, monkeypatch):
        # Capture the outgoing request body and verify the smoke test
        # actively asks the server to skip chain-of-thought and gives
        # reasoning models headroom. These two together are what stops
        # a Qwen3-style model from eating its whole budget on <think>.
        import urllib.request

        captured = {}

        def fake_urlopen(req, *a, **kw):
            captured["payload"] = json.loads(req.data.decode())
            return _FakeChatResp("READY", usage={"completion_tokens": 1})

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        pb.smoke_test_llamacpp_model("any-model.gguf")
        assert captured["payload"]["max_tokens"] >= 256
        assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


class TestSmokeTestLmStudioModel:
    def test_returns_ok_true_with_usage_and_timing(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("READY", usage={"completion_tokens": 5}),
        )
        result = pb.smoke_test_lmstudio_model("qwen3-coder:30b")
        assert result["ok"] is True
        assert result["response"] == "READY"
        assert result["completion_tokens"] == 5
        assert result["duration_seconds"] > 0
        assert isinstance(result["tokens_per_second"], float)
        assert result["tokens_per_second"] > 0

    def test_returns_ok_false_when_response_not_ready(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("nope", usage={"completion_tokens": 1}),
        )
        result = pb.smoke_test_lmstudio_model("qwen3-coder:30b")
        assert result["ok"] is False

    def test_missing_usage_leaves_tokens_per_second_none(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeChatResp("READY"),
        )
        result = pb.smoke_test_lmstudio_model("qwen3-coder:30b")
        assert result["ok"] is True
        assert result["tokens_per_second"] is None
        assert result["completion_tokens"] is None

    def test_returns_ok_false_on_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        result = pb.smoke_test_lmstudio_model("qwen3-coder:30b")
        assert result["ok"] is False
        assert "error" in result


class TestLlamaCppAdapter:
    def test_name_and_recommend_params(self):
        adapter = pb.LlamaCppAdapter()
        assert adapter.name == "llamacpp"
        assert adapter.recommend_params("balanced") == {"provider": "llamacpp", "extra_flags": []}
        assert adapter.recommend_params("fast") == {"provider": "llamacpp", "extra_flags": []}
        assert adapter.recommend_params("quality") == {"provider": "llamacpp", "extra_flags": []}

    def test_healthcheck_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": False,
                "binary": "",
                "server_running": False,
                "server_port": 8001,
                "model": None,
            },
        )
        adapter = pb.LlamaCppAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is False
        assert "not found" in result["detail"]

    def test_healthcheck_when_binary_present_but_server_down(self, monkeypatch):
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
        adapter = pb.LlamaCppAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is False
        assert "not running" in result["detail"]

    def test_healthcheck_when_server_running(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "binary": "llama-server",
                "server_running": True,
                "server_port": 8001,
                "model": "q.gguf",
            },
        )
        adapter = pb.LlamaCppAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is True
        assert "8001" in result["detail"]

    def test_list_models_when_server_running_with_model(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_info",
            lambda: {
                "present": True,
                "binary": "llama-server",
                "server_running": True,
                "server_port": 8001,
                "model": "qwen.gguf",
            },
        )
        adapter = pb.LlamaCppAdapter()
        models = adapter.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "qwen.gguf"
        assert models[0]["format"] == "gguf"
        assert models[0]["local"] is True

    def test_list_models_when_server_not_running(self, monkeypatch):
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
        adapter = pb.LlamaCppAdapter()
        assert adapter.list_models() == []

    def test_run_test_delegates_to_smoke_test(self, monkeypatch):
        monkeypatch.setattr(
            pb, "smoke_test_llamacpp_model", lambda m: {"ok": True, "response": "READY"}
        )
        adapter = pb.LlamaCppAdapter()
        result = adapter.run_test("qwen.gguf")
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Router9Adapter — 9router runtime adapter (issue #51).
# ---------------------------------------------------------------------------


class _FakeModelsResp:
    """Minimal fake OpenAI-compatible /v1/models response for urllib mocking."""

    def __init__(self, model_ids: list[str], status: int = 200):
        self._body = json.dumps({"data": [{"id": mid} for mid in model_ids]}).encode()
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeOpenRouterChatResp:
    """Minimal fake OpenAI-compatible chat completion response."""

    status = 200

    def __init__(self, content: str, completion_tokens: int = 1):
        self._body = json.dumps(
            {
                "choices": [{"message": {"content": content}}],
                "usage": {"completion_tokens": completion_tokens},
            }
        ).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class TestRouter9Adapter:
    def test_name_and_recommend_params(self):
        adapter = pb.Router9Adapter()
        assert adapter.name == "9router"
        assert adapter.recommend_params("balanced") == {"provider": "9router", "extra_flags": []}
        assert adapter.recommend_params("fast") == {"provider": "9router", "extra_flags": []}

    def test_detect_present_when_models_endpoint_responds(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda *a, **kw: _FakeModelsResp(["kr/claude-sonnet-4.5"])
        )
        adapter = pb.Router9Adapter()
        result = adapter.detect()
        assert result["present"] is True
        assert result["base_url"] == pb.ROUTER9_BASE_URL

    def test_detect_returns_not_present_on_url_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        adapter = pb.Router9Adapter()
        result = adapter.detect()
        assert result["present"] is False

    def test_healthcheck_ok_when_models_listed(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["kr/claude-sonnet-4.5", "kr/gpt-4o"]),
        )
        adapter = pb.Router9Adapter()
        result = adapter.healthcheck()
        assert result["ok"] is True
        assert "2" in result["detail"]

    def test_healthcheck_fails_when_unreachable(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        adapter = pb.Router9Adapter()
        result = adapter.healthcheck()
        assert result["ok"] is False

    def test_list_models_returns_cloud_routed_entries(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["kr/claude-sonnet-4.5"]),
        )
        adapter = pb.Router9Adapter()
        models = adapter.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "kr/claude-sonnet-4.5"
        assert models[0]["local"] is False
        assert models[0]["format"] == "cloud-routed"

    def test_list_models_empty_on_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        adapter = pb.Router9Adapter()
        assert adapter.list_models() == []

    def test_run_test_delegates_to_smoke_test_models(self, monkeypatch):
        sentinel = {"ok": True, "models": ["kr/claude-sonnet-4.5"], "response": "1 models"}
        monkeypatch.setattr(pb, "smoke_test_router9_models", lambda *a, **kw: sentinel)
        adapter = pb.Router9Adapter()
        # run_test deliberately ignores the model arg; it only probes /v1/models.
        assert adapter.run_test("kr/anything") == sentinel


class TestSmokeTestRouter9Models:
    def test_returns_ok_with_model_ids(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["kr/claude-sonnet-4.5", "or/gpt-5"]),
        )
        result = pb.smoke_test_router9_models()
        assert result["ok"] is True
        assert "kr/claude-sonnet-4.5" in result["models"]
        assert "or/gpt-5" in result["models"]

    def test_returns_ok_false_on_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        result = pb.smoke_test_router9_models()
        assert result["ok"] is False
        assert "error" in result

    def test_uses_custom_base_url(self, monkeypatch):
        import urllib.request

        seen: dict[str, str] = {}

        def fake_urlopen(req, *a, **kw):
            seen["url"] = req.full_url
            return _FakeModelsResp([])

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        pb.smoke_test_router9_models("http://other-host:9999/v1")
        assert seen["url"] == "http://other-host:9999/v1/models"

    def test_source_does_not_call_chat_completions(self):
        """Mechanical pin: smoke_test_router9_models source must NOT reference chat/completions.

        Routes to paid cloud models — we can never call a chat endpoint
        as part of detection or smoke testing, even by mistake. We strip
        the docstring (which mentions /chat/completions for the explicit
        warning) and inspect only the executable code lines.
        """
        from pathlib import Path

        source = Path(pb.__file__).read_text()
        marker = "def smoke_test_router9_models("
        start = source.index(marker)
        # Slice until the next top-level def; \n at column 0.
        rest = source[start + len(marker) :]
        end_idx = rest.find("\ndef ")
        body = rest[:end_idx] if end_idx != -1 else rest
        # Strip the triple-quoted docstring before the assertion so the
        # warning text in the docstring doesn't trip the pin.
        if '"""' in body:
            first = body.index('"""')
            second = body.index('"""', first + 3)
            body = body[:first] + body[second + 3 :]
        assert "chat/completions" not in body, (
            "smoke_test_router9_models must never call /chat/completions — "
            "that would burn paid quota."
        )


# ---------------------------------------------------------------------------
# OpenRouterAdapter — OpenRouter hosted-SaaS runtime adapter (issue #83).
# ---------------------------------------------------------------------------


class TestOpenRouterAdapter:
    def test_name_and_recommend_params(self):
        adapter = pb.OpenRouterAdapter()
        assert adapter.name == "openrouter"
        assert adapter.recommend_params("balanced") == {
            "provider": "openrouter",
            "extra_flags": [],
        }
        assert adapter.recommend_params("fast") == {
            "provider": "openrouter",
            "extra_flags": [],
        }

    def test_detect_present_when_models_endpoint_responds(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["anthropic/claude-sonnet-4.6"]),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.detect()
        assert result["present"] is True
        assert result["base_url"] == pb.OPENROUTER_BASE_URL

    def test_detect_returns_not_present_on_url_error(self, monkeypatch):
        """Offline machines must not crash detect() — wizard discover step
        runs unconditionally and a raised exception would abort it."""
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("dns failure")),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.detect()
        assert result["present"] is False

    def test_detect_returns_not_present_on_timeout(self, monkeypatch):
        """Network timeout (5s budget) must NOT propagate."""
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("timed out")),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.detect()
        assert result["present"] is False

    def test_detect_returns_not_present_on_non_2xx(self, monkeypatch):
        """Non-2xx response (e.g. 503) must not be treated as present."""
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp([], status=503),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.detect()
        assert result["present"] is False

    def test_healthcheck_ok_when_models_listed(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["anthropic/claude-sonnet-4.6", "openai/gpt-4o"]),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is True
        assert "2" in result["detail"]

    def test_healthcheck_fails_when_unreachable(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        adapter = pb.OpenRouterAdapter()
        result = adapter.healthcheck()
        assert result["ok"] is False

    def test_list_models_returns_cloud_routed_entries(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["anthropic/claude-sonnet-4.6"]),
        )
        adapter = pb.OpenRouterAdapter()
        models = adapter.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "anthropic/claude-sonnet-4.6"
        assert models[0]["local"] is False
        assert models[0]["format"] == "cloud-routed"

    def test_list_models_empty_on_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        adapter = pb.OpenRouterAdapter()
        assert adapter.list_models() == []

    def test_run_test_delegates_to_selected_model_smoke_test(self, monkeypatch):
        sentinel = {
            "ok": True,
            "model": "anthropic/anything",
            "response": "READY",
        }

        def fake_smoke(model, **kw):
            assert model == "anthropic/anything"
            return sentinel

        monkeypatch.setattr(pb, "smoke_test_openrouter_model", fake_smoke)
        adapter = pb.OpenRouterAdapter()
        assert adapter.run_test("anthropic/anything") == sentinel


class TestSmokeTestOpenRouterModels:
    def test_returns_ok_with_model_ids(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeModelsResp(["anthropic/claude-sonnet-4.6", "openai/gpt-4o"]),
        )
        result = pb.smoke_test_openrouter_models()
        assert result["ok"] is True
        assert "anthropic/claude-sonnet-4.6" in result["models"]
        assert "openai/gpt-4o" in result["models"]

    def test_returns_ok_false_on_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("refused")),
        )
        result = pb.smoke_test_openrouter_models()
        assert result["ok"] is False
        assert "error" in result

    def test_uses_custom_base_url(self, monkeypatch):
        import urllib.request

        seen: dict[str, str] = {}

        def fake_urlopen(req, *a, **kw):
            seen["url"] = req.full_url
            return _FakeModelsResp([])

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        pb.smoke_test_openrouter_models("http://other-host:9999/api/v1")
        assert seen["url"] == "http://other-host:9999/api/v1/models"

    def test_catalog_probe_source_does_not_call_chat_completions(self):
        """The /models catalog probe stays metadata-only.

        The selected-model smoke test lives in smoke_test_openrouter_model().
        """
        from pathlib import Path

        source = Path(pb.__file__).read_text()
        marker = "def smoke_test_openrouter_models("
        start = source.index(marker)
        rest = source[start + len(marker) :]
        end_idx = rest.find("\ndef ")
        body = rest[:end_idx] if end_idx != -1 else rest
        if '"""' in body:
            first = body.index('"""')
            second = body.index('"""', first + 3)
            body = body[:first] + body[second + 3 :]
        assert "chat/completions" not in body


class TestSmokeTestOpenRouterModel:
    def test_posts_minimal_prompt_to_selected_model(self, monkeypatch):
        import urllib.request

        seen: dict[str, Any] = {}

        def fake_urlopen(req, timeout):
            seen["url"] = req.full_url
            seen["payload"] = json.loads(req.data)
            seen["headers"] = dict(req.headers)
            return _FakeOpenRouterChatResp("READY", completion_tokens=4)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = pb.smoke_test_openrouter_model(
            "anthropic/claude-sonnet-4.6",
            api_key="openrouter-test-key",  # pragma: allowlist secret
        )

        assert result["ok"] is True
        assert result["model"] == "anthropic/claude-sonnet-4.6"
        assert seen["url"] == f"{pb.OPENROUTER_BASE_URL}/chat/completions"
        assert seen["payload"]["model"] == "anthropic/claude-sonnet-4.6"
        assert seen["payload"]["messages"][0]["content"] == "Reply with exactly READY"
        assert seen["headers"].get("Authorization") == "Bearer openrouter-test-key"

    def test_failure_mentions_selected_model(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("bad request")),
        )
        result = pb.smoke_test_openrouter_model("anthropic/bad-model")
        assert result["ok"] is False
        assert result["model"] == "anthropic/bad-model"
        assert "anthropic/bad-model" in result["error"]


class TestFetchOpenRouterFreeModels:
    """Tests for fetch_openrouter_free_models — catalog fetch + free-tier filter."""

    def _fake_response(self, models: list[dict[str, Any]]) -> _FakeModelsResp:
        """Build a fake /models response with full model objects."""
        return _FakeModelsResp.__new__(_FakeModelsResp)

    def _make_free_model(self, **overrides) -> dict[str, Any]:
        """Build a single free-tier model dict matching the OpenRouter schema."""
        base = {
            "id": "google/gemma-4-31b-it:free",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {
                "tokenizer": "gemini",
                "modality": "text->text",
            },
            "supported_parameters": ["tools", "response_format"],
            "description": "A free model by Google.",
        }
        base.update(overrides)
        return base

    def _make_paid_model(self, **overrides) -> dict[str, Any]:
        """Build a paid-tier model dict."""
        base = {
            "id": "anthropic/claude-sonnet-4.6",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "architecture": {
                "tokenizer": "claude",
                "modality": "text->text",
            },
            "supported_parameters": ["tools"],
            "description": "A paid model.",
        }
        base.update(overrides)
        return base

    def _fake_urlopen_factory(self, models: list[dict[str, Any]]):
        """Return a urlopen replacement that serves the given model list."""

        body = json.dumps({"data": models}).encode()

        class _Resp:
            status = 200

            def read(self):
                return body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        return lambda *a, **kw: _Resp()

    def test_returns_only_free_models(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(id="google/gemma-4-31b-it:free"),
            self._make_paid_model(id="anthropic/claude-sonnet-4.6"),
            self._make_free_model(
                id="mistralai/mistral-7b-instruct:free",
                context_length=32768,
                architecture={"tokenizer": "mistral", "modality": "text->text"},
                supported_parameters=[],
            ),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is True
        ids = [m["id"] for m in result["models"]]
        assert "google/gemma-4-31b-it:free" in ids
        assert "mistralai/mistral-7b-instruct:free" in ids
        assert "anthropic/claude-sonnet-4.6" not in ids

    def test_sorts_by_context_length_descending(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(id="small-model:free", context_length=8192),
            self._make_free_model(id="large-model:free", context_length=262144),
            self._make_free_model(id="mid-model:free", context_length=32768),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        ids = [m["id"] for m in result["models"]]
        assert ids == ["large-model:free", "mid-model:free", "small-model:free"]

    def test_derived_capabilities_include_function_calling(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(
                id="google/gemma-4-31b-it:free",
                supported_parameters=["tools", "response_format"],
            ),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        caps = result["models"][0]["capabilities"]
        assert "function-calling" in caps
        assert "structured-output" in caps

    def test_derived_capabilities_text_only(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(
                id="simple-model:free",
                supported_parameters=["temperature"],
            ),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        caps = result["models"][0]["capabilities"]
        assert "text" in caps
        assert "function-calling" not in caps

    def test_image_modality(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(
                id="vision-model:free",
                architecture={"tokenizer": "clip", "modality": "text+image->text"},
            ),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        caps = result["models"][0]["capabilities"]
        assert "image" in caps

    def test_returns_error_on_connection_failure(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(urllib.error.URLError("Connection refused")),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is False
        assert len(result["models"]) == 0
        assert "error" in result

    def test_truncates_long_descriptions(self, monkeypatch):
        import urllib.request

        long_desc = "A" * 200
        models = [
            self._make_free_model(
                id="verbose-model:free",
                description=long_desc,
            ),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        desc = result["models"][0]["description"]
        assert len(desc) <= 80
        assert desc.endswith("...")

    def test_handles_zero_context_length(self, monkeypatch):
        import urllib.request

        models = [
            self._make_free_model(id="noctx:free", context_length=0),
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is True
        assert result["models"][0]["context_length"] == 0

    def test_handles_missing_context_length(self, monkeypatch):
        import urllib.request

        m = self._make_free_model(id="nolen:free")
        del m["context_length"]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory([m]),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is True
        assert result["models"][0]["context_length"] == 0

    def test_empty_catalog_returns_ok_with_empty_list(self, monkeypatch):
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory([]),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is True
        assert result["models"] == []

    def test_paid_models_with_zero_prompt_but_nonzero_completion_excluded(self, monkeypatch):
        import urllib.request

        models = [
            {
                "id": "partial-free",
                "context_length": 8192,
                "pricing": {"prompt": "0", "completion": "0.001"},
                "architecture": {"tokenizer": "x", "modality": "text->text"},
                "supported_parameters": [],
                "description": "",
            },
        ]
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            self._fake_urlopen_factory(models),
        )
        result = pb.fetch_openrouter_free_models()
        assert result["ok"] is True
        assert len(result["models"]) == 0

    def test_uses_custom_base_url(self, monkeypatch):
        import urllib.request

        seen: dict[str, str] = {}

        def fake_urlopen(req, *a, **kw):
            seen["url"] = req.full_url

            class _Resp:
                status = 200

                def read(self):
                    return json.dumps({"data": []}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        pb.fetch_openrouter_free_models("http://custom:9999/api/v1")
        assert seen["url"] == "http://custom:9999/api/v1/models"


# ---------------------------------------------------------------------------
# llama.cpp server lifecycle helpers (issue #53).
# ---------------------------------------------------------------------------


class TestLlamaCppArgBuilders:
    def test_safe_repo_slug_basic(self):
        assert pb.safe_repo_slug("bartowski/Qwen2.5-Coder-7B") == "bartowski-Qwen2.5-Coder-7B"

    def test_safe_repo_slug_strips_unsafe_chars(self):
        assert pb.safe_repo_slug("a/b c?d&e") == "a-b-c-d-e"

    def test_safe_repo_slug_falls_back_when_empty(self):
        assert pb.safe_repo_slug("////") == "model"

    def test_default_ctx_size_fits_real_coding_session(self):
        # 128k is the floor that survives a real coding turn — Claude Code's
        # system prompt (~26k) plus a single diff-read or multi-file tool
        # call already pushed the previous 32k default to 41k+ and 400'd in
        # production. All current coding-tuned models advertise ≥128k native
        # context, so the auto-started llama-server must match. Lowering
        # this default regresses the wizard's first user-visible task.
        assert pb.LLAMACPP_CTX_SIZE >= 131072

    def test_build_argv_shape(self):
        argv = pb.build_llamacpp_server_args(
            binary="/usr/local/bin/llama-server",
            model_path="/tmp/model.gguf",
            port=8001,
            host="127.0.0.1",
            ctx_size=4096,
            n_gpu_layers=-1,
            threads=8,
        )
        assert argv[0] == "/usr/local/bin/llama-server"
        assert "--model" in argv and argv[argv.index("--model") + 1] == "/tmp/model.gguf"
        assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
        assert "--port" in argv and argv[argv.index("--port") + 1] == "8001"
        assert "--ctx-size" in argv and argv[argv.index("--ctx-size") + 1] == "4096"
        assert "--n-gpu-layers" in argv and argv[argv.index("--n-gpu-layers") + 1] == "-1"
        assert "--threads" in argv and argv[argv.index("--threads") + 1] == "8"


class TestLlamaCppGpuOffload:
    def test_env_override_wins_over_profile(self, monkeypatch):
        monkeypatch.setattr(pb, "LLAMACPP_N_GPU_LAYERS", "33")
        out = pb.detect_llamacpp_gpu_offload({"llmfit_system": {"system": {"has_gpu": False}}})
        assert out["n_gpu_layers"] == 33
        assert out["kind"] == "env-override"

    def test_apple_silicon_profile_uses_metal(self, monkeypatch):
        monkeypatch.setattr(pb, "LLAMACPP_N_GPU_LAYERS", None)
        profile = {
            "llmfit_system": {"system": {"has_gpu": True, "gpu_name": "apple-m2"}},
        }
        out = pb.detect_llamacpp_gpu_offload(profile)
        assert out["n_gpu_layers"] == -1
        assert out["kind"] == "metal"

    def test_cpu_only_when_no_signal(self, monkeypatch):
        import platform as _platform

        monkeypatch.setattr(pb, "LLAMACPP_N_GPU_LAYERS", None)
        monkeypatch.setattr(pb.shutil, "which", lambda name: None)
        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        out = pb.detect_llamacpp_gpu_offload(None)
        assert out["n_gpu_layers"] == 0
        assert out["kind"] == "cpu"

    def test_cuda_detected_when_nvidia_smi_present(self, monkeypatch):
        import platform as _platform

        monkeypatch.setattr(pb, "LLAMACPP_N_GPU_LAYERS", None)
        monkeypatch.setattr(
            pb.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
        )
        monkeypatch.setattr(_platform, "system", lambda: "Linux")
        monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
        out = pb.detect_llamacpp_gpu_offload(None)
        assert out["n_gpu_layers"] == -1
        assert out["kind"] == "cuda"

    def test_threads_uses_profile_cpu_cores(self, monkeypatch):
        monkeypatch.setattr(pb, "LLAMACPP_THREADS", None)
        profile = {"llmfit_system": {"system": {"cpu_cores": 12}}}
        assert pb.detect_llamacpp_threads(profile) == 12

    def test_threads_caps_at_16(self, monkeypatch):
        monkeypatch.setattr(pb, "LLAMACPP_THREADS", None)
        profile = {"llmfit_system": {"system": {"cpu_cores": 64}}}
        assert pb.detect_llamacpp_threads(profile) == 16

    def test_threads_env_override(self, monkeypatch):
        monkeypatch.setattr(pb, "LLAMACPP_THREADS", "6")
        assert pb.detect_llamacpp_threads(None) == 6


class TestLlamaCppWaitUntilReady:
    def test_returns_true_when_health_responds_200(self, monkeypatch):
        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_urlopen(url, timeout=2):
            assert "/health" in url
            return _Resp()

        monkeypatch.setattr(pb, "LLAMACPP_SERVER_HOST", "127.0.0.1")
        import urllib.request as _u

        monkeypatch.setattr(_u, "urlopen", _fake_urlopen)
        assert pb.llamacpp_wait_until_ready(port=18001, timeout=2.0, poll_interval=0.01) is True

    def test_returns_false_on_persistent_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request as _u

        def _fake_urlopen(url, timeout=2):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(_u, "urlopen", _fake_urlopen)
        # Tiny timeout so the test is fast.
        assert pb.llamacpp_wait_until_ready(port=18002, timeout=0.2, poll_interval=0.05) is False

    def test_short_circuits_when_proc_already_exited(self, monkeypatch):
        import urllib.error
        import urllib.request as _u

        called = {"hits": 0}

        def _fake_urlopen(url, timeout=2):
            called["hits"] += 1
            raise urllib.error.URLError("should not be called")

        monkeypatch.setattr(_u, "urlopen", _fake_urlopen)

        class _DeadProc:
            def poll(self):
                return 1

        # The dead-proc short-circuit must bail out without burning the timeout.
        assert (
            pb.llamacpp_wait_until_ready(
                port=18099, timeout=30.0, poll_interval=5.0, proc=_DeadProc()
            )
            is False
        )
        # And without ever attempting the HTTP probe.
        assert called["hits"] == 0


class TestLlamaCppStartServer:
    def test_returns_error_when_binary_missing(self, monkeypatch, isolated_state):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(pb_mod, "llamacpp_detect", lambda: {"present": False})
        out = pb_mod.llamacpp_start_server(model_path="/tmp/model.gguf")
        assert out["ok"] is False
        assert "binary not found" in out["error"]
        assert out["argv"] == []

    def test_returns_error_when_model_path_missing(self, monkeypatch, isolated_state, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(
            pb_mod, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server"}
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        missing = tmp_path / "ghost.gguf"
        out = pb_mod.llamacpp_start_server(model_path=str(missing))
        assert out["ok"] is False
        assert "not found" in out["error"]

    def test_spawn_succeeds_when_health_responds(self, monkeypatch, isolated_state, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        # Pretend the binary is present.
        monkeypatch.setattr(
            pb_mod, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server"}
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        # Pretend the model file exists.
        model_file = tmp_path / "fake.gguf"
        model_file.write_bytes(b"\x00")

        spawned: dict = {}

        class _FakeProc:
            pid = 424242

            def poll(self):
                return None  # still running

        def _fake_popen(argv, **kwargs):
            spawned["argv"] = argv
            spawned["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(pb_mod.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(pb_mod, "llamacpp_wait_until_ready", lambda **kw: True)

        out = pb_mod.llamacpp_start_server(model_path=str(model_file), port=18003)
        assert out["ok"] is True
        handle = out["handle"]
        assert handle is not None
        assert handle.pid == 424242
        assert handle.port == 18003
        assert handle.model_path == str(model_file)
        # argv shape sanity-check
        assert "--model" in handle.argv
        assert str(model_file) in handle.argv
        assert "--port" in handle.argv
        # Pid file written under STATE_DIR/run
        assert (pb_mod.LLAMACPP_PID_DIR / "llama-server-18003.pid").exists()

    def test_spawn_failure_returns_argv_for_manual_run(self, monkeypatch, isolated_state, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(
            pb_mod, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server"}
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        model_file = tmp_path / "fake.gguf"
        model_file.write_bytes(b"\x00")

        def _fail_popen(argv, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'llama-server'")

        monkeypatch.setattr(pb_mod.subprocess, "Popen", _fail_popen)
        out = pb_mod.llamacpp_start_server(model_path=str(model_file), port=18004)
        assert out["ok"] is False
        assert "failed to spawn" in out["error"]
        # Caller can echo argv to the user even on spawn failure.
        assert any("--model" in a for a in out["argv"])
        assert any(str(model_file) == a for a in out["argv"])

    def test_readiness_timeout_terminates_child(self, monkeypatch, isolated_state, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(
            pb_mod, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server"}
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        model_file = tmp_path / "fake.gguf"
        model_file.write_bytes(b"\x00")

        class _FakeProc:
            pid = 99999

            def poll(self):
                return None

        monkeypatch.setattr(pb_mod.subprocess, "Popen", lambda argv, **kw: _FakeProc())
        monkeypatch.setattr(pb_mod, "llamacpp_wait_until_ready", lambda **kw: False)
        stop_calls: list = []
        monkeypatch.setattr(
            pb_mod, "llamacpp_stop_server", lambda h, **kw: stop_calls.append(h) or True
        )

        out = pb_mod.llamacpp_start_server(model_path=str(model_file), port=18005)
        assert out["ok"] is False
        assert "did not become ready" in out["error"]
        assert len(stop_calls) == 1

    def test_post_readiness_exit_cleans_up_pid_file(self, monkeypatch, isolated_state, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(
            pb_mod, "llamacpp_detect", lambda: {"present": True, "binary": "llama-server"}
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        model_file = tmp_path / "fake.gguf"
        model_file.write_bytes(b"\x00")

        class _FlappingProc:
            """Returns None during wait_until_ready, then exited afterwards.

            The wait_until_ready monkeypatch swallows the proc kwarg so this
            class only needs to report the exit status when the post-readiness
            assertion calls poll().
            """

            pid = 88888
            returncode = 13

            def poll(self):
                return 13

        monkeypatch.setattr(pb_mod.subprocess, "Popen", lambda argv, **kw: _FlappingProc())
        monkeypatch.setattr(pb_mod, "llamacpp_wait_until_ready", lambda **kw: True)

        out = pb_mod.llamacpp_start_server(model_path=str(model_file), port=18006)
        assert out["ok"] is False
        assert "after readiness probe" in out["error"]
        # Pid file must NOT remain on disk pointing at a dead process.
        assert not (pb_mod.LLAMACPP_PID_DIR / "llama-server-18006.pid").exists()


class TestLlamaCppStopServer:
    def test_refuses_to_stop_servers_we_did_not_start(self, isolated_state):
        pb_mod, _wiz, _ = isolated_state
        handle = pb_mod.LlamaServerHandle(
            pid=12345,
            port=8001,
            host="127.0.0.1",
            model_path="/tmp/x.gguf",
            argv=[],
            log_path="",
            pid_file="",
            we_started_it=False,
        )
        assert pb_mod.llamacpp_stop_server(handle) is False

    def test_returns_true_when_pid_already_gone(self, monkeypatch, isolated_state):
        pb_mod, _wiz, _ = isolated_state
        handle = pb_mod.LlamaServerHandle(
            pid=99999999,
            port=8001,
            host="127.0.0.1",
            model_path="/tmp/x.gguf",
            argv=[],
            log_path="",
            pid_file=str(_wiz_temp_pid_path(isolated_state)),
            we_started_it=True,
        )

        def _raise_lookup(*a, **kw):
            raise ProcessLookupError()

        monkeypatch.setattr(pb_mod.os, "kill", _raise_lookup)
        if hasattr(pb_mod.os, "killpg"):
            monkeypatch.setattr(pb_mod.os, "killpg", _raise_lookup)
        assert pb_mod.llamacpp_stop_server(handle, grace_seconds=0.1) is True

    def test_uses_popen_wait_when_proc_attached(self, isolated_state, monkeypatch):
        """When handle.proc is set, stop_server prefers Popen.wait over raw pid
        polling — that closes the pid-recycle race window."""
        pb_mod, _wiz, _ = isolated_state

        wait_calls: list[float | None] = []

        class _FakeProc:
            pid = 99999999
            returncode = 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

            def wait(self, timeout: float | None = None) -> int:
                wait_calls.append(timeout)
                return 0

        # Stub _signal_process so it doesn't actually fire on the host.
        monkeypatch.setattr(pb_mod, "_signal_process", lambda pid, sig: None)

        fake = _FakeProc()
        handle = pb_mod.LlamaServerHandle(
            pid=fake.pid,
            port=8001,
            host="127.0.0.1",
            model_path="/tmp/x.gguf",
            argv=[],
            log_path="",
            pid_file=str(_wiz_temp_pid_path(isolated_state)),
            we_started_it=True,
            proc=fake,  # type: ignore[arg-type]
        )
        assert pb_mod.llamacpp_stop_server(handle, grace_seconds=0.5) is True
        # First wait should use the grace window; SIGKILL-fallback wait is not
        # expected because the first wait returned cleanly.
        assert wait_calls == [0.5]

    def test_escalates_to_kill_when_terminate_times_out(self, isolated_state, monkeypatch):
        pb_mod, _wiz, _ = isolated_state
        wait_calls: list[float | None] = []
        kill_called = []

        class _StuckProc:
            pid = 99999998
            returncode = -9

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                kill_called.append(True)

            def wait(self, timeout: float | None = None) -> int:
                wait_calls.append(timeout)
                # First call (after SIGTERM) times out; second call (after
                # SIGKILL) returns cleanly.
                if len(wait_calls) == 1:
                    raise pb_mod.subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
                return -9

        monkeypatch.setattr(pb_mod, "_signal_process", lambda pid, sig: None)
        proc = _StuckProc()
        handle = pb_mod.LlamaServerHandle(
            pid=proc.pid,
            port=8001,
            host="127.0.0.1",
            model_path="/tmp/x.gguf",
            argv=[],
            log_path="",
            pid_file=str(_wiz_temp_pid_path(isolated_state)),
            we_started_it=True,
            proc=proc,  # type: ignore[arg-type]
        )
        assert pb_mod.llamacpp_stop_server(handle, grace_seconds=0.1) is True
        assert len(wait_calls) == 2  # one for SIGTERM, one after SIGKILL


def _wiz_temp_pid_path(isolated_state):
    """Return a writable pid-file path inside the isolated state dir."""
    _pb, _wiz, state_dir = isolated_state
    pid_dir = state_dir / "run"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "test.pid"
    pid_file.write_text("99999999")
    return pid_file


# ---------------------------------------------------------------------------
# Lazy-llmfit machine_profile (issue #79).
# ---------------------------------------------------------------------------


def _stub_machine_internals(monkeypatch, pb_mod):
    """
    Patch the slow / network-touching parts of machine_profile so we can
    isolate the run_llmfit branch without mocking llmfit_system itself.
    """

    def fake_command_version(name):
        return {"present": True, "version": f"{name} 1.0.0"}

    monkeypatch.setattr(pb_mod, "command_version", fake_command_version)
    monkeypatch.setattr(pb_mod, "lms_info", lambda: {"present": False, "models": []})
    monkeypatch.setattr(pb_mod, "llamacpp_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(pb_mod, "huggingface_cli_detect", lambda: {"present": False, "version": ""})
    monkeypatch.setattr(
        pb_mod,
        "vllm_info",
        lambda: {"present": False, "version": "", "base_url": "http://localhost:8000"},
    )
    monkeypatch.setattr(pb_mod, "parse_ollama_list", lambda: [])

    class _StubRouter9:
        def detect(self):
            return {"present": False, "version": ""}

        def healthcheck(self):
            return {"ok": False}

    monkeypatch.setattr(pb_mod, "Router9Adapter", _StubRouter9)
    monkeypatch.setattr(
        pb_mod,
        "disk_usage_for",
        lambda _path: {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "free_gib": 0.0,
            "total_gib": 0.0,
        },
    )


class TestMachineProfileLazyLlmfit:
    """Issue #79: machine_profile must accept run_llmfit and respect lazy/skip semantics."""

    def test_machine_profile_skips_llmfit_when_disabled(self, isolated_state, monkeypatch):
        pb_mod, _wiz, _state_dir = isolated_state
        _stub_machine_internals(monkeypatch, pb_mod)
        calls: list[int] = []

        def fake_llmfit():
            calls.append(1)
            return {"system": {"available_ram_gb": 32}}

        monkeypatch.setattr(pb_mod, "llmfit_system", fake_llmfit)

        profile = pb_mod.machine_profile(run_llmfit=False)

        assert calls == [], "llmfit_system must not run when run_llmfit=False"
        assert pb_mod._is_llmfit_skipped(profile.get("llmfit_system"))

    def test_machine_profile_runs_llmfit_when_force(self, isolated_state, monkeypatch):
        pb_mod, _wiz, _state_dir = isolated_state
        _stub_machine_internals(monkeypatch, pb_mod)
        calls: list[int] = []

        def fake_llmfit():
            calls.append(1)
            return {"system": {"available_ram_gb": 32}}

        monkeypatch.setattr(pb_mod, "llmfit_system", fake_llmfit)

        profile = pb_mod.machine_profile(run_llmfit=True)

        assert calls == [1]
        assert profile.get("llmfit_system") == {"system": {"available_ram_gb": 32}}

    def test_machine_profile_force_with_populated_cache_does_not_re_run(
        self, isolated_state, monkeypatch
    ):
        """Cache holds populated llmfit data; even run_llmfit=True must not re-run."""
        pb_mod, _wiz, _state_dir = isolated_state
        _stub_machine_internals(monkeypatch, pb_mod)
        calls: list[int] = []

        def fake_llmfit():
            calls.append(1)
            return {"system": {"available_ram_gb": 32}}

        monkeypatch.setattr(pb_mod, "llmfit_system", fake_llmfit)

        # First call (run_llmfit=True): builds + populates llmfit_system in cache.
        first = pb_mod.machine_profile(run_llmfit=True)
        assert calls == [1]
        assert "system" in first.get("llmfit_system", {})

        # Reset the in-process cache so the next call reads from disk cache.
        pb_mod.invalidate_machine_profile_inproc_cache()

        # Second call (run_llmfit=True): cache hit with populated llmfit_system
        # — must NOT re-run llmfit.
        second = pb_mod.machine_profile(run_llmfit=True)
        assert calls == [1], "populated cache must not trigger a second llmfit run"
        assert second.get("llmfit_system") == first.get("llmfit_system")

    def test_machine_profile_skipped_then_force_re_runs(self, isolated_state, monkeypatch):
        """Cache holds the skip sentinel; forcing run_llmfit must re-run llmfit and persist."""
        pb_mod, _wiz, _state_dir = isolated_state
        _stub_machine_internals(monkeypatch, pb_mod)
        calls: list[int] = []

        def fake_llmfit():
            calls.append(1)
            return {"system": {"available_ram_gb": 64}}

        monkeypatch.setattr(pb_mod, "llmfit_system", fake_llmfit)

        # First call: defer the scan — sentinel is persisted.
        first = pb_mod.machine_profile(run_llmfit=False)
        assert pb_mod._is_llmfit_skipped(first.get("llmfit_system"))
        assert calls == []

        # Reset in-process cache so the next call hits the disk cache (which
        # holds the sentinel) rather than returning the in-memory profile.
        pb_mod.invalidate_machine_profile_inproc_cache()

        # Second call: ask for llmfit. Even though the disk cache exists, the
        # sentinel must trigger a fresh llmfit_system() invocation.
        second = pb_mod.machine_profile(run_llmfit=True)
        assert calls == [1], "llmfit_system must run on the second call"
        assert second.get("llmfit_system") == {"system": {"available_ram_gb": 64}}


# ---------------------------------------------------------------------------
# merge_models_for_engine (issue #79).
# ---------------------------------------------------------------------------


class TestMergeModelsForEngine:
    def test_dedups_installed_over_cached(self, monkeypatch):
        """Same id in installed + cached → one entry, source='installed'."""
        profile = {
            "ollama": {"models": [{"name": "qwen2.5-coder:7b", "local": True, "size": "4.1 GB"}]},
            "llmfit_system": {"system": {"available_ram_gb": 16}},
        }

        # Stub llmfit_coding_candidates to return a candidate whose ollama_tag
        # is already installed.
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda **_kw: [
                {
                    "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "score": 80,
                    "estimated_tps": 30,
                    "ollama_tag": "qwen2.5-coder:7b",
                }
            ],
        )

        merged = pb.merge_models_for_engine(profile, "ollama")
        assert len(merged) == 1
        assert merged[0]["tag"] == "qwen2.5-coder:7b"
        assert merged[0]["source"] == "installed"

    def test_tags_cached_when_not_installed(self, monkeypatch):
        """Cached candidate id NOT in installed list → one entry, source='cached'."""
        profile = {
            "ollama": {"models": []},
            "llmfit_system": {"system": {"available_ram_gb": 32}},
        }
        monkeypatch.setattr(
            pb,
            "llmfit_coding_candidates",
            lambda **_kw: [
                {
                    "name": "Qwen/Qwen3-Coder-30B",
                    "score": 95,
                    "estimated_tps": 12,
                    "ollama_tag": "qwen3-coder:30b",
                    "ram_gb": 18,
                }
            ],
        )

        merged = pb.merge_models_for_engine(profile, "ollama")
        assert len(merged) == 1
        assert merged[0]["tag"] == "qwen3-coder:30b"
        assert merged[0]["source"] == "cached"
        assert merged[0]["candidate"]["score"] == 95

    def test_handles_skipped_llmfit(self):
        """When llmfit_system carries the skip sentinel, only return installed."""
        profile = {
            "ollama": {"models": [{"name": "qwen2.5-coder:7b", "local": True}]},
            "llmfit_system": pb.LLMFIT_SKIPPED_SENTINEL,
        }
        merged = pb.merge_models_for_engine(profile, "ollama")
        assert len(merged) == 1
        assert merged[0]["tag"] == "qwen2.5-coder:7b"
        assert merged[0]["source"] == "installed"

    def test_handles_none_llmfit(self):
        """llmfit_system=None is the not-installed branch — return only installed."""
        profile = {
            "ollama": {"models": [{"name": "qwen2.5-coder:7b", "local": True}]},
            "llmfit_system": None,
        }
        merged = pb.merge_models_for_engine(profile, "ollama")
        assert len(merged) == 1
        assert merged[0]["source"] == "installed"

    def test_empty_both(self, monkeypatch):
        """No installed models AND no cached candidates → empty list."""
        profile = {"ollama": {"models": []}}
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda **_kw: [])
        # No llmfit_system at all → falls through to return-only-installed branch.
        assert pb.merge_models_for_engine(profile, "ollama") == []


# ---------------------------------------------------------------------------
# invalidate_machine_profile_inproc_cache (issue #79).
# ---------------------------------------------------------------------------


class TestInvalidateInprocCache:
    def test_invalidate_clears_inproc_data(self):
        # Seed the cache and verify it's populated.
        pb._set_machine_profile_in_process_cache({"foo": "bar"})
        assert pb._machine_profile_in_process_cache() == {"foo": "bar"}

        pb.invalidate_machine_profile_inproc_cache()
        assert pb._machine_profile_in_process_cache() is None
