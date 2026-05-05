"""
Unit tests for core — pure functions + subprocess-mockable helpers.

These tests never touch real ollama / lms / claude / codex binaries. Anything
that would shell out is either patched or routed through the `fake_bin`
fixture defined in conftest.py.
"""

from __future__ import annotations

import json

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

    def test_llamacpp_returns_raw_name(self, monkeypatch):
        monkeypatch.setattr(pb, "llmfit_coding_candidates", self._candidates)
        rec = pb.recommend_for_mode(_empty_profile(), "balanced", "llamacpp")
        assert rec is not None
        # llama.cpp uses the HF name directly.
        assert rec["engine_tag"] == "Qwen/Qwen3-Coder-30B"

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

    def test_all_adapters_registry_contains_all_five(self):
        names = {a.name for a in pb.ALL_ADAPTERS}
        assert names == {"ollama", "lmstudio", "llamacpp", "vllm", "9router"}


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

    def test_server_running_when_models_endpoint_responds(self, monkeypatch):
        monkeypatch.setattr(
            pb,
            "llamacpp_detect",
            lambda: {"present": True, "binary": "llama-server", "version": "b1234"},
        )

        class _FakeResp:
            def read(self):
                return json.dumps({"data": [{"id": "my-model.gguf"}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
        result = pb.llamacpp_info()
        assert result["server_running"] is True
        assert result["model"] == "my-model.gguf"

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

    def __init__(self, content: str, usage: dict | None = None):
        body = {"choices": [{"message": {"content": content}}]}
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
# llama.cpp server lifecycle helpers (issue #53).
# ---------------------------------------------------------------------------


class TestLlamaCppArgBuilders:
    def test_safe_repo_slug_basic(self):
        assert pb.safe_repo_slug("bartowski/Qwen2.5-Coder-7B") == "bartowski-Qwen2.5-Coder-7B"

    def test_safe_repo_slug_strips_unsafe_chars(self):
        assert pb.safe_repo_slug("a/b c?d&e") == "a-b-c-d-e"

    def test_safe_repo_slug_falls_back_when_empty(self):
        assert pb.safe_repo_slug("////") == "model"

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


def _wiz_temp_pid_path(isolated_state):
    """Return a writable pid-file path inside the isolated state dir."""
    _pb, _wiz, state_dir = isolated_state
    pid_dir = state_dir / "run"
    pid_dir.mkdir(parents=True, exist_ok=True)
    pid_file = pid_dir / "test.pid"
    pid_file.write_text("99999999")
    return pid_file
