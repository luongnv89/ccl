"""
Unit tests for poc_bridge — pure functions + subprocess-mockable helpers.

These tests never touch real ollama / lms / claude / codex binaries. Anything
that would shell out is either patched or routed through the `fake_bin`
fixture defined in conftest.py.
"""

from __future__ import annotations

import json

import poc_bridge as pb

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
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda: [])
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
            lambda: [
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
            lambda: [
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
            lambda: [
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
            lambda: [
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
        monkeypatch.setattr(pb, "llmfit_coding_candidates", lambda: [])
        rec = pb.select_best_model(_empty_profile(), mode="bogus")
        assert rec["mode"] == "balanced"


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

    def test_all_adapters_registry_contains_all_three(self):
        names = {a.name for a in pb.ALL_ADAPTERS}
        assert names == {"ollama", "lmstudio", "llamacpp"}


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


class TestSmokeTestLlamaCppModel:
    def test_returns_ok_true_when_ready_in_response(self, monkeypatch):
        class _FakeResp:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "READY"}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is True
        assert result["response"] == "READY"

    def test_returns_ok_false_when_response_not_ready(self, monkeypatch):
        class _FakeResp:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Hello!"}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeResp())
        result = pb.smoke_test_llamacpp_model("my-model.gguf")
        assert result["ok"] is False

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
