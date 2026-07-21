from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

import claude_codex_local.core as pb
import claude_codex_local.engines as engines_pkg
import claude_codex_local.engines.registry as registry
from claude_codex_local.engines import (
    ACTIONS,
    engine_action_matrix,
    engine_names,
    run_engine_action,
)
from claude_codex_local.engines.registry import EngineLifecycleError

# Required engines are those that must have a complete action surface
# (all five actions).  Derived from the registry so adding a new engine
# automatically includes it here.
REQUIRED_ENGINES = {"ollama", "lmstudio", "llamacpp", "vllm", "9router"}


class TestEngineLifecycleRegistry:
    def test_discovers_required_engines(self):
        assert REQUIRED_ENGINES.issubset(set(engine_names()))

    def test_each_required_engine_has_complete_action_surface(self):
        matrix = engine_action_matrix()
        for engine in REQUIRED_ENGINES:
            assert set(matrix[engine]) == set(ACTIONS)

    def test_unknown_engine_fails_before_importing_action(self):
        with pytest.raises(EngineLifecycleError, match="Unknown engine"):
            run_engine_action("totally-new-engine", "config")

    def test_unknown_action_fails_with_contract_message(self):
        with pytest.raises(EngineLifecycleError, match="Unknown action"):
            run_engine_action("ollama", "reindex")

    def test_core_lifecycle_cli_uses_uniform_dispatch(self):
        source = inspect.getsource(pb.main)
        engine_block = source.split('elif args.command == "engine":', 1)[1]

        assert "run_engine_action" in engine_block
        for engine in REQUIRED_ENGINES:
            assert f'== "{engine}"' not in engine_block
            assert f"== '{engine}'" not in engine_block

    def test_sixth_engine_is_discovered_without_core_changes(self, tmp_path, monkeypatch):
        package = tmp_path / "customsixth"
        package.mkdir()
        (package / "__init__.py").write_text('ENGINE_NAME = "customsixth"\n')
        for action in ACTIONS:
            (package / f"{action}.py").write_text(
                "def run(**kwargs):\n"
                f"    return {{'engine': 'customsixth', 'action': '{action}', 'ok': True}}\n"
            )

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(engines_pkg, "__path__", [*engines_pkg.__path__, str(tmp_path)])
        registry._engine_packages.cache_clear()
        try:
            assert "customsixth" in engine_names()
            result = run_engine_action("customsixth", "benchmark")
            assert result == {"engine": "customsixth", "action": "benchmark", "ok": True}
        finally:
            registry._engine_packages.cache_clear()


class TestEngineLifecycleIntegration:
    def test_ollama_test_action_delegates_to_engine_script(self, monkeypatch):
        calls: list[str] = []

        def fake_smoke(model: str):
            calls.append(model)
            return {"ok": True, "response": "READY", "tokens_per_second": 12.5}

        monkeypatch.setattr(pb, "smoke_test_ollama_model", fake_smoke)

        result = run_engine_action("ollama", "test", model="qwen3-coder:30b", dry_run=False)

        assert result["ok"] is True
        assert result["engine"] == "ollama"
        assert result["action"] == "test"
        assert result["data"]["tokens_per_second"] == 12.5
        assert calls == ["qwen3-coder:30b"]

    def test_llamacpp_optimize_uses_profile_without_core_branching(self):
        profile = {
            "llmfit_system": {
                "system": {
                    "has_gpu": True,
                    "gpu_name": "Apple M4",
                    "cpu_cores": 12,
                }
            }
        }

        result = run_engine_action("llamacpp", "optimize", profile=profile)

        assert result["ok"] is True
        assert result["env"]["LLAMACPP_N_GPU_LAYERS"] == "-1"
        assert result["env"]["LLAMACPP_THREADS"] == "12"

    def test_9router_benchmark_does_not_call_paid_chat_endpoint(self):
        result = run_engine_action("9router", "benchmark", dry_run=False)

        assert result["ok"] is True
        assert result["data"]["skipped_chat"] is True
        assert "paid" in result["detail"]


class TestEngineLifecycleCliE2E:
    def test_core_engine_config_cli_prints_json(self):
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "claude_codex_local.core",
                "engine",
                "ollama",
                "config",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads(cp.stdout)
        assert data["engine"] == "ollama"
        assert data["action"] == "config"
        assert data["data"]["base_url"].startswith("http")

    def test_core_engine_benchmark_cli_dry_runs_without_live_engine(self):
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "claude_codex_local.core",
                "engine",
                "vllm",
                "benchmark",
                "--model",
                "Qwen/Qwen2.5-0.5B-Instruct",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        data = json.loads(cp.stdout)
        assert data["engine"] == "vllm"
        assert data["action"] == "benchmark"
        assert data["data"]["dry_run"] is True
