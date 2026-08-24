from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

import claude_codex_local._llamacpp_lifecycle as lifecycle_mod
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


def _write_pid_record_file(path, pid: int, **overrides):
    record = {"pid": pid, "create_time": "1000", "boot_id": "b0", "image": "llama-server"}
    record.update(overrides)
    path.write_text(json.dumps(record))
    return record


class TestLlamaCppLifecyclePidSafety:
    """F-BUG-005: a persisted pid file must never cause an unrelated process
    to be signalled. The recorded (pid, create_time, boot_id) plus the live
    process image are re-verified before any signal; signalling is
    single-PID (no killpg)."""

    def test_recycled_pid_with_foreign_create_time_is_not_signalled(
        self, isolated_state, monkeypatch, tmp_path
    ):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(lifecycle_mod, "LLAMACPP_PID_DIR", tmp_path)
        pid_file = tmp_path / "llama-server-8001.pid"
        _write_pid_record_file(pid_file, 4242)

        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            lifecycle_mod, "_signal_process", lambda pid, sig: signals.append((pid, sig))
        )
        monkeypatch.setattr(lifecycle_mod, "_pid_gone", lambda pid: False)
        monkeypatch.setattr(lifecycle_mod, "_read_boot_id", lambda: "b0")
        # The live process at PID 4242 was created after the recorded one:
        # the PID was recycled by an unrelated process.
        monkeypatch.setattr(lifecycle_mod, "_process_start_marker", lambda pid: "9999")

        out = pb_mod.llamacpp_stop_server_by_port(8001)

        assert out["ok"] is False
        assert out["pid"] == 4242
        assert signals == [], "a recycled PID belonging to an unrelated process was signalled"
        assert "create-time-mismatch" in out["error"]
        assert not pid_file.exists()

    def test_boot_id_mismatch_is_not_signalled(self, isolated_state, monkeypatch, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(lifecycle_mod, "LLAMACPP_PID_DIR", tmp_path)
        pid_file = tmp_path / "llama-server-8001.pid"
        _write_pid_record_file(pid_file, 4242, boot_id="previous-boot")

        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            lifecycle_mod, "_signal_process", lambda pid, sig: signals.append((pid, sig))
        )
        monkeypatch.setattr(lifecycle_mod, "_pid_gone", lambda pid: False)
        monkeypatch.setattr(lifecycle_mod, "_read_boot_id", lambda: "current-boot")
        monkeypatch.setattr(lifecycle_mod, "_process_start_marker", lambda pid: "1000")
        monkeypatch.setattr(lifecycle_mod, "_process_image_name", lambda pid: "llama-server")

        out = pb_mod.llamacpp_stop_server_by_port(8001)

        assert out["ok"] is False
        assert signals == []
        assert "boot-id-mismatch" in out["error"]

    def test_matching_record_signals_single_recorded_pid(
        self, isolated_state, monkeypatch, tmp_path
    ):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(lifecycle_mod, "LLAMACPP_PID_DIR", tmp_path)
        pid_file = tmp_path / "llama-server-8001.pid"
        _write_pid_record_file(pid_file, 4242)

        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            lifecycle_mod, "_signal_process", lambda pid, sig: signals.append((pid, sig))
        )
        monkeypatch.setattr(lifecycle_mod, "_pid_gone", lambda pid: False)
        monkeypatch.setattr(lifecycle_mod, "_read_boot_id", lambda: "b0")
        monkeypatch.setattr(lifecycle_mod, "_process_start_marker", lambda pid: "1000")
        monkeypatch.setattr(lifecycle_mod, "_process_image_name", lambda pid: "llama-server")

        def _gone(pid):
            # Alive until the first signal lands, then gone.
            return bool(signals)

        monkeypatch.setattr(lifecycle_mod, "_pid_gone", _gone)
        monkeypatch.setattr(pb_mod.time, "sleep", lambda _s: None)

        out = pb_mod.llamacpp_stop_server_by_port(8001, grace_seconds=0.1)

        assert out["ok"] is True
        # Exactly the recorded PID was signalled — no process-group signal.
        assert (4242, 15) in signals
        assert all(pid == 4242 for pid, _sig in signals)
        assert not pid_file.exists()

    def test_process_image_mismatch_is_not_signalled(self, isolated_state, monkeypatch, tmp_path):
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(lifecycle_mod, "LLAMACPP_PID_DIR", tmp_path)
        pid_file = tmp_path / "llama-server-8001.pid"
        _write_pid_record_file(pid_file, 4242)

        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            lifecycle_mod, "_signal_process", lambda pid, sig: signals.append((pid, sig))
        )
        monkeypatch.setattr(lifecycle_mod, "_pid_gone", lambda pid: False)
        monkeypatch.setattr(lifecycle_mod, "_read_boot_id", lambda: "b0")
        monkeypatch.setattr(lifecycle_mod, "_process_start_marker", lambda pid: "1000")
        # Same boot, same create time, but the image is now something else.
        monkeypatch.setattr(lifecycle_mod, "_process_image_name", lambda pid: "yes")

        out = pb_mod.llamacpp_stop_server_by_port(8001)

        assert out["ok"] is False
        assert signals == []
        assert "process-image-mismatch" in out["error"]

    def test_legacy_plain_pid_file_is_refused(self, isolated_state, monkeypatch, tmp_path):
        """Pid files written by older ccl versions carry no verifiable
        identity — they must never be signalled."""
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(lifecycle_mod, "LLAMACPP_PID_DIR", tmp_path)
        pid_file = tmp_path / "llama-server-8001.pid"
        pid_file.write_text("4242")

        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            lifecycle_mod, "_signal_process", lambda pid, sig: signals.append((pid, sig))
        )

        out = pb_mod.llamacpp_stop_server_by_port(8001)

        assert out["ok"] is False
        assert out["pid"] is None
        assert signals == []
        assert "unrecognized pid-file format" in out["error"]

    def test_keyboard_interrupt_during_readiness_wait_stops_server_and_removes_pid_file(
        self, isolated_state, monkeypatch, tmp_path
    ):
        """F-BUG-006: Ctrl-C while waiting for readiness must stop the child
        and remove the pid file before re-raising."""
        pb_mod, _wiz, _ = isolated_state
        monkeypatch.setattr(
            lifecycle_mod,
            "llamacpp_detect",
            lambda: {"present": True, "binary": "llama-server"},
        )
        monkeypatch.setattr(
            pb_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/llama-server" if name == "llama-server" else None,
        )
        model_file = tmp_path / "fake.gguf"
        model_file.write_bytes(b"\x00")

        class _FakeProc:
            pid = 424242

            def poll(self):
                return None

        monkeypatch.setattr(pb_mod.subprocess, "Popen", lambda argv, **kw: _FakeProc())
        # Keep the pid-record probes off the real /proc and ps so the fake
        # PID never triggers a live process lookup or a ps subprocess.
        monkeypatch.setattr(lifecycle_mod, "_read_boot_id", lambda: "b0")
        monkeypatch.setattr(lifecycle_mod, "_process_start_marker", lambda pid: "1000")
        monkeypatch.setattr(lifecycle_mod, "_process_image_name", lambda pid: "llama-server")

        def _raise_interrupt(**kw):
            raise KeyboardInterrupt

        monkeypatch.setattr(lifecycle_mod, "llamacpp_wait_until_ready", _raise_interrupt)

        stopped: list = []
        monkeypatch.setattr(
            lifecycle_mod,
            "llamacpp_stop_server",
            lambda h, **kw: stopped.append(h) or True,
        )

        with pytest.raises(KeyboardInterrupt):
            pb_mod.llamacpp_start_server(model_path=str(model_file), port=18042)

        assert len(stopped) == 1
        assert stopped[0].pid == 424242
        assert not (pb_mod.LLAMACPP_PID_DIR / "llama-server-18042.pid").exists()
