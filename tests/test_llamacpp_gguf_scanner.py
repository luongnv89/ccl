"""
Unit tests for llama.cpp GGUF cache scanning.

Tests the scan_huggingface_gguf_cache() function and its integration with
installed_models_for_engine() for the llamacpp engine.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import claude_codex_local.core as core


class TestScanHuggingfaceGgufCache:
    """Unit tests for scan_huggingface_gguf_cache()."""

    def test_empty_cache_returns_empty_list(self, tmp_path):
        """When HF cache doesn't exist, return empty list."""
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "nonexistent")}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()
            assert result == []

    def test_scans_single_gguf_model(self, tmp_path):
        """Detect a single GGUF model in HF cache."""
        # Create mock HF cache structure
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--bartowski--Qwen2.5-Coder-7B-Instruct-GGUF"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Create a GGUF file
        gguf_file = snapshot_dir / "model-Q4_K_M.gguf"
        gguf_file.write_bytes(b"x" * (7 * 1024**3))  # 7 GB

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            assert len(result) == 1
            assert result[0]["display"].startswith(
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF-Q4_K_M"
            )
            assert "7.0 GB" in result[0]["display"]
            assert result[0]["path"] == str(gguf_file)
            assert result[0]["size_gb"] == pytest.approx(7.0, rel=0.1)

    def test_scans_multiple_gguf_models(self, tmp_path):
        """Detect multiple GGUF models across different repos."""
        cache_dir = tmp_path / "hub"

        # Model 1: bartowski/Qwen2.5-Coder-7B-Instruct-GGUF
        model1_dir = cache_dir / "models--bartowski--Qwen2.5-Coder-7B-Instruct-GGUF"
        snapshot1_dir = model1_dir / "snapshots" / "abc123"
        snapshot1_dir.mkdir(parents=True)
        gguf1 = snapshot1_dir / "model-Q4_K_M.gguf"
        gguf1.write_bytes(b"x" * (7 * 1024**3))

        # Model 2: TheBloke/deepseek-coder-6.7B-GGUF
        model2_dir = cache_dir / "models--TheBloke--deepseek-coder-6.7B-GGUF"
        snapshot2_dir = model2_dir / "snapshots" / "def456"
        snapshot2_dir.mkdir(parents=True)
        gguf2 = snapshot2_dir / "model-Q5_K_M.gguf"
        gguf2.write_bytes(b"x" * (5 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            assert len(result) == 2
            displays = [m["display"] for m in result]
            assert any("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF" in d for d in displays)
            assert any("TheBloke/deepseek-coder-6.7B-GGUF" in d for d in displays)

    def test_respects_hf_home_env_var(self, tmp_path):
        """Use HF_HOME when set."""
        custom_cache = tmp_path / "custom_hf"
        cache_dir = custom_cache / "hub"
        model_dir = cache_dir / "models--org--repo"
        snapshot_dir = model_dir / "snapshots" / "snap1"
        snapshot_dir.mkdir(parents=True)

        gguf_file = snapshot_dir / "model.gguf"
        gguf_file.write_bytes(b"x" * (3 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(custom_cache)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            assert len(result) == 1
            assert "org/repo" in result[0]["display"]

    def test_handles_symlinks(self, tmp_path):
        """Resolve symlinks to actual GGUF files."""
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--org--model"
        snapshot_dir = model_dir / "snapshots" / "snap1"
        snapshot_dir.mkdir(parents=True)

        # Create actual file in blobs directory
        blobs_dir = cache_dir / "blobs"
        blobs_dir.mkdir(parents=True)
        actual_file = blobs_dir / "blob123"
        actual_file.write_bytes(b"x" * (4 * 1024**3))

        # Create symlink in snapshot
        symlink = snapshot_dir / "model.gguf"
        symlink.symlink_to(actual_file)

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            assert len(result) == 1
            # Should resolve to actual file
            assert result[0]["path"] == str(actual_file)

    def test_caches_results_for_5_minutes(self, tmp_path):
        """Results are cached for 5 minutes to avoid repeated scans."""
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--org--model"
        snapshot_dir = model_dir / "snapshots" / "snap1"
        snapshot_dir.mkdir(parents=True)

        gguf_file = snapshot_dir / "model.gguf"
        gguf_file.write_bytes(b"x" * (2 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            # First call
            result1 = core.scan_huggingface_gguf_cache()
            assert len(result1) == 1

            # Add another model
            model2_dir = cache_dir / "models--org--model2"
            snapshot2_dir = model2_dir / "snapshots" / "snap2"
            snapshot2_dir.mkdir(parents=True)
            gguf2 = snapshot2_dir / "model2.gguf"
            gguf2.write_bytes(b"x" * (3 * 1024**3))

            # Second call should return cached result (still 1 model)
            result2 = core.scan_huggingface_gguf_cache()
            assert len(result2) == 1

    def test_handles_permission_errors_gracefully(self, tmp_path):
        """Return empty list when cache isn't readable."""
        cache_dir = tmp_path / "hub"
        cache_dir.mkdir(parents=True)

        # Make directory unreadable
        cache_dir.chmod(0o000)

        try:
            with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
                # Clear cache
                if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                    delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

                result = core.scan_huggingface_gguf_cache()
                assert result == []
        finally:
            # Restore permissions for cleanup
            cache_dir.chmod(0o755)

    def test_extracts_quantization_from_filename(self, tmp_path):
        """Extract quantization level from GGUF filename."""
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--org--model"
        snapshot_dir = model_dir / "snapshots" / "snap1"
        snapshot_dir.mkdir(parents=True)

        # Different quantization formats
        test_cases = [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model-Q5_K_S.gguf", "Q5_K_S"),
            ("model-Q8_0.gguf", "Q8_0"),
        ]

        for filename, expected_quant in test_cases:
            gguf_file = snapshot_dir / filename
            gguf_file.write_bytes(b"x" * (1 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            assert len(result) == 3
            for model in result:
                # Check that quantization is in display name
                assert any(quant in model["display"] for _, quant in test_cases)

    def test_handles_invalid_directory_names(self, tmp_path):
        """Skip directories that don't match models--org--repo pattern."""
        cache_dir = tmp_path / "hub"

        # Invalid directory names
        invalid_dirs = [
            cache_dir / "not-a-model-dir",
            cache_dir / "models--only-one-part",
            cache_dir / "random",
        ]

        for d in invalid_dirs:
            d.mkdir(parents=True)
            snapshot = d / "snapshots" / "snap1"
            snapshot.mkdir(parents=True)
            (snapshot / "model.gguf").write_bytes(b"x" * 1024)

        # Valid directory
        valid_dir = cache_dir / "models--org--repo"
        valid_snapshot = valid_dir / "snapshots" / "snap1"
        valid_snapshot.mkdir(parents=True)
        (valid_snapshot / "model.gguf").write_bytes(b"x" * (2 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            result = core.scan_huggingface_gguf_cache()

            # Should only find the valid one
            assert len(result) == 1
            assert "org/repo" in result[0]["display"]


class TestInstalledModelsForEngineLlamacpp:
    """Integration tests for installed_models_for_engine() with llamacpp."""

    def test_returns_downloaded_gguf_models(self, tmp_path):
        """List downloaded GGUF models from HF cache."""
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--bartowski--Qwen2.5-Coder-7B-Instruct-GGUF"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        gguf_file = snapshot_dir / "model-Q4_K_M.gguf"
        gguf_file.write_bytes(b"x" * (7 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            profile = {"llamacpp": {"present": True, "server_running": False}}
            result = core.installed_models_for_engine(profile, "llamacpp")

            assert len(result) == 1
            assert result[0]["source"] == "llamacpp"
            assert "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF" in result[0]["display"]
            assert result[0]["tag"] == str(gguf_file)
            assert "size_gb" in result[0]

    def test_includes_running_model_if_not_in_cache(self, tmp_path):
        """Include currently running model even if not in HF cache."""
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "empty")}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            profile = {
                "llamacpp": {
                    "present": True,
                    "server_running": True,
                    "server_port": 8001,
                    "model": "/custom/path/model.gguf",
                }
            }
            result = core.installed_models_for_engine(profile, "llamacpp")

            assert len(result) == 1
            assert result[0]["tag"] == "/custom/path/model.gguf"
            assert "running on port 8001" in result[0]["display"]
            assert result[0].get("running") is True

    def test_deduplicates_running_model_if_in_cache(self, tmp_path):
        """Don't duplicate running model if it's already in HF cache."""
        cache_dir = tmp_path / "hub"
        model_dir = cache_dir / "models--org--model"
        snapshot_dir = model_dir / "snapshots" / "snap1"
        snapshot_dir.mkdir(parents=True)

        gguf_file = snapshot_dir / "model.gguf"
        gguf_file.write_bytes(b"x" * (5 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            profile = {
                "llamacpp": {
                    "present": True,
                    "server_running": True,
                    "server_port": 8001,
                    "model": str(gguf_file),
                }
            }
            result = core.installed_models_for_engine(profile, "llamacpp")

            # Should only appear once
            assert len(result) == 1
            assert result[0]["tag"] == str(gguf_file)

    def test_sorts_coder_models_first(self, tmp_path):
        """Coding models appear before general models."""
        cache_dir = tmp_path / "hub"

        # General model
        general_dir = cache_dir / "models--org--general-model"
        general_snapshot = general_dir / "snapshots" / "snap1"
        general_snapshot.mkdir(parents=True)
        (general_snapshot / "model.gguf").write_bytes(b"x" * (3 * 1024**3))

        # Coder model
        coder_dir = cache_dir / "models--org--qwen2.5-coder"
        coder_snapshot = coder_dir / "snapshots" / "snap2"
        coder_snapshot.mkdir(parents=True)
        (coder_snapshot / "model.gguf").write_bytes(b"x" * (7 * 1024**3))

        with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            profile = {"llamacpp": {"present": True, "server_running": False}}
            result = core.installed_models_for_engine(profile, "llamacpp")

            assert len(result) == 2
            # Coder model should be first
            assert "coder" in result[0]["display"].lower()
            assert "general" in result[1]["display"].lower()

    def test_returns_empty_list_when_no_models(self, tmp_path):
        """Return empty list when no GGUF models are downloaded."""
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "empty")}):
            # Clear cache
            if hasattr(core.scan_huggingface_gguf_cache, "_gguf_cache"):
                delattr(core.scan_huggingface_gguf_cache, "_gguf_cache")

            profile = {"llamacpp": {"present": True, "server_running": False}}
            result = core.installed_models_for_engine(profile, "llamacpp")

            assert result == []
