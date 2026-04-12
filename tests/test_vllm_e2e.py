"""
E2E tests for vLLM backend — Integration tests with monkeypatched HTTP mocks.

These tests verify integration between the adapter methods without requiring
a real vLLM server.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

import claude_codex_local.core as pb

# ---------------------------------------------------------------------------
# Fixtures for mocked HTTP responses
# ---------------------------------------------------------------------------


class FakeVLLMResponse:
    """Mock HTTP response for vLLM server responses."""

    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status = status
        self._headers = headers or {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self):
        if self._closed:
            raise RuntimeError("Already read")
        self._closed = True
        return json.dumps(self.data).encode()

    @property
    def headers(self):
        class HeaderDict:
            def __init__(self, data):
                self._data = data

            def get(self, key, default=""):
                return self._data.get(key, default)

        return HeaderDict(self._headers)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestVLLMAdapterIntegration:
    """Integration tests combining multiple adapter methods."""

    def test_full_workflow_with_mock_server(self, monkeypatch):
        """Test full workflow: detect -> healthcheck -> list_models -> run_test."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000")

        call_count = [0]

        def mock_urlopen(req, timeout):
            call_count[0] += 1
            if "models" in req.full_url:
                return FakeVLLMResponse(
                    {
                        "data": [
                            {"id": "model-a", "object": "model"},
                            {"id": "model-b", "object": "model"},
                        ],
                        "object": "list",
                    },
                    headers={"X-VLLM-Version": "0.5.0"},
                )
            else:
                return FakeVLLMResponse(
                    {
                        "choices": [{"message": {"content": "READY"}}],
                        "usage": {"completion_tokens": 42, "prompt_tokens": 10},
                    }
                )

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()

            # detect
            detect_result = adapter.detect()
            assert detect_result["present"] is True
            assert detect_result["version"] == "0.5.0"

            # healthcheck
            health_result = adapter.healthcheck()
            assert health_result["ok"] is True

            # list_models
            models = adapter.list_models()
            assert len(models) == 2

            # run_test
            test_result = adapter.run_test("model-a")
            assert test_result["ok"] is True
            assert test_result["completion_tokens"] == 42

    def test_detect_failure_short_circuits_healthcheck(self, monkeypatch):
        """Test that failure in detect short-circuits healthcheck."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000")

        def mock_urlopen(*a, **kw):
            raise Exception("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()

            # detect fails
            detect_result = adapter.detect()
            assert detect_result["present"] is False

            # healthcheck also fails
            health_result = adapter.healthcheck()
            assert health_result["ok"] is False


class TestSmokeTestIntegration:
    """Integration tests for smoke_test_vllm_model function."""

    def test_smoke_test_full_response_with_timing(self, monkeypatch):
        """Test smoke test includes timing information."""

        def mock_urlopen(req, timeout):
            time.sleep(0.01)  # Simulate network delay
            return FakeVLLMResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 100, "prompt_tokens": 20},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://127.0.0.1:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )

            assert result["ok"] is True
            assert "READY" in result["response"]
            assert result["completion_tokens"] == 100
            assert result["duration_seconds"] is not None
            assert result["duration_seconds"] > 0.01
            assert result["tokens_per_second"] is not None

    def test_smoke_test_with_api_key_in_request(self, monkeypatch):
        """Test that API key is included in request headers."""
        captured_headers = {}

        def mock_urlopen(req, timeout):
            captured_headers.update(dict(req.headers))
            return FakeVLLMResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 10},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://127.0.0.1:8000",
                api_key="secret-key-123",
                timeout=60,
                max_tokens=2048,
            )

            assert result["ok"] is True
            assert captured_headers.get("Authorization") == "Bearer secret-key-123"

    def test_smoke_test_with_custom_max_tokens(self, monkeypatch):
        """Test that max_tokens is included in request payload."""
        captured_data = []

        def mock_urlopen(req, timeout):
            import json

            captured_data.append(json.loads(req.data.decode()))
            return FakeVLLMResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 50},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://127.0.0.1:8000",
                api_key="",
                timeout=60,
                max_tokens=8192,
            )

            assert result["ok"] is True
            assert captured_data[0]["max_tokens"] == 8192


class TestVLLMConfigurationIntegration:
    """Integration tests for vLLM configuration."""

    def test_env_var_configuration(self, monkeypatch):
        """Test environment variables are properly loaded."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://custom-host:9999")
        monkeypatch.setenv("VLLM_API_KEY", "test-key")
        monkeypatch.setenv("VLLM_TIMEOUT", "120")
        monkeypatch.setenv("VLLM_MAX_TOKENS", "4096")

        adapter = pb.VLLMAdapter()
        assert adapter._base_url == "http://custom-host:9999"
        assert adapter._api_key == "test-key"
        assert adapter._timeout == 120
        assert adapter._max_tokens == 4096

    def test_base_url_trailing_slash_cleanup(self, monkeypatch):
        """Test trailing slashes are removed from base URL."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/")
        adapter = pb.VLLMAdapter()
        assert adapter._base_url == "http://localhost:8000"


class TestVLLMErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_smoke_test_handles_connection_error(self, monkeypatch):
        """Test smoke test handles connection errors gracefully."""

        def mock_urlopen(*a, **kw):
            raise Exception("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://127.0.0.1:59999",
                api_key="",
                timeout=2,
                max_tokens=2048,
            )

            assert result["ok"] is False
            assert "error" in result

    def test_smoke_test_handles_missing_usage(self, monkeypatch):
        """Test smoke test handles responses without usage field."""

        def mock_urlopen(req, timeout):
            return FakeVLLMResponse({"choices": [{"message": {"content": "READY"}}]})

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://127.0.0.1:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )

            assert result["ok"] is True
            assert result["completion_tokens"] is None
            assert result["tokens_per_second"] is None

    def test_adapter_handles_empty_model_list(self, monkeypatch):
        """Test adapter handles empty model list gracefully."""

        def mock_urlopen(req, timeout):
            return FakeVLLMResponse({"data": [], "object": "list"})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            models = adapter.list_models()
            assert models == []


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration
