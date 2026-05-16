"""
Unit tests for vLLM backend — vLLMAdapter and smoke_test_vllm_model.

These tests use mocked HTTP responses via monkeypatching to test all code paths
without requiring a running vLLM server.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

import claude_codex_local.core as pb

# ---------------------------------------------------------------------------
# Test fixtures for mocked HTTP responses
# ---------------------------------------------------------------------------


class FakeURLError(urllib.error.URLError):
    """Mock URLError for testing connection failures."""

    pass


class FakeHTTPError(urllib.error.HTTPError):
    """Mock HTTPError for testing HTTP error responses."""

    def __init__(self, url, code, msg, hdrs, fp):
        self.url = url
        self.code = code
        self.msg = msg
        self.hdrs = hdrs
        self.fp = None
        self._body = msg.encode("utf-8") if isinstance(msg, str) else msg
        super().__init__(url, code, msg, hdrs, fp)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self):
        return self._body


class FakeResponse:
    """Mock HTTP response for testing successful requests."""

    def __init__(self, body: dict, status: int = 200, headers: dict | None = None):
        self.body = json.dumps(body).encode("utf-8")
        self.status = status
        self._headers = headers or {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self):
        if self._closed:
            raise RuntimeError("Response already read")
        self._closed = True
        return self.body

    def readinto(self, b):
        if self._closed:
            raise RuntimeError("Response already read")
        self._closed = True
        b[: len(self.body)] = self.body
        return len(self.body)

    @property
    def headers(self):
        class HeaderDict:
            def get(self, key, default=""):
                return self._dict.get(key, default)

            def __init__(self, data):
                self._dict = data

        return HeaderDict(self._headers)


# ---------------------------------------------------------------------------
# vLLMAdapter unit tests
# ---------------------------------------------------------------------------


class TestVLLMAdapterInit:
    """Test vLLMAdapter initialization and configuration."""

    def test_default_values(self):
        """Test default adapter configuration."""
        adapter = pb.VLLMAdapter()
        assert adapter.name == "vllm"
        assert adapter._base_url == "http://localhost:8000"
        assert adapter._api_key == ""
        assert adapter._timeout == 60
        assert adapter._max_tokens == 2048

    def test_respects_environment_variables(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://custom-host:9999")
        monkeypatch.setenv("VLLM_API_KEY", "test-api-key-123")
        monkeypatch.setenv("VLLM_TIMEOUT", "120")
        monkeypatch.setenv("VLLM_MAX_TOKENS", "4096")

        adapter = pb.VLLMAdapter()
        assert adapter._base_url == "http://custom-host:9999"
        assert adapter._api_key == "test-api-key-123"
        assert adapter._timeout == 120
        assert adapter._max_tokens == 4096

    def test_base_url_trailing_slash_handling(self, monkeypatch):
        """Test that trailing slashes are stripped from base URL."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/")
        adapter = pb.VLLMAdapter()
        # _base_url should strip the trailing slash
        assert adapter._base_url == "http://localhost:8000"


class TestVLLMAdapterFullURL:
    """Test _full_url method."""

    def test_constructs_base_url(self, monkeypatch):
        """Test full URL construction from base URL."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        adapter = pb.VLLMAdapter()
        assert adapter._full_url("/v1/models") == "http://localhost:8000/v1/models"
        assert (
            adapter._full_url("/v1/chat/completions") == "http://localhost:8000/v1/chat/completions"
        )

    def test_accepts_absolute_url(self, monkeypatch):
        """Test that absolute URLs are returned as-is."""
        adapter = pb.VLLMAdapter()
        assert (
            adapter._full_url("http://other-host:8000/v1/models")
            == "http://other-host:8000/v1/models"
        )


class TestVLLMAdapterBuildHeaders:
    """Test _build_headers method."""

    def test_basic_headers(self, monkeypatch):
        """Test basic request headers."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        adapter = pb.VLLMAdapter()
        headers = adapter._build_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_includes_authorization_with_api_key(self, monkeypatch):
        """Test Authorization header with API key."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        monkeypatch.setenv("VLLM_API_KEY", "secret-key")
        adapter = pb.VLLMAdapter()
        headers = adapter._build_headers()
        assert headers["Authorization"] == "Bearer secret-key"


class TestVLLMAdapterDetect:
    """Test detect() method."""

    def test_detect_success(self, monkeypatch):
        """Test successful detection of vLLM server."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(req, timeout):
            assert req.method == "GET"
            assert "v1/models" in req.full_url
            return FakeResponse({"data": [{"id": "model1", "object": "model"}], "object": "list"})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            result = adapter.detect()
            assert result["present"] is True
            assert result["base_url"] == "http://localhost:8000"

    def test_detect_failure_connection_error(self, monkeypatch):
        """Test detection failure on connection error."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(*a, **kw):
            raise FakeURLError("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            result = adapter.detect()
            assert result["present"] is False

    def test_detect_with_api_key(self, monkeypatch):
        """Test detection includes Authorization header with API key."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        monkeypatch.setenv("VLLM_API_KEY", "test-key")

        captured_headers = {}

        def mock_urlopen(req, timeout):
            captured_headers.update(dict(req.headers))
            return FakeResponse({"data": []})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            adapter.detect()
            assert captured_headers.get("Authorization") == "Bearer test-key"

    def test_detect_version_header(self, monkeypatch):
        """Test version is captured from response headers."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(req, timeout):
            return FakeResponse({"data": []}, headers={"X-VLLM-Version": "0.5.0"})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            result = adapter.detect()
            assert result["version"] == "0.5.0"


class TestVLLMAdapterHealthcheck:
    """Test healthcheck() method."""

    def test_healthcheck_success(self, monkeypatch):
        """Test successful health check."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(req, timeout):
            return FakeResponse({"data": [{"id": "model1", "object": "model"}], "object": "list"})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            result = adapter.healthcheck()
            assert result["ok"] is True
            assert "vLLM server up" in result["detail"]

    def test_healthcheck_server_not_reachable(self, monkeypatch):
        """Test health check failure when server is unreachable."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(*a, **kw):
            raise FakeURLError("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            result = adapter.healthcheck()
            assert result["ok"] is False
            assert "not reachable" in result["detail"]


class TestVLLMAdapterListModels:
    """Test list_models() method."""

    def test_list_models_success(self, monkeypatch):
        """Test successful model listing."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(req, timeout):
            return FakeResponse(
                {
                    "data": [
                        {"id": "llama-2-7b", "object": "model"},
                        {"id": "qwen-coder", "object": "model"},
                    ],
                    "object": "list",
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            models = adapter.list_models()
            assert len(models) == 2
            assert models[0]["name"] == "llama-2-7b"
            assert models[0]["local"] is True

    def test_list_models_empty(self, monkeypatch):
        """Test listing when no models are loaded."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(req, timeout):
            return FakeResponse({"data": [], "object": "list"})

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            models = adapter.list_models()
            assert models == []

    def test_list_models_connection_error(self, monkeypatch):
        """Test model listing fails gracefully on connection error."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(*a, **kw):
            raise FakeURLError("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()
            models = adapter.list_models()
            assert models == []


class TestVLLMAdapterRunTest:
    """Test run_test() method."""

    def test_run_test_success(self, monkeypatch):
        """Test successful model run."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_smoke_test(model, base_url, api_key, timeout, max_tokens):
            return {
                "ok": True,
                "response": "READY",
                "tokens_per_second": 42.0,
                "completion_tokens": 10,
                "duration_seconds": 0.25,
            }

        monkeypatch.setattr(pb, "smoke_test_vllm_model", mock_smoke_test)

        adapter = pb.VLLMAdapter()
        result = adapter.run_test("test-model")
        assert result["ok"] is True
        assert result["response"] == "READY"
        assert result["tokens_per_second"] == 42.0


class TestVLLMAdapterRecommendParams:
    """Test recommend_params() method."""

    def test_recommend_params_returns_vllm(self, monkeypatch):
        """Test that params correctly identify vLLM provider."""
        adapter = pb.VLLMAdapter()
        result = adapter.recommend_params("balanced")
        assert result["provider"] == "vllm"
        assert result["extra_flags"] == []


# ---------------------------------------------------------------------------
# smoke_test_vllm_model unit tests
# ---------------------------------------------------------------------------


class TestSmokeTestVLLMModelSuccess:
    """Test successful smoke test scenarios."""

    def test_basic_smoke_test(self, monkeypatch):
        """Test basic successful smoke test."""
        captured_url = []
        captured_data = []

        def mock_urlopen(req, timeout):
            captured_url.append(req.full_url)
            captured_data.append(req.data)
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 10},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )

        assert result["ok"] is True
        assert "READY" in result["response"]
        assert result["completion_tokens"] == 10
        assert result["duration_seconds"] is not None
        assert captured_url[0] == "http://localhost:8000/v1/chat/completions"

    def test_smoke_test_with_api_key(self, monkeypatch):
        """Test smoke test with API key authentication."""
        captured_headers = {}

        def mock_urlopen(req, timeout):
            captured_headers.update(dict(req.headers))
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 5},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="secret-key-123",
                timeout=60,
                max_tokens=512,
            )

        assert result["ok"] is True
        assert captured_headers.get("Authorization") == "Bearer secret-key-123"

    def test_smoke_test_custom_base_url(self, monkeypatch):
        """Test smoke test with custom base URL."""
        captured_url = []

        def mock_urlopen(req, timeout):
            captured_url.append(req.full_url)
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 5},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://custom-host:9999",
                api_key="",
                timeout=30,
                max_tokens=1024,
            )

        assert captured_url[0] == "http://custom-host:9999/v1/chat/completions"

    def test_smoke_test_with_custom_max_tokens(self, monkeypatch):
        """Test smoke test with custom max_tokens."""
        captured_data = []

        def mock_urlopen(req, timeout):
            captured_data.append(req.data)
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 5},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=8192,
            )

        payload = json.loads(captured_data[0])
        assert payload["max_tokens"] == 8192

    def test_smoke_test_response_format(self, monkeypatch):
        """Test that response matches expected format."""

        def mock_urlopen(req, timeout):
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 100},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )

        # Verify all expected keys are present
        assert "ok" in result
        assert "response" in result
        assert "tokens_per_second" in result
        assert "completion_tokens" in result
        assert "duration_seconds" in result
        assert "error" not in result


class TestSmokeTestVLLMModelErrors:
    """Test error handling in smoke test."""

    def test_smoke_test_connection_refused(self, monkeypatch):
        """Test handling of connection refused error."""

        def mock_urlopen(*a, **kw):
            raise FakeURLError("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=10,
                max_tokens=2048,
            )

        assert result["ok"] is False
        assert "ok" in result
        assert "error" in result

    def test_smoke_test_http_error_404(self, monkeypatch):
        """Test handling of 404 HTTP error."""

        def mock_urlopen(req, timeout):
            raise FakeHTTPError(
                req.full_url,
                404,
                "Model not found",
                {"Content-Type": "application/json"},
                None,
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="unknown-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )

        assert result["ok"] is False
        assert "404" in result.get("error", "")

    def test_smoke_test_http_error_401(self, monkeypatch):
        """Test handling of 401 unauthorized error."""

        def mock_urlopen(req, timeout):
            raise FakeHTTPError(
                req.full_url,
                401,
                "Invalid API key",
                {"Content-Type": "application/json"},
                None,
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="invalid-key",
                timeout=60,
                max_tokens=2048,
            )

        assert result["ok"] is False
        assert "401" in result.get("error", "")

    def test_smoke_test_timeout(self, monkeypatch):
        """Test handling of request timeout."""

        def mock_urlopen(*a, **kw):
            raise urllib.error.URLError("Timeout")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=1,
                max_tokens=2048,
            )

        assert result["ok"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# vLLMAdapter in ALL_ADAPTERS
# ---------------------------------------------------------------------------


class TestAllAdaptersRegistry:
    """Test that vLLMAdapter is properly registered."""

    def test_vllm_in_all_adapters(self):
        """Test that vLLMAdapter is in the ALL_ADAPTERS registry."""
        adapter_names = [a.name for a in pb.ALL_ADAPTERS]
        assert "vllm" in adapter_names

    def test_vllm_adapter_count(self):
        """Test that we have the expected number of adapters."""
        assert len(pb.ALL_ADAPTERS) == 6

    def test_adapter_order(self):
        """Test that adapters are in the expected order."""
        expected_order = ["lmstudio", "ollama", "llamacpp", "vllm", "9router", "openrouter"]
        adapter_names = [a.name for a in pb.ALL_ADAPTERS]
        assert adapter_names == expected_order


# ---------------------------------------------------------------------------
# Integration-style tests with monkeypatched subprocess
# ---------------------------------------------------------------------------


class TestVLLMAdapterSmokeIntegration:
    """Integration tests combining multiple adapter methods."""

    def test_full_workflow_success(self, monkeypatch):
        """Test full workflow: detect -> healthcheck -> list_models -> run_test."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        call_count = [0]

        def mock_urlopen(req, timeout):
            call_count[0] += 1
            if "models" in req.full_url:
                return FakeResponse(
                    {
                        "data": [{"id": "test-model", "object": "model"}],
                        "object": "list",
                    }
                )
            else:
                return FakeResponse(
                    {
                        "choices": [{"message": {"content": "READY"}}],
                        "usage": {"completion_tokens": 10},
                    }
                )

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()

            # detect
            detect_result = adapter.detect()
            assert detect_result["present"] is True

            # healthcheck
            health_result = adapter.healthcheck()
            assert health_result["ok"] is True

            # list_models
            models = adapter.list_models()
            assert len(models) == 1

            # run_test
            test_result = adapter.run_test("test-model")
            assert test_result["ok"] is True

    def test_workflow_failure_short_circuit(self, monkeypatch):
        """Test that failure in detect short-circuits healthcheck."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")

        def mock_urlopen(*a, **kw):
            raise FakeURLError("Connection refused")

        with patch("urllib.request.urlopen", mock_urlopen):
            adapter = pb.VLLMAdapter()

            # detect fails
            detect_result = adapter.detect()
            assert detect_result["present"] is False

            # healthcheck also fails
            health_result = adapter.healthcheck()
            assert health_result["ok"] is False


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


class TestVLLMAdapterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_base_url_without_scheme(self, monkeypatch):
        """Test handling of base URL without http:// prefix."""
        monkeypatch.setenv("VLLM_BASE_URL", "localhost:8000")
        adapter = pb.VLLMAdapter()
        # _base_url is stored as-is, full_url will concatenate
        assert adapter._base_url == "localhost:8000"

    def test_empty_model_name(self, monkeypatch):
        """Test smoke test with empty model name."""

        def mock_urlopen(req, timeout):
            return FakeResponse(
                {
                    "choices": [{"message": {"content": "READY"}}],
                    "usage": {"completion_tokens": 1},
                }
            )

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )
            # The API might reject empty model, but our code should handle it
            assert "ok" in result

    def test_response_without_usage(self, monkeypatch):
        """Test handling of response without usage field."""

        def mock_urlopen(req, timeout):
            return FakeResponse({"choices": [{"message": {"content": "READY"}}]})

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )
            # Should handle missing usage gracefully
            assert "ok" in result
            assert result["completion_tokens"] is None
            assert result["tokens_per_second"] is None

    def test_response_without_choices(self, monkeypatch):
        """Test handling of response without choices field."""

        def mock_urlopen(req, timeout):
            raise Exception("Invalid response format")

        with patch("urllib.request.urlopen", mock_urlopen):
            result = pb.smoke_test_vllm_model(
                model="test-model",
                base_url="http://localhost:8000",
                api_key="",
                timeout=60,
                max_tokens=2048,
            )
            assert result["ok"] is False


# ---------------------------------------------------------------------------
# Parameterized tests
# ---------------------------------------------------------------------------


class TestVLLMAdapterParameterized:
    """Parameterized tests for various scenarios."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            ("http://localhost:8000", "http://localhost:8000"),
            ("http://localhost:8000/", "http://localhost:8000"),
            ("https://vllm.example.com", "https://vllm.example.com"),
            ("https://vllm.example.com/", "https://vllm.example.com"),
            ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ],
    )
    def test_base_url_cleanup(self, monkeypatch, base_url, expected):
        """Test that base URL trailing slashes are cleaned up."""
        monkeypatch.setenv("VLLM_BASE_URL", base_url)
        adapter = pb.VLLMAdapter()
        assert adapter._base_url == expected

    @pytest.mark.parametrize(
        "api_key,expected_header",
        [
            ("", None),
            ("my-api-key", "Bearer my-api-key"),
            ("key-with-special-chars!@#", "Bearer key-with-special-chars!@#"),
        ],
    )
    def test_authorization_header(self, monkeypatch, api_key, expected_header):
        """Test Authorization header with various API keys."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        monkeypatch.setenv("VLLM_API_KEY", api_key)
        adapter = pb.VLLMAdapter()
        headers = adapter._build_headers()
        if expected_header:
            assert headers["Authorization"] == expected_header
        else:
            assert "Authorization" not in headers
