from __future__ import annotations

import ipaddress
import json
import os
import re
import warnings
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ORIG_HOME = Path(os.environ.get("HOME", str(Path.home())))
STATE_DIR = Path(os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", ORIG_HOME / ".claude-codex-local"))

LMS_SERVER_PORT = int(os.environ.get("LMS_SERVER_PORT", "1234"))


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ResponseTooLargeError(ValueError):
    """An HTTP response exceeded the bounded-read cap."""


def _normalize_base_url(
    value: str,
    *,
    default_scheme: str = "http",
    allow_path: bool = False,
) -> str:
    base = value.strip().rstrip("/")
    if "://" not in base:
        # Scheme-less input ("host:port", "gpu-box.local") gets the default
        # scheme; anything that already carries a scheme is validated below.
        base = f"{default_scheme}://{base}"
    parsed = urlparse(base)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Engine endpoint base URL {value!r} must use http:// or https:// "
            f"(got scheme {parsed.scheme!r}); refusing to send API keys or "
            f"requests over other schemes"
        )
    if not allow_path and (parsed.path or parsed.query or parsed.fragment):
        warnings.warn(
            f"Engine endpoint base URL {value!r} should not include a path, "
            f"query, or fragment (engine probes append their own paths). "
            f"Stripped to {parsed.scheme}://{parsed.netloc!r}.",
            stacklevel=2,
        )
        base = f"{parsed.scheme}://{parsed.netloc}"
    return base


def _ensure_http_url(url: str) -> str:
    """Return *url* unchanged unless its scheme is not http/https."""
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ValueError(f"Refusing non-http(s) endpoint URL: {url!r}")
    return url


def _read_bounded(resp: Any, *, max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    """Read at most *max_bytes* from *resp*, honouring Content-Length.

    Raises ResponseTooLargeError when the declared Content-Length or the
    actual byte stream exceeds the cap, so untrusted endpoints cannot make
    the CLI buffer unbounded data into memory.
    """
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            declared = int(headers.get("Content-Length"))
        except (TypeError, ValueError, AttributeError):
            declared = None
        if declared is not None and declared > max_bytes:
            raise ResponseTooLargeError(
                f"Response Content-Length {declared} exceeds the {max_bytes}-byte cap"
            )
    try:
        data = resp.read(max_bytes + 1)
    except TypeError:
        # Test doubles and non-HTTP responses may only support read().
        data = resp.read()
    if len(data) > max_bytes:
        raise ResponseTooLargeError(f"Response exceeded the {max_bytes}-byte cap")
    return data


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme.lower() not in ("http", "https"):
        # No explicit http(s) scheme: cannot classify the endpoint as a
        # local engine ("evil.com:8001" parses as scheme 'evil.com').
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


OLLAMA_BASE_URL = _normalize_base_url(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_KEY_FILE = STATE_DIR / "ollama-api-key"
LLAMACPP_SERVER_PORT = int(os.environ.get("LLAMACPP_SERVER_PORT", "8001"))
LLAMACPP_SERVER_HOST = os.environ.get("LLAMACPP_SERVER_HOST", "127.0.0.1")
LLAMACPP_BASE_URL = _normalize_base_url(
    os.environ.get("LLAMACPP_BASE_URL", f"http://localhost:{LLAMACPP_SERVER_PORT}")
)
LLAMACPP_API_KEY = os.environ.get("LLAMACPP_API_KEY", "")
LLAMACPP_KEY_FILE = STATE_DIR / "llamacpp-api-key"
LLAMACPP_CTX_SIZE = int(os.environ.get("LLAMACPP_CTX_SIZE", "131072"))
LLAMACPP_N_GPU_LAYERS = os.environ.get("LLAMACPP_N_GPU_LAYERS")
LLAMACPP_THREADS = os.environ.get("LLAMACPP_THREADS")
LLAMACPP_MTP_ENABLED = os.environ.get("LLAMACPP_MTP_ENABLED")
LLAMACPP_SPEC_DRAFT_N_MAX = os.environ.get("LLAMACPP_SPEC_DRAFT_N_MAX")
LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX = 5
LLAMACPP_LOG_DIR = STATE_DIR / "logs"
LLAMACPP_PID_DIR = STATE_DIR / "run"

ROUTER9_BASE_URL = _normalize_base_url(
    os.environ.get("CCL_9ROUTER_BASE_URL", "http://localhost:20128/v1"),
    allow_path=True,
)
ROUTER9_KEY_FILE = STATE_DIR / "9router-api-key"

OPENROUTER_BASE_URL = _normalize_base_url(
    os.environ.get("CCL_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    allow_path=True,
)
OPENROUTER_KEY_FILE = STATE_DIR / "openrouter-api-key"

VLLM_BASE_URL = _normalize_base_url(os.environ.get("VLLM_BASE_URL", "http://localhost:8000"))
VLLM_KEY_FILE = STATE_DIR / "vllm-api-key"
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

MACHINE_PROFILE_CACHE_FILE = STATE_DIR / "machine-profile.json"
MACHINE_PROFILE_TTL_SECONDS = 3600

HF_TO_OLLAMA: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Qwen3-Coder-30B", re.IGNORECASE), "qwen3-coder:30b"),
    (re.compile(r"Qwen3-Coder-14B", re.IGNORECASE), "qwen3-coder:14b"),
    (re.compile(r"Qwen3-Coder-7B", re.IGNORECASE), "qwen3-coder:7b"),
    (re.compile(r"Qwen3-Coder-4B", re.IGNORECASE), "qwen3-coder:4b"),
    (re.compile(r"Qwen3-Coder-1\.5B", re.IGNORECASE), "qwen3-coder:1.5b"),
    (re.compile(r"Qwen2\.5-Coder-32B", re.IGNORECASE), "qwen2.5-coder:32b"),
    (re.compile(r"Qwen2\.5-Coder-14B", re.IGNORECASE), "qwen2.5-coder:14b"),
    (re.compile(r"Qwen2\.5-Coder-7B", re.IGNORECASE), "qwen2.5-coder:7b"),
    (re.compile(r"Qwen2\.5-Coder-3B", re.IGNORECASE), "qwen2.5-coder:3b"),
    (re.compile(r"Qwen2\.5-Coder-1\.5B", re.IGNORECASE), "qwen2.5-coder:1.5b"),
    (re.compile(r"DeepSeek-Coder-V2-Lite", re.IGNORECASE), "deepseek-coder-v2:16b"),
    (re.compile(r"DeepSeek-Coder-V2", re.IGNORECASE), "deepseek-coder-v2"),
    (re.compile(r"deepseek-coder.*33b", re.IGNORECASE), "deepseek-coder:33b"),
    (re.compile(r"deepseek-coder.*6\.7b", re.IGNORECASE), "deepseek-coder:6.7b"),
    (re.compile(r"CodeLlama-34b", re.IGNORECASE), "codellama:34b"),
    (re.compile(r"CodeLlama-13b", re.IGNORECASE), "codellama:13b"),
    (re.compile(r"CodeLlama-7b", re.IGNORECASE), "codellama:7b"),
    (re.compile(r"starcoder2-15b", re.IGNORECASE), "starcoder2:15b"),
    (re.compile(r"starcoder2-7b", re.IGNORECASE), "starcoder2:7b"),
    (re.compile(r"starcoder2-3b", re.IGNORECASE), "starcoder2:3b"),
    (re.compile(r"granite-code.*34b", re.IGNORECASE), "granite-code:34b"),
    (re.compile(r"granite-code.*20b", re.IGNORECASE), "granite-code:20b"),
    (re.compile(r"granite-code.*8b", re.IGNORECASE), "granite-code:8b"),
    (re.compile(r"granite-code.*3b", re.IGNORECASE), "granite-code:3b"),
    (re.compile(r"WizardCoder-15B", re.IGNORECASE), "wizardcoder:15b"),
    (re.compile(r"WizardCoder-7B", re.IGNORECASE), "wizardcoder:7b"),
]

MLX_QUANT_RANK = {"mlx-4bit": 0, "mlx-5bit": 1, "mlx-6bit": 2, "mlx-8bit": 3}

MLX_QUANT_SUFFIX = {
    "mlx-4bit": "MLX-4bit",
    "mlx-5bit": "MLX-5bit",
    "mlx-6bit": "MLX-6bit",
    "mlx-8bit": "MLX-8bit",
}

HF_TO_LMS_HUB: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Qwen3-Coder-Next", re.IGNORECASE), "qwen/qwen3-coder-next"),
    (re.compile(r"Qwen3-Coder-480B", re.IGNORECASE), "qwen/qwen3-coder-480b"),
    (re.compile(r"Qwen3-Coder-30B", re.IGNORECASE), "qwen/qwen3-coder-30b"),
    (re.compile(r"Qwen3-Coder-14B", re.IGNORECASE), "qwen/qwen3-coder-14b"),
    (re.compile(r"Qwen3-Coder-7B", re.IGNORECASE), "qwen/qwen3-coder-7b"),
    (re.compile(r"Qwen3-Coder-4B", re.IGNORECASE), "qwen/qwen3-coder-4b"),
    (re.compile(r"Qwen3-Coder-1\.5B", re.IGNORECASE), "qwen/qwen3-coder-1.5b"),
    (re.compile(r"Qwen2\.5-Coder-32B", re.IGNORECASE), "qwen/qwen2.5-coder-32b"),
    (re.compile(r"Qwen2\.5-Coder-14B", re.IGNORECASE), "qwen/qwen2.5-coder-14b"),
    (re.compile(r"Qwen2\.5-Coder-7B", re.IGNORECASE), "qwen/qwen2.5-coder-7b"),
    (re.compile(r"Qwen2\.5-Coder-3B", re.IGNORECASE), "qwen/qwen2.5-coder-3b"),
    (re.compile(r"Qwen2\.5-Coder-1\.5B", re.IGNORECASE), "qwen/qwen2.5-coder-1.5b"),
    (re.compile(r"Qwen2\.5-Coder-0\.5B", re.IGNORECASE), "qwen/qwen2.5-coder-0.5b"),
    (re.compile(r"DeepSeek-Coder-V2-Lite", re.IGNORECASE), "deepseek/deepseek-coder-v2-lite"),
    (re.compile(r"DeepSeek-Coder-V2", re.IGNORECASE), "deepseek/deepseek-coder-v2"),
    (re.compile(r"CodeLlama-34b", re.IGNORECASE), "meta-llama/codellama-34b"),
    (re.compile(r"CodeLlama-13b", re.IGNORECASE), "meta-llama/codellama-13b"),
    (re.compile(r"CodeLlama-7b", re.IGNORECASE), "meta-llama/codellama-7b"),
    (re.compile(r"starcoder2-15b", re.IGNORECASE), "bigcode/starcoder2-15b"),
    (re.compile(r"starcoder2-7b", re.IGNORECASE), "bigcode/starcoder2-7b"),
    (re.compile(r"starcoder2-3b", re.IGNORECASE), "bigcode/starcoder2-3b"),
]


def _probe_openai_models_endpoint(
    base_url: str,
    *,
    service_name: str,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    base = _ensure_http_url(base_url.rstrip("/"))
    url = f"{base}/models"
    req = urllib.request.Request(
        url,
        headers=headers or {"Content-Type": "application/json"},
        method="GET",
    )
    try:
        # base_url is scheme-checked by _normalize_base_url/_ensure_http_url upstream.
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            status = getattr(resp, "status", 200)
            raw_body = _read_bounded(resp)
            response_headers = getattr(resp, "headers", {})
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} endpoint returned HTTP {exc.code} at {url}",
            "error_type": "auth_failed" if exc.code in (401, 403) else "http_error",
            "status": exc.code,
            "url": url,
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} unreachable at {url}: {exc}",
            "error_type": "network_error",
            "url": url,
        }
    except ResponseTooLargeError as exc:
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} response at {url} too large: {exc}",
            "error_type": "response_too_large",
            "url": url,
        }

    if not 200 <= status < 300:
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} endpoint returned HTTP {status} at {url}",
            "error_type": "http_error",
            "status": status,
            "url": url,
        }
    try:
        body = json.loads(raw_body)
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} returned malformed JSON at {url}: {exc}",
            "error_type": "malformed_json",
            "status": status,
            "url": url,
        }

    if not isinstance(body, dict):
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} returned malformed models payload at {url}",
            "error_type": "malformed_response",
            "status": status,
            "url": url,
        }
    raw_models = body.get("data")
    if not isinstance(raw_models, list):
        return {
            "ok": False,
            "models": [],
            "error": f"{service_name} returned malformed models payload at {url}",
            "error_type": "malformed_response",
            "status": status,
            "url": url,
        }
    models = [m.get("id", "") for m in raw_models if isinstance(m, dict)]
    return {
        "ok": True,
        "models": models,
        "response": f"{len(models)} models",
        "status": status,
        "url": url,
        "headers": response_headers,
    }
