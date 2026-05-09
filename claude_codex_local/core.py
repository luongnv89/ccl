#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ORIG_HOME = Path(os.environ.get("HOME", str(Path.home())))
STATE_DIR = Path(os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", ORIG_HOME / ".claude-codex-local"))

LMS_SERVER_PORT = int(os.environ.get("LMS_SERVER_PORT", "1234"))
LLAMACPP_SERVER_PORT = int(os.environ.get("LLAMACPP_SERVER_PORT", "8001"))
LLAMACPP_SERVER_HOST = os.environ.get("LLAMACPP_SERVER_HOST", "127.0.0.1")
# 131072 (128k) is the minimum that survives a real coding session: Claude
# Code's system prompt is ~26k, but a single tool turn that reads a diff or
# a few source files routinely pushes the request past 40k. We saw real
# 400s at the previous 32k default within the first user-visible task.
# Modern coding-tuned models (Qwen2.5-Coder, Qwen3-Coder, DeepSeek-Coder-V2,
# Llama-3.1-Coder) all advertise ≥128k native context, so this is just
# matching the model. KV cache for 128k on a 7B–35B model is well within a
# single 24 GB GPU; bigger models or smaller GPUs should override via env.
LLAMACPP_CTX_SIZE = int(os.environ.get("LLAMACPP_CTX_SIZE", "131072"))
# When set explicitly, overrides GPU-offload auto-detection (e.g. "0" forces CPU,
# "-1" offloads everything to GPU, "33" offloads N layers).
LLAMACPP_N_GPU_LAYERS = os.environ.get("LLAMACPP_N_GPU_LAYERS")
LLAMACPP_THREADS = os.environ.get("LLAMACPP_THREADS")
# Per-port pid + log files live under STATE_DIR; we never assume /tmp.
LLAMACPP_LOG_DIR = STATE_DIR / "logs"
LLAMACPP_PID_DIR = STATE_DIR / "run"

# 9router exposes an OpenAI-compatible API; the dashboard issues an API key
# the user pastes into ~/.claude-codex-local/9router-api-key (chmod 600).
ROUTER9_BASE_URL = os.environ.get("CCL_9ROUTER_BASE_URL", "http://localhost:20128/v1")
ROUTER9_KEY_FILE = STATE_DIR / "9router-api-key"

# vLLM exposes an OpenAI-compatible API. Unlike ollama / llama.cpp the wizard
# does not start the server (vllm needs a Python venv with CUDA/ROCm wheels);
# we only probe reachability. The optional API key sits in a chmod-600 file
# next to the 9router one — same security boundary.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_KEY_FILE = STATE_DIR / "vllm-api-key"

# Machine profile cache: persist the full scan so subsequent setup runs
# do not re-probe every tool on the host.
MACHINE_PROFILE_CACHE_FILE = STATE_DIR / "machine-profile.json"
MACHINE_PROFILE_TTL_SECONDS = 3600  # 1 hour

# Mapping from HuggingFace model name patterns → Ollama registry tags.
# Ordered from newest/best to older fallbacks; first match wins.
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

# Quantization preference order for MLX on Apple Silicon.
# llmfit uses best_quant="mlx-4bit" as its recommended default; we prefer that,
# then fall to progressively heavier quants as tiebreakers.
MLX_QUANT_RANK = {"mlx-4bit": 0, "mlx-5bit": 1, "mlx-6bit": 2, "mlx-8bit": 3}

# Canonical MLX quantization suffix as it appears in lmstudio-community model names.
MLX_QUANT_SUFFIX = {
    "mlx-4bit": "MLX-4bit",
    "mlx-5bit": "MLX-5bit",
    "mlx-6bit": "MLX-6bit",
    "mlx-8bit": "MLX-8bit",
}

# Mapping from HuggingFace model name patterns → LM Studio Hub names.
# `lms get <hub_name> -y` auto-selects the best quant for your hardware.
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


# ---------------------------------------------------------------------------
# Runtime adapter contract (Task 1.1)
# ---------------------------------------------------------------------------


class RuntimeAdapter(Protocol):
    """
    Shared contract every runtime adapter must satisfy.

    All methods return plain dicts so callers never need to know which
    concrete adapter is in use — the scoring and setup flows operate on
    the normalised output only.
    """

    name: str  # e.g. "ollama", "lmstudio", "llamacpp"

    def detect(self) -> dict[str, Any]:
        """Return presence info: {"present": bool, "version": str, ...}"""
        ...

    def healthcheck(self) -> dict[str, Any]:
        """Return server/process health: {"ok": bool, "detail": str}"""
        ...

    def list_models(self) -> list[dict[str, Any]]:
        """Return installed models: [{"name": str, "local": bool, ...}]"""
        ...

    def run_test(self, model: str) -> dict[str, Any]:
        """Smoke-test a model: {"ok": bool, "response"?: str, "error"?: str}"""
        ...

    def recommend_params(self, mode: str) -> dict[str, Any]:
        """
        Return runtime-specific launch params for the given mode.
        mode is one of "balanced", "fast", "quality".
        Returns dict with at minimum: {"provider": str, "extra_flags": list[str]}
        """
        ...


@dataclass
class OllamaAdapter:
    """RuntimeAdapter implementation for Ollama."""

    name: str = "ollama"

    def detect(self) -> dict[str, Any]:
        return command_version("ollama")

    def healthcheck(self) -> dict[str, Any]:
        info = command_version("ollama")
        if not info.get("present"):
            return {"ok": False, "detail": "ollama not found in PATH"}
        models = parse_ollama_list()
        return {"ok": True, "detail": f"{len(models)} model(s) installed"}

    def list_models(self) -> list[dict[str, Any]]:
        return parse_ollama_list()

    def run_test(self, model: str) -> dict[str, Any]:
        return smoke_test_ollama_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        # Ollama does not expose per-request param overrides via its CLI;
        # mode differences are expressed through model selection upstream.
        return {"provider": "ollama", "extra_flags": []}


@dataclass
class LMStudioAdapter:
    """RuntimeAdapter implementation for LM Studio."""

    name: str = "lmstudio"

    def detect(self) -> dict[str, Any]:
        lms = lms_binary()
        if not lms:
            return {"present": False, "version": ""}
        return command_version(lms, ["--version"])

    def healthcheck(self) -> dict[str, Any]:
        info = lms_info()
        if not info.get("present"):
            return {"ok": False, "detail": "lms CLI not found"}
        if not info.get("server_running"):
            return {
                "ok": False,
                "detail": f"LM Studio server not running on port {info['server_port']}. Run: lms server start",
            }
        return {
            "ok": True,
            "detail": f"server up on port {info['server_port']}, {len(info['models'])} model(s) installed",
        }

    def list_models(self) -> list[dict[str, Any]]:
        info = lms_info()
        return [
            {"name": m["path"], "format": m["format"], "local": True}
            for m in info.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        return smoke_test_lmstudio_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "lmstudio", "extra_flags": []}


@dataclass
class LlamaCppAdapter:
    """RuntimeAdapter implementation for llama.cpp (llama-server)."""

    name: str = "llamacpp"

    def detect(self) -> dict[str, Any]:
        return llamacpp_detect()

    def healthcheck(self) -> dict[str, Any]:
        info = llamacpp_info()
        if not info.get("present"):
            return {"ok": False, "detail": "llama-server binary not found in PATH"}
        if not info.get("server_running"):
            return {
                "ok": False,
                "detail": f"llama.cpp server not running on port {info['server_port']}. "
                f"The wizard's Step 5 will auto-start it after a model download; "
                f"run `ccl --resume` to continue.",
            }
        return {
            "ok": True,
            "detail": f"server up on port {info['server_port']}",
        }

    def list_models(self) -> list[dict[str, Any]]:
        # llama.cpp loads one model at server start; users manage GGUF files manually.
        info = llamacpp_info()
        if not info.get("server_running"):
            return []
        model = info.get("model")
        if model:
            return [{"name": model, "format": "gguf", "local": True}]
        return []

    def run_test(self, model: str) -> dict[str, Any]:
        return smoke_test_llamacpp_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "llamacpp", "extra_flags": []}


@dataclass
class VLLMAdapter:
    """RuntimeAdapter implementation for vLLM server.

    vLLM provides an OpenAI-compatible HTTP API at http://localhost:8000/v1/* by default.
    Supports /v1/chat/completions for chat-based inference and /v1/models for listing.

    Configuration via environment variables:
      - VLLM_BASE_URL: vLLM server URL (default: http://localhost:8000)
      - VLLM_API_KEY: Optional API key for authentication
      - VLLM_TIMEOUT: Request timeout in seconds (default: 60)
      - VLLM_MAX_TOKENS: Default max_tokens for requests (default: 2048)
    """

    name: str = "vllm"
    _base_url: str | None = None
    _api_key: str | None = None
    _timeout: int = 60
    _max_tokens: int = 2048

    def __post_init__(self):
        """Initialize configuration from environment variables."""
        base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self._base_url = base.rstrip("/") if isinstance(base, str) else base
        self._api_key = os.environ.get("VLLM_API_KEY", "")
        self._timeout = int(os.environ.get("VLLM_TIMEOUT", "60"))
        self._max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "2048"))

    def _full_url(self, endpoint: str) -> str:
        """Construct full URL for a given endpoint."""
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{self._base_url}{endpoint}"

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def detect(self) -> dict[str, Any]:
        """Check if vLLM server is accessible.

        Uses a short fixed timeout (not VLLM_TIMEOUT) so the wizard's discover
        step doesn't hang for a full minute when the server is down.
        """
        import urllib.request

        try:
            url = self._full_url("/v1/models")
            req = urllib.request.Request(url, headers=self._build_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return {
                        "present": True,
                        "version": resp.headers.get("X-VLLM-Version", "unknown"),
                        "base_url": self._base_url,
                    }
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        return {"present": False, "version": ""}

    def healthcheck(self) -> dict[str, Any]:
        """Check vLLM server health and report status."""
        detect_info = self.detect()
        if not detect_info.get("present"):
            return {
                "ok": False,
                "detail": f"vLLM server not reachable at {self._base_url}. "
                "Start vLLM server: vllm server --model <model_path>",
            }
        try:
            import urllib.request

            url = self._full_url("/v1/models")
            req = urllib.request.Request(url, headers=self._build_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
                models = body.get("data", [])
                return {
                    "ok": True,
                    "detail": f"vLLM server up at {self._base_url}, {len(models)} model(s) available",
                }
        except urllib.error.URLError:
            return {
                "ok": False,
                "detail": f"vLLM server at {self._base_url} is not responding",
            }
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}

    def list_models(self) -> list[dict[str, Any]]:
        """List available models from vLLM server.

        Same short-timeout reasoning as detect(): this is a metadata probe,
        not an inference call, so it shouldn't share VLLM_TIMEOUT.
        """
        import urllib.request

        try:
            url = self._full_url("/v1/models")
            req = urllib.request.Request(url, headers=self._build_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                models = body.get("data", [])
                return [{"name": m["id"], "format": "unknown", "local": True} for m in models]
        except Exception:
            return []

    def run_test(self, model: str) -> dict[str, Any]:
        """Smoke-test a model via vLLM's chat API."""
        return smoke_test_vllm_model(
            model,
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_tokens=self._max_tokens,
        )

    def recommend_params(self, mode: str) -> dict[str, Any]:
        """Return runtime-specific params for the given mode."""
        return {"provider": "vllm", "extra_flags": []}


@dataclass
class Router9Adapter:
    """RuntimeAdapter implementation for 9router.

    9router is a local proxy that exposes an OpenAI-compatible API on
    http://localhost:20128/v1 and forwards calls to cloud models like
    `kr/claude-sonnet-4.5`. Because every chat call costs paid quota, this
    adapter NEVER calls /chat/completions for detection or smoke tests; we
    use /v1/models reachability instead.
    """

    name: str = "9router"

    def detect(self) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"{ROUTER9_BASE_URL.rstrip('/')}/models"
        req = urllib.request.Request(
            url, headers={"Content-Type": "application/json"}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return {"present": True, "version": "", "base_url": ROUTER9_BASE_URL}
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        return {"present": False, "version": ""}

    def healthcheck(self) -> dict[str, Any]:
        result = smoke_test_router9_models()
        if result.get("ok"):
            count = len(result.get("models", []))
            return {
                "ok": True,
                "detail": f"server up at {ROUTER9_BASE_URL}, {count} model(s) available",
            }
        return {"ok": False, "detail": result.get("error", "unreachable")}

    def list_models(self) -> list[dict[str, Any]]:
        result = smoke_test_router9_models()
        if not result.get("ok"):
            return []
        return [
            {"name": m, "format": "cloud-routed", "local": False} for m in result.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        # CRITICAL: this routes to paid cloud models. We deliberately do
        # NOT call /chat/completions; we only verify the endpoint is
        # reachable. See smoke_test_router9_models for the rationale.
        return smoke_test_router9_models()

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "9router", "extra_flags": []}


# Registry of adapters in preference order (LM Studio MLX first on Apple Silicon).
ALL_ADAPTERS: list[
    OllamaAdapter | LMStudioAdapter | LlamaCppAdapter | VLLMAdapter | Router9Adapter
] = [
    LMStudioAdapter(),
    OllamaAdapter(),
    LlamaCppAdapter(),
    VLLMAdapter(),
    Router9Adapter(),
]


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------


def ensure_path(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(os.environ if env is None else env)
    # Include ~/.lmstudio/bin so `lms` is reachable even in stripped environments.
    extra_bins = [
        ORIG_HOME / ".lmstudio" / "bin",
        ORIG_HOME / ".local" / "bin",
    ]
    current_entries = set(merged.get("PATH", "").split(os.pathsep))
    prepend = [str(p) for p in extra_bins if p.exists() and str(p) not in current_entries]
    if prepend:
        merged["PATH"] = os.pathsep.join(prepend + [merged.get("PATH", "")]).strip(os.pathsep)
    return merged


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        env=ensure_path(env),
        timeout=timeout,
    )


def command_version(name: str, args: list[str] | None = None) -> dict[str, Any]:
    try:
        cp = run([name, *(args or ["--version"])])
        text = (cp.stdout or cp.stderr).strip().splitlines()
        return {"present": True, "version": text[0] if text else ""}
    except Exception as exc:
        return {"present": False, "error": str(exc)}


def state_env() -> dict[str, str]:
    return ensure_path()


def ensure_state_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "bin").mkdir(parents=True, exist_ok=True)


def require(cmd: str) -> None:
    if not command_version(cmd).get("present"):
        print(f"missing required command: {cmd}", file=sys.stderr)
        sys.exit(1)


def run_shell(
    command: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return run(["bash", "-lc", command], env=env)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def parse_ollama_list() -> list[dict[str, Any]]:
    try:
        cp = run(["ollama", "list"])
    except Exception:
        return []
    lines = [line.rstrip() for line in cp.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    models: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            continue
        name, model_id, size, modified = parts[0], parts[1], parts[2], parts[3]
        models.append(
            {"name": name, "id": model_id, "size": size, "modified": modified, "local": size != "-"}
        )
    return models


def hf_name_to_ollama_tag(hf_name: str) -> str | None:
    for pattern, tag in HF_TO_OLLAMA:
        if pattern.search(hf_name):
            return tag
    return None


def hf_name_to_lms_hub(hf_name: str) -> str | None:
    """Map a HuggingFace model name to its LM Studio Hub name, or None if unknown."""
    for pattern, hub in HF_TO_LMS_HUB:
        if pattern.search(hf_name):
            return hub
    return None


def smoke_test_ollama_model(model: str) -> dict[str, Any]:
    """
    Smoke-test an Ollama model via its HTTP API (/api/generate).

    Uses the HTTP endpoint — instead of `ollama run` — so we can harvest
    the `eval_count` and `eval_duration` (nanoseconds) fields Ollama
    returns and compute tokens-per-second throughput. Falls back to the
    CLI if the HTTP call fails (e.g. the daemon is not exposing the API).
    """
    import time
    import urllib.error
    import urllib.request

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not ollama_host.startswith(("http://", "https://")):
        ollama_host = "http://" + ollama_host
    url = f"{ollama_host}/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": "Reply with exactly READY",
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        wall_seconds = time.time() - start
        text = str(body.get("response", "")).strip()

        eval_count = body.get("eval_count")
        eval_duration_ns = body.get("eval_duration")
        tokens_per_second: float | None = None
        duration_seconds: float | None = None
        completion_tokens: int | None = None
        if (
            isinstance(eval_count, int)
            and isinstance(eval_duration_ns, int)
            and eval_duration_ns > 0
        ):
            duration_seconds = eval_duration_ns / 1e9
            completion_tokens = eval_count
            tokens_per_second = eval_count / duration_seconds
        elif wall_seconds > 0 and text:
            # Fallback: approximate from wall-clock time and response length.
            duration_seconds = wall_seconds

        return {
            "ok": "READY" in text.upper(),
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
        }
    except urllib.error.URLError:
        # Fall back to the CLI path — the HTTP daemon may not be running.
        return _smoke_test_ollama_cli(model)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _smoke_test_ollama_cli(model: str) -> dict[str, Any]:
    """Legacy CLI-based smoke test for Ollama; no timing info available."""
    try:
        cp = run(["ollama", "run", model, "Reply with exactly READY"], timeout=180)
        text = cp.stdout.strip()
        return {
            "ok": "READY" in text.upper(),
            "response": text,
            "tokens_per_second": None,
            "completion_tokens": None,
            "duration_seconds": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout after 180s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# LM Studio helpers
# ---------------------------------------------------------------------------


def lms_binary() -> str | None:
    """Return the path to the lms CLI if present, else None."""
    lms_path = ORIG_HOME / ".lmstudio" / "bin" / "lms"
    if lms_path.exists():
        return str(lms_path)
    # Also try PATH
    info = command_version("lms")
    return "lms" if info.get("present") else None


def lms_info() -> dict[str, Any]:
    """
    Probe LM Studio: presence, server status, and installed models.

    Returns:
        present:        bool — lms CLI found
        server_running: bool — server is up on LMS_SERVER_PORT
        server_port:    int
        models:         list of {"path": str, "format": "mlx"|"gguf"|"unknown"}
    """
    lms = lms_binary()
    if not lms:
        return {
            "present": False,
            "server_running": False,
            "server_port": LMS_SERVER_PORT,
            "models": [],
        }

    # Check server status
    server_running = False
    try:
        cp = run([lms, "server", "status"])
        server_running = str(LMS_SERVER_PORT) in (cp.stdout + cp.stderr)
    except Exception:
        pass

    # List installed models
    models: list[dict[str, Any]] = []
    try:
        cp = run([lms, "ls"])
        for line in cp.stdout.splitlines():
            # Lines look like: "  lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit (1 variant)   ..."
            # or:              "  liquid/lfm2.5-1.2b (1 variant)    1.2B ..."
            # We only care about the model path (first token-like field).
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("LLM")
                or stripped.startswith("EMBEDDING")
                or stripped.startswith("You have")
            ):
                continue
            # Remove trailing "(N variant)" annotation
            path_part = re.split(r"\s+\(\d+ variant", stripped)[0].strip()
            if "/" not in path_part:
                continue
            fmt = "unknown"
            lower = path_part.lower()
            if "mlx" in lower:
                fmt = "mlx"
            elif "gguf" in lower:
                fmt = "gguf"
            models.append({"path": path_part, "format": fmt})
    except Exception:
        pass

    return {
        "present": True,
        "server_running": server_running,
        "server_port": LMS_SERVER_PORT,
        "models": models,
    }


def lms_responses_api_ok(model: str) -> bool:
    """
    Return True only if LM Studio's /v1/responses endpoint supports streaming SSE
    as Codex requires.  LM Studio may accept the request and return HTTP 200 for
    non-streaming calls while returning an empty body for streaming — the streaming
    case is what Codex actually uses, so we test that.
    """
    import urllib.error
    import urllib.request

    url = f"http://localhost:{LMS_SERVER_PORT}/v1/responses"
    payload = json.dumps(
        {
            "model": model,
            "input": "Reply with exactly: OK",
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Read the first chunk; if it's empty the endpoint is broken for streaming.
            chunk = resp.read(256)
            return bool(chunk and chunk.strip())
    except Exception:
        return False


def lms_start_server() -> bool:
    """Start the LM Studio server if not running. Returns True if server is up."""
    lms = lms_binary()
    if not lms:
        return False
    try:
        run([lms, "server", "start"])
        return True
    except Exception:
        return False


def lms_running_models() -> set[str]:
    """Return the set of model identifiers currently loaded in LM Studio."""
    lms = lms_binary()
    if not lms:
        return set()
    try:
        cp = run([lms, "ps"])
        running: set[str] = set()
        for line in cp.stdout.splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                running.add(parts[0])
        return running
    except Exception:
        return set()


def lms_load_model(model_path: str) -> dict[str, Any]:
    """Load a model into the LM Studio server (non-interactive).
    If the model is already loaded, returns ok=True immediately."""
    lms = lms_binary()
    if not lms:
        return {"ok": False, "error": "lms CLI not found"}
    if model_path in lms_running_models():
        return {"ok": True, "stdout": "already loaded"}
    try:
        cp = run([lms, "load", model_path, "-y"], timeout=60)
        return {"ok": True, "stdout": cp.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout loading model"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def lms_download_model(hub_name: str) -> dict[str, Any]:
    """
    Download a model via `lms get <hub_name> -y`.

    Pass the LM Studio Hub name (e.g. "qwen/qwen3-coder-30b") — lms auto-selects
    the best quantization for your hardware.  Do NOT pass the full
    lmstudio-community/... artifact path here; the --mlx flag is incompatible
    with exact artifact names and the hub search form handles quant selection.
    """
    lms = lms_binary()
    if not lms:
        return {"ok": False, "error": "lms CLI not found"}
    try:
        cp = run([lms, "get", hub_name, "-y"], timeout=600)
        return {"ok": True, "stdout": cp.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout downloading model"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def smoke_test_lmstudio_model(model_path: str) -> dict[str, Any]:
    """
    Smoke-test a model loaded in the LM Studio server via its OpenAI-compatible API.
    Requires the server to be running and the model loaded.

    Reports tokens-per-second using `usage.completion_tokens` from the response and
    wall-clock time around the HTTP call.
    """
    import time
    import urllib.error
    import urllib.request

    url = f"http://localhost:{LMS_SERVER_PORT}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model_path,
            "messages": [{"role": "user", "content": "Reply with exactly READY"}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        duration_seconds = max(time.time() - start, 1e-6)
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage") or {}
        raw_completion = usage.get("completion_tokens")
        completion_tokens = int(raw_completion) if isinstance(raw_completion, int) else None
        tokens_per_second: float | None = None
        if completion_tokens is not None and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration_seconds
        return {
            "ok": "READY" in text.upper(),
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# 9router helpers
# ---------------------------------------------------------------------------


def smoke_test_router9_models(base_url: str | None = None) -> dict[str, Any]:
    """
    Reachability + presence check for the 9router endpoint.

    Issues GET {base_url}/models (where {base_url} ends in /v1).
    NEVER calls /chat/completions — that would burn paid quota since 9router
    routes to cloud models like kr/claude-sonnet-4.5.

    Returns: {"ok": bool, "models": list[str], "response"?: str, "error"?: str}
    """
    import urllib.error
    import urllib.request

    base = (base_url or ROUTER9_BASE_URL).rstrip("/")
    url = f"{base}/models"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        models = [m.get("id", "") for m in body.get("data", [])]
        return {"ok": True, "models": models, "response": f"{len(models)} models"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"9router unreachable at {url}: {exc}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------


def vllm_info() -> dict[str, Any]:
    """
    Probe vLLM: CLI installation status and server reachability at VLLM_BASE_URL.
    Returns a profile-shaped dict regardless of reachability so callers can render
    a uniform discover row.

    Shape:
        {
            "present": bool,         # vllm CLI installed
            "base_url": str,         # resolved URL (no trailing slash)
            "version": str,          # CLI version or X-VLLM-Version header
            "models": list[dict],    # [{"name": str, "format": "...", "local": True}, ...]
            "error": str,            # populated when present is False
            "server_reachable": bool,  # /v1/models returned 200 (optional)
        }
    """
    # First check if vllm CLI is installed
    cli_info = command_version("vllm")
    base_url = VLLM_BASE_URL

    info: dict[str, Any] = {
        "present": cli_info.get("present", False),
        "base_url": base_url,
        "version": cli_info.get("version", ""),
        "models": [],
    }

    if not info["present"]:
        info["error"] = "vllm CLI not installed"
        return info

    # If CLI is installed, also check if server is reachable
    adapter = VLLMAdapter()
    detect = adapter.detect()

    info["server_reachable"] = bool(detect.get("present"))
    if detect.get("version"):
        info["version"] = detect.get("version", "")

    if info["server_reachable"]:
        info["models"] = adapter.list_models()
    else:
        info["error"] = f"vLLM server not reachable at {base_url} (but CLI is installed)"

    return info


def smoke_test_vllm_model(
    model: str,
    base_url: str | None = "http://localhost:8000",
    api_key: str | None = "",
    timeout: int = 60,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Smoke-test a model hosted by a vLLM server via its OpenAI-compatible API.

    The vLLM server should be running and serving the specified model.
    Uses /v1/chat/completions endpoint with a test prompt.

    Args:
        model: Model name/id to test
        base_url: vLLM server base URL (default: http://localhost:8000)
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 60)
        max_tokens: Maximum tokens to generate (default: 2048)

    Returns:
        {
            "ok": bool,
            "response": str,
            "tokens_per_second": float | None,
            "completion_tokens": int | None,
            "duration_seconds": float | None,
            "error": str | None
        }
    """
    import time
    import urllib.error
    import urllib.request

    url = f"{(base_url or 'http://localhost:8000').rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly READY"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers)
    start = time.time()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())

        duration_seconds = max(time.time() - start, 1e-6)
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage") or {}
        raw_completion = usage.get("completion_tokens")
        completion_tokens = int(raw_completion) if isinstance(raw_completion, int) else None

        tokens_per_second: float | None = None
        if completion_tokens is not None and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration_seconds

        return {
            "ok": "READY" in text.upper(),
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
            "detail": None,
        }

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {error_body}",
            "detail": str(exc),
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Connection failed: {exc.reason}", "detail": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# llmfit helpers
# ---------------------------------------------------------------------------


def llmfit_system() -> dict[str, Any] | None:
    if not command_version("llmfit").get("present"):
        return None
    try:
        cp = run(["llmfit", "system", "--json"])
        return json.loads(cp.stdout)
    except Exception:
        return None


def llmfit_info(model_name: str) -> dict[str, Any] | None:
    """
    Look up a single model via `llmfit info <name> --json`.

    Returns the first matching model dict (with fields like `total_memory_gb`,
    `params_b`, `best_quant`) or None if llmfit is missing, the lookup fails,
    or the query is ambiguous.
    """
    if not command_version("llmfit").get("present"):
        return None
    try:
        cp = run(["llmfit", "info", model_name, "--json"])
    except Exception:
        return None
    try:
        data = json.loads(cp.stdout)
    except Exception:
        return None
    models = data.get("models") or []
    if len(models) != 1:
        return None  # ambiguous or no match
    return models[0]


def llmfit_estimate_size_bytes(candidate_or_name: dict[str, Any] | str) -> int | None:
    """
    Best-effort disk-size estimate for an llmfit candidate or a free-form model
    name. Prefers `total_memory_gb` from the candidate dict; falls back to
    `llmfit info` when only a name is given; falls back to a
    params_b × quant-bits calculation if `total_memory_gb` is missing.
    """
    if isinstance(candidate_or_name, str):
        candidate = llmfit_info(candidate_or_name)
        if candidate is None:
            return None
    else:
        candidate = candidate_or_name

    gb = candidate.get("total_memory_gb") or candidate.get("memory_required_gb")
    if not gb:
        params_b = candidate.get("params_b")
        quant = (candidate.get("best_quant") or "").lower()
        bits_per_param = {
            "mlx-4bit": 4,
            "q4_k_m": 4,
            "q4_0": 4,
            "q4_1": 4,
            "mlx-5bit": 5,
            "q5_k_m": 5,
            "q5_0": 5,
            "mlx-6bit": 6,
            "q6_k": 6,
            "mlx-8bit": 8,
            "q8_0": 8,
        }.get(quant)
        if params_b and bits_per_param:
            gb = params_b * bits_per_param / 8.0
    if not gb:
        return None
    return int(gb * (1024**3))


def llmfit_coding_candidates(ram_gb: float | None = None) -> list[dict[str, Any]]:
    """
    Run `llmfit fit --json`, filter to Coding category, and return a deduplicated
    list of candidates enriched with:
      - ollama_tag:   Ollama registry name (or None)
      - lms_mlx_path: lmstudio-community MLX model path for the recommended quant (or None)

    When `ram_gb` is provided, `--ram` is passed to llmfit so rankings account
    for currently-available memory instead of total system RAM.
    """
    if not command_version("llmfit").get("present"):
        return []
    cmd: list[str] = ["llmfit"]
    if ram_gb is not None and ram_gb > 0:
        cmd.extend(["--ram", f"{ram_gb:.2f}G"])
    cmd.extend(["fit", "--json"])
    try:
        cp = run(cmd)
        data = json.loads(cp.stdout)
    except Exception:
        return []

    all_models: list[dict[str, Any]] = data.get("models", [])
    coding = [m for m in all_models if m.get("category", "").lower() in ("coding", "code")]

    # Group by canonical base model identity (HF org/name without quant suffix).
    # We want one entry per logical model, preferring the entry whose name is the
    # canonical HF name (no lmstudio-community prefix, no MLX-Xbit suffix).
    # Within each group, pick the variant whose best_quant is lowest rank (most efficient).
    groups: dict[str, dict[str, Any]] = {}

    for m in coding:
        ollama_tag = hf_name_to_ollama_tag(m["name"])
        lms_mlx_path = _derive_lms_mlx_path(m)
        lms_hub_name = hf_name_to_lms_hub(m["name"])

        # Build a stable group key: strip org prefix and MLX-quant suffix from name
        key = _canonical_key(m["name"])

        existing = groups.get(key)
        if existing is None:
            groups[key] = {
                **m,
                "ollama_tag": ollama_tag,
                "lms_mlx_path": lms_mlx_path,
                "lms_hub_name": lms_hub_name,
            }
        else:
            # Prefer: higher llmfit score, then lower MLX quant rank (more efficient)
            cur_rank = MLX_QUANT_RANK.get(m.get("best_quant", ""), 99)
            ex_rank = MLX_QUANT_RANK.get(existing.get("best_quant", ""), 99)
            cur_score = m.get("score", 0)
            ex_score = existing.get("score", 0)
            if cur_score > ex_score or (cur_score == ex_score and cur_rank < ex_rank):
                groups[key] = {
                    **m,
                    "ollama_tag": ollama_tag,
                    "lms_mlx_path": lms_mlx_path,
                    "lms_hub_name": lms_hub_name,
                }

    # Sort by score descending, then return
    return sorted(groups.values(), key=lambda m: m.get("score", 0), reverse=True)


def _canonical_key(name: str) -> str:
    """Strip org prefix and MLX-quant suffix to get a stable group key."""
    # Remove org prefix (everything up to and including the first '/')
    base = name.split("/", 1)[-1]
    # Remove trailing -MLX-Xbit or -FP8 / -FP4 suffixes
    base = re.sub(r"[-_](MLX[-_]\w+|FP\d+)$", "", base, flags=re.IGNORECASE)
    return base.lower()


def _derive_lms_mlx_path(m: dict[str, Any]) -> str | None:
    """
    Derive the lmstudio-community MLX model path for the recommended quant.

    llmfit returns entries like:
      name="Qwen/Qwen3-Coder-30B-A3B-Instruct", best_quant="mlx-4bit"  # pragma: allowlist secret
      name="lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit"  # pragma: allowlist secret

    For the canonical HF entry (with a best_quant), construct the lmstudio-community path.
    For entries that are already lmstudio-community models, use the name directly.
    """
    name: str = m.get("name", "")
    best_quant: str = m.get("best_quant", "")

    if name.startswith("lmstudio-community/") and "MLX" in name:
        return name

    if not best_quant or best_quant not in MLX_QUANT_SUFFIX:
        return None

    # Extract the model basename (after org/)
    basename = name.split("/", 1)[-1]
    # Remove any existing quant suffix
    basename = re.sub(r"[-_](MLX[-_]\w+|FP\d+)$", "", basename, flags=re.IGNORECASE)
    suffix = MLX_QUANT_SUFFIX[best_quant]
    return f"lmstudio-community/{basename}-{suffix}"


# ---------------------------------------------------------------------------
# Machine profile
# ---------------------------------------------------------------------------


def disk_usage_for(path: Path) -> dict[str, Any]:
    """Return free/total bytes for the filesystem holding `path` (or its nearest existing parent)."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        return {
            "path": str(probe),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024**3), 2),
            "total_gib": round(usage.total / (1024**3), 2),
        }
    except Exception as exc:
        return {"path": str(probe), "error": str(exc)}


def huggingface_cli_detect() -> dict[str, Any]:
    """Detect the HuggingFace Hub CLI on PATH.

    Tries both ``hf`` (the modern entry-point introduced in huggingface_hub
    ≥0.20) and the legacy ``huggingface-cli`` name so that either installation
    is recognised.  Returns a ``binary`` key with the name that was found,
    mirroring the convention used by ``llamacpp_detect``.
    """
    for candidate in ("hf", "huggingface-cli"):
        if shutil.which(candidate):
            return {"present": True, "binary": candidate, "version": ""}
    return {"present": False, "binary": "", "version": ""}


def huggingface_download_gguf(
    repo_id: str,
    filename: str | None = None,
    local_dir: str | None = None,
    *,
    include: str | None = None,
    stream: bool = True,
) -> dict[str, Any]:
    """
    Download a GGUF model file from Hugging Face Hub via the HuggingFace CLI.

    Args:
        repo_id:   HF repo ID, e.g. "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        filename:  Specific file to download (e.g. "model-Q4_K_M.gguf").
                   When None and ``include`` is also None, the entire repo is
                   fetched — only safe when the caller has already verified
                   the repo holds a single quant.
        local_dir: Directory to store the file. Defaults to the HF cache.
        include:   Glob pattern passed to ``hf download --include``. Use this
                   to grab one quant from a multi-quant GGUF repo (e.g.
                   ``"*Q4_K_M*.gguf"`` for top-level files or ``"BF16/*"`` for
                   sharded subfolders). Mutually exclusive with ``filename``;
                   if both are given, ``filename`` wins.
        stream:    When True (default) the CLI's stdout/stderr stream to the
                   terminal so the user can see the HF CLI's built-in progress
                   bar (download speed / bytes / ETA). When False, output is
                   captured and returned only in error paths — useful for unit
                   tests that assert on the returned path.

    Returns:
        {"ok": bool, "path": str | None, "error": str | None,
         "bytes_downloaded": int | None, "elapsed_seconds": float | None,
         "not_found": bool}

        ``not_found`` is True only when the HF CLI reports the repo does not
        exist (e.g. 404 / RepositoryNotFoundError). This lets callers trigger a
        fuzzy-search fallback without scraping the raw error text.
    """
    import time

    det = huggingface_cli_detect()
    if not det.get("present"):
        return {
            "ok": False,
            "path": None,
            "error": "HuggingFace CLI (hf / huggingface-cli) not found — install with: pip install 'huggingface_hub[cli]'",
            "bytes_downloaded": None,
            "elapsed_seconds": None,
            "not_found": False,
        }

    cmd = [det["binary"], "download", repo_id]
    if filename:
        cmd.append(filename)
    elif include:
        cmd += ["--include", include]
    if local_dir:
        cmd += ["--local-dir", local_dir]

    start = time.monotonic()
    try:
        if stream:
            # Inherit stdout+stderr so the HF CLI's progress bar is visible to
            # the user. We lose stdout capture, so we can only recover a precise
            # path when the caller supplied ``local_dir``. When not supplied,
            # return ``path=None`` and let the caller rely on HF's default
            # cache resolution.
            proc = subprocess.Popen(cmd, env=ensure_path(None))
            try:
                rc = proc.wait(timeout=3600)
            except KeyboardInterrupt:
                # Don't leave the child HF CLI running when the user hits
                # Ctrl-C — escalate terminate → wait → kill, then re-raise
                # so the wizard's KI handler can print its clean message.
                import contextlib

                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            proc.wait(timeout=3)
                except Exception:
                    # Best-effort cleanup: the child may already be gone.
                    pass
                raise
            elapsed = time.monotonic() - start
            if rc != 0:
                return {
                    "ok": False,
                    "path": None,
                    "error": f"huggingface-cli exited with status {rc}",
                    "bytes_downloaded": None,
                    "elapsed_seconds": elapsed,
                    "not_found": False,  # can't scrape stderr we didn't capture
                }
            resolved_path = None
            size_bytes: int | None = None
            if local_dir:
                if filename:
                    candidate = Path(local_dir) / filename
                    if candidate.exists():
                        resolved_path = str(candidate)
                        try:
                            size_bytes = candidate.stat().st_size
                        except OSError:
                            size_bytes = None
                else:
                    resolved_path = local_dir
                    try:
                        size_bytes = _dir_size_bytes(Path(local_dir))
                    except OSError:
                        size_bytes = None
            return {
                "ok": True,
                "path": resolved_path,
                "error": None,
                "bytes_downloaded": size_bytes,
                "elapsed_seconds": elapsed,
                "not_found": False,
            }
        # Non-streaming path preserves the original contract — capture stdout
        # and read the final path from its last line. Used by unit tests.
        cp = run(cmd, timeout=600, check=False)
        elapsed = time.monotonic() - start
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout).strip()
            return {
                "ok": False,
                "path": None,
                "error": err,
                "bytes_downloaded": None,
                "elapsed_seconds": elapsed,
                "not_found": _looks_like_not_found(err),
            }
        path = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else None
        size_bytes = None
        if path:
            try:
                p = Path(path)
                if p.is_file():
                    size_bytes = p.stat().st_size
                elif p.is_dir():
                    size_bytes = _dir_size_bytes(p)
            except OSError:
                size_bytes = None
        return {
            "ok": True,
            "path": path,
            "error": None,
            "bytes_downloaded": size_bytes,
            "elapsed_seconds": elapsed,
            "not_found": False,
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {
            "ok": False,
            "path": None,
            "error": str(exc),
            "bytes_downloaded": None,
            "elapsed_seconds": elapsed,
            "not_found": _looks_like_not_found(str(exc)),
        }


# ---------------------------------------------------------------------------
# Helpers for the HF download flow (#38, #39)
# ---------------------------------------------------------------------------


_HF_NOT_FOUND_MARKERS = (
    "repository not found",
    "repositorynotfounderror",
    "404 client error",
    "not found for url",
    "entry not found",
    "revisionnotfounderror",
)


def _looks_like_not_found(text: str) -> bool:
    """Heuristic predicate for HF "repo/file does not exist" errors."""
    t = (text or "").lower()
    return any(marker in t for marker in _HF_NOT_FOUND_MARKERS)


def _dir_size_bytes(root: Path) -> int:
    """Sum of every regular file under ``root`` (best-effort)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            try:
                total += (Path(dirpath) / fname).stat().st_size
            except OSError:
                continue
    return total


def huggingface_search_models(
    query: str,
    limit: int = 10,
    *,
    timeout: float = 10.0,
    raise_on_error: bool = False,
) -> list[str]:
    """
    Search the Hugging Face Hub for model IDs matching ``query``.

    Uses the public `/api/models?search=` endpoint via ``urllib.request`` so
    there is no new runtime dependency. Returns a flat list of model IDs (e.g.
    ``["bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", ...]``) ordered as HF
    returned them.

    By default this silently returns ``[]`` on any network or parsing error
    so callers can gracefully degrade to a re-prompt. Pass
    ``raise_on_error=True`` when the caller needs to distinguish "no hits"
    from "the search API itself failed" — the raw exception is propagated.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    if not query or not query.strip():
        return []
    try:
        params = urllib.parse.urlencode({"search": query.strip(), "limit": limit})
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "claude-codex-local"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        if raise_on_error:
            raise
        return []
    if not isinstance(body, list):
        return []
    ids: list[str] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        mid = item.get("id") or item.get("modelId")
        if isinstance(mid, str) and mid:
            ids.append(mid)
    return ids


def huggingface_fuzzy_find(query: str, *, max_results: int = 3) -> list[str]:
    """
    Suggest up to ``max_results`` Hugging Face model IDs that closely resemble
    ``query``. Combines the HF search endpoint with ``difflib.get_close_matches``
    for ranking so typos like "qwen2.5-codr" still resolve.

    Returns an empty list when the query is blank, the network is unavailable,
    or nothing plausibly matches.
    """
    import difflib

    candidates = huggingface_search_models(query, limit=10)
    if not candidates:
        return []
    # Rank by similarity to the user's query (case-insensitive). `get_close_matches`
    # with a low cutoff keeps partial matches in; we clamp to max_results at the end.
    ranked = difflib.get_close_matches(
        query.lower(),
        [c.lower() for c in candidates],
        n=max_results,
        cutoff=0.3,
    )
    if ranked:
        # Map lowercase picks back to their original-cased IDs, preserving
        # difflib's ranking order.
        lut = {c.lower(): c for c in candidates}
        deduped: list[str] = []
        for low in ranked:
            orig = lut.get(low)
            if orig and orig not in deduped:
                deduped.append(orig)
            if len(deduped) >= max_results:
                break
        return deduped
    # difflib found nothing — fall back to HF's own ordering so we still
    # surface *some* suggestion instead of silently failing.
    return candidates[:max_results]


def huggingface_list_repo_files(
    repo_id: str,
    *,
    timeout: float = 10.0,
) -> list[str]:
    """
    Return the filenames stored in a Hugging Face model repo, or ``[]`` on
    any error (network down, repo gone, malformed payload).

    Uses the public `/api/models/{repo}` endpoint and reads ``siblings`` —
    same dependency-free urllib pattern as ``huggingface_search_models``.

    The empty-list-on-error contract is intentional: callers downstream
    treat an empty result as "ambiguous, proceed" so a temporary HF outage
    never blocks a download. Use ``[".gguf" in f for f in result]`` to test
    repo content; only act on a *non-empty* file list.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    if not repo_id or not repo_id.strip():
        return []
    try:
        encoded = urllib.parse.quote(repo_id.strip(), safe="/")
        url = f"https://huggingface.co/api/models/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "claude-codex-local"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https only
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []
    if not isinstance(body, dict):
        return []
    siblings = body.get("siblings")
    if not isinstance(siblings, list):
        return []
    files: list[str] = []
    for entry in siblings:
        if isinstance(entry, dict):
            name = entry.get("rfilename")
            if isinstance(name, str) and name:
                files.append(name)
    return files


def huggingface_repo_has_gguf(repo_id: str) -> bool | None:
    """
    Tri-state check: does ``repo_id`` contain at least one ``.gguf`` file?

    Returns:
      True  — the file listing was retrieved and contains a GGUF.
      False — the file listing was retrieved and contains *no* GGUF.
      None  — the file listing could not be retrieved (network/404/empty);
              caller should treat as "unknown" and not block on it.
    """
    files = huggingface_list_repo_files(repo_id)
    if not files:
        return None
    return any(f.lower().endswith(".gguf") for f in files)


# Cache of "candidate base name" → resolved GGUF repo id (or None when no
# mirror was found). Avoids repeated HF API hits while the wizard re-renders
# its model picker. Cleared at process exit; the wizard's TTL is fine.
_GGUF_MIRROR_CACHE: dict[str, str | None] = {}

# Authors who reliably publish high-quality GGUF conversions of popular
# open-weight LLMs. Probed in order — first hit wins.
_GGUF_MIRROR_AUTHORS: tuple[str, ...] = (
    "bartowski",
    "unsloth",
    "lmstudio-community",
    "TheBloke",
)


def _candidate_base_name(name: str) -> str:
    """
    Strip org prefix and quant/format suffixes from an HF model name so we can
    construct GGUF mirror repo ids. Example:
      "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit" → "Qwen3-Coder-30B-A3B-Instruct"  # pragma: allowlist secret
      "NexVeridian/Qwen3-Coder-Next-8bit" → "Qwen3-Coder-Next"
    """
    base = (name or "").split("/", 1)[-1]
    # Trailing quant/format markers we don't want in a GGUF mirror name.
    base = re.sub(
        r"[-_](MLX[-_]?\w*|FP\d+|BF\d+|GPTQ|AWQ|\d+bit|GGUF)$",
        "",
        base,
        flags=re.IGNORECASE,
    )
    return base


def resolve_gguf_mirror(name: str) -> str | None:
    """
    Resolve an HF model name (typically from llmfit's coding catalog, which is
    MLX-centric) to a Hugging Face repo that actually contains GGUF files.

    Strategy, first hit wins:
      1. The original repo, if it already contains GGUF files.
      2. ``<author>/<base>-GGUF`` for each well-known mirror author.
      3. HF search for ``<base>-GGUF`` and pick the first hit that verifies.

    Returns the resolved repo id, or ``None`` when no mirror could be located.
    Results are memoised by base-name so the wizard can call this on every
    re-render of the picker without hammering the HF API.

    Network errors degrade to ``None`` rather than blocking — the wizard
    treats that the same as "no mirror found" and silently omits the
    affected llmfit recommendation from the llama.cpp profile picker.
    """
    if not name:
        return None
    base = _candidate_base_name(name)
    if not base:
        return None
    cache_key = base.lower()
    if cache_key in _GGUF_MIRROR_CACHE:
        return _GGUF_MIRROR_CACHE[cache_key]

    def _remember(value: str | None) -> str | None:
        _GGUF_MIRROR_CACHE[cache_key] = value
        return value

    if huggingface_repo_has_gguf(name) is True:
        return _remember(name)

    for author in _GGUF_MIRROR_AUTHORS:
        candidate = f"{author}/{base}-GGUF"
        if huggingface_repo_has_gguf(candidate) is True:
            return _remember(candidate)

    for hit in huggingface_search_models(f"{base}-GGUF", limit=5):
        if huggingface_repo_has_gguf(hit) is True:
            return _remember(hit)

    return _remember(None)


def llamacpp_detect() -> dict[str, Any]:
    """Detect a usable llama.cpp server binary on PATH."""
    for candidate in ("llama-server", "llama-cpp-server", "server"):
        info = command_version(candidate, ["--version"])
        if info.get("present"):
            # "server" is a generic name; verify it's actually a llama.cpp binary.
            if candidate == "server" and "llama" not in info.get("version", "").lower():
                continue
            return {"present": True, "binary": candidate, "version": info.get("version", "")}
    return {"present": False, "version": ""}


def llamacpp_info() -> dict[str, Any]:
    """
    Probe llama.cpp: binary presence and server health via HTTP.

    Returns:
        present:        bool — llama-server binary found on PATH
        binary:         str  — binary name used (e.g. "llama-server")
        server_running: bool — server is responding on LLAMACPP_SERVER_PORT
        server_port:    int
        model:          str | None — model reported by /v1/models endpoint (if running)
    """
    import urllib.error
    import urllib.request

    detect = llamacpp_detect()
    base: dict[str, Any] = {
        "present": detect.get("present", False),
        "binary": detect.get("binary", ""),
        "server_running": False,
        "server_port": LLAMACPP_SERVER_PORT,
        "model": None,
    }
    if not base["present"]:
        return base

    # Liveness via /health — returns 200 *or* 503 ("loading model") as long
    # as the server process is alive and bound to the port. Using
    # /v1/models for liveness was wrong: large models (35B+) can take 30s+
    # to load before /v1/models responds, during which a second probe would
    # mistakenly think the server was down and try to spawn a duplicate
    # (which then crashes on the bound port).
    health_url = f"http://{LLAMACPP_SERVER_HOST}:{LLAMACPP_SERVER_PORT}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as resp:
            base["server_running"] = resp.status in (200, 503)
    except (urllib.error.URLError, OSError):
        return base
    except Exception:
        return base

    # Read the served model name from /v1/models — only meaningful once the
    # model is loaded; missing during the loading window is expected and not
    # an error.
    models_url = f"http://{LLAMACPP_SERVER_HOST}:{LLAMACPP_SERVER_PORT}/v1/models"
    try:
        with urllib.request.urlopen(models_url, timeout=2) as resp:
            body = json.loads(resp.read())
            models = body.get("data", [])
            base["model"] = models[0]["id"] if models else None
    except (urllib.error.URLError, OSError):
        pass
    except Exception:
        pass
    return base


# ---------------------------------------------------------------------------
# llama.cpp server lifecycle (start / wait / stop)
# ---------------------------------------------------------------------------


@dataclass
class LlamaServerHandle:
    """Minimal handle returned when we spawn a llama-server process.

    ``we_started_it`` is True only when this process owns the lifecycle of the
    spawned server — Step 5 uses it to decide whether ``stop_server`` is allowed
    to terminate the underlying process.

    ``proc`` is the original ``Popen`` from the spawn site, when available.
    ``llamacpp_stop_server`` prefers it over raw pid signalling because
    ``Popen.terminate``/``kill``/``wait`` track the original child and cannot
    misfire onto a recycled pid.
    """

    pid: int
    port: int
    host: str
    model_path: str
    argv: list[str]
    log_path: str
    pid_file: str
    we_started_it: bool = True
    # field(repr=False) keeps the dataclass repr stable for tests/logs.
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False, compare=False)


def safe_repo_slug(repo_id: str) -> str:
    """Filesystem-safe slug for a HuggingFace repo id like ``org/repo``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.strip())
    return cleaned.strip("-") or "model"


def detect_llamacpp_gpu_offload(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Decide a sensible ``--n-gpu-layers`` for llama-server on this host.

    Returns ``{"n_gpu_layers": int, "kind": str, "reason": str}``.
    Order: explicit env override → llmfit-detected GPU → Apple Silicon (Metal) →
    nvidia-smi (CUDA) → CPU only.
    """
    if LLAMACPP_N_GPU_LAYERS is not None:
        try:
            n = int(LLAMACPP_N_GPU_LAYERS)
            return {"n_gpu_layers": n, "kind": "env-override", "reason": "LLAMACPP_N_GPU_LAYERS"}
        except ValueError:
            pass

    if profile:
        sys_info = profile.get("llmfit_system") or {}
        sys_info = sys_info.get("system", sys_info) if isinstance(sys_info, dict) else {}
        if isinstance(sys_info, dict) and sys_info.get("has_gpu"):
            gpu_name = str(sys_info.get("gpu_name") or "").lower()
            if "apple" in gpu_name or "metal" in gpu_name or gpu_name.startswith("m"):
                return {
                    "n_gpu_layers": -1,
                    "kind": "metal",
                    "reason": f"llmfit detected {gpu_name}",
                }
            return {"n_gpu_layers": -1, "kind": "gpu", "reason": f"llmfit detected {gpu_name}"}

    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine in ("arm64", "aarch64"):
        return {"n_gpu_layers": -1, "kind": "metal", "reason": "Apple Silicon detected"}

    if shutil.which("nvidia-smi"):
        return {"n_gpu_layers": -1, "kind": "cuda", "reason": "nvidia-smi found on PATH"}

    return {"n_gpu_layers": 0, "kind": "cpu", "reason": "no GPU detected"}


def detect_llamacpp_threads(profile: dict[str, Any] | None = None) -> int:
    """Pick a reasonable ``--threads`` value, honoring LLAMACPP_THREADS."""
    if LLAMACPP_THREADS is not None:
        try:
            n = int(LLAMACPP_THREADS)
            if n > 0:
                return n
        except ValueError:
            pass
    if profile:
        sys_info = profile.get("llmfit_system") or {}
        sys_info = sys_info.get("system", sys_info) if isinstance(sys_info, dict) else {}
        if isinstance(sys_info, dict):
            cores = sys_info.get("cpu_cores")
            if isinstance(cores, int) and cores > 0:
                # Use physical cores; oversubscribing slows llama-server.
                return min(cores, 16)
    cpu = os.cpu_count() or 4
    return min(cpu, 16)


def build_llamacpp_server_args(
    *,
    binary: str,
    model_path: str,
    port: int = LLAMACPP_SERVER_PORT,
    host: str = LLAMACPP_SERVER_HOST,
    ctx_size: int = LLAMACPP_CTX_SIZE,
    n_gpu_layers: int = 0,
    threads: int = 4,
) -> list[str]:
    """Compose the argv list for llama-server with the wizard's defaults."""
    return [
        binary,
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--threads",
        str(threads),
    ]


def llamacpp_wait_until_ready(
    *,
    port: int = LLAMACPP_SERVER_PORT,
    host: str = LLAMACPP_SERVER_HOST,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
    proc: subprocess.Popen[bytes] | None = None,
) -> bool:
    """Poll ``/health`` until the server responds 200 or the deadline passes.

    When ``proc`` is supplied, the loop also bails out the moment the
    underlying process has exited — avoids waiting the full timeout for a
    crashed server (e.g. bad GGUF, port already bound).
    """
    import time
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def diagnose_llama_server_log(log_path: str | Path, *, tail_bytes: int = 16384) -> str | None:
    """
    Read the tail of a llama-server log and return a human-friendly hint if a
    known startup-failure signature is present, else ``None``.

    Recognised signatures (today):
      - ``unknown model architecture`` — llama.cpp doesn't even parse this arch.
      - ``missing tensor 'blk.N.<name>'`` after ``arch = X`` — arch is parsed
        but the tensor loader for X is incomplete (e.g. qwen3next/Mamba/SSM
        support not yet built into this binary).
      - ``failed to allocate``/``out of memory``/``CUDA error: out of memory``
        — VRAM exhaustion; suggest lowering ``--n-gpu-layers``.
    """
    try:
        path = Path(log_path)
        if not path.is_file():
            return None
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(-tail_bytes, os.SEEK_END)
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return None

    lower = text.lower()

    if "unknown model architecture" in lower:
        match = re.search(r"unknown model architecture[^\n]*?'([^']+)'", text)
        arch = match.group(1) if match else "this model"
        return (
            f"Your llama.cpp build doesn't recognise the model architecture "
            f"({arch}). Update llama.cpp from source (git pull && rebuild) and "
            f"retry, or pick a model GGUF for an architecture this build "
            f"already supports."
        )

    if "missing tensor" in lower:
        arch_match = re.search(r"arch\s*=\s*(\S+)", text)
        arch = arch_match.group(1) if arch_match else "this model's"
        tensor_match = re.search(r"missing tensor '([^']+)'", text)
        tensor = tensor_match.group(1) if tensor_match else None
        ssm_hint = ""
        if tensor and ("ssm" in tensor.lower() or "mamba" in tensor.lower()):
            ssm_hint = (
                " The missing tensor is a state-space (Mamba/SSM) weight, "
                "which means this build predates full support for hybrid "
                "Mamba/Attention models like Qwen3-Next."
            )
        elif tensor:
            ssm_hint = f" Missing tensor: {tensor}."
        return (
            f"Your llama.cpp build is too old for the {arch} architecture: "
            f"it parses the GGUF metadata but the tensor loader is "
            f"incomplete.{ssm_hint} Update llama.cpp from source "
            f"(git pull && rebuild) and retry, or use a GGUF for an "
            f"architecture this build already supports."
        )

    if "out of memory" in lower or "failed to allocate" in lower:
        return (
            "llama-server ran out of memory while loading the model. Try "
            "lowering --n-gpu-layers (set LLAMACPP_N_GPU_LAYERS=0 to force "
            "CPU), reducing --ctx-size, or pick a smaller quantisation."
        )

    return None


def llamacpp_start_server(
    *,
    model_path: str,
    profile: dict[str, Any] | None = None,
    port: int = LLAMACPP_SERVER_PORT,
    host: str = LLAMACPP_SERVER_HOST,
    ctx_size: int = LLAMACPP_CTX_SIZE,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """
    Spawn ``llama-server`` with the just-downloaded model and the wizard's
    default parameter set, then poll ``/health`` until it is ready.

    Returns ``{"ok": bool, "handle": LlamaServerHandle | None, "argv": list[str],
    "error": str | None, "log_path": str}``. Always returns the argv that was
    attempted (or would have been attempted) so callers can echo it on failure.
    """
    detect = llamacpp_detect()
    if not detect.get("present"):
        return {
            "ok": False,
            "handle": None,
            "argv": [],
            "error": "llama-server binary not found on PATH",
            "log_path": "",
        }
    binary = shutil.which(detect["binary"]) or detect["binary"]

    if not Path(model_path).is_file():
        return {
            "ok": False,
            "handle": None,
            "argv": [],
            "error": f"model file not found: {model_path}",
            "log_path": "",
        }

    gpu = detect_llamacpp_gpu_offload(profile)
    threads = detect_llamacpp_threads(profile)
    argv = build_llamacpp_server_args(
        binary=binary,
        model_path=model_path,
        port=port,
        host=host,
        ctx_size=ctx_size,
        n_gpu_layers=int(gpu["n_gpu_layers"]),
        threads=threads,
    )

    LLAMACPP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    LLAMACPP_PID_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LLAMACPP_LOG_DIR / f"llama-server-{port}.log"
    pid_file = LLAMACPP_PID_DIR / f"llama-server-{port}.pid"

    popen_kwargs: dict[str, Any] = {
        "env": ensure_path(None),
    }
    # POSIX: detach into a new session so SIGINT in the wizard doesn't
    # propagate to the long-lived child, and cleanup can target the group.
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        # Subprocess inherits this fd; we close our copy after Popen returns,
        # so a `with` block is not appropriate here.
        log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115
    except OSError as exc:
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"could not open log file {log_path}: {exc}",
            "log_path": str(log_path),
        }

    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        log_handle.close()
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"failed to spawn llama-server: {exc}",
            "log_path": str(log_path),
        }
    except OSError as exc:
        log_handle.close()
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"failed to spawn llama-server: {exc}",
            "log_path": str(log_path),
        }
    finally:
        # The child inherits the file descriptor; we don't need our copy.
        with contextlib.suppress(Exception):
            log_handle.close()

    with contextlib.suppress(OSError):
        pid_file.write_text(str(proc.pid))

    handle = LlamaServerHandle(
        pid=proc.pid,
        port=port,
        host=host,
        model_path=model_path,
        argv=argv,
        log_path=str(log_path),
        pid_file=str(pid_file),
        we_started_it=True,
        proc=proc,
    )

    ready = llamacpp_wait_until_ready(port=port, host=host, timeout=timeout, proc=proc)
    if not ready:
        # Process never came up (or already exited) — best-effort terminate so
        # we don't leak it, and ensure the pid file is gone.
        llamacpp_stop_server(handle, grace_seconds=3.0)
        _cleanup_pid_file(str(pid_file))
        if proc.poll() is not None:
            err = (
                f"llama-server exited with status {proc.returncode} during startup; see {log_path}"
            )
        else:
            err = f"server did not become ready within {timeout:.0f}s"
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": err,
            "hint": diagnose_llama_server_log(log_path),
            "log_path": str(log_path),
        }

    # Confirm the process is still running after the readiness probe; some
    # error paths (model load failure, bind error) make /health flap before
    # the process exits.
    if proc.poll() is not None:
        _cleanup_pid_file(str(pid_file))
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"llama-server exited with status {proc.returncode} after readiness probe",
            "hint": diagnose_llama_server_log(log_path),
            "log_path": str(log_path),
        }

    return {
        "ok": True,
        "handle": handle,
        "argv": argv,
        "error": None,
        "log_path": str(log_path),
    }


def llamacpp_stop_server(handle: LlamaServerHandle, *, grace_seconds: float = 5.0) -> bool:
    """Terminate a server we started.

    Returns True if (and only if) the underlying process is gone afterwards.
    Returns False in two distinct cases:

    1. We refused to act because ``handle.we_started_it`` is False (the wizard
       never owned this server's lifecycle).
    2. SIGTERM + SIGKILL both failed to clear the pid within their grace
       windows.

    Callers that need to distinguish the two should check
    ``handle.we_started_it`` before calling.

    When ``handle.proc`` is set (the common path — set by
    ``llamacpp_start_server``), the function uses ``Popen.terminate`` /
    ``Popen.kill`` / ``Popen.wait`` to act on the original child. That avoids
    the pid-recycle race where ``os.kill(pid, sig)`` could land on an
    unrelated process if the kernel reused the spawned child's pid between
    the exit and the signal.
    """
    if not handle.we_started_it:
        return False

    proc = handle.proc
    if proc is not None:
        # Fast path: act on the actual child object. terminate() == SIGTERM on
        # POSIX. We still target the process group on POSIX so any helpers
        # llama-server forked (CUDA workers, etc.) get signalled too.
        if os.name == "posix":
            _signal_process(handle.pid, 15)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
        try:
            proc.wait(timeout=max(grace_seconds, 0.0))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                _signal_process(handle.pid, 9)
            else:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return False
        _cleanup_pid_file(handle.pid_file)
        return True

    # Fallback: handle was reconstructed without a Popen (e.g. loaded from
    # disk in some hypothetical future code path). Use raw pid signalling and
    # accept the small pid-recycle risk.
    pid = handle.pid
    _signal_process(pid, 15)

    import time

    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if _pid_gone(pid):
            _cleanup_pid_file(handle.pid_file)
            return True
        time.sleep(0.1)

    _signal_process(pid, 9)

    deadline2 = time.monotonic() + 2.0
    while time.monotonic() < deadline2:
        if _pid_gone(pid):
            _cleanup_pid_file(handle.pid_file)
            return True
        time.sleep(0.1)
    return _pid_gone(pid)


def _pid_gone(pid: int) -> bool:
    """Best-effort check that ``pid`` is no longer alive."""
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        # Different user owns it — we can't tell, but it's not ours to wait on.
        return True
    except OSError:
        return True


def _signal_process(pid: int, sig: int) -> None:
    """Best-effort SIGTERM/SIGKILL to a process; targets the session group on POSIX."""
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)


def _cleanup_pid_file(pid_file: str) -> None:
    with contextlib.suppress(OSError):
        Path(pid_file).unlink(missing_ok=True)


def smoke_test_llamacpp_model(model: str) -> dict[str, Any]:
    """
    Smoke-test a model loaded in the llama.cpp server via its OpenAI-compatible API.
    Requires the server to be running with the model loaded.

    Reports tokens-per-second using `usage.completion_tokens` from the response and
    wall-clock time around the HTTP call.
    """
    import time
    import urllib.error
    import urllib.request

    url = f"http://{LLAMACPP_SERVER_HOST}:{LLAMACPP_SERVER_PORT}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly READY"}],
            # Reasoning models (Qwen3+, DeepSeek-R1, …) consume tokens inside
            # <think> before producing the visible answer. 16 tokens is far
            # too tight; 256 leaves room for a brief thinking pass without
            # letting a runaway model hang the wizard.
            "max_tokens": 256,
            "temperature": 0,
            # Suppress chain-of-thought when the chat template understands
            # this kwarg (Qwen3, etc.). Templates that don't reference it
            # ignore the field, so this is safe across all models.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        duration_seconds = max(time.time() - start, 1e-6)
        choice = body["choices"][0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        # Some thinking-aware servers (llama.cpp Qwen3 template, etc.) split
        # the trace into a separate `reasoning_content` field. Even if the
        # final answer is empty, finding READY there proves the engine,
        # model, and chat template are wired up correctly.
        reasoning = (message.get("reasoning_content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = body.get("usage") or {}
        raw_completion = usage.get("completion_tokens")
        completion_tokens = int(raw_completion) if isinstance(raw_completion, int) else None
        tokens_per_second: float | None = None
        if completion_tokens is not None and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration_seconds
        ok_flag = "READY" in text.upper() or "READY" in reasoning.upper()
        result: dict[str, Any] = {
            "ok": ok_flag,
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
            "finish_reason": finish_reason,
        }
        if not ok_flag:
            # The wizard prints `error or response` on failure; without an
            # explicit error, an empty `content` (e.g. all tokens consumed
            # by reasoning, finish_reason=length) shows nothing useful.
            snippet = (text or reasoning).replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            parts = [f"finish_reason={finish_reason}"]
            if reasoning and not text:
                parts.append("model produced reasoning but no final answer")
            if snippet:
                parts.append(f"saw: '{snippet}'")
            result["error"] = "; ".join(parts)
        return result
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _compute_machine_fingerprint(profile: dict) -> str:
    """Compute a hash of key hardware identifiers for cache invalidation.

    Reads hardware identifiers from the already-built profile dict rather
    than calling ``platform.*`` again — avoids redundant work and makes the
    function pure (profile-only input, no external I/O).
    """
    host = profile.get("host", {})
    keys = [
        host.get("system", ""),
        host.get("machine", ""),
        host.get("release", ""),
        host.get("platform", ""),
    ]
    # Include llmfit_system hash if present
    sys_block = profile.get("llmfit_system", {}).get("system", {})
    if sys_block:
        keys.append(str(sys_block.get("available_ram_gb", "")))
        keys.append(str(sys_block.get("cpu_model", "")))
    fingerprint_input = "|".join(str(k) for k in keys if k)
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]


def _load_machine_profile_cache() -> dict | None:
    """Load a cached machine profile if it exists and is not expired."""
    try:
        if not MACHINE_PROFILE_CACHE_FILE.exists():
            return None
        with open(MACHINE_PROFILE_CACHE_FILE) as f:
            data = json.load(f)
        # Check TTL
        cached_ts = data.get("_cached_at", 0)
        if time.time() - cached_ts > MACHINE_PROFILE_TTL_SECONDS:
            return None  # Expired
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _save_machine_profile_cache(profile: dict, fingerprint: str) -> None:
    """Persist machine profile to cache file with fingerprint and timestamp."""
    try:
        MACHINE_PROFILE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            **profile,
            "_cached_at": time.time(),
            "_fingerprint": fingerprint,
        }
        with open(MACHINE_PROFILE_CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except OSError:
        pass  # Silently fail — a cache miss is not a failure


def _machine_profile_in_process_cache() -> dict | None:
    """In-process memoization for repeated calls within the same session."""
    cache_key = "_inproc_cache"
    if not hasattr(_machine_profile_in_process_cache, cache_key):
        setattr(_machine_profile_in_process_cache, cache_key, {"timestamp": 0, "data": None})
    cache = getattr(_machine_profile_in_process_cache, cache_key)
    # Same-process cache lasts for 30 seconds
    if time.time() - cache["timestamp"] < 30:
        return cache["data"]
    return None


def _set_machine_profile_in_process_cache(data: dict) -> None:
    cache_key = "_inproc_cache"
    setattr(_machine_profile_in_process_cache, cache_key, {"timestamp": time.time(), "data": data})


def machine_profile() -> dict[str, Any]:
    # Check in-process cache first
    cached = _machine_profile_in_process_cache()
    if cached is not None:
        return cached

    # Check file-based cache
    file_cache = _load_machine_profile_cache()
    if file_cache is not None:
        _set_machine_profile_in_process_cache(file_cache)
        return file_cache

    # Full scan — build profile
    llmfit_sys = llmfit_system()
    lms = lms_info()
    llamacpp = llamacpp_detect()
    hf_cli = huggingface_cli_detect()
    vllm = vllm_info()

    ollama_info = command_version("ollama")
    claude_info = command_version("claude")
    codex_info = command_version("codex")
    llmfit_info = command_version("llmfit")

    # Presence summary used by the wizard's discover step.
    harnesses_present = [
        name
        for name, info in (("claude", claude_info), ("codex", codex_info))
        if info.get("present")
    ]
    engines_present = []
    if ollama_info.get("present"):
        engines_present.append("ollama")
    if lms.get("present"):
        engines_present.append("lmstudio")
    if llamacpp.get("present"):
        engines_present.append("llamacpp")
    if vllm.get("present"):
        engines_present.append("vllm")
    router9_info = Router9Adapter().detect()
    router9_health = (
        Router9Adapter().healthcheck()
        if router9_info.get("present")
        else {
            "ok": False,
            "detail": "9router endpoint not reachable",
        }
    )
    if router9_info.get("present"):
        engines_present.append("9router")

    profile: dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": {
            "ollama": ollama_info,
            "lmstudio": {
                "present": lms["present"],
                "version": command_version("lms")["version"] if lms["present"] else "",
            },
            "llamacpp": llamacpp,
            "vllm": {
                "present": vllm["present"],
                "version": vllm.get("version", ""),
                "base_url": vllm["base_url"],
            },
            "huggingface_cli": hf_cli,
            "claude": claude_info,
            "codex": codex_info,
            "llmfit": llmfit_info,
            "9router": {
                "present": bool(router9_info.get("present")),
                "version": router9_info.get("version", ""),
                "base_url": ROUTER9_BASE_URL,
            },
        },
        "presence": {
            "harnesses": harnesses_present,
            "engines": engines_present,
            "llmfit": llmfit_info.get("present", False),
            "has_minimum": bool(harnesses_present) and bool(engines_present),
        },
        "ollama": {"models": parse_ollama_list()},
        "lmstudio": lms,
        "llamacpp": llamacpp,
        "vllm": vllm,
        "9router": {
            "present": bool(router9_info.get("present")),
            "base_url": ROUTER9_BASE_URL,
            "healthcheck": router9_health,
        },
        "disk": disk_usage_for(STATE_DIR),
        "state_dir": str(STATE_DIR),
    }
    if llmfit_sys:
        profile["llmfit_system"] = llmfit_sys

    # Persist to file cache for future runs
    fingerprint = _compute_machine_fingerprint(profile)
    profile["_fingerprint"] = fingerprint
    _save_machine_profile_cache(profile, fingerprint)
    _set_machine_profile_in_process_cache(profile)

    return profile


# ---------------------------------------------------------------------------
# Model selection — runtime-aware, llmfit-driven
# ---------------------------------------------------------------------------


RECOMMENDATION_MODES: tuple[str, ...] = ("balanced", "fast", "quality")


def _available_ram_gb(profile: dict[str, Any]) -> float | None:
    """Return `available_ram_gb` from the llmfit system block, if present."""
    sys_block = (
        (profile.get("llmfit_system") or {}).get("system") or profile.get("llmfit_system") or {}
    )
    val = sys_block.get("available_ram_gb")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# Human-readable descriptions for each recommendation profile. Shown in the
# wizard model picker so users understand the speed/quality tradeoff before
# choosing a profile. Keep these short (one sentence each).
RECOMMENDATION_MODE_DESCRIPTIONS: dict[str, str] = {
    "balanced": "Best score within comfortable memory headroom — good default for most machines.",
    "fast": "Prioritises tokens/second — smallest model that still fits, snappiest replies.",
    "quality": "Highest llmfit score regardless of size — best output, may be slower.",
}


def rank_candidates_for_mode(candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """
    Re-rank a list of llmfit coding candidates according to the requested mode.

    Mirrors the sort logic in `select_best_model` so UI callers can preview the
    per-mode winner without triggering smoke tests or model loads.

    Returns a new list (does not mutate the input).
    """
    mode = mode if mode in RECOMMENDATION_MODES else "balanced"
    if not candidates:
        return []
    if mode == "fast":
        return sorted(
            candidates,
            key=lambda c: (-(c.get("estimated_tps") or 0), -(c.get("score") or 0)),
        )
    if mode == "quality":
        return sorted(candidates, key=lambda c: -(c.get("score") or 0))
    # "balanced" uses the default order (score descending) as produced by
    # llmfit_coding_candidates().
    return list(candidates)


def recommend_for_mode(profile: dict[str, Any], mode: str, engine: str) -> dict[str, Any] | None:
    """
    Lightweight recommendation helper used by the wizard's profile picker.

    Returns the best llmfit coding candidate for `mode` that maps to `engine`,
    or None when llmfit is unavailable or no candidate applies to the engine.

    Unlike `select_best_model`, this helper does NOT run smoke tests or touch
    LM Studio / Ollama — it is safe to call many times while building UI choices.

    The returned dict is the raw llmfit candidate augmented with:
      - engine_tag: engine-specific tag for the chosen engine
      - mode:       the mode this recommendation was computed for
    """
    if engine not in ("ollama", "lmstudio", "llamacpp"):
        return None

    candidates = llmfit_coding_candidates(ram_gb=_available_ram_gb(profile))
    ranked = rank_candidates_for_mode(candidates, mode)
    if not ranked:
        return None

    for c in ranked:
        tag = _candidate_tag_for_engine(c, engine)
        if tag:
            return {**c, "engine_tag": tag, "mode": mode}
    return None


def _candidate_tag_for_engine(c: dict[str, Any], engine: str) -> str | None:
    """
    Pull the engine-specific tag from an llmfit candidate.

    Mirrors wizard._candidate_tag but lives in core so the recommendation
    helpers do not depend on the wizard module.
    """
    if engine == "ollama":
        return c.get("ollama_tag")
    if engine == "lmstudio":
        return c.get("lms_hub_name") or c.get("lms_mlx_path")
    if engine == "llamacpp":
        # llama.cpp can only load GGUF files. llmfit's coding catalog is
        # MLX-centric (for LM Studio), so the candidate's own repo is often
        # safetensors-only — handing that name to the HF download path
        # produced 80+ GiB of unusable files in #58. Resolve to a known GGUF
        # mirror (bartowski/unsloth/etc.) instead, and silently drop the
        # candidate when no mirror can be located.
        name = c.get("name")
        return resolve_gguf_mirror(name) if name else None
    return None


def scan_huggingface_gguf_cache() -> list[dict[str, Any]]:
    """
    Scan the HuggingFace cache for downloaded GGUF model files.

    Returns a list of dicts with:
      - path:    absolute path to the GGUF file
      - display: human-readable name with size (e.g. "org/repo-Q4_K_M (7.2 GB)")
      - size_gb: file size in gigabytes

    Respects HF_HOME environment variable. Returns empty list on errors.
    """
    import logging
    import time

    # Check cache (5-minute TTL)
    cache_key = "_gguf_cache"
    if not hasattr(scan_huggingface_gguf_cache, cache_key):
        setattr(scan_huggingface_gguf_cache, cache_key, {"timestamp": 0, "models": []})

    cache = getattr(scan_huggingface_gguf_cache, cache_key)
    now = time.time()
    if now - cache["timestamp"] < 300:  # 5 minutes
        return cache["models"]

    # Determine HF cache directory
    hf_home = os.getenv("HF_HOME")
    cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"

    models: list[dict[str, Any]] = []

    try:
        if not cache_dir.exists():
            cache["timestamp"] = now
            cache["models"] = []
            return []

        # Scan for GGUF files in models--org--repo/snapshots/*/
        for model_dir in cache_dir.glob("models--*"):
            if not model_dir.is_dir():
                continue

            # Extract model name from directory: models--org--repo → org/repo
            dir_name = model_dir.name
            if not dir_name.startswith("models--"):
                continue

            parts = dir_name[8:].split("--", 1)
            if len(parts) != 2:
                continue

            org, repo = parts
            base_name = f"{org}/{repo}"

            # Look for GGUF files in snapshots
            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue

                for gguf_file in snapshot_dir.glob("*.gguf"):
                    try:
                        # Resolve symlinks to get actual file
                        resolved = gguf_file.resolve()
                        if not resolved.exists():
                            continue

                        size_bytes = resolved.stat().st_size
                        size_gb = size_bytes / (1024**3)

                        quant = _gguf_quant_from_filename(gguf_file.stem)

                        # Build display name
                        if quant:
                            display_name = f"{base_name}-{quant} ({size_gb:.1f} GB)"
                        else:
                            display_name = f"{base_name} ({size_gb:.1f} GB)"

                        models.append(
                            {
                                "path": str(resolved),
                                "display": display_name,
                                "size_gb": size_gb,
                            }
                        )
                    except (OSError, ValueError) as exc:
                        logging.debug(f"Skipping {gguf_file}: {exc}")
                        continue

    except PermissionError:
        # Cache isn't readable
        logging.debug(f"Permission denied reading HF cache: {cache_dir}")
    except Exception as exc:
        logging.debug(f"Error scanning HF cache: {exc}")

    # Update cache
    cache["timestamp"] = now
    cache["models"] = models

    return models


def _gguf_quant_from_filename(stem: str) -> str:
    """Extract quantization tag (e.g. ``Q4_K_M``) from a GGUF stem, or "" if absent."""
    if "-" not in stem:
        return ""
    potential = stem.split("-")[-1]
    if potential.startswith("Q") and any(c.isdigit() for c in potential):
        return potential
    return ""


def scan_state_dir_gguf_models() -> list[dict[str, Any]]:
    """
    Scan ``STATE_DIR/models`` for GGUF files downloaded by the wizard.

    The wizard's HF download path writes to ``STATE_DIR/models/<slug>/`` (see
    ``wizard.py:_step_5*``), which is *outside* the HuggingFace cache, so the
    HF-cache scanner alone misses these. Returns the same shape as
    ``scan_huggingface_gguf_cache``: ``[{"path", "display", "size_gb"}]``.
    """
    import logging
    import time

    cache_key = "_state_dir_gguf_cache"
    if not hasattr(scan_state_dir_gguf_models, cache_key):
        setattr(scan_state_dir_gguf_models, cache_key, {"timestamp": 0, "models": []})

    cache = getattr(scan_state_dir_gguf_models, cache_key)
    now = time.time()
    if now - cache["timestamp"] < 300:  # 5 minutes
        return cache["models"]

    models_root = STATE_DIR / "models"
    models: list[dict[str, Any]] = []

    try:
        if not models_root.exists():
            cache["timestamp"] = now
            cache["models"] = []
            return []

        # Wizard layout: STATE_DIR/models/<safe_repo_slug>/<file>.gguf, but
        # rglob handles arbitrary nesting (e.g. include="BF16/*").
        for gguf_file in models_root.rglob("*.gguf"):
            try:
                resolved = gguf_file.resolve()
                if not resolved.is_file():
                    continue
                size_bytes = resolved.stat().st_size
                size_gb = size_bytes / (1024**3)

                # Reconstruct a friendly base name from the repo slug directory.
                # safe_repo_slug("org/repo") → "org-repo"; we display it verbatim
                # since we can't reliably split arbitrary slugs back to org/repo.
                rel = gguf_file.relative_to(models_root)
                slug = rel.parts[0] if rel.parts else gguf_file.stem
                quant = _gguf_quant_from_filename(gguf_file.stem)
                if quant:
                    display_name = f"{slug}-{quant} ({size_gb:.1f} GB)"
                else:
                    display_name = f"{slug}/{gguf_file.name} ({size_gb:.1f} GB)"

                models.append(
                    {
                        "path": str(resolved),
                        "display": display_name,
                        "size_gb": size_gb,
                    }
                )
            except (OSError, ValueError) as exc:
                logging.debug(f"Skipping {gguf_file}: {exc}")
                continue
    except PermissionError:
        logging.debug(f"Permission denied reading STATE_DIR/models: {models_root}")
    except Exception as exc:
        logging.debug(f"Error scanning STATE_DIR/models: {exc}")

    cache["timestamp"] = now
    cache["models"] = models
    return models


def installed_models_for_engine(profile: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    """
    Return locally-installed models for the chosen engine, cached in `profile`.

    Each entry is a small dict with at least:
      - tag:     engine-specific identifier usable as `engine_model_tag`
      - display: human-readable label for the UI
      - source:  short label ("ollama", "lmstudio", "llamacpp")

    The list is ordered so that recognisable coding models (qwen-coder,
    deepseek-coder, codellama, starcoder, …) come first — the same preference
    the wizard already applies in its non-interactive model-auto pick.

    This function never hits the network; it only reads fields already
    collected by `machine_profile()`.
    """
    coder_keywords = (
        "qwen3-coder",
        "qwen2.5-coder",
        "deepseek-coder",
        "codellama",
        "starcoder",
        "granite-code",
        "wizardcoder",
        "coder",
        "code",
    )

    # Embedding / reranker / TTS / vision-only models are not usable as chat
    # coding models — hide them from the picker to avoid surprising failures.
    excluded_keywords = (
        "embed",
        "embedding",
        "reranker",
        "rerank",
    )

    def _is_coder(text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in coder_keywords)

    def _is_excluded(text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in excluded_keywords)

    entries: list[dict[str, Any]] = []
    if engine == "ollama":
        for m in profile.get("ollama", {}).get("models", []) or []:
            if not m.get("local"):
                continue
            name = m.get("name")
            if not name or _is_excluded(name):
                continue
            entries.append(
                {
                    "tag": name,
                    "display": name,
                    "source": "ollama",
                    "size": m.get("size"),
                }
            )
    elif engine == "lmstudio":
        for m in profile.get("lmstudio", {}).get("models", []) or []:
            path = m.get("path")
            if not path or _is_excluded(path):
                continue
            entries.append(
                {
                    "tag": path,
                    "display": path,
                    "source": "lmstudio",
                    "format": m.get("format"),
                }
            )
    elif engine == "llamacpp":
        # Scan both the HuggingFace cache *and* STATE_DIR/models — the wizard's
        # HF download writes to STATE_DIR/models/<slug>/ (issue #59), which the
        # HF-cache scan alone wouldn't see.
        seen_paths: set[str] = set()
        for m in scan_huggingface_gguf_cache() + scan_state_dir_gguf_models():
            path = m["path"]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            entries.append(
                {
                    "tag": path,
                    "display": m["display"],
                    "source": "llamacpp",
                    "size_gb": m["size_gb"],
                }
            )

        # Also include the currently running model if server is up
        status = profile.get("llamacpp") or {}
        if status.get("server_running") and status.get("model"):
            # Check if it's not already in the list
            running_model = status["model"]
            if not any(e["tag"] == running_model for e in entries):
                entries.append(
                    {
                        "tag": running_model,
                        "display": f"{running_model} (running on port {status.get('server_port')})",
                        "source": "llamacpp",
                        "running": True,
                    }
                )

    # Stable sort: coder-likely models first, then alphabetic by display.
    entries.sort(key=lambda e: (0 if _is_coder(e["display"]) else 1, e["display"]))
    return entries


def select_best_model(profile: dict[str, Any], mode: str = "balanced") -> dict[str, Any]:
    """
    Use llmfit to pick the best coding model for the requested mode.

    mode:
      "balanced" (default) — best score within comfortable memory headroom
      "fast"               — smallest model that still fits; prioritises tok/s
      "quality"            — highest-score model regardless of size

    Priority:
      1. Already-installed LM Studio MLX model that matches a top llmfit pick.
      2. Already-installed Ollama model that matches a top llmfit pick.
      3. Recommend the top llmfit MLX pick for download via lms (if LM Studio present).
      4. Recommend the top llmfit Ollama pick for download via ollama pull.
      5. Safe hardcoded fallback if llmfit is unavailable.
    """
    mode = mode if mode in ("balanced", "fast", "quality") else "balanced"

    ollama_installed = {
        m["name"]: m for m in profile.get("ollama", {}).get("models", []) if m.get("local")
    }
    lms_data: dict[str, Any] = profile.get("lmstudio", {})
    lms_present = lms_data.get("present", False)
    lms_installed = {m["path"]: m for m in lms_data.get("models", [])}
    lms_usable = lms_present  # set to False if Responses API check fails

    candidates = llmfit_coding_candidates(ram_gb=_available_ram_gb(profile))

    # Re-rank candidates according to mode before any selection pass.
    if mode == "fast" and candidates:
        # Sort by estimated_tps descending (fastest first), then score as tiebreak.
        candidates = sorted(
            candidates,
            key=lambda c: (-(c.get("estimated_tps") or 0), -(c.get("score") or 0)),
        )
    elif mode == "quality" and candidates:
        # Sort by score descending (highest quality first).
        candidates = sorted(candidates, key=lambda c: -(c.get("score") or 0))

    rationale: list[str] = []
    caveats: list[str] = []
    next_steps: list[str] = []
    smoke: dict[str, Any] | None = None
    selected_candidate: dict[str, Any] | None = None
    runtime = "ollama"
    status = "ready"
    selected_tag: str = ""

    # --- Pass 1: installed LM Studio MLX match ---
    # lms ls can report models under two naming schemes:
    #   lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit  (old/community)
    #   qwen/qwen3-coder-30b                                        (hub short name)
    # We match against both lms_mlx_path and lms_hub_name so either scheme is found.
    if lms_present:
        for c in candidates:
            lms_path = c.get("lms_mlx_path")
            lms_hub = c.get("lms_hub_name")
            matched_key = None
            if lms_path and lms_path in lms_installed:
                matched_key = lms_path
            elif lms_hub and lms_hub in lms_installed:
                matched_key = lms_hub
            if matched_key:
                server_up = lms_data.get("server_running", False)

                if not server_up:
                    # Model is on disk but server is not running — can't use it right now.
                    caveats.append(
                        f"LM Studio has '{matched_key}' installed but the server is not running. "
                        "Falling back to Ollama. Start LM Studio server with: lms server start"
                    )
                    lms_usable = False
                    break

                # Verify streaming Responses API works — Codex requires it.
                # LM Studio may return HTTP 200 for non-streaming but nothing for streaming.
                if not lms_responses_api_ok(matched_key):
                    caveats.append(
                        f"LM Studio server is running but its /v1/responses streaming endpoint "
                        f"returned no data for '{matched_key}'. "
                        "Falling back to Ollama. Upgrade LM Studio or use Ollama."
                    )
                    lms_usable = False
                    break

                selected_candidate = c
                selected_tag = matched_key
                runtime = "lmstudio"
                rationale.append(
                    f"LM Studio is installed and '{matched_key}' is already on disk — "
                    f"using it (score={c.get('score')}, fit={c.get('fit_level')}, "
                    f"~{c.get('estimated_tps')} tok/s, MLX)."
                )
                load_result = lms_load_model(matched_key)
                if load_result.get("ok"):
                    smoke = smoke_test_lmstudio_model(matched_key)
                    if smoke.get("ok"):
                        rationale.append("LM Studio server smoke test passed.")
                    else:
                        caveats.append(
                            f"LM Studio smoke test failed: {smoke.get('error') or smoke.get('response', '')}"
                        )
                else:
                    caveats.append(
                        f"Could not load model in LM Studio: {load_result.get('error', '')}"
                    )
                break

    # --- Pass 2: installed Ollama match ---
    if not selected_tag:
        for c in candidates:
            tag = c.get("ollama_tag")
            if tag and tag in ollama_installed:
                selected_candidate = c
                selected_tag = tag
                runtime = "ollama"
                rationale.append(
                    f"llmfit ranked '{c['name']}' as the best-fit coding model "
                    f"(score={c.get('score')}, fit={c.get('fit_level')}, ~{c.get('estimated_tps')} tok/s). "
                    f"Ollama tag '{tag}' is already installed."
                )
                smoke = smoke_test_ollama_model(tag)
                if smoke.get("ok"):
                    rationale.append("Live ollama smoke test passed.")
                else:
                    caveats.append(
                        f"Ollama smoke test failed: {smoke.get('error') or smoke.get('response', '')}"
                    )
                break

    # --- Pass 2b: any installed Ollama model as a best-effort fallback ---
    # If llmfit candidates don't match any installed tag (e.g. user has a general-purpose
    # model like qwen3.5:27b), use the largest installed local model rather than requiring
    # a fresh download.
    if not selected_tag and ollama_installed:
        # Prefer models with a numeric size suffix (larger = higher quality heuristic).
        def _ollama_size_key(name: str) -> float:
            m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
            return float(m.group(1)) if m else 0.0

        best_installed = max(ollama_installed.keys(), key=_ollama_size_key)
        selected_tag = best_installed
        runtime = "ollama"
        rationale.append(
            f"No llmfit coding model is installed in Ollama. "
            f"Using the largest installed model '{best_installed}' as a best-effort fallback."
        )
        smoke = smoke_test_ollama_model(best_installed)
        if smoke.get("ok"):
            rationale.append("Live ollama smoke test passed.")
        else:
            caveats.append(
                f"Ollama smoke test failed: {smoke.get('error') or smoke.get('response', '')}"
            )

    # --- Pass 3: LM Studio present and usable but model not installed → recommend MLX download ---
    if not selected_tag and lms_usable and candidates:
        best = candidates[0]
        lms_hub = best.get("lms_hub_name")
        lms_path = best.get("lms_mlx_path")
        if lms_hub or lms_path:
            status = "download-required"
            selected_candidate = best
            # Use the lmstudio-community path as the selected_model identifier;
            # the download command uses the Hub name.
            selected_tag = lms_path or lms_hub or best["name"]
            runtime = "lmstudio"
            rationale.append(
                f"LM Studio is installed. llmfit recommends '{best['name']}' "
                f"(score={best.get('score')}, fit={best.get('fit_level')}, "
                f"mem={best.get('memory_required_gb')}GB, ~{best.get('estimated_tps')} tok/s, MLX)."
            )
            rationale.append(
                "MLX runs natively on Apple Silicon — faster and lower power than GGUF/Ollama."
            )
            # `lms get <hub_name> -y` lets lms pick the right quant automatically.
            # Do not pass --mlx here; it is only valid for search terms, not exact paths.
            dl_cmd = f"lms get {lms_hub} -y" if lms_hub else f"lms get {lms_path} -y"
            next_steps.append(dl_cmd)
            next_steps.append("lms server start")
            caveats.append(
                "Download the model above, then re-run this command to confirm readiness."
            )

    # --- Pass 4: Ollama fallback download ---
    if not selected_tag and candidates:
        best = candidates[0]
        tag = best.get("ollama_tag")
        if tag:
            status = "download-required"
            selected_candidate = best
            selected_tag = tag
            runtime = "ollama"
            rationale.append(
                f"llmfit recommends '{best['name']}' as the best coding model for this hardware "
                f"(score={best.get('score')}, fit={best.get('fit_level')}, "
                f"mem={best.get('memory_required_gb')}GB, ~{best.get('estimated_tps')} tok/s)."
            )
            next_steps.append(f"ollama pull {tag}")
            next_steps.append("./bin/codex-local")
            caveats.append(
                "Run `ollama pull` above, then re-run this command to confirm readiness."
            )

    # --- Pass 5: no llmfit candidates at all ---
    if not selected_tag:
        status = "download-required"
        selected_tag = "qwen2.5-coder:7b"
        runtime = "ollama"
        rationale.append(
            "llmfit returned no candidates. Defaulting to qwen2.5-coder:7b as a safe fallback."
        )
        next_steps.append(f"ollama pull {selected_tag}")

    modes: dict[str, str | None] = {
        "balanced": selected_tag,
        "fast": selected_tag,
        "quality": selected_tag
        if (selected_candidate and selected_candidate.get("fit_level") in ("Perfect", "Good"))
        else None,
    }

    return {
        "runtime": runtime,
        "mode": mode,
        "status": status,
        "selected_model": selected_tag,
        "modes": modes,
        "rationale": rationale,
        "caveats": list(dict.fromkeys(caveats)),
        "next_steps": next_steps,
        "smoke_test": smoke,
        "llmfit": {
            "score": selected_candidate.get("score") if selected_candidate else None,
            "fit_level": selected_candidate.get("fit_level") if selected_candidate else None,
            "estimated_tps": selected_candidate.get("estimated_tps")
            if selected_candidate
            else None,
            "memory_required_gb": selected_candidate.get("memory_required_gb")
            if selected_candidate
            else None,
            "hf_name": selected_candidate.get("name") if selected_candidate else None,
            "best_quant": selected_candidate.get("best_quant") if selected_candidate else None,
            "candidates_evaluated": len(candidates),
        },
        "state_dir": str(STATE_DIR),
    }


# ---------------------------------------------------------------------------
# Codex smoke test
# ---------------------------------------------------------------------------


def smoke_test_codex(model: str, runtime: str = "ollama") -> dict[str, Any]:
    env = state_env()
    provider = "lmstudio" if runtime == "lmstudio" else "ollama"
    try:
        cp = run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--oss",
                "--local-provider",
                provider,
                "-m",
                model,
                "Reply with exactly READY",
            ],
            env=env,
            timeout=240,
        )
        merged = (cp.stdout + "\n" + cp.stderr).strip()
        normalized = re.sub(r"[^a-z]", "", merged.lower())
        ok = "ready" in normalized
        auth_noise = (
            "failed to refresh available models" in merged.lower()
            or "401 unauthorized" in merged.lower()
        )
        return {
            "ok": ok,
            "output": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
            "auth_noise": auth_noise,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout after 240s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


def doctor(run_codex_smoke: bool, mode: str = "balanced") -> dict[str, Any]:
    profile = machine_profile()
    recommendation = select_best_model(profile, mode)
    issues: list[str] = []
    fixes: list[str] = []

    for tool_name, tool_info in profile["tools"].items():
        if not tool_info.get("present"):
            issues.append(f"Missing tool: {tool_name}")

    if not profile["ollama"]["models"] and not profile["lmstudio"].get("models"):
        issues.append("No models found in Ollama or LM Studio.")

    if recommendation["status"] == "download-required":
        issues.append("No suitable local coding model is installed.")
        fixes.extend(recommendation["next_steps"])

    codex_smoke = (
        smoke_test_codex(recommendation["selected_model"], recommendation["runtime"])
        if run_codex_smoke
        else None
    )
    if codex_smoke and not codex_smoke.get("ok"):
        issues.append("Codex local smoke test failed.")
    elif codex_smoke and codex_smoke.get("auth_noise"):
        fixes.append("Codex emits a harmless 401 model-refresh warning in local-only mode.")

    return {
        "profile": profile,
        "recommendation": recommendation,
        "issues": issues,
        "fixes": fixes,
        "codex_smoke": codex_smoke,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


MODE_CHOICES = ["balanced", "fast", "quality"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m claude_codex_local.core",
        description=(
            "claude-codex-local debug CLI — machine profile, model recommendation, "
            "doctor, and adapter introspection. These commands dump JSON for "
            "scripting and debugging; the user-facing binary is `ccl`."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile")

    rec_cmd = sub.add_parser("recommend")
    rec_cmd.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="balanced",
        help="Preset: balanced (default), fast (smallest/fastest), quality (highest score)",
    )

    doctor_cmd = sub.add_parser("doctor")
    doctor_cmd.add_argument("--run-codex-smoke", action="store_true")
    doctor_cmd.add_argument("--mode", choices=MODE_CHOICES, default="balanced")

    # adapters: expose the RuntimeAdapter contract for inspection
    sub.add_parser("adapters")

    args = parser.parse_args()

    if args.command == "profile":
        print_payload(machine_profile())
    elif args.command == "recommend":
        print_payload(select_best_model(machine_profile(), args.mode))
    elif args.command == "doctor":
        print_payload(doctor(args.run_codex_smoke, args.mode))
    elif args.command == "adapters":
        result = []
        for adapter in ALL_ADAPTERS:
            result.append(
                {
                    "name": adapter.name,
                    "detect": adapter.detect(),
                    "healthcheck": adapter.healthcheck(),
                    "models": adapter.list_models(),
                    "recommend_params": {m: adapter.recommend_params(m) for m in MODE_CHOICES},
                }
            )
        print_payload({"adapters": result})


if __name__ == "__main__":
    main()
