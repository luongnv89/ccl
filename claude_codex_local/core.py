#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ORIG_HOME = Path(os.environ.get("HOME", str(Path.home())))
STATE_DIR = Path(os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", ORIG_HOME / ".claude-codex-local"))

LMS_SERVER_PORT = int(os.environ.get("LMS_SERVER_PORT", "1234"))
LLAMACPP_SERVER_PORT = int(os.environ.get("LLAMACPP_SERVER_PORT", "8001"))

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
    (re.compile(r"Qwen2\.5-Coder-0\.5B", re.IGNORECASE), "qwen2.5-coder:0.5b"),
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
    (re.compile(r"DeepSeek-Coder-V2-Lite", re.IGNORECASE), "deepseek-ai/deepseek-coder-v2-lite"),
    (re.compile(r"DeepSeek-Coder-V2", re.IGNORECASE), "deepseek-ai/deepseek-coder-v2"),
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
                f"Run: llama-server --port {info['server_port']} --model <path/to/model.gguf>",
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


# Registry of adapters in preference order (LM Studio MLX first on Apple Silicon).
ALL_ADAPTERS: list[OllamaAdapter | LMStudioAdapter | LlamaCppAdapter] = [
    LMStudioAdapter(),
    OllamaAdapter(),
    LlamaCppAdapter(),
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


def llmfit_coding_candidates() -> list[dict[str, Any]]:
    """
    Run `llmfit fit --json`, filter to Coding category, and return a deduplicated
    list of candidates enriched with:
      - ollama_tag:   Ollama registry name (or None)
      - lms_mlx_path: lmstudio-community MLX model path for the recommended quant (or None)
    """
    if not command_version("llmfit").get("present"):
        return []
    try:
        cp = run(["llmfit", "fit", "--json"])
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
    """Detect huggingface-cli (from huggingface_hub[cli] package) on PATH."""
    if shutil.which("huggingface-cli"):
        return {"present": True, "version": ""}
    return {"present": False, "version": ""}


def huggingface_download_gguf(
    repo_id: str,
    filename: str | None = None,
    local_dir: str | None = None,
) -> dict[str, Any]:
    """
    Download a GGUF model file from Hugging Face Hub via huggingface-cli.

    Args:
        repo_id:   HF repo ID, e.g. "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        filename:  Specific file to download (e.g. "model-Q4_K_M.gguf").
                   When None the entire repo is fetched (picks the first GGUF).
        local_dir: Directory to store the file. Defaults to the HF cache.

    Returns:
        {"ok": bool, "path": str | None, "error": str | None}
    """
    if not huggingface_cli_detect().get("present"):
        return {
            "ok": False,
            "path": None,
            "error": "huggingface-cli not found — install with: pip install 'huggingface_hub[cli]'",
        }

    cmd = ["huggingface-cli", "download", repo_id]
    if filename:
        cmd.append(filename)
    if local_dir:
        cmd += ["--local-dir", local_dir]

    try:
        cp = run(cmd, timeout=600, check=False)
        if cp.returncode != 0:
            return {"ok": False, "path": None, "error": (cp.stderr or cp.stdout).strip()}
        # huggingface-cli download prints the resolved path on stdout
        path = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else None
        return {"ok": True, "path": path, "error": None}
    except Exception as exc:
        return {"ok": False, "path": None, "error": str(exc)}


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

    # Probe /v1/models — llama-server exposes this endpoint when running.
    url = f"http://localhost:{LLAMACPP_SERVER_PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read())
            models = body.get("data", [])
            base["server_running"] = True
            base["model"] = models[0]["id"] if models else None
    except (urllib.error.URLError, OSError):
        pass
    except Exception:
        pass
    return base


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

    url = f"http://localhost:{LLAMACPP_SERVER_PORT}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
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


def machine_profile() -> dict[str, Any]:
    llmfit_sys = llmfit_system()
    lms = lms_info()
    llamacpp = llamacpp_detect()
    hf_cli = huggingface_cli_detect()

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
            "huggingface_cli": hf_cli,
            "claude": claude_info,
            "codex": codex_info,
            "llmfit": llmfit_info,
        },
        "presence": {
            "harnesses": harnesses_present,
            "engines": engines_present,
            "llmfit": llmfit_info.get("present", False),
            "has_minimum": bool(harnesses_present)
            and bool(engines_present)
            and llmfit_info.get("present", False),
        },
        "ollama": {"models": parse_ollama_list()},
        "lmstudio": lms,
        "llamacpp": llamacpp,
        "disk": disk_usage_for(STATE_DIR),
        "state_dir": str(STATE_DIR),
    }
    if llmfit_sys:
        profile["llmfit_system"] = llmfit_sys.get("system", llmfit_sys)
    return profile


# ---------------------------------------------------------------------------
# Model selection — runtime-aware, llmfit-driven
# ---------------------------------------------------------------------------


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

    candidates = llmfit_coding_candidates()

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
