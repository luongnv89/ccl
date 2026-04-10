#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent
ORIG_HOME = Path(os.environ.get("HOME", str(Path.home())))
STATE_DIR = Path(os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", ROOT / ".claude-codex-local"))
STATE_HOME = STATE_DIR / "home"

LMS_SERVER_PORT = int(os.environ.get("LMS_SERVER_PORT", "1234"))

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
    (re.compile(r"Qwen3-Coder-Next", re.IGNORECASE),    "qwen/qwen3-coder-next"),
    (re.compile(r"Qwen3-Coder-480B", re.IGNORECASE),    "qwen/qwen3-coder-480b"),
    (re.compile(r"Qwen3-Coder-30B", re.IGNORECASE),     "qwen/qwen3-coder-30b"),
    (re.compile(r"Qwen3-Coder-14B", re.IGNORECASE),     "qwen/qwen3-coder-14b"),
    (re.compile(r"Qwen3-Coder-7B", re.IGNORECASE),      "qwen/qwen3-coder-7b"),
    (re.compile(r"Qwen3-Coder-4B", re.IGNORECASE),      "qwen/qwen3-coder-4b"),
    (re.compile(r"Qwen3-Coder-1\.5B", re.IGNORECASE),   "qwen/qwen3-coder-1.5b"),
    (re.compile(r"Qwen2\.5-Coder-32B", re.IGNORECASE),  "qwen/qwen2.5-coder-32b"),
    (re.compile(r"Qwen2\.5-Coder-14B", re.IGNORECASE),  "qwen/qwen2.5-coder-14b"),
    (re.compile(r"Qwen2\.5-Coder-7B", re.IGNORECASE),   "qwen/qwen2.5-coder-7b"),
    (re.compile(r"Qwen2\.5-Coder-3B", re.IGNORECASE),   "qwen/qwen2.5-coder-3b"),
    (re.compile(r"Qwen2\.5-Coder-1\.5B", re.IGNORECASE),"qwen/qwen2.5-coder-1.5b"),
    (re.compile(r"Qwen2\.5-Coder-0\.5B", re.IGNORECASE),"qwen/qwen2.5-coder-0.5b"),
    (re.compile(r"DeepSeek-Coder-V2-Lite", re.IGNORECASE), "deepseek-ai/deepseek-coder-v2-lite"),
    (re.compile(r"DeepSeek-Coder-V2", re.IGNORECASE),   "deepseek-ai/deepseek-coder-v2"),
    (re.compile(r"CodeLlama-34b", re.IGNORECASE),        "meta-llama/codellama-34b"),
    (re.compile(r"CodeLlama-13b", re.IGNORECASE),        "meta-llama/codellama-13b"),
    (re.compile(r"CodeLlama-7b", re.IGNORECASE),         "meta-llama/codellama-7b"),
    (re.compile(r"starcoder2-15b", re.IGNORECASE),       "bigcode/starcoder2-15b"),
    (re.compile(r"starcoder2-7b", re.IGNORECASE),        "bigcode/starcoder2-7b"),
    (re.compile(r"starcoder2-3b", re.IGNORECASE),        "bigcode/starcoder2-3b"),
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
            return {"ok": False, "detail": f"LM Studio server not running on port {info['server_port']}. Run: lms server start"}
        return {"ok": True, "detail": f"server up on port {info['server_port']}, {len(info['models'])} model(s) installed"}

    def list_models(self) -> list[dict[str, Any]]:
        info = lms_info()
        return [{"name": m["path"], "format": m["format"], "local": True} for m in info.get("models", [])]

    def run_test(self, model: str) -> dict[str, Any]:
        return smoke_test_lmstudio_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "lmstudio", "extra_flags": []}


# Registry of adapters in preference order (LM Studio MLX first on Apple Silicon).
ALL_ADAPTERS: list[OllamaAdapter | LMStudioAdapter] = [
    LMStudioAdapter(),
    OllamaAdapter(),
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
        cmd, check=check, capture_output=True, text=True,
        env=ensure_path(env), timeout=timeout,
    )


def command_version(name: str, args: list[str] | None = None) -> dict[str, Any]:
    try:
        cp = run([name, *(args or ["--version"])])
        text = (cp.stdout or cp.stderr).strip().splitlines()
        return {"present": True, "version": text[0] if text else ""}
    except Exception as exc:
        return {"present": False, "error": str(exc)}


def state_env() -> dict[str, str]:
    env = ensure_path()
    env["HOME"] = str(STATE_HOME)
    env["XDG_CONFIG_HOME"] = str(STATE_HOME / ".config")
    env["XDG_DATA_HOME"] = str(STATE_HOME / ".local/share")
    return env


def ensure_state_dirs() -> None:
    (STATE_HOME / ".config").mkdir(parents=True, exist_ok=True)
    (STATE_HOME / ".local/share").mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def require(cmd: str) -> None:
    if not command_version(cmd).get("present"):
        print(f"missing required command: {cmd}", file=sys.stderr)
        sys.exit(1)


def run_shell(command: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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
        models.append({"name": name, "id": model_id, "size": size, "modified": modified, "local": size != "-"})
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
    try:
        cp = run(["ollama", "run", model, "Reply with exactly READY"], timeout=180)
        text = cp.stdout.strip()
        return {"ok": "READY" in text.upper(), "response": text}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout after 180s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def configure_ollama_integration(target: str, model: str) -> dict[str, Any]:
    ensure_state_dirs()
    require("ollama")
    shell_cmd = f"printf 'n\\n' | ollama launch {shlex.quote(target)} --config --model {shlex.quote(model)}"
    cp = run_shell(shell_cmd, env=state_env())
    return {"target": target, "model": model, "state_dir": str(STATE_DIR), "home": str(STATE_HOME), "stdout": cp.stdout.strip()}


# ---------------------------------------------------------------------------
# Ollama "no-think" variant builder
#
# Claude Code sends a `thinking` field in its chat payload that confuses Qwen3
# reasoning models (they emit <think> blocks that never close, blowing the
# context budget) and wastes latency on Gemma4. The fix is to bake a derived
# model via `ollama create` with a family-appropriate Modelfile:
#
#   - Qwen3 / Qwen3-Coder  → SYSTEM "/no_think" + recommended sampling
#   - Gemma4               → no think toggle; just ensure 64K ctx + sampling
#   - Qwen2.5-Coder        → already non-thinking; just ensure 64K ctx
#
# Reference: blog-posts/2026-04-03-run-claude-code-codex-local-gemma4.
# ---------------------------------------------------------------------------

NOTHINK_VARIANT_SUFFIX = "-cclocal"

def _ollama_model_family(tag: str) -> str:
    t = tag.lower()
    # Strip version suffix so we match against the model name only.
    name = t.split(":", 1)[0]
    if name.endswith(NOTHINK_VARIANT_SUFFIX):
        return "already-patched"
    if "qwen3" in t:
        return "qwen3"
    if "qwen2.5" in t or "qwen2_5" in t:
        return "qwen2.5"
    if "gemma4" in t or "gemma-4" in t:
        return "gemma4"
    return "unknown"


def ollama_nothink_modelfile(base_tag: str) -> str | None:
    """
    Return the Modelfile body for a Claude-Code-friendly variant of `base_tag`,
    or None if no patch is needed/known for this model family.
    """
    family = _ollama_model_family(base_tag)
    if family == "qwen3":
        return (
            f"FROM {base_tag}\n"
            f"PARAMETER num_ctx 65536\n"
            f"PARAMETER temperature 0.7\n"
            f"PARAMETER top_p 0.8\n"
            f"PARAMETER top_k 20\n"
            f'SYSTEM "/no_think"\n'
        )
    if family == "gemma4":
        return (
            f"FROM {base_tag}\n"
            f"PARAMETER num_ctx 65536\n"
            f"PARAMETER temperature 1.0\n"
            f"PARAMETER top_p 0.95\n"
            f"PARAMETER top_k 64\n"
        )
    if family == "qwen2.5":
        return (
            f"FROM {base_tag}\n"
            f"PARAMETER num_ctx 65536\n"
        )
    return None


def ollama_variant_tag(base_tag: str) -> str:
    """Return the derived tag name for the Claude-Code-friendly variant."""
    if ":" in base_tag:
        name, ver = base_tag.split(":", 1)
        return f"{name}{NOTHINK_VARIANT_SUFFIX}:{ver}"
    return f"{base_tag}{NOTHINK_VARIANT_SUFFIX}"


def ollama_ensure_nothink_variant(base_tag: str) -> tuple[str, dict[str, Any]]:
    """
    Ensure a Claude-Code-friendly Ollama variant of `base_tag` exists.

    Returns (effective_tag, info). If no patch is applicable or creation fails,
    the original tag is returned unchanged and info['patched'] is False.
    """
    info: dict[str, Any] = {"base_tag": base_tag, "patched": False, "reason": ""}
    body = ollama_nothink_modelfile(base_tag)
    if body is None:
        info["reason"] = f"no known no-think fix for '{base_tag}'"
        return base_tag, info

    variant = ollama_variant_tag(base_tag)

    # Skip if the variant already exists on disk.
    for m in parse_ollama_list():
        if m.get("name") == variant:
            info.update(patched=True, variant_tag=variant, reused=True)
            return variant, info

    ensure_state_dirs()
    modelfile_path = STATE_DIR / f"Modelfile.{variant.replace(':', '_').replace('/', '_')}"
    modelfile_path.write_text(body)

    try:
        run(["ollama", "create", variant, "-f", str(modelfile_path)], timeout=600)
    except subprocess.TimeoutExpired:
        info["reason"] = "ollama create timed out"
        return base_tag, info
    except subprocess.CalledProcessError as exc:
        info["reason"] = f"ollama create failed: {exc.stderr or exc.stdout or exc}"
        return base_tag, info
    except Exception as exc:
        info["reason"] = f"ollama create error: {exc}"
        return base_tag, info

    info.update(patched=True, variant_tag=variant, reused=False, modelfile=str(modelfile_path))
    return variant, info


def configure_lmstudio_integration(target: str, model: str) -> dict[str, Any]:
    """
    For LM Studio the 'config' step is just ensuring state dirs exist and
    recording the selection — the server is managed separately via `lms server start`.
    """
    ensure_state_dirs()
    return {
        "target": target,
        "model": model,
        "runtime": "lmstudio",
        "state_dir": str(STATE_DIR),
        "home": str(STATE_HOME),
        "note": "LM Studio manages its own server; use `lms server start` and `lms load` separately.",
    }


def ensure_config(target: str, model: str, runtime: str) -> dict[str, Any]:
    """Dispatch to the right integration config based on runtime."""
    if runtime == "lmstudio":
        return configure_lmstudio_integration(target, model)
    return configure_ollama_integration(target, model)


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
        return {"present": False, "server_running": False, "server_port": LMS_SERVER_PORT, "models": []}

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
            if not stripped or stripped.startswith("LLM") or stripped.startswith("EMBEDDING") or stripped.startswith("You have"):
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
    import urllib.request
    import urllib.error

    url = f"http://localhost:{LMS_SERVER_PORT}/v1/responses"
    payload = json.dumps({
        "model": model,
        "input": "Reply with exactly: OK",
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
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
    """
    import urllib.request
    import urllib.error

    url = f"http://localhost:{LMS_SERVER_PORT}/v1/chat/completions"
    payload = json.dumps({
        "model": model_path,
        "messages": [{"role": "user", "content": "Reply with exactly READY"}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
            text = body["choices"][0]["message"]["content"].strip()
            return {"ok": "READY" in text.upper(), "response": text}
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
            groups[key] = {**m, "ollama_tag": ollama_tag, "lms_mlx_path": lms_mlx_path, "lms_hub_name": lms_hub_name}
        else:
            # Prefer: higher llmfit score, then lower MLX quant rank (more efficient)
            cur_rank = MLX_QUANT_RANK.get(m.get("best_quant", ""), 99)
            ex_rank = MLX_QUANT_RANK.get(existing.get("best_quant", ""), 99)
            cur_score = m.get("score", 0)
            ex_score = existing.get("score", 0)
            if cur_score > ex_score or (cur_score == ex_score and cur_rank < ex_rank):
                groups[key] = {**m, "ollama_tag": ollama_tag, "lms_mlx_path": lms_mlx_path, "lms_hub_name": lms_hub_name}

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
      name="Qwen/Qwen3-Coder-30B-A3B-Instruct", best_quant="mlx-4bit"
      name="lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit"

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
            "free_gib": round(usage.free / (1024 ** 3), 2),
            "total_gib": round(usage.total / (1024 ** 3), 2),
        }
    except Exception as exc:
        return {"path": str(probe), "error": str(exc)}


def llamacpp_detect() -> dict[str, Any]:
    """Detect a usable llama.cpp server binary on PATH."""
    for candidate in ("llama-server", "llama-cpp-server", "server"):
        info = command_version(candidate, ["--version"])
        if info.get("present"):
            return {"present": True, "binary": candidate, "version": info.get("version", "")}
    return {"present": False, "version": ""}


def machine_profile() -> dict[str, Any]:
    llmfit_sys = llmfit_system()
    lms = lms_info()
    llamacpp = llamacpp_detect()

    ollama_info = command_version("ollama")
    claude_info = command_version("claude")
    codex_info = command_version("codex")
    llmfit_info = command_version("llmfit")

    # Presence summary used by the wizard's discover step.
    harnesses_present = [name for name, info in (("claude", claude_info), ("codex", codex_info)) if info.get("present")]
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
            "lmstudio": {"present": lms["present"], "version": command_version("lms")["version"] if lms["present"] else ""},
            "llamacpp": llamacpp,
            "claude": claude_info,
            "codex": codex_info,
            "llmfit": llmfit_info,
        },
        "presence": {
            "harnesses": harnesses_present,
            "engines": engines_present,
            "llmfit": llmfit_info.get("present", False),
            "has_minimum": bool(harnesses_present) and bool(engines_present) and llmfit_info.get("present", False),
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

    ollama_installed = {m["name"]: m for m in profile.get("ollama", {}).get("models", []) if m.get("local")}
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
                        caveats.append(f"LM Studio smoke test failed: {smoke.get('error') or smoke.get('response','')}")
                else:
                    caveats.append(f"Could not load model in LM Studio: {load_result.get('error','')}")
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
                    caveats.append(f"Ollama smoke test failed: {smoke.get('error') or smoke.get('response','')}")
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
            caveats.append(f"Ollama smoke test failed: {smoke.get('error') or smoke.get('response','')}")

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
            caveats.append("Download the model above, then re-run this command to confirm readiness.")

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
            caveats.append("Run `ollama pull` above, then re-run this command to confirm readiness.")

    # --- Pass 5: no llmfit candidates at all ---
    if not selected_tag:
        status = "download-required"
        selected_tag = "qwen2.5-coder:7b"
        runtime = "ollama"
        rationale.append("llmfit returned no candidates. Defaulting to qwen2.5-coder:7b as a safe fallback.")
        next_steps.append(f"ollama pull {selected_tag}")

    modes: dict[str, str | None] = {
        "balanced": selected_tag,
        "fast": selected_tag,
        "quality": selected_tag if (selected_candidate and selected_candidate.get("fit_level") in ("Perfect", "Good")) else None,
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
            "estimated_tps": selected_candidate.get("estimated_tps") if selected_candidate else None,
            "memory_required_gb": selected_candidate.get("memory_required_gb") if selected_candidate else None,
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
            ["codex", "exec", "--skip-git-repo-check", "--oss", "--local-provider", provider, "-m", model, "Reply with exactly READY"],
            env=env,
            timeout=240,
        )
        merged = (cp.stdout + "\n" + cp.stderr).strip()
        normalized = re.sub(r"[^a-z]", "", merged.lower())
        ok = "ready" in normalized
        auth_noise = "failed to refresh available models" in merged.lower() or "401 unauthorized" in merged.lower()
        return {"ok": ok, "output": cp.stdout.strip(), "stderr": cp.stderr.strip(), "auth_noise": auth_noise}
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

    rt = recommendation["runtime"]
    config_written: dict[str, Any] = {
        "codex": ensure_config("codex", recommendation["selected_model"], rt),
        "claude": ensure_config("claude", recommendation["selected_model"], rt),
    }

    codex_smoke = smoke_test_codex(recommendation["selected_model"], recommendation["runtime"]) if run_codex_smoke else None
    if codex_smoke and not codex_smoke.get("ok"):
        issues.append("Codex local smoke test failed.")
    elif codex_smoke and codex_smoke.get("auth_noise"):
        fixes.append("Codex emits a harmless 401 model-refresh warning in local-only mode.")

    return {
        "profile": profile,
        "recommendation": recommendation,
        "issues": issues,
        "fixes": fixes,
        "config_written": config_written,
        "codex_smoke": codex_smoke,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


MODE_CHOICES = ["balanced", "fast", "quality"]


def main() -> None:
    parser = argparse.ArgumentParser(description="POC helper for claude-codex-local")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile")

    rec_cmd = sub.add_parser("recommend")
    rec_cmd.add_argument(
        "--mode", choices=MODE_CHOICES, default="balanced",
        help="Preset: balanced (default), fast (smallest/fastest), quality (highest score)",
    )

    ensure_cmd = sub.add_parser("ensure-config")
    ensure_cmd.add_argument("target", choices=["codex", "claude"])
    ensure_cmd.add_argument("--model")
    ensure_cmd.add_argument("--mode", choices=MODE_CHOICES, default="balanced")

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
    elif args.command == "ensure-config":
        rec = select_best_model(machine_profile(), args.mode)
        model = args.model or rec["selected_model"]
        print_payload(ensure_config(args.target, model, rec["runtime"]))
    elif args.command == "doctor":
        print_payload(doctor(args.run_codex_smoke, args.mode))
    elif args.command == "adapters":
        result = []
        for adapter in ALL_ADAPTERS:
            result.append({
                "name": adapter.name,
                "detect": adapter.detect(),
                "healthcheck": adapter.healthcheck(),
                "models": adapter.list_models(),
                "recommend_params": {m: adapter.recommend_params(m) for m in MODE_CHOICES},
            })
        print_payload({"adapters": result})


if __name__ == "__main__":
    main()
