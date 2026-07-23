from __future__ import annotations

import json
import time
from typing import Any

from claude_codex_local._config import (
    OPENROUTER_BASE_URL,
    _probe_openai_models_endpoint,
)


def smoke_test_openrouter_models(base_url: str | None = None) -> dict[str, Any]:
    return _probe_openai_models_endpoint(
        base_url or OPENROUTER_BASE_URL,
        service_name="OpenRouter",
        timeout=15,
        headers={"Content-Type": "application/json"},
    )


def smoke_test_openrouter_model(
    model: str,
    base_url: str | None = None,
    api_key: str | None = "",
    timeout: int = 60,
    max_tokens: int = 16,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"{(base_url or OPENROUTER_BASE_URL).rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly READY"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/luongnv89/ccl",
        "X-Title": "claude-codex-local",
    }
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
            "model": model,
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "model": model, "error": f"OpenRouter model {model} failed: {exc}"}
    except Exception as exc:
        return {"ok": False, "model": model, "error": f"OpenRouter model {model} failed: {exc}"}


def fetch_openrouter_free_models(
    base_url: str | None = None,
    timeout: int = 15,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"{(base_url or OPENROUTER_BASE_URL).rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        return {"ok": False, "models": [], "error": f"OpenRouter unreachable: {exc}"}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}

    raw_models = body.get("data", [])
    free_models: list[dict[str, Any]] = []
    for model in raw_models:
        pricing = model.get("pricing", {})
        prompt_price = pricing.get("prompt", "1")
        completion_price = pricing.get("completion", "1")
        if prompt_price != "0" or completion_price != "0":
            continue

        arch = model.get("architecture", {})
        supported = model.get("supported_parameters", [])

        capabilities: list[str] = []
        modality = arch.get("modality", "")
        if "text" in modality or modality == "":
            capabilities.append("text")
        if "image" in modality:
            capabilities.append("image")
        if "function_call" in supported or "tools" in supported:
            capabilities.append("function-calling")
        if "response_format" in supported or "structured_outputs" in supported:
            capabilities.append("structured-output")
        if not capabilities:
            capabilities.append("text")

        ctx_len = model.get("context_length", 0)
        if not isinstance(ctx_len, int):
            ctx_len = 0

        desc = model.get("description", "") or ""
        if len(desc) > 80:
            desc = desc[:77] + "..."

        free_models.append(
            {
                "id": model.get("id", ""),
                "context_length": ctx_len,
                "architecture": arch.get("tokenizer", "unknown"),
                "capabilities": capabilities,
                "description": desc,
            }
        )

    free_models.sort(key=lambda m: m["context_length"], reverse=True)

    return {"ok": True, "models": free_models, "error": None}
