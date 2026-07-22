from __future__ import annotations

import json
import os
import time
from typing import Any

from claude_codex_local._config import VLLM_BASE_URL, _is_local_base_url
from claude_codex_local._shell import command_version


def vllm_info() -> dict[str, Any]:
    cli_info = command_version("vllm")
    from claude_codex_local._adapters import VLLMAdapter
    adapter = VLLMAdapter()
    base_url = adapter._base_url or VLLM_BASE_URL
    detect = adapter.detect()
    server_reachable = bool(detect.get("present"))

    info: dict[str, Any] = {
        "present": server_reachable,
        "base_url": base_url,
        "version": detect.get("version") or cli_info.get("version", ""),
        "models": adapter.list_models() if server_reachable else [],
        "server_reachable": server_reachable,
        "remote": not _is_local_base_url(base_url),
    }
    if not server_reachable:
        info["error"] = f"vLLM server not reachable at {base_url}"
    return info


def smoke_test_vllm_model(
    model: str,
    base_url: str | None = "http://localhost:8000",
    api_key: str | None = "",
    timeout: int = 60,
    max_tokens: int = 2048,
    prompt: str = "Reply with exactly READY",
    expected: str | None = "READY",
) -> dict[str, Any]:
    import time
    import urllib.error
    import urllib.request

    url = f"{(base_url or 'http://localhost:8000').rstrip('/')}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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

        ok_flag = bool(text) if expected is None else expected.upper() in text.upper()
        return {
            "ok": ok_flag,
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
