from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any

from claude_codex_local._config import LMS_SERVER_PORT, ORIG_HOME
from claude_codex_local._shell import command_version, run


def lms_binary() -> str | None:
    lms_path = ORIG_HOME / ".lmstudio" / "bin" / "lms"
    if lms_path.exists():
        return str(lms_path)
    info = command_version("lms")
    return "lms" if info.get("present") else None


def lms_info() -> dict[str, Any]:
    lms = lms_binary()
    if not lms:
        return {
            "present": False,
            "server_running": False,
            "server_port": LMS_SERVER_PORT,
            "models": [],
        }

    server_running = False
    try:
        cp = run([lms, "server", "status"])
        server_running = str(LMS_SERVER_PORT) in (cp.stdout + cp.stderr)
    except Exception:
        pass

    models: list[dict[str, Any]] = []
    try:
        cp = run([lms, "ls"])
        for line in cp.stdout.splitlines():
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("LLM")
                or stripped.startswith("EMBEDDING")
                or stripped.startswith("You have")
            ):
                continue
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
            chunk = resp.read(256)
            return bool(chunk and chunk.strip())
    except Exception:
        return False


def lms_start_server() -> bool:
    lms = lms_binary()
    if not lms:
        return False
    try:
        run([lms, "server", "start"])
        return True
    except Exception:
        return False


def lms_running_models() -> set[str]:
    lms = lms_binary()
    if not lms:
        return set()
    try:
        cp = run([lms, "ps"])
        running: set[str] = set()
        for line in cp.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                running.add(parts[0])
        return running
    except Exception:
        return set()


def lms_load_model(model_path: str) -> dict[str, Any]:
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


def smoke_test_lmstudio_model(
    model_path: str,
    prompt: str = "Reply with exactly READY",
    expected: str | None = "READY",
    max_tokens: int = 16,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = f"http://localhost:{LMS_SERVER_PORT}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
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
        ok_flag = bool(text) if expected is None else expected.upper() in text.upper()
        return {
            "ok": ok_flag,
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
        }
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
