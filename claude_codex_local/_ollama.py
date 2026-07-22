from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any

from claude_codex_local._config import OLLAMA_API_KEY, _is_local_base_url, HF_TO_OLLAMA
from claude_codex_local._shell import ollama_base_url, _auth_headers, run, command_version


def _ollama_http_models(timeout: int = 5) -> list[dict[str, Any]] | None:
    if "OLLAMA_HOST" not in os.environ:
        return None
    import urllib.error
    import urllib.request

    url = f"{ollama_base_url()}/api/tags"
    req = urllib.request.Request(url, headers=_auth_headers(OLLAMA_API_KEY), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, OSError):
        return None
    except Exception:
        return None

    models = body.get("models", []) if isinstance(body, dict) else []
    if not isinstance(models, list):
        return []
    is_local = _is_local_base_url(ollama_base_url())
    result: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = str(model.get("name") or model.get("model") or "")
        if not name:
            continue
        result.append(
            {
                "name": name,
                "id": str(model.get("digest") or model.get("id") or ""),
                "size": model.get("size", ""),
                "modified": str(model.get("modified_at") or model.get("modified") or ""),
                "local": is_local,
            }
        )
    return result


def _parse_ollama_list_cli() -> list[dict[str, Any]]:
    import claude_codex_local.core as _core

    try:
        cp = _core.run(["ollama", "list"])
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


def parse_ollama_list() -> list[dict[str, Any]]:
    models = _ollama_http_models()
    if models is not None:
        return models
    return _parse_ollama_list_cli()


def ollama_info() -> dict[str, Any]:
    # Import core at call time (never at module level) so that test
    # monkeypatches on core.parse_ollama_list / core._ollama_http_models
    # take effect — the functions live in this module but tests patch the
    # re-exports on the core (pb) facade.
    import claude_codex_local.core as _core

    cli_info = command_version("ollama")
    models = _core._ollama_http_models()
    server_reachable = models is not None
    if models is None and cli_info.get("present"):
        models = _core.parse_ollama_list()
    return {
        "present": server_reachable or bool(cli_info.get("present")),
        "version": cli_info.get("version", ""),
        "base_url": ollama_base_url(),
        "models": models or [],
        "server_reachable": server_reachable,
        "remote": not _is_local_base_url(ollama_base_url()),
        "error": None
        if server_reachable
        else f"Ollama server not reachable at {ollama_base_url()}",
    }


def hf_name_to_ollama_tag(hf_name: str) -> str | None:
    for pattern, tag in HF_TO_OLLAMA:
        if pattern.search(hf_name):
            return tag
    return None


def smoke_test_ollama_model(
    model: str,
    prompt: str = "Reply with exactly READY",
    expected: str | None = "READY",
    max_tokens: int | None = None,
) -> dict[str, Any]:
    import time
    import urllib.error
    import urllib.request

    url = f"{ollama_base_url()}/api/generate"
    payload_dict: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if max_tokens is not None:
        payload_dict["options"] = {"num_predict": max_tokens}
    payload = json.dumps(payload_dict).encode()
    req = urllib.request.Request(url, data=payload, headers=_auth_headers(OLLAMA_API_KEY))
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
            duration_seconds = wall_seconds

        ok_flag = bool(text) if expected is None else expected.upper() in text.upper()
        return {
            "ok": ok_flag,
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
        }
    except urllib.error.URLError:
        return _smoke_test_ollama_cli(model, prompt=prompt, expected=expected)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _smoke_test_ollama_cli(
    model: str,
    prompt: str = "Reply with exactly READY",
    expected: str | None = "READY",
) -> dict[str, Any]:
    import claude_codex_local.core as _core

    try:
        cp = _core.run(["ollama", "run", model, prompt], timeout=180)
        text = cp.stdout.strip()
        ok_flag = bool(text) if expected is None else expected.upper() in text.upper()
        return {
            "ok": ok_flag,
            "response": text,
            "tokens_per_second": None,
            "completion_tokens": None,
            "duration_seconds": None,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout after 180s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

