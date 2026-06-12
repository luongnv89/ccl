from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result

_MIN_MODEL_MATCH_LEN = 12
_VARIANT_TOKENS = ("instruct", "chat", "base", "it")


def _variant_token(normalized: str) -> str | None:
    for tok in _VARIANT_TOKENS:
        if (
            normalized == tok
            or normalized.startswith(f"{tok}-")
            or normalized.endswith(f"-{tok}")
            or f"-{tok}-" in normalized
        ):
            return tok
    return None


def _normalize_model_id(value: str) -> str:
    raw = value.strip().lower()
    raw = raw.rsplit("/", 1)[-1]
    if raw.endswith(".gguf"):
        raw = raw[: -len(".gguf")]
    if raw.endswith("-gguf"):
        raw = raw[: -len("-gguf")]
    return raw


def _models_match(running: str, wanted: str) -> bool:
    if not running or not wanted:
        return False
    a = _normalize_model_id(running)
    b = _normalize_model_id(wanted)
    if a == b:
        return True
    short = a if len(a) <= len(b) else b
    long_ = b if short is a else a
    if len(short) < _MIN_MODEL_MATCH_LEN or short not in long_:
        return False
    a_variant = _variant_token(a)
    b_variant = _variant_token(b)
    return not (a_variant and b_variant and a_variant != b_variant)


def _render_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _wrap(result: dict[str, Any]) -> dict[str, object]:
    return lifecycle_result(
        "llamacpp",
        "test",
        ok=bool(result.get("ok")),
        detail=result.get("error") or result.get("response") or "",
        data=result,
    )


def _spawn_and_smoke(model: str, model_path: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Verify the GGUF path, start llama-server, then smoke-test it.

    Returns the raw ``{"ok": bool, ...}`` smoke dict (callers ``_wrap`` it).
    """
    if not model_path or not Path(model_path).is_file():
        return {
            "ok": False,
            "error": f"no resolved GGUF path for model '{model}'; re-run wizard step 4",
        }

    start_result = pb.llamacpp_start_server(
        model_path=model_path,
        profile=profile,
        port=pb.LLAMACPP_SERVER_PORT,
        host=pb.LLAMACPP_SERVER_HOST,
    )
    argv = start_result.get("argv") or []
    manual_cmd = _render_command(argv) if argv else ""
    if not start_result.get("ok"):
        return {
            "ok": False,
            "error": start_result.get("error") or "unknown error",
            "hint": start_result.get("hint"),
            "manual_command": manual_cmd,
            "log_path": start_result.get("log_path") or "",
        }

    handle = start_result["handle"]
    smoke = pb.smoke_test_llamacpp_model(model)
    if not smoke.get("ok"):
        smoke = dict(smoke)
        smoke.setdefault("manual_command", manual_cmd)
        smoke.setdefault("log_path", handle.log_path)
    return smoke


def run(
    model: str = "",
    dry_run: bool = True,
    profile: dict[str, Any] | None = None,
    non_interactive: bool = True,
    **_: object,
) -> dict[str, object]:
    if dry_run:
        return lifecycle_result(
            "llamacpp",
            "test",
            detail=f"Would smoke-test llama.cpp model {model!r}",
            data={"dry_run": True, "model": model},
        )

    profile = profile or {}
    model_path = profile.get("llamacpp_model_path") or ""
    info = pb.llamacpp_info()
    if info.get("remote"):
        base_url = info.get("base_url") or pb.llamacpp_base_url()
        if info.get("server_running"):
            return _wrap(pb.smoke_test_llamacpp_model(model))
        return _wrap(
            {
                "ok": False,
                "error": (
                    f"remote llama.cpp server at {base_url} is not reachable "
                    f"(GET /health failed); check the remote host and "
                    f"LLAMACPP_BASE_URL before re-running"
                ),
            }
        )

    if info.get("server_running"):
        running_model = (info.get("model") or "").strip()
        if _models_match(running_model, model):
            return _wrap(pb.smoke_test_llamacpp_model(model))

        # Different model on our port. If WE started it (pid-file gated), stop
        # and restart with the wanted model (issue #149).
        def _switch() -> dict[str, Any]:
            stop_res = pb.llamacpp_stop_server_by_port(info["server_port"])
            if not stop_res.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"could not stop the running llama-server on port "
                        f"{info['server_port']}: {stop_res.get('error')}; "
                        f"stop it manually and re-run with --resume"
                    ),
                }
            return _spawn_and_smoke(model, model_path, profile)

        if non_interactive:
            return _wrap(_switch())
        # Interactive: caller drives the prompt elsewhere; default to using the
        # already-running model for the smoke test rather than killing it.
        smoke_target = running_model or model
        return _wrap(pb.smoke_test_llamacpp_model(smoke_target))

    return _wrap(_spawn_and_smoke(model, model_path, profile))
