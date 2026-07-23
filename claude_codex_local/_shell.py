import os
import subprocess
import sys
from typing import Any

from claude_codex_local._config import (
    ORIG_HOME,
    STATE_DIR,
)


def ensure_path(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(os.environ if env is None else env)
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


def ollama_base_url() -> str:
    # Import core at call time so test monkeypatches on
    # ``core.OLLAMA_BASE_URL`` (set by wizard._apply_remote_endpoint)
    # propagate into code that reads the URL via this function.
    import claude_codex_local.core as _core

    return _core.OLLAMA_BASE_URL


def ollama_openai_base_url() -> str:
    return f"{ollama_base_url()}/v1"


def llamacpp_base_url() -> str:
    import claude_codex_local.core as _core

    return _core.LLAMACPP_BASE_URL


def _auth_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
