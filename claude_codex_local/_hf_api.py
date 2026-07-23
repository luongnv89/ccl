from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from claude_codex_local._shell import ensure_path


def disk_usage_for(path: Path) -> dict[str, Any]:
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
            proc = subprocess.Popen(cmd, env=ensure_path(None))
            try:
                rc = proc.wait(timeout=3600)
            except KeyboardInterrupt:
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
                    "not_found": False,
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
        import claude_codex_local.core as _core

        cp = _core.run(cmd, timeout=600, check=False)
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


_HF_NOT_FOUND_MARKERS = (
    "repository not found",
    "repositorynotfounderror",
    "404 client error",
    "not found for url",
    "entry not found",
    "revisionnotfounderror",
)


def _looks_like_not_found(text: str) -> bool:
    t = (text or "").lower()
    return any(marker in t for marker in _HF_NOT_FOUND_MARKERS)


def _dir_size_bytes(root: Path) -> int:
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
    import urllib.error
    import urllib.parse
    import urllib.request

    if not query or not query.strip():
        return []
    try:
        params = urllib.parse.urlencode({"search": query.strip(), "limit": limit})
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "claude-codex-local"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    import difflib

    import claude_codex_local.core as _core

    candidates = _core.huggingface_search_models(query, limit=10)
    if not candidates:
        return []
    ranked = difflib.get_close_matches(
        query.lower(),
        [c.lower() for c in candidates],
        n=max_results,
        cutoff=0.3,
    )
    if ranked:
        lut = {c.lower(): c for c in candidates}
        deduped: list[str] = []
        for low in ranked:
            orig = lut.get(low)
            if orig and orig not in deduped:
                deduped.append(orig)
            if len(deduped) >= max_results:
                break
        return deduped
    return candidates[:max_results]


def huggingface_list_repo_files(
    repo_id: str,
    *,
    timeout: float = 10.0,
) -> list[str]:
    import urllib.error
    import urllib.parse
    import urllib.request

    if not repo_id or not repo_id.strip():
        return []
    try:
        encoded = urllib.parse.quote(repo_id.strip(), safe="/")
        url = f"https://huggingface.co/api/models/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "claude-codex-local"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    # Call through core so test monkeypatches on core.huggingface_list_repo_files
    # take effect (tests patch the re-export on the core facade).
    import claude_codex_local.core as _core

    files = _core.huggingface_list_repo_files(repo_id)
    if not files:
        return None
    return any(f.lower().endswith(".gguf") for f in files)


_GGUF_MIRROR_CACHE: dict[str, str | None] = {}

_GGUF_MIRROR_AUTHORS: tuple[str, ...] = (
    "bartowski",
    "unsloth",
    "lmstudio-community",
    "TheBloke",
)


def _candidate_base_name(name: str) -> str:
    base = (name or "").split("/", 1)[-1]
    base = re.sub(
        r"[-_](MLX[-_]?\w*|FP\d+|BF\d+|GPTQ|AWQ|\d+bit|GGUF)$",
        "",
        base,
        flags=re.IGNORECASE,
    )
    return base


def resolve_gguf_mirror(name: str) -> str | None:
    # Import core at call time so test monkeypatches on
    # ``core.huggingface_repo_has_gguf`` and ``core.huggingface_search_models``
    # propagate.  Since core.py now imports these directly from _hf_api,
    # ``core.huggingface_repo_has_gguf`` IS ``_hf_api.huggingface_repo_has_gguf``
    # (same function object), so patches on either propagate.
    import claude_codex_local.core as _core

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

    if _core.huggingface_repo_has_gguf(name) is True:
        return _remember(name)

    for author in _GGUF_MIRROR_AUTHORS:
        candidate = f"{author}/{base}-GGUF"
        if _core.huggingface_repo_has_gguf(candidate) is True:
            return _remember(candidate)

    for hit in _core.huggingface_search_models(f"{base}-GGUF", limit=5):
        if _core.huggingface_repo_has_gguf(hit) is True:
            return _remember(hit)

    return _remember(None)
