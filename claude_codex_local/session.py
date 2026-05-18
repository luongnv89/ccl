#!/usr/bin/env python3
"""Shared conversation session storage between local agents.

The session store is an opt-in JSONL protocol under
``~/.claude-codex-local/sessions``. Each agent writes to its own file and may
copy redacted messages into another agent's file via ``ccl session sync``.

The module intentionally stores only conversation content and lightweight
metadata. Secret-looking values are redacted before writes so a sync operation
cannot copy credentials between agents.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AGENT_ID_ENV = "__CCL_AGENT_ID"
_MAX_AGENT_ID_LENGTH = 80
_SECRET_PATTERNS = [
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk_live_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(r"gh[opsu]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
]


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _get_state_dir() -> Path:
    """Get the state directory from environment or the default location."""
    return Path(
        os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", str(Path.home() / ".claude-codex-local"))
    )


def _get_sessions_dir() -> Path:
    """Get the sessions directory."""
    return _get_state_dir() / "sessions"


def _ensure_sessions_dir() -> Path:
    """Ensure the sessions directory exists and return it."""
    path = _get_sessions_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_agent_id(agent_id: str) -> str:
    """Normalize an agent id so it cannot escape the sessions directory."""
    value = (agent_id or get_current_agent_id() or "default").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return (value or "default")[:_MAX_AGENT_ID_LENGTH]


def _parse_timestamp(value: Any) -> datetime | str:
    """Parse timestamps when possible while preserving unknown strings."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return _utcnow()


def _redact_string(value: str) -> str:
    """Replace secret-looking tokens with a redaction marker."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    """Recursively redact secret-looking strings from JSON-compatible data."""
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, dict):
        return {str(key): redact_secrets(item) for key, item in value.items()}
    return value


@dataclass
class SessionMessage:
    """A single message in a shared conversation session."""

    role: str  # "user" | "assistant" | "system" | "tool" | "tool_result"
    content: str | dict[str, Any] | list[Any]
    timestamp: datetime | str = field(default_factory=_utcnow)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMessage:
        return cls(
            role=str(data.get("role", "")),
            content=data.get("content", ""),
            timestamp=_parse_timestamp(data.get("timestamp")),
            session_id=str(data.get("session_id", "")),
            agent_id=str(data.get("agent_id", "")),
        )

    def to_dict(self, *, agent_id: str | None = None) -> dict[str, Any]:
        timestamp = (
            self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else str(self.timestamp)
        )
        return {
            "role": self.role,
            "content": redact_secrets(self.content),
            "timestamp": timestamp,
            "session_id": self.session_id,
            "agent_id": _safe_agent_id(agent_id or self.agent_id),
        }


def get_current_agent_id() -> str:
    """Get the current agent identifier from the environment."""
    return os.environ.get(_AGENT_ID_ENV, "").strip()


def get_session_path(agent_id: str) -> Path:
    """Get the path to a session file for a specific agent."""
    return _ensure_sessions_dir() / f"{_safe_agent_id(agent_id)}.jsonl"


def _load_jsonl(path: Path) -> list[SessionMessage]:
    if not path.exists():
        return []
    messages: list[SessionMessage] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    messages.append(SessionMessage.from_dict(data))
    except OSError:
        return []
    return messages


def load_session(agent_id: str) -> list[SessionMessage]:
    """Load messages from a session file for a specific agent."""
    return _load_jsonl(get_session_path(agent_id))


def _append_message(path: Path, agent_id: str, message: SessionMessage) -> None:
    _ensure_sessions_dir()
    payload = message.to_dict(agent_id=agent_id)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_message(agent_id: str, message: SessionMessage) -> None:
    """Append a redacted message to an agent's session file."""
    _append_message(get_session_path(agent_id), _safe_agent_id(agent_id), message)


def clear_session(agent_id: str) -> dict[str, Any]:
    """Clear the session file for a specific agent."""
    path = get_session_path(agent_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        return {"success": False, "agent_id": _safe_agent_id(agent_id), "error": str(exc)}
    return {"success": True, "agent_id": _safe_agent_id(agent_id), "error": None}


def get_session_summary(agent_id: str) -> dict[str, Any]:
    """Get a summary of the session for a specific agent."""
    messages = load_session(agent_id)
    last = messages[-1].timestamp if messages else None
    last_message = last.isoformat() if isinstance(last, datetime) else (str(last) if last else None)
    return {
        "agent_id": _safe_agent_id(agent_id),
        "message_count": len(messages),
        "last_message": last_message,
        "session_ids": sorted({message.session_id for message in messages if message.session_id}),
    }


def _message_key(message: SessionMessage) -> tuple[str, str, str, str]:
    timestamp = (
        message.timestamp.isoformat()
        if isinstance(message.timestamp, datetime)
        else str(message.timestamp)
    )
    return (
        message.session_id,
        message.role,
        timestamp,
        json.dumps(redact_secrets(message.content), sort_keys=True),
    )


def sync_session(agent_id: str, other_agent_id: str) -> dict[str, Any]:
    """Copy redacted messages from one agent session into another."""
    target = _safe_agent_id(agent_id)
    source = _safe_agent_id(other_agent_id)
    source_messages = load_session(source)
    target_path = get_session_path(target)
    existing_keys = {_message_key(message) for message in load_session(target)}
    copied = 0
    for message in source_messages:
        key = _message_key(message)
        if key in existing_keys:
            continue
        _append_message(
            target_path,
            source,
            SessionMessage(
                role=message.role,
                content=message.content,
                timestamp=message.timestamp,
                session_id=message.session_id,
                agent_id=source,
            ),
        )
        existing_keys.add(key)
        copied += 1
    return {
        "success": True,
        "agent_id": target,
        "other_agent_id": source,
        "message_count": copied,
        "error": None,
    }


def get_all_sessions() -> list[dict[str, Any]]:
    """Get summaries for all discovered session files."""
    sessions_dir = _ensure_sessions_dir()
    return [get_session_summary(path.stem) for path in sorted(sessions_dir.glob("*.jsonl"))]


def truncate_session(agent_id: str, keep_last: int | None = None) -> dict[str, Any]:
    """Truncate a session file, keeping only the last ``keep_last`` messages.

    ``keep_last`` of ``None`` or ``0`` deletes every message in the file. The
    CLI surface (``ccl session truncate``) requires ``--keep`` so this wipe
    behavior is never accidental; callers using the function directly should
    pass an explicit value and use :func:`clear_session` to remove the file.
    """
    keep = max(0, keep_last or 0)
    path = get_session_path(agent_id)
    if not path.exists():
        return {"success": True, "agent_id": _safe_agent_id(agent_id), "kept": 0, "error": None}
    messages = load_session(agent_id)
    kept_messages = messages[-keep:] if keep else []
    try:
        with path.open("w", encoding="utf-8") as handle:
            for message in kept_messages:
                handle.write(
                    json.dumps(
                        message.to_dict(agent_id=message.agent_id or agent_id), ensure_ascii=False
                    )
                    + "\n"
                )
    except OSError as exc:
        return {
            "success": False,
            "agent_id": _safe_agent_id(agent_id),
            "kept": 0,
            "error": str(exc),
        }
    return {
        "success": True,
        "agent_id": _safe_agent_id(agent_id),
        "kept": len(kept_messages),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Cross-harness bridge: locate native session files, import them, render a
# context prefix for the consumer side. The producer (ccl run) imports its own
# harness's latest session file after the harness exits; the consumer reads
# the *opposite* agent's accumulated transcript and prepends it to the prompt.
# ---------------------------------------------------------------------------

_CWD_BRIDGE_SCAN_DAYS = 14
_CONTEXT_PREFIX_CHAR_BUDGET = 16_000  # ~4k tokens at 4 chars/token rule of thumb
_BRIDGE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7-day staleness cap


def _normalize_cwd(cwd: str) -> str:
    """Resolve symlinks so encoded directory names match what the harnesses see.

    macOS aliases ``/tmp`` → ``/private/tmp``, ``/var`` → ``/private/var``, and
    similar. ``os.getcwd()`` returns the resolved form, but a caller may pass
    a non-resolved path; harnesses encode whatever they themselves saw at
    launch. Use :py:meth:`Path.resolve` to make both sides comparable.
    """
    try:
        return str(Path(cwd).resolve())
    except OSError:
        return cwd


def _encode_cwd_claude(cwd: str) -> str:
    """Claude Code's project-dir encoding: leading dash, slashes → dashes."""
    return "-" + _normalize_cwd(cwd).lstrip("/").replace("/", "-")


def _encode_cwd_pi(cwd: str) -> str:
    """Pi's session-dir encoding: double-dash leading and trailing."""
    return "--" + _normalize_cwd(cwd).lstrip("/").replace("/", "-") + "--"


def _native_root(harness: str) -> Path:
    """Resolve the harness's native state root (overridable for tests)."""
    home = Path(os.environ.get("CCL_NATIVE_HOME_OVERRIDE", str(Path.home())))
    if harness == "claude":
        return home / ".claude"
    if harness == "codex":
        return home / ".codex"
    if harness == "pi":
        return home / ".pi" / "agent"
    return home


def find_latest_native_session(
    harness: str, cwd: str, *, max_age_seconds: int = _BRIDGE_MAX_AGE_SECONDS
) -> Path | None:
    """Return the most recently modified native session file for ``cwd``.

    Claude and Pi shard by encoded cwd directory; we return the newest
    ``*.jsonl`` under that directory. Codex shards by date and stores cwd in
    each file's ``session_meta`` line, so we walk the last
    ``_CWD_BRIDGE_SCAN_DAYS`` date directories newest-first and confirm the
    cwd match by reading the first line. Files older than ``max_age_seconds``
    are excluded so stale months-old transcripts don't get silently
    re-injected. Returns ``None`` if no match.
    """
    root = _native_root(harness)
    if not root.exists():
        return None

    cwd_resolved = _normalize_cwd(cwd)
    cutoff = datetime.now(tz=timezone.utc).timestamp() - max_age_seconds

    def _fresh(path: Path) -> bool:
        try:
            return path.stat().st_mtime >= cutoff
        except OSError:
            return False

    if harness == "claude":
        project_dir = root / "projects" / _encode_cwd_claude(cwd_resolved)
        if not project_dir.exists():
            return None
        candidates = sorted(
            (p for p in project_dir.glob("*.jsonl") if _fresh(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    if harness == "pi":
        project_dir = root / "sessions" / _encode_cwd_pi(cwd_resolved)
        if not project_dir.exists():
            return None
        candidates = sorted(
            (p for p in project_dir.glob("*.jsonl") if _fresh(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    if harness == "codex":
        sessions_dir = root / "sessions"
        if not sessions_dir.exists():
            return None
        # Walk all rollout-*.jsonl across recent dates, newest first, and
        # pick the first one whose session_meta.cwd matches. The staleness
        # cap also bounds how far back we scan in practice.
        candidates = sorted(
            (p for p in sessions_dir.rglob("rollout-*.jsonl") if _fresh(p)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates[: 50 * _CWD_BRIDGE_SCAN_DAYS]:  # cap scan size
            try:
                with path.open("r", encoding="utf-8") as handle:
                    first = handle.readline()
            except OSError:
                continue
            try:
                meta = json.loads(first)
            except json.JSONDecodeError:
                continue
            payload = meta.get("payload") or {}
            stored_cwd = _normalize_cwd(str(payload.get("cwd", "")))
            if stored_cwd == cwd_resolved:
                return path
        return None

    return None


def import_native_session(harness: str, agent_id: str, cwd: str) -> dict[str, Any]:
    """Import the latest native session for ``harness`` into ``<agent_id>.jsonl``.

    Idempotent: messages already present (matched by the same content-hash
    dedup key as :func:`sync_session`) are skipped. Returns a status dict
    with the number of messages imported and the source path.
    """
    # Local import: keeps the adapters module's import of session.py from
    # cycling back through session.py at top-level load time.
    from claude_codex_local.session_adapters import read_native_session

    safe_agent = _safe_agent_id(agent_id)
    source_path = find_latest_native_session(harness, cwd)
    if source_path is None:
        return {
            "success": True,
            "agent_id": safe_agent,
            "harness": harness,
            "source_path": None,
            "imported": 0,
            "error": None,
        }
    messages = read_native_session(harness, source_path)
    if not messages:
        return {
            "success": True,
            "agent_id": safe_agent,
            "harness": harness,
            "source_path": str(source_path),
            "imported": 0,
            "error": None,
        }
    target_path = get_session_path(safe_agent)
    existing_keys = {_message_key(message) for message in load_session(safe_agent)}
    imported = 0
    for message in messages:
        key = _message_key(message)
        if key in existing_keys:
            continue
        _append_message(target_path, safe_agent, message)
        existing_keys.add(key)
        imported += 1
    return {
        "success": True,
        "agent_id": safe_agent,
        "harness": harness,
        "source_path": str(source_path),
        "imported": imported,
        "error": None,
    }


def _content_to_text(content: Any) -> str:
    """Reduce SessionMessage.content (string | dict | list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else json.dumps(content, ensure_ascii=False)
    return ""


def build_context_prefix(agent_id: str, *, char_budget: int = _CONTEXT_PREFIX_CHAR_BUDGET) -> str:
    """Render an agent's accumulated transcript as a prompt-prefix block.

    The block is tail-keep: most-recent messages first wins, walking backward
    until we hit the character budget. Returns an empty string when the agent
    has no messages, so callers can no-op cleanly. The output is wrapped in
    explicit ``[prior context …]`` / ``[end prior context]`` delimiters so
    the consumer harness recognizes it as preamble rather than user intent.
    """
    safe_agent = _safe_agent_id(agent_id)
    messages = load_session(safe_agent)
    if not messages:
        return ""
    rendered: list[str] = []
    used = 0
    for message in reversed(messages):
        text = _content_to_text(message.content).strip()
        if not text:
            continue
        line = f"{message.role}: {text}"
        if used + len(line) + 2 > char_budget:
            break
        rendered.append(line)
        used += len(line) + 2
    if not rendered:
        return ""
    rendered.reverse()
    body = "\n".join(rendered)
    return (
        f"[prior context, agent={safe_agent}, {len(rendered)} messages]\n"
        f"{body}\n"
        f"[end prior context]\n\n"
    )
