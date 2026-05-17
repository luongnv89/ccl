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


def _get_session_file() -> Path:
    """Get the shared session file path."""
    return _get_sessions_dir() / "shared.jsonl"


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


@dataclass
class Session:
    """A snapshot of a shared session at a point in time."""

    session_id: str
    messages: list[SessionMessage]
    agent_id: str
    timestamp: datetime = field(default_factory=_utcnow)


def get_current_agent_id() -> str:
    """Get the current agent identifier from the environment."""
    return os.environ.get(_AGENT_ID_ENV, "").strip()


def get_session_path(agent_id: str) -> Path:
    """Get the path to a session file for a specific agent."""
    return _ensure_sessions_dir() / f"{_safe_agent_id(agent_id)}.jsonl"


def get_shared_session_path() -> Path:
    """Get the path to the shared session file."""
    _ensure_sessions_dir()
    return _get_session_file()


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


def save_message_to_shared(agent_id: str, message: SessionMessage) -> None:
    """Append a redacted message to the shared session file."""
    _append_message(get_shared_session_path(), _safe_agent_id(agent_id), message)


def get_shared_messages() -> list[SessionMessage]:
    """Load all messages from the shared session file."""
    return _load_jsonl(get_shared_session_path())


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
    existing_keys = {_message_key(message) for message in load_session(target)}
    copied = 0
    for message in source_messages:
        key = _message_key(message)
        if key in existing_keys:
            continue
        save_message(
            target,
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


def get_shared_session_data() -> list[dict[str, Any]]:
    """Get all shared-session messages as dictionaries."""
    return [
        message.to_dict(agent_id=message.agent_id or "shared") for message in get_shared_messages()
    ]


def truncate_session(agent_id: str, keep_last: int | None = None) -> dict[str, Any]:
    """Truncate a session file, keeping only the last N messages."""
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
