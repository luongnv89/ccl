#!/usr/bin/env python3
"""Per-harness readers that translate native session JSONL into SessionMessage.

Each harness persists its own session format under its home directory:

- Claude Code: ``~/.claude/projects/<cwd-encoded>/<uuid>.jsonl`` with one
  event per line. We keep ``type=user`` and ``type=assistant`` entries whose
  ``message.content`` resolves to plain text.
- Codex: ``~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-...jsonl`` with a
  ``{timestamp,type,payload}`` envelope. We keep ``type=response_item`` where
  ``payload.type=message`` and the content is ``input_text`` (user) or
  ``output_text`` (assistant).
- Pi: ``~/.pi/agent/sessions/<cwd-encoded>/<timestamp_uuid>.jsonl`` with a
  typed event stream. We keep ``type=message`` and pull ``message.role`` /
  ``message.content`` (string or list-of-blocks).

The readers drop harness-internal events (tool calls, attachments, reasoning
traces, environment context dumps) and the AGENTS.md / CLAUDE.md re-dump that
every harness injects as the first synthetic user message — those are
boilerplate, not real conversation. Every surviving message goes through
:func:`session.redact_secrets` before being returned.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claude_codex_local.session import SessionMessage, redact_secrets

_BOILERPLATE_PREFIXES = (
    "# AGENTS.md instructions",
    "# CLAUDE.md instructions",
    "<INSTRUCTIONS>",
    "<environment_context>",
    "<system-reminder>",
    "<user_instructions>",
    "<local-command-caveat>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<skill name=",
)


def _is_boilerplate(text: str) -> bool:
    """Detect harness-injected synthetic user messages (AGENTS.md / CLAUDE.md dumps, slash-command echo, skill loads)."""
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES)


def _flatten_content(content: Any) -> str:
    """Reduce a message-content value (string or list-of-blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                # Anthropic-style: {type, text} | Codex: {type:input_text, text} | etc.
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _iter_jsonl(path: Path):
    """Yield decoded JSON objects from a JSONL file, skipping malformed lines."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _build_message(
    role: str, text: str, timestamp: str, session_id: str, fallback_ts: str
) -> SessionMessage | None:
    """Wrap a (role, text) pair in SessionMessage after filtering boilerplate.

    ``fallback_ts`` is used when the event has no native timestamp — the
    bridge's dedup key includes the timestamp, so we need a deterministic
    value across re-imports of the same file (file mtime works for that).
    """
    if role not in {"user", "assistant"}:
        return None
    if not text.strip():
        return None
    if _is_boilerplate(text):
        return None
    return SessionMessage(
        role=role,
        content=redact_secrets(text),
        timestamp=timestamp or fallback_ts,
        session_id=session_id or "",
    )


def _file_fallback_ts(path: Path) -> str:
    """Return ``path``'s mtime as an ISO timestamp, or empty on failure."""
    try:
        ts = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def read_claude(path: Path) -> list[SessionMessage]:
    """Read a Claude Code session JSONL into normalized SessionMessages.

    Claude Code writes one event per line. User / assistant events have a
    ``message`` field carrying the Anthropic message envelope; everything
    else (``attachment``, ``queue-operation``, ``last-prompt``,
    ``permission-mode``) is harness internal and skipped.
    """
    messages: list[SessionMessage] = []
    session_id = path.stem  # filename is the session uuid
    fallback = _file_fallback_ts(path)
    for event in _iter_jsonl(path):
        event_type = event.get("type")
        if event_type not in {"user", "assistant"}:
            continue
        envelope = event.get("message") or {}
        role = envelope.get("role") or event_type
        text = _flatten_content(envelope.get("content"))
        msg = _build_message(role, text, event.get("timestamp", ""), session_id, fallback)
        if msg is not None:
            messages.append(msg)
    return messages


def read_codex(path: Path) -> list[SessionMessage]:
    """Read a Codex rollout JSONL into normalized SessionMessages.

    Codex wraps each event as ``{timestamp, type, payload}``. We keep
    ``type=response_item`` with ``payload.type=message`` (input_text for
    user, output_text for assistant). ``reasoning``, ``function_call``,
    ``function_call_output``, ``custom_tool_call(_output)``, and
    ``ghost_snapshot`` are all tool-channel chatter and dropped.
    """
    messages: list[SessionMessage] = []
    session_id = ""
    fallback = _file_fallback_ts(path)
    for event in _iter_jsonl(path):
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if event_type == "session_meta":
            session_id = str(payload.get("id", ""))
            continue
        if event_type != "response_item":
            continue
        if payload.get("type") != "message":
            continue
        role = payload.get("role", "")
        text = _flatten_content(payload.get("content"))
        msg = _build_message(role, text, event.get("timestamp", ""), session_id, fallback)
        if msg is not None:
            messages.append(msg)
    return messages


def read_pi(path: Path) -> list[SessionMessage]:
    """Read a Pi session JSONL into normalized SessionMessages.

    Pi writes a typed event stream. We keep ``type=message`` and pull
    ``message.role`` / ``message.content``. The leading ``session`` /
    ``model_change`` / ``thinking_level_change`` events are skipped.
    """
    messages: list[SessionMessage] = []
    session_id = ""
    fallback = _file_fallback_ts(path)
    for event in _iter_jsonl(path):
        event_type = event.get("type")
        if event_type == "session":
            session_id = str(event.get("id", ""))
            continue
        if event_type != "message":
            continue
        envelope = event.get("message") or {}
        role = envelope.get("role", "")
        text = _flatten_content(envelope.get("content"))
        msg = _build_message(role, text, event.get("timestamp", ""), session_id, fallback)
        if msg is not None:
            messages.append(msg)
    return messages


_READERS: dict[str, Callable[[Path], list[SessionMessage]]] = {
    "claude": read_claude,
    "codex": read_codex,
    "pi": read_pi,
}


def read_native_session(harness: str, path: Path) -> list[SessionMessage]:
    """Dispatch to the per-harness reader. Returns [] for unknown harnesses."""
    reader = _READERS.get(harness)
    if reader is None:
        return []
    return reader(path)
