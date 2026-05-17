#!/usr/bin/env python3
"""Unit tests for the session module."""

import json
import os
from datetime import datetime, timezone

import pytest

from claude_codex_local import session as sess
from claude_codex_local.session import SessionMessage


@pytest.fixture
def temp_state_dir(tmp_path):
    """Create a temporary state directory for testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # Mock STATE_DIR by setting environment variable
    original_state_dir = os.environ.get("CLAUDE_CODEX_LOCAL_STATE_DIR", "")
    os.environ["CLAUDE_CODEX_LOCAL_STATE_DIR"] = str(state_dir)
    yield state_dir
    if original_state_dir:
        os.environ["CLAUDE_CODEX_LOCAL_STATE_DIR"] = original_state_dir


@pytest.fixture
def temp_session_dir(temp_state_dir):
    """Create a temporary sessions directory."""
    session_dir = temp_state_dir / "sessions"
    session_dir.mkdir()
    yield session_dir


def test_ensure_sessions_dir(temp_state_dir):
    """Test that sessions directory is created."""
    # Should not raise
    sess._ensure_sessions_dir()
    assert (temp_state_dir / "sessions").exists()


def test_get_session_path(temp_state_dir, temp_session_dir):
    """Test getting the session path for an agent."""
    agent_id = "test-agent"
    path = sess.get_session_path(agent_id)
    expected = temp_session_dir / f"{agent_id}.jsonl"
    assert path == expected


def test_get_shared_session_path(temp_state_dir, temp_session_dir):
    """Test getting the shared session path."""
    path = sess.get_shared_session_path()
    expected = temp_session_dir / "shared.jsonl"
    assert path == expected


def test_load_empty_session(temp_state_dir, temp_session_dir):
    """Test loading an empty session."""
    agent_id = "test-agent"
    messages = sess.load_session(agent_id)
    assert messages == []


def test_load_session_with_messages(temp_state_dir, temp_session_dir):
    """Test loading a session with messages."""
    agent_id = "test-agent"
    path = sess.get_session_path(agent_id)

    # Write some test messages
    messages_data = [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T00:00:00",
            "session_id": "abc123",
            "agent_id": "test-agent",
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "timestamp": "2024-01-01T00:00:01",
            "session_id": "abc123",
            "agent_id": "test-agent",
        },
    ]

    with path.open("w") as f:
        for msg in messages_data:
            f.write(json.dumps(msg) + "\n")

    messages = sess.load_session(agent_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"


def test_save_message(temp_state_dir, temp_session_dir):
    """Test saving a message."""
    from claude_codex_local.session import SessionMessage

    agent_id = "test-agent"

    # Save a message
    message = SessionMessage(
        role="user",
        content="Test message",
        timestamp=datetime.now(timezone.utc),
        session_id="test123",
        agent_id=agent_id,
    )
    sess.save_message(agent_id, message)

    # Verify it was saved
    messages = sess.load_session(agent_id)
    assert len(messages) == 1
    assert messages[0].content == "Test message"


def test_save_message_auto_agent_id(temp_state_dir, temp_session_dir):
    """Test that save_message auto-detects agent_id from environment."""
    # Set the agent_id environment variable
    os.environ["__CCL_AGENT_ID"] = "auto-detect-agent"

    # Save a message without specifying agent_id
    message = SessionMessage(
        role="user",
        content="Test auto-detect",
        timestamp=datetime.now(timezone.utc),
        session_id="auto123",
    )
    sess.save_message(agent_id="auto-detect-agent", message=message)  # No agent_id specified

    # Verify it was saved with auto-detect-agent
    messages = sess.load_session("auto-detect-agent")
    assert len(messages) == 1
    assert messages[0].content == "Test auto-detect"
    # Clean up
    del os.environ["__CCL_AGENT_ID"]


def test_save_message_to_shared(temp_state_dir, temp_session_dir):
    """Test saving a message to the shared session."""
    from claude_codex_local.session import SessionMessage

    message = SessionMessage(
        role="user",
        content="Shared message",
        timestamp=datetime.now(timezone.utc),
        session_id="shared123",
        agent_id="test-agent",
    )
    sess.save_message_to_shared("test-agent", message)

    # Verify it was saved to shared
    shared_messages = sess.get_shared_messages()
    assert len(shared_messages) == 1
    assert shared_messages[0].content == "Shared message"


def test_save_message_redacts_secrets(temp_state_dir, temp_session_dir):
    """Redaction must scrub secret patterns before they hit the JSONL file."""
    agent_id = "redact-agent"
    secrets = {
        "openai": "sk-1234567890abcdef1234567890abcdef",
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "github": "ghp_1234567890abcdef1234567890abcdef1234",
    }
    message = SessionMessage(
        role="user",
        content=(f"openai={secrets['openai']} aws={secrets['aws']} github={secrets['github']}"),
        timestamp=datetime.now(timezone.utc),
        session_id="redact1",
        agent_id=agent_id,
    )
    sess.save_message(agent_id, message)

    raw = sess.get_session_path(agent_id).read_text(encoding="utf-8")
    for label, secret in secrets.items():
        assert secret not in raw, f"{label} secret leaked into persisted JSONL"
    assert "[REDACTED]" in raw


def test_sync_session_redacts_secrets(temp_state_dir, temp_session_dir):
    """Synced rows must be redacted and keep the source agent_id."""
    source = "claude"
    target = "codex"
    secret = "sk-1234567890abcdef1234567890abcdef"
    message = SessionMessage(
        role="user",
        content=f"token={secret}",
        timestamp=datetime.now(timezone.utc),
        session_id="sync-redact",
        agent_id=source,
    )
    sess.save_message(source, message)

    result = sess.sync_session(target, source)
    assert result["success"] is True
    assert result["message_count"] == 1

    target_path = sess.get_session_path(target)
    raw = target_path.read_text(encoding="utf-8")
    assert secret not in raw, "secret leaked into target JSONL during sync"
    assert "[REDACTED]" in raw

    with target_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert rows, "expected at least one row in target after sync"
    assert rows[0]["agent_id"] == source, "synced row must preserve source agent_id"


def test_get_session_summary(temp_state_dir, temp_session_dir):
    """Test getting a session summary."""
    agent_id = "test-agent"
    summary = sess.get_session_summary(agent_id)
    assert summary["agent_id"] == agent_id
    # Empty session
    assert summary["message_count"] == 0


def test_sync_session(temp_state_dir, temp_session_dir):
    """Test syncing session between agents."""

    # First, write messages to claude's session
    claude_path = temp_session_dir / "claude.jsonl"
    with claude_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "role": "user",
                    "content": "Hello from Claude",
                    "timestamp": "2024-01-01T00:00:00",
                    "session_id": "claude123",
                    "agent_id": "claude",
                }
            )
            + "\n"
        )

    # Sync from claude to codex
    result = sess.sync_session("codex", "claude")
    assert result["success"] is True
    assert result["message_count"] == 1

    # Verify codex now has the messages
    codex_path = temp_session_dir / "codex.jsonl"
    assert codex_path.exists()
    messages = sess.load_session("codex")
    assert len(messages) == 1
    assert messages[0].content == "Hello from Claude"
    # Provenance: synced rows must keep the source agent_id, not the target's.
    assert messages[0].agent_id == "claude"
    with codex_path.open("r", encoding="utf-8") as handle:
        raw_rows = [json.loads(line) for line in handle if line.strip()]
    assert raw_rows[0]["agent_id"] == "claude"


def test_clear_session(temp_state_dir, temp_session_dir):
    """Test clearing a session."""
    # Write some messages first
    path = temp_session_dir / "test-agent.jsonl"
    with path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "role": "user",
                    "content": "test",
                    "timestamp": "2024-01-01T00:00:00",
                    "session_id": "1",
                    "agent_id": "test-agent",
                }
            )
            + "\n"
        )

    # Clear the session
    result = sess.clear_session("test-agent")
    assert result["success"] is True

    # Verify it's empty
    messages = sess.load_session("test-agent")
    assert messages == []


def test_truncate_session(temp_state_dir, temp_session_dir):
    """Test truncating a session."""
    # Write 5 messages
    path = temp_session_dir / "test-agent.jsonl"
    for i in range(5):
        with path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "role": "user",
                        "content": f"message {i}",
                        "timestamp": "2024-01-01T00:00:00",
                        "session_id": str(i),
                        "agent_id": "test-agent",
                    }
                )
                + "\n"
            )

    # Truncate to keep last 2
    result = sess.truncate_session("test-agent", keep_last=2)
    assert result["success"] is True
    assert result["kept"] == 2

    # Verify only last 2 remain
    messages = sess.load_session("test-agent")
    assert len(messages) == 2
    assert messages[0].content == "message 3"
    assert messages[1].content == "message 4"


def test_get_all_sessions(temp_state_dir, temp_session_dir):
    """Test getting all sessions."""
    # Write messages to different agents
    for agent in ["claude", "codex", "pi"]:
        path = temp_session_dir / f"{agent}.jsonl"
        with path.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "role": "user",
                        "content": "test",
                        "timestamp": "2024-01-01T00:00:00",
                        "session_id": "1",
                        "agent_id": agent,
                    }
                )
                + "\n"
            )

    sessions = sess.get_all_sessions()
    assert len(sessions) >= 3


def test_get_shared_session_data(temp_state_dir, temp_session_dir):
    """Test getting all shared session data."""
    # Write to shared
    shared_path = temp_session_dir / "shared.jsonl"
    with shared_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "role": "user",
                    "content": "shared test",
                    "timestamp": "2024-01-01T00:00:00",
                    "session_id": "shared1",
                    "agent_id": "test-agent",
                }
            )
            + "\n"
        )

    data = sess.get_shared_session_data()
    assert len(data) == 1
    assert data[0]["content"] == "shared test"
