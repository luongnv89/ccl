#!/usr/bin/env python3
"""Unit tests for the cross-harness session bridge.

Covers the native-session adapters (claude/codex/pi), cwd-scoped file
resolution, idempotent import, the context-prefix renderer, and the
auto-capture/auto-inject wiring in ``run_session``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_codex_local import session as sess
from claude_codex_local import session_adapters as adapters
from claude_codex_local import wizard
from claude_codex_local.session import SessionMessage

# ---------------------------------------------------------------------------
# Fixtures: isolated state dir, isolated native home, fixture session files.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """CCL state dir + native-home overrides — every test starts on a clean slate."""
    state_dir = tmp_path / "ccl-state"
    native_home = tmp_path / "native-home"
    state_dir.mkdir()
    native_home.mkdir()
    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CCL_NATIVE_HOME_OVERRIDE", str(native_home))
    yield {
        "state": state_dir,
        "native_home": native_home,
        "cwd": "/repo/example",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _seed_claude_native(native_home: Path, cwd: str, rows: list[dict]) -> Path:
    encoded = "-" + cwd.lstrip("/").replace("/", "-")
    path = native_home / ".claude" / "projects" / encoded / "sess-1.jsonl"
    _write_jsonl(path, rows)
    return path


def _seed_codex_native(
    native_home: Path, cwd: str, rows: list[dict], date_dir: str = "2026/05/17"
) -> Path:
    path = native_home / ".codex" / "sessions" / date_dir / "rollout-2026-05-17T10-00-00-aaaa.jsonl"
    full_rows = [
        {
            "timestamp": "2026-05-17T10:00:00Z",
            "type": "session_meta",
            "payload": {"id": "abc-codex", "cwd": cwd},
        }
    ] + rows
    _write_jsonl(path, full_rows)
    return path


def _seed_pi_native(native_home: Path, cwd: str, rows: list[dict]) -> Path:
    encoded = "--" + cwd.lstrip("/").replace("/", "-") + "--"
    path = native_home / ".pi" / "agent" / "sessions" / encoded / "sess-pi.jsonl"
    full_rows = [{"type": "session", "version": 3, "id": "abc-pi", "cwd": cwd}] + rows
    _write_jsonl(path, full_rows)
    return path


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


def test_claude_adapter_keeps_user_assistant_drops_internals(isolated_state):
    path = _seed_claude_native(
        isolated_state["native_home"],
        isolated_state["cwd"],
        [
            {"type": "permission-mode", "permissionMode": "default"},
            {"type": "attachment", "attachment": {"x": 1}},
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi back"}],
                },
            },
            {"type": "queue-operation", "operation": "x"},
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "<system-reminder>boilerplate</system-reminder>",
                },
            },
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "<local-command-stdout>fake output</local-command-stdout>",
                },
            },
        ],
    )
    msgs = adapters.read_claude(path)
    assert [(m.role, m.content) for m in msgs] == [
        ("user", "hello"),
        ("assistant", "hi back"),
    ]


def test_codex_adapter_keeps_message_drops_tool_chatter(isolated_state):
    path = _seed_codex_native(
        isolated_state["native_home"],
        isolated_state["cwd"],
        [
            {
                "timestamp": "t1",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "what is 2+2"}],
                },
            },
            {
                "timestamp": "t2",
                "type": "response_item",
                "payload": {"type": "reasoning", "summary": []},
            },
            {
                "timestamp": "t3",
                "type": "response_item",
                "payload": {"type": "function_call", "name": "shell", "arguments": "{}"},
            },
            {
                "timestamp": "t4",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "four"}],
                },
            },
            {
                "timestamp": "t5",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<user_instructions>boilerplate</user_instructions>",
                        }
                    ],
                },
            },
        ],
    )
    msgs = adapters.read_codex(path)
    assert [(m.role, m.content) for m in msgs] == [
        ("user", "what is 2+2"),
        ("assistant", "four"),
    ]


def test_pi_adapter_keeps_message_drops_metadata(isolated_state):
    path = _seed_pi_native(
        isolated_state["native_home"],
        isolated_state["cwd"],
        [
            {"type": "model_change", "modelId": "gpt-5"},
            {"type": "thinking_level_change", "thinkingLevel": "high"},
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "ping"}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "pong"}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": '<skill name="foo">load</skill>'}],
                },
            },
        ],
    )
    msgs = adapters.read_pi(path)
    assert [(m.role, m.content) for m in msgs] == [("user", "ping"), ("assistant", "pong")]


def test_adapter_redacts_secrets_in_content(isolated_state):
    path = _seed_claude_native(
        isolated_state["native_home"],
        isolated_state["cwd"],
        [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "my key is sk-proj-" + "A" * 30,
                },
            },
        ],
    )
    msgs = adapters.read_claude(path)
    assert len(msgs) == 1
    assert "sk-proj-" not in msgs[0].content
    assert "[REDACTED]" in msgs[0].content


def test_read_native_session_dispatches_by_harness(isolated_state):
    cwd = isolated_state["cwd"]
    home = isolated_state["native_home"]
    cp = _seed_claude_native(
        home, cwd, [{"type": "user", "message": {"role": "user", "content": "c"}}]
    )
    xp = _seed_codex_native(
        home,
        cwd,
        [
            {
                "timestamp": "t1",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "x"}],
                },
            }
        ],
    )
    pp = _seed_pi_native(
        home, cwd, [{"type": "message", "message": {"role": "user", "content": "p"}}]
    )
    assert [m.content for m in adapters.read_native_session("claude", cp)] == ["c"]
    assert [m.content for m in adapters.read_native_session("codex", xp)] == ["x"]
    assert [m.content for m in adapters.read_native_session("pi", pp)] == ["p"]
    assert adapters.read_native_session("unknown", cp) == []


# ---------------------------------------------------------------------------
# find_latest_native_session
# ---------------------------------------------------------------------------


def test_find_latest_native_session_claude_cwd_match(isolated_state):
    cwd = isolated_state["cwd"]
    p = _seed_claude_native(
        isolated_state["native_home"],
        cwd,
        [{"type": "user", "message": {"role": "user", "content": "x"}}],
    )
    assert sess.find_latest_native_session("claude", cwd) == p
    # Different cwd → no match.
    assert sess.find_latest_native_session("claude", "/repo/other") is None


def test_find_latest_native_session_codex_matches_via_session_meta(isolated_state):
    cwd = isolated_state["cwd"]
    home = isolated_state["native_home"]
    # Older file for a different cwd, newer file for ours: must pick ours.
    _seed_codex_native(
        home,
        "/repo/other",
        [
            {
                "timestamp": "t",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "n"}],
                },
            }
        ],
        date_dir="2026/05/15",
    )
    expected = _seed_codex_native(
        home,
        cwd,
        [
            {
                "timestamp": "t",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "y"}],
                },
            }
        ],
        date_dir="2026/05/17",
    )
    assert sess.find_latest_native_session("codex", cwd) == expected


def test_find_latest_native_session_returns_none_when_missing(isolated_state):
    assert sess.find_latest_native_session("claude", "/no/such/cwd") is None
    assert sess.find_latest_native_session("codex", "/no/such/cwd") is None
    assert sess.find_latest_native_session("pi", "/no/such/cwd") is None
    assert sess.find_latest_native_session("unknown", "/anywhere") is None


# ---------------------------------------------------------------------------
# import_native_session
# ---------------------------------------------------------------------------


def test_import_native_session_idempotent(isolated_state):
    cwd = isolated_state["cwd"]
    _seed_claude_native(
        isolated_state["native_home"],
        cwd,
        [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "hi"},
            },
        ],
    )
    r1 = sess.import_native_session("claude", "claude", cwd)
    assert r1["imported"] == 2
    r2 = sess.import_native_session("claude", "claude", cwd)
    assert r2["imported"] == 0  # idempotent
    msgs = sess.load_session("claude")
    assert [m.role for m in msgs] == ["user", "assistant"]


def test_import_native_session_no_source_is_clean(isolated_state):
    result = sess.import_native_session("claude", "claude", "/nope")
    assert result["success"] is True
    assert result["imported"] == 0
    assert result["source_path"] is None


# ---------------------------------------------------------------------------
# build_context_prefix
# ---------------------------------------------------------------------------


def test_build_context_prefix_empty_when_no_messages(isolated_state):
    assert sess.build_context_prefix("nobody") == ""


def test_build_context_prefix_wraps_in_delimiters_and_tail_keeps(isolated_state):
    for i in range(5):
        sess.save_message(
            "claude",
            SessionMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}"),
        )
    prefix = sess.build_context_prefix("claude", char_budget=500)
    assert prefix.startswith("[prior context, agent=claude")
    assert prefix.rstrip().endswith("[end prior context]")
    # All 5 short messages fit comfortably under 500 chars.
    for i in range(5):
        assert f"m{i}" in prefix


def test_build_context_prefix_drops_oldest_when_over_budget(isolated_state):
    for i in range(5):
        sess.save_message(
            "claude",
            SessionMessage(role="user", content="X" * 100 + f"_{i}"),
        )
    prefix = sess.build_context_prefix("claude", char_budget=300)
    # Budget ~ 300 chars, each message ~ 105+ chars → keep only the tail.
    assert "_4" in prefix  # newest survives
    assert "_0" not in prefix  # oldest dropped


# ---------------------------------------------------------------------------
# run_session bridge wiring — stub the subprocess so no real harness is invoked.
# ---------------------------------------------------------------------------


def _stub_wizard_state(
    monkeypatch, tmp_path: Path, harness: str = "claude", engine: str = "ollama"
):
    """Pretend wizard state exists and is valid."""
    fake_state = tmp_path / "ccl-wizard-state.json"
    fake_state.write_text("{}")
    monkeypatch.setattr(wizard, "STATE_FILE", fake_state)

    class _State:
        primary_harness = harness
        primary_engine = engine
        engine_model_tag = "test-model"
        wire_result = {"argv": ["echo", "stub"]}
        helper_script_path = None

    monkeypatch.setattr(wizard.WizardState, "load", staticmethod(lambda: _State()))
    monkeypatch.setattr(wizard, "_resolve_wire_env", lambda wr: {})
    monkeypatch.setattr(wizard, "_build_oneshot_cmd", lambda *a, **kw: ["echo", "harness-stub"])


def test_run_session_one_shot_injects_other_agent_prefix(isolated_state, monkeypatch, tmp_path):
    """When codex has a native session for cwd and we run as claude, the
    prefix from the codex transcript must be prepended to the prompt."""
    cwd = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Codex has prior conversation for this cwd.
    _seed_codex_native(
        isolated_state["native_home"],
        cwd,
        [
            {
                "timestamp": "t1",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "earlier codex question"}],
                },
            },
            {
                "timestamp": "t2",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "earlier codex answer"}],
                },
            },
        ],
    )
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    captured: dict = {}

    def fake_run(cmd, env=None):
        captured["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(wizard.subprocess, "run", fake_run)
    rc = wizard.run_session(prompt="new question")
    assert rc == 0
    # The harness was invoked with our stubbed _build_oneshot_cmd return value.
    assert captured["cmd"] == ["echo", "harness-stub"]
    # The prompt we passed into _build_oneshot_cmd is captured via a side
    # channel: redo the test by spying on _build_oneshot_cmd's prompt arg.


def test_run_session_one_shot_prefix_threaded_to_oneshot_builder(
    isolated_state, monkeypatch, tmp_path
):
    """Spy on _build_oneshot_cmd to assert the prefix is actually in the prompt."""
    cwd = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    _seed_codex_native(
        isolated_state["native_home"],
        cwd,
        [
            {
                "timestamp": "t1",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "remember the X bug"}],
                },
            },
        ],
    )
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    received: dict = {}

    def spy_build(harness, engine, tag, wire_result, prompt):
        received["prompt"] = prompt
        return ["echo", "stub"]

    monkeypatch.setattr(wizard, "_build_oneshot_cmd", spy_build)
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="new question")
    assert received["prompt"].startswith("[prior context, agent=codex")
    assert "remember the X bug" in received["prompt"]
    assert received["prompt"].endswith("new question")


def test_run_session_no_context_flag_skips_bridge(isolated_state, monkeypatch, tmp_path):
    cwd = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    _seed_codex_native(
        isolated_state["native_home"],
        cwd,
        [
            {
                "timestamp": "t",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "should not appear"}],
                },
            }
        ],
    )
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    received: dict = {}

    def spy_build(*args):
        received["prompt"] = args[-1]
        return ["echo", "stub"]

    monkeypatch.setattr(wizard, "_build_oneshot_cmd", spy_build)
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="bare prompt", no_context=True)
    assert received["prompt"] == "bare prompt"
    # And no capture happened: claude.jsonl should not exist.
    assert not sess.get_session_path("claude").exists()


def test_run_session_env_opt_out_skips_bridge(isolated_state, monkeypatch, tmp_path):
    monkeypatch.setenv("CCL_SESSION_BRIDGE", "0")
    monkeypatch.chdir(tmp_path)
    _seed_claude_native(
        isolated_state["native_home"],
        str(tmp_path),
        [{"type": "user", "message": {"role": "user", "content": "x"}}],
    )
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="hi")
    # Capture suppressed.
    assert not sess.get_session_path("claude").exists()


def test_run_session_post_run_capture_imports_native(isolated_state, monkeypatch, tmp_path):
    cwd = str(tmp_path)
    monkeypatch.chdir(tmp_path)
    _seed_claude_native(
        isolated_state["native_home"],
        cwd,
        [
            {"type": "user", "message": {"role": "user", "content": "ran something"}},
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "did the thing"},
            },
        ],
    )
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="hi")
    # Capture pulled the native session into claude.jsonl.
    msgs = sess.load_session("claude")
    assert [(m.role, m.content) for m in msgs] == [
        ("user", "ran something"),
        ("assistant", "did the thing"),
    ]


def test_run_session_capture_failure_does_not_break_run(isolated_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )

    def boom(*a, **kw):
        raise RuntimeError("simulated capture failure")

    monkeypatch.setattr(sess, "import_native_session", boom)
    # Should swallow the exception and return the harness's rc.
    assert wizard.run_session(prompt="hi") == 0
