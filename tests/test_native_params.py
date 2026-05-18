#!/usr/bin/env python3
"""Unit tests for the ``ccl run --native-params`` passthrough flag.

Covers the argv pre-processor, threading through ``run_session`` to both the
one-shot and interactive paths, per-harness insertion position (especially
Codex's post-``exec`` rule), the ``--`` separator convention, and verbatim
handling of arguments containing spaces and quotes.
"""

from __future__ import annotations

from pathlib import Path

from claude_codex_local import wizard

# ---------------------------------------------------------------------------
# _extract_native_params: argv slicing
# ---------------------------------------------------------------------------


def test_extract_returns_none_when_flag_absent():
    cleaned, native = wizard._extract_native_params(["run", "-p", "hi"])
    assert cleaned == ["run", "-p", "hi"]
    assert native is None


def test_extract_captures_tail_after_flag():
    cleaned, native = wizard._extract_native_params(
        ["run", "-p", "hi", "--native-params", "--foo", "bar"]
    )
    assert cleaned == ["run", "-p", "hi"]
    assert native == ["--foo", "bar"]


def test_extract_strips_leading_double_dash_separator():
    """The recommended invocation uses `--` to disambiguate ccl flags from
    harness-native ones; the separator must not be forwarded to the harness."""
    cleaned, native = wizard._extract_native_params(
        ["run", "--native-params", "--", "--dangerously-skip-permissions"]
    )
    assert cleaned == ["run"]
    assert native == ["--dangerously-skip-permissions"]


def test_extract_with_empty_tail_returns_empty_list():
    """An empty list (vs None) means the flag was present without any args.
    Acceptance criterion #5: empty must still match the omitted-flag behavior
    when passed to `_build_oneshot_cmd`."""
    cleaned, native = wizard._extract_native_params(["run", "--native-params"])
    assert cleaned == ["run"]
    assert native == []


def test_extract_preserves_args_with_spaces_verbatim():
    cleaned, native = wizard._extract_native_params(
        ["run", "--native-params", "--system", 'You are "helpful" & terse']
    )
    assert cleaned == ["run"]
    assert native == ["--system", 'You are "helpful" & terse']


def test_extract_treats_first_occurrence_as_boundary():
    """Only the first ``--native-params`` is the slice point; later
    occurrences pass through as ordinary harness args. Pins the documented
    behavior so future refactors can't silently regress it."""
    cleaned, native = wizard._extract_native_params(
        ["run", "--native-params", "--foo", "--native-params", "--bar"]
    )
    assert cleaned == ["run"]
    assert native == ["--foo", "--native-params", "--bar"]


# ---------------------------------------------------------------------------
# _build_oneshot_cmd: per-harness insertion position
# ---------------------------------------------------------------------------


def test_oneshot_claude_non_ollama_inserts_before_prompt_tail():
    wire = {"argv": ["claude"]}
    cmd = wizard._build_oneshot_cmd(
        "claude", "lmstudio", "tag", wire, "hello", native_params=["--foo", "--bar"]
    )
    assert cmd == ["claude", "--foo", "--bar", "-p", "hello"]


def test_oneshot_claude_ollama_inserts_after_double_dash_before_prompt():
    cmd = wizard._build_oneshot_cmd(
        "claude", "ollama", "modeltag", {}, "hello", native_params=["--foo"]
    )
    assert cmd == [
        "ollama",
        "launch",
        "claude",
        "--model",
        "modeltag",
        "--",
        "--foo",
        "-p",
        "hello",
        "--model",
        "modeltag",
    ]


def test_oneshot_codex_non_ollama_inserts_after_exec_and_model_before_prompt():
    """Codex's `exec` flags must land *after* `exec --skip-git-repo-check -m TAG`
    and *before* the prompt, or they become part of the prompt text."""
    cmd = wizard._build_oneshot_cmd(
        "codex", "lmstudio", "tag", {}, "hello", native_params=["--foo"]
    )
    assert cmd == [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-m",
        "tag",
        "--foo",
        "hello",
    ]


def test_oneshot_codex_ollama_inserts_after_local_provider_before_prompt():
    cmd = wizard._build_oneshot_cmd("codex", "ollama", "tag", {}, "hello", native_params=["--foo"])
    assert cmd == [
        "ollama",
        "launch",
        "codex",
        "--model",
        "tag",
        "--",
        "exec",
        "--skip-git-repo-check",
        "--oss",
        "--local-provider=ollama",
        "--foo",
        "hello",
    ]


def test_oneshot_pi_inserts_before_prompt_tail():
    wire = {"argv": ["pi", "--provider", "openrouter"]}
    cmd = wizard._build_oneshot_cmd(
        "pi", "openrouter", "tag", wire, "hello", native_params=["--foo"]
    )
    assert cmd == ["pi", "--provider", "openrouter", "--foo", "-p", "hello"]


def test_oneshot_omitting_native_params_preserves_existing_argv():
    """Acceptance criterion #5: omission must not change launch behavior."""
    wire = {"argv": ["claude"]}
    before = wizard._build_oneshot_cmd("claude", "lmstudio", "tag", wire, "hello")
    after_none = wizard._build_oneshot_cmd(
        "claude", "lmstudio", "tag", wire, "hello", native_params=None
    )
    after_empty = wizard._build_oneshot_cmd(
        "claude", "lmstudio", "tag", wire, "hello", native_params=[]
    )
    assert before == ["claude", "-p", "hello"]
    assert after_none == before
    assert after_empty == before


def test_oneshot_native_params_preserves_args_with_spaces_and_quotes():
    """Acceptance criterion #4: arguments are forwarded verbatim. The argv
    list never re-tokenizes, so a value with embedded spaces or quotes
    arrives at the harness exactly as the user wrote it."""
    weird = ["--system", 'You are "helpful" & terse']
    cmd = wizard._build_oneshot_cmd(
        "claude", "lmstudio", "tag", {"argv": ["claude"]}, "hi", native_params=weird
    )
    assert cmd == ["claude", "--system", 'You are "helpful" & terse', "-p", "hi"]


# ---------------------------------------------------------------------------
# run_session: threading into the one-shot and interactive paths
# ---------------------------------------------------------------------------


def _stub_wizard_state(
    monkeypatch, tmp_path: Path, harness: str = "claude", engine: str = "ollama"
):
    """Mirror tests/test_session_bridge.py: pretend wizard state is valid."""
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
    # Disable the session bridge so we don't need a real native-home setup.
    monkeypatch.setenv("CCL_SESSION_BRIDGE", "0")


def test_run_session_one_shot_threads_native_params_to_builder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    received: dict = {}

    def spy_build(harness, engine, tag, wire_result, prompt, native_params=None):
        received["native_params"] = native_params
        return ["echo", "stub"]

    monkeypatch.setattr(wizard, "_build_oneshot_cmd", spy_build)
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="hi", native_params=["--foo", "bar"])
    assert received["native_params"] == ["--foo", "bar"]


def test_run_session_one_shot_omitted_native_params_threads_none(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    received: dict = {}

    def spy_build(harness, engine, tag, wire_result, prompt, native_params=None):
        received["native_params"] = native_params
        return ["echo", "stub"]

    monkeypatch.setattr(wizard, "_build_oneshot_cmd", spy_build)
    monkeypatch.setattr(
        wizard.subprocess, "run", lambda *a, **kw: type("R", (), {"returncode": 0})()
    )
    wizard.run_session(prompt="hi")
    assert received["native_params"] is None


def test_run_session_interactive_forwards_native_params_to_helper(monkeypatch, tmp_path):
    """No prompt → interactive helper path. native_params must land as extra
    argv after the helper-script path so the helper's trailing ``$@``
    forwards them to the harness."""
    monkeypatch.chdir(tmp_path)
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    monkeypatch.setattr(wizard.pb, "STATE_DIR", tmp_path, raising=False)
    (tmp_path / "bin").mkdir(exist_ok=True)
    fallback_helper = (
        tmp_path / "bin" / wizard._helper_script_basename(wizard._fence_tag_for("claude", "ollama"))
    )
    fallback_helper.write_text("#!/usr/bin/env bash\nexit 0\n")
    fallback_helper.chmod(0o755)

    captured: dict = {}

    def fake_run(cmd, env=None):
        captured["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(wizard.subprocess, "run", fake_run)
    rc = wizard.run_session(prompt=None, native_params=["--foo", "bar"])
    assert rc == 0
    assert captured["cmd"][0] == str(fallback_helper)
    assert captured["cmd"][1:] == ["--foo", "bar"]


def test_run_session_interactive_no_native_params_invokes_helper_alone(monkeypatch, tmp_path):
    """Acceptance criterion #5 on the interactive path: omitting the flag
    must not append anything to the helper invocation."""
    monkeypatch.chdir(tmp_path)
    _stub_wizard_state(monkeypatch, tmp_path, harness="claude")
    monkeypatch.setattr(wizard.pb, "STATE_DIR", tmp_path, raising=False)
    (tmp_path / "bin").mkdir(exist_ok=True)
    fallback_helper = (
        tmp_path / "bin" / wizard._helper_script_basename(wizard._fence_tag_for("claude", "ollama"))
    )
    fallback_helper.write_text("#!/usr/bin/env bash\nexit 0\n")
    fallback_helper.chmod(0o755)

    captured: dict = {}

    def fake_run(cmd, env=None):
        captured["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(wizard.subprocess, "run", fake_run)
    wizard.run_session(prompt=None)
    assert captured["cmd"] == [str(fallback_helper)]


# ---------------------------------------------------------------------------
# main(): end-to-end argv handling through the pre-processor
# ---------------------------------------------------------------------------


def test_main_strips_leading_double_dash_and_threads_to_run_session(monkeypatch):
    received: dict = {}

    def fake_run_session(prompt=None, no_context=False, native_params=None):
        received["prompt"] = prompt
        received["native_params"] = native_params
        return 0

    monkeypatch.setattr(wizard, "run_session", fake_run_session)
    monkeypatch.setattr(
        wizard.sys,
        "argv",
        ["ccl", "run", "-p", "hi", "--native-params", "--", "--foo", "bar"],
    )
    wizard.main()
    assert received["prompt"] == "hi"
    assert received["native_params"] == ["--foo", "bar"]


def test_main_handles_native_params_without_double_dash_separator(monkeypatch):
    """Users may invoke without the separator: `--native-params --foo`."""
    received: dict = {}

    def fake_run_session(prompt=None, no_context=False, native_params=None):
        received["native_params"] = native_params
        return 0

    monkeypatch.setattr(wizard, "run_session", fake_run_session)
    monkeypatch.setattr(
        wizard.sys,
        "argv",
        ["ccl", "run", "-p", "hi", "--native-params", "--foo", "bar"],
    )
    wizard.main()
    assert received["native_params"] == ["--foo", "bar"]


def test_main_without_native_params_passes_none(monkeypatch):
    """Acceptance criterion #5 at the CLI seam: omitted flag → None."""
    received: dict = {}

    def fake_run_session(prompt=None, no_context=False, native_params=None):
        received["native_params"] = native_params
        return 0

    monkeypatch.setattr(wizard, "run_session", fake_run_session)
    monkeypatch.setattr(wizard.sys, "argv", ["ccl", "run", "-p", "hi"])
    wizard.main()
    assert received["native_params"] is None
