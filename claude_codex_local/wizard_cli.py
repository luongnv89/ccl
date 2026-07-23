"""
Wizard CLI — orchestration, doctor, status, serve, session, run, parser, and main.

Exports:
    STEPS — ordered list of (step_id, title, step_fn) tuples
    run_wizard — main wizard orchestration
    run_doctor — read-only triage command
    run_status — status report command
    run_serve — serve a model command
    run_session — run a session command
    run_find_model_standalone — find-model standalone command
    _build_parser — argparse parser builder
    _extract_native_params — native params extraction
    main — entry point
"""

from __future__ import annotations

from collections.abc import Callable

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path
from typing import Any

from claude_codex_local import __version__
from claude_codex_local import core as pb
from claude_codex_local.wizard_discovery import (
    step_2_1_discover,
    step_2_2_install_missing,
)
from claude_codex_local.wizard_state import GUIDE_PATH, STATE_DIR, STATE_FILE, WizardState
from claude_codex_local.wizard_steps import (
    _alias_names_for,
    _ensure_llamacpp_server_running,
    _fence_tag_for,
    _helper_script_basename,
    _model_already_installed,
    _OPENROUTER_MODEL_RE,
    _ROUTER9_MODEL_RE,
    step_2_4_pick_model,
    step_2_5_5_benchmark,
    step_2_5_smoke_test,
    step_2_6_wire_harness,
    step_2_65_install_aliases,
    step_2_7_verify,
    step_2_8_generate_guide,
    step_2_select_harness,
    step_3_select_engine,
)
from claude_codex_local.wizard_ui import Panel, console, fail, header, info, ok, warn
from rich.table import Table

STEPS: list[tuple[str, str, Callable]] = [
    ("1", "Discover environment", step_2_1_discover),
    ("2", "Select harness", step_2_select_harness),
    ("3", "Select engine", step_3_select_engine),
    ("4", "Pick a model", step_2_4_pick_model),
    ("5", "Smoke test engine + model", step_2_5_smoke_test),
    ("5.5", "Optional benchmark", step_2_5_5_benchmark),
    ("6", "Wire up harness", step_2_6_wire_harness),
    ("7", "Install helper script + shell aliases", step_2_65_install_aliases),
    ("8", "Verify launch command", step_2_7_verify),
    ("9", "Generate guide.md", step_2_8_generate_guide),
]


def run_wizard(
    *,
    resume: bool = False,
    non_interactive: bool = False,
    start_step: str | None = None,
    force_harness: str | None = None,
    force_engine: str | None = None,
    force_scan: bool = False,
    run_llmfit_flag: bool = False,
) -> int:
    state = WizardState.load() if resume else WizardState()
    if not resume and not non_interactive and sys.stdout.isatty():
        print_welcome_banner()
    if resume and state.completed_steps:
        info(f"Resuming. Already completed: {', '.join(state.completed_steps)}")
    if force_harness:
        state.primary_harness = force_harness
    if force_engine:
        state.primary_engine = force_engine

    # Resolve STEPS via the wizard module so that test monkeypatches on
    # ``wizard.STEPS`` take effect (the original monolithic wizard.py had
    # everything in one namespace).
    import claude_codex_local.wizard as _wiz_mod

    for step_id, title, fn in _wiz_mod.STEPS:
        if resume and step_id in state.completed_steps and step_id != start_step:
            continue
        if step_id == "1":
            ok_step = fn(  # type: ignore[call-arg]
                state,
                non_interactive,
                force_scan=force_scan,
                run_llmfit_flag=run_llmfit_flag,
            )
        elif step_id == "4":
            ok_step = fn(  # type: ignore[call-arg]
                state, non_interactive, run_llmfit_flag=run_llmfit_flag
            )
        else:
            ok_step = fn(state, non_interactive)
        if not ok_step:
            fail(f"Step {step_id} ({title}) did not complete. Re-run with --resume to continue.")
            return 1

    if state.alias_names:
        alias_short = state.alias_names[0]
    elif state.primary_harness:
        alias_short = _alias_names_for(_fence_tag_for(state.primary_harness, state.primary_engine))[
            0
        ]
    else:
        alias_short = "cc"
    if state.shell_rc_path:
        completion_body = (
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Reload your shell so the new alias is picked up:\n"
            f"  [cyan]source {state.shell_rc_path}[/cyan]  (or open a new terminal)\n\n"
            f"Then run: [cyan]{alias_short}[/cyan]\n\n"
            f"See [bold]{GUIDE_PATH}[/bold] for the full guide."
        )
    else:
        helper = state.helper_script_path or "(helper script)"
        completion_body = (
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Alias install was skipped, so no shell reload is required.\n\n"
            f"Run the helper directly:\n"
            f"  [cyan]{helper}[/cyan]\n\n"
            f"See [bold]{GUIDE_PATH}[/bold] for the full guide."
        )
    console.print()
    console.print(
        Panel.fit(
            completion_body,
            border_style="green",
        )
    )
    return 0


def run_doctor() -> int:
    """
    Read-only triage command. Prints the current wizard state and re-checks
    presence of the tools/models the wizard selected. Exit 0 when healthy,
    1 when regressions are detected.
    """
    header("doctor — wizard state + presence re-check")

    if not STATE_FILE.exists():
        warn(f"No wizard state found at {STATE_FILE}. Run `ccl setup` first.")
        return 1

    state = WizardState.load()

    # --- Stored wizard state ---
    state_table = Table(title="Stored wizard state", show_header=False, box=None)
    state_table.add_column("key", style="bold")
    state_table.add_column("value")
    state_table.add_row("state file", str(STATE_FILE))
    state_table.add_row("completed steps", ", ".join(state.completed_steps) or "(none)")
    state_table.add_row("harness", state.primary_harness or "(unset)")
    state_table.add_row("engine", state.primary_engine or "(unset)")
    state_table.add_row("model (raw)", state.model_name or "(unset)")
    state_table.add_row("engine tag", state.engine_model_tag or "(unset)")
    state_table.add_row("model source", state.model_source or "(unset)")
    state_table.add_row(
        "launch command",
        " ".join(shlex.quote(x) for x in state.launch_command)
        if state.launch_command
        else "(unset)",
    )
    last_verify = state.verify_result.get("ok")
    state_table.add_row(
        "last verify",
        "[green]ok[/green]"
        if last_verify
        else ("[red]failed[/red]" if state.verify_result else "(never run)"),
    )
    console.print(state_table)
    console.print()

    # --- Live presence re-check ---
    info("Re-running machine presence check...")
    profile = pb.machine_profile()
    presence = profile.get("presence", {})

    issues: list[str] = []

    check_table = Table(title="Presence re-check", show_header=True)
    check_table.add_column("component")
    check_table.add_column("expected")
    check_table.add_column("status")

    def add_row(name: str, expected: str, ok_flag: bool, detail: str = "") -> None:
        mark = "[green]✓[/green]" if ok_flag else "[red]✗[/red]"
        check_table.add_row(name, expected, f"{mark} {detail}".strip())
        if not ok_flag:
            issues.append(f"{name}: {detail or 'missing'}")

    # Harness
    harnesses = presence.get("harnesses", []) or []
    if state.primary_harness:
        add_row(
            "harness",
            state.primary_harness,
            state.primary_harness in harnesses,
            "found"
            if state.primary_harness in harnesses
            else f"not in PATH (have: {harnesses or 'none'})",
        )

    # Engine
    engines = presence.get("engines", []) or []
    if state.primary_engine:
        add_row(
            "engine",
            state.primary_engine,
            state.primary_engine in engines,
            "found"
            if state.primary_engine in engines
            else f"not installed (have: {engines or 'none'})",
        )

    # Model presence on the engine
    if state.engine_model_tag and state.primary_engine:
        installed = _model_already_installed(state.primary_engine, state.engine_model_tag, profile)
        add_row(
            f"{state.primary_engine} model",
            state.engine_model_tag,
            installed,
            "installed" if installed else "missing — re-run wizard to re-create/pull",
        )

    # Helper script (cc / cx / cc9 / cx9 / cco / cxo / cpo)
    if state.helper_script_path:
        script_path = Path(state.helper_script_path)
        add_row(
            "helper script",
            state.helper_script_path,
            script_path.exists(),
            "present" if script_path.exists() else "missing — re-run step 6.5",
        )

    # 9router-specific checks: key-file mode, key non-empty, model name regex.
    if state.primary_engine == "9router":
        key_file = pb.ROUTER9_KEY_FILE
        if key_file.exists():
            try:
                mode = key_file.stat().st_mode & 0o777
            except OSError:
                mode = -1
            mode_ok = mode != -1 and (mode & 0o077) == 0
            add_row(
                "9router key file mode",
                "owner-only (0600)",
                mode_ok,
                f"{mode:04o}" if mode != -1 else "stat failed",
            )
            content = ""
            with contextlib.suppress(OSError):
                content = key_file.read_text().strip()
            add_row(
                "9router key file content",
                "non-empty",
                bool(content),
                "ok" if content else "empty — re-run step 4 to set the key",
            )
        else:
            add_row(
                "9router key file",
                str(key_file),
                False,
                "missing — re-run step 4 to set the key",
            )
        if state.engine_model_tag:
            valid_model = len(state.engine_model_tag) <= 256 and bool(
                _ROUTER9_MODEL_RE.match(state.engine_model_tag)
            )
            add_row(
                "9router model name",
                "<provider>/<model-id>",
                valid_model,
                state.engine_model_tag if valid_model else "invalid — re-run step 4",
            )

    # OpenRouter-specific checks (mirrors the 9router shape — same key-file
    # boundary, same provider/model-id regex).
    if state.primary_engine == "openrouter":
        key_file = pb.OPENROUTER_KEY_FILE
        if key_file.exists():
            try:
                mode = key_file.stat().st_mode & 0o777
            except OSError:
                mode = -1
            mode_ok = mode != -1 and (mode & 0o077) == 0
            add_row(
                "openrouter key file mode",
                "owner-only (0600)",
                mode_ok,
                f"{mode:04o}" if mode != -1 else "stat failed",
            )
            content = ""
            with contextlib.suppress(OSError):
                content = key_file.read_text().strip()
            add_row(
                "openrouter key file content",
                "non-empty",
                bool(content),
                "ok" if content else "empty — re-run step 4 to set the key",
            )
        else:
            add_row(
                "openrouter key file",
                str(key_file),
                False,
                "missing — re-run step 4 to set the key",
            )
        if state.engine_model_tag:
            valid_model = len(state.engine_model_tag) <= 256 and bool(
                _OPENROUTER_MODEL_RE.match(state.engine_model_tag)
            )
            add_row(
                "openrouter model name",
                "<provider>/<model-id>[:variant]",
                valid_model,
                state.engine_model_tag if valid_model else "invalid — re-run step 4",
            )

    # guide.md
    add_row(
        "guide.md",
        str(GUIDE_PATH),
        GUIDE_PATH.exists(),
        "present" if GUIDE_PATH.exists() else "missing — re-run step 8",
    )

    console.print(check_table)
    console.print()

    if issues:
        fail(f"{len(issues)} issue(s) detected:")
        for i in issues:
            console.print(f"  [red]•[/red] {i}")
        console.print()
        info("Suggested fix: `ccl setup --resume`")
        return 1

    ok("All checks passed.")
    return 0


def _build_oneshot_cmd(
    harness: str,
    engine: str,
    tag: str,
    wire_result: dict[str, Any],
    prompt: str,
    native_params: list[str] | None = None,
) -> list[str] | None:
    """
    Build the harness/engine-specific argv for a one-shot session driven by
    `ccl run -p PROMPT`.

    Mirrors the verify step's per-backend argv shape (see step_2_7_verify) so
    automation drivers get the same dispatch path the wizard already exercises.
    The Codex+Ollama branch sidesteps the documented top-level-flag limitation
    by placing `--oss --local-provider=ollama` AFTER the `exec` subcommand.

    `native_params`, when provided, is inserted verbatim immediately before
    the prompt-bearing tail so the harness (not ccl, not the prompt) consumes
    the flags. See `ccl run --native-params` for the user-facing surface.
    """
    extra = list(native_params or [])
    if harness == "claude":
        if engine == "ollama":
            return [
                "ollama",
                "launch",
                "claude",
                "--model",
                tag,
                "--",
                *extra,
                "-p",
                prompt,
                "--model",
                tag,
            ]
        return list(wire_result.get("argv", [])) + extra + ["-p", prompt]
    if harness == "codex":
        if engine == "ollama":
            return [
                "ollama",
                "launch",
                "codex",
                "--model",
                tag,
                "--",
                "exec",
                "--skip-git-repo-check",
                "--oss",
                "--local-provider=ollama",
                *extra,
                prompt,
            ]
        return [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-m",
            tag,
            *extra,
            prompt,
        ]
    if harness == "pi":
        return list(wire_result.get("argv", [])) + extra + ["-p", prompt]
    return None


def _resolve_wire_env(wire_result: dict[str, Any]) -> dict[str, str]:
    """
    Materialize the env dict the harness should run under.

    `wire_result.env` values are literal strings. `wire_result.raw_env` values
    are bash expressions evaluated at exec-time by the helper script (e.g.
    `"$(cat /path/to/key)"` for the 9router/openrouter/vllm key files — see WireResult).
    `ccl run` bypasses the helper script, so we evaluate raw_env in a one-shot
    bash subshell to keep the secret-on-disk boundary intact.
    """
    env: dict[str, str] = dict(wire_result.get("env", {}))
    raw_env: dict[str, str] = dict(wire_result.get("raw_env", {}))
    if not raw_env:
        return env
    script_lines = [f"export {shlex.quote(k)}={v}" for k, v in raw_env.items()]
    keys_alt = "|".join(re.escape(k) for k in raw_env)
    script_lines.append(f"env | grep -E '^({keys_alt})='")
    try:
        proc = subprocess.run(
            ["bash", "-c", "\n".join(script_lines)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return env
    if proc.returncode != 0:
        return env
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k in raw_env:
            env[k] = v
    return env


_BRIDGE_HARNESSES = ("claude", "codex", "pi")


def _bridge_disabled(no_context: bool) -> bool:
    """Resolve the bridge opt-out: --no-context flag or CCL_SESSION_BRIDGE=0."""
    if no_context:
        return True
    return os.environ.get("CCL_SESSION_BRIDGE", "").strip() == "0"


def _pick_source_agent(harness: str, cwd: str) -> str | None:
    """Pick the most recently active *other* harness with native session content for ``cwd``.

    Returns the harness name (== agent_id) of the freshest match, or ``None``
    when no other harness has a session for this cwd. Self is always
    excluded so we don't replay our own transcript to ourselves.
    """
    from claude_codex_local.session import find_latest_native_session

    best: tuple[float, str] | None = None
    for candidate in _BRIDGE_HARNESSES:
        if candidate == harness:
            continue
        path = find_latest_native_session(candidate, cwd)
        if path is None:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, candidate)
    return best[1] if best else None


def _bridge_pre_run_prefix(harness: str, cwd: str) -> tuple[str, str | None, float | None]:
    """Build the cross-agent context prefix for the consumer side.

    Returns ``(prefix_text, source_agent, age_seconds)``. ``prefix_text`` is
    empty when there's nothing to inject. ``source_agent`` is the harness
    name we pulled from, useful for the info banner. ``age_seconds`` is the
    wall-clock age of the source's native file (useful for surfacing in the
    info banner so the user can spot a stale handoff). Imports the other
    agent's latest native session first so the prefix reflects the freshest
    state.
    """
    import time

    from claude_codex_local.session import (
        build_context_prefix,
        find_latest_native_session,
        import_native_session,
    )

    source = _pick_source_agent(harness, cwd)
    if source is None:
        return "", None, None
    source_path = find_latest_native_session(source, cwd)
    age = None
    if source_path is not None:
        try:
            age = time.time() - source_path.stat().st_mtime
        except OSError:
            age = None
    # Refresh the source agent's CCL JSONL from its native store before
    # rendering, so the prefix carries the latest turn even if `ccl session
    # sync` hasn't been run since the producer harness exited.
    import_native_session(source, source, cwd)
    prefix = build_context_prefix(source)
    return prefix, source, age


def _format_age(seconds: float) -> str:
    """Render a wall-clock age as a short, scannable string for the banner."""
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:  # < 90 min
        return f"{int(seconds / 60)}m ago"
    if seconds < 36 * 3600:  # < 36h
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


def _bridge_post_run_capture(harness: str, cwd: str) -> None:
    """Import the just-run harness's native session into ``<harness>.jsonl``.

    Best-effort: errors are surfaced as a single info line and never
    propagate, because failing the user's session over a capture issue
    would be hostile. The dedup key inside ``import_native_session`` makes
    repeated runs idempotent.
    """
    from claude_codex_local.session import import_native_session

    try:
        result = import_native_session(harness, harness, cwd)
    except Exception as exc:  # noqa: BLE001 - capture must never break the run
        info(f"session bridge: capture failed ({exc})")
        return
    imported = int(result.get("imported", 0) or 0)
    if imported > 0:
        info(f"session bridge: captured {imported} message(s) into {harness}.jsonl")


def run_session(
    prompt: str | None = None,
    no_context: bool = False,
    native_params: list[str] | None = None,
) -> int:
    """
    Exposed as `ccl run [-p PROMPT] [--no-context] [--native-params ...]`.
    Launch the configured harness with an optional initial prompt so external
    agents can drive CCL non-interactively, with an optional cross-agent
    context bridge.

    `native_params`, when provided, is forwarded verbatim to the launched
    harness — Claude Code, Codex, or Pi — so users can opt into harness-native
    flags (e.g. Claude Code's `--dangerously-skip-permissions`) without ccl
    needing a first-class wrapper for every option.

    With `-p PROMPT`, the harness runs in one-shot mode (Claude Code's `-p` /
    Codex's `exec` subcommand) and exits when the response is complete — the
    common automation case. Without `-p`, behavior is identical to invoking
    the alias (`cc` / `cx` / `cc9` / `cx9`): the helper script execs the
    wired argv and the user gets an interactive session.

    The session bridge runs in two halves unless ``--no-context`` (or
    ``CCL_SESSION_BRIDGE=0``) opts out:

    * **Pre-run injection** — one-shot only. If another harness has a
      native session for ``$PWD``, its transcript is rendered as a
      ``[prior context …]`` prefix and prepended to ``PROMPT``. Interactive
      sessions don't inject, because each harness already exposes its own
      ``--resume``/``--continue`` and stdin is the user's TTY.
    * **Post-run capture** — both paths. After the harness exits we import
      its newest native session file for ``$PWD`` into
      ``~/.claude-codex-local/sessions/<harness>.jsonl``. Idempotent.

    Returns the harness's exit code, or a non-zero CCL-level code when
    preconditions fail (no setup, missing helper script, unknown harness).
    """
    if not STATE_FILE.exists():
        fail(f"No wizard state found at {STATE_FILE}. Run `ccl setup` first.")
        return 1
    state = WizardState.load()
    harness = state.primary_harness
    engine = state.primary_engine
    tag = state.engine_model_tag
    if not harness or not engine:
        fail("Wizard state is incomplete — re-run `ccl setup`.")
        return 1
    if not state.wire_result:
        fail("No wired launch found on state — re-run `ccl setup`.")
        return 1
    if prompt is not None and not prompt.strip():
        fail("--prompt cannot be empty.")
        return 1

    # llama.cpp's backing server may have been killed since the last session;
    # bring it back before any harness invocation hits ConnectionRefused.
    if engine == "llamacpp":
        ensure = _ensure_llamacpp_server_running(state)
        if not ensure.get("ok"):
            fail(f"Cannot start llama-server: {ensure.get('error')}")
            return 1

    bridge_off = _bridge_disabled(no_context)
    cwd = os.getcwd()
    effective_prompt = prompt

    # Pre-run injection: one-shot only. Builds the prior-context prefix from
    # the freshest *other* harness's native session for this cwd.
    if prompt is not None and not bridge_off and harness in _BRIDGE_HARNESSES:
        prefix, source, age = _bridge_pre_run_prefix(harness, cwd)
        if prefix:
            age_str = f" (last activity {_format_age(age)})" if age is not None else ""
            info(f"session bridge: injecting context from {source}{age_str}")
            effective_prompt = prefix + prompt

    if effective_prompt is None:
        # Interactive: defer to the helper script so the user gets the same
        # behavior as `cc` / `cx`, including any pre-flight stanzas baked in
        # by step 6.5.
        helper_path = state.helper_script_path or str(
            pb.STATE_DIR / "bin" / _helper_script_basename(_fence_tag_for(harness, engine))
        )
        helper = Path(helper_path)
        if not helper.exists():
            fail(f"Helper script missing at {helper}. Re-run `ccl setup`.")
            return 1
        helper_cmd = [str(helper), *(native_params or [])]
        try:
            proc = subprocess.run(helper_cmd)
        except KeyboardInterrupt:
            return 130
        rc = proc.returncode
        if not bridge_off and harness in _BRIDGE_HARNESSES:
            _bridge_post_run_capture(harness, cwd)
        return rc

    # Only pass `native_params` when the flag was actually used. Calling
    # without the kwarg keeps the function-call shape identical to pre-#97
    # behavior and avoids breaking external/test code that monkeypatches
    # _build_oneshot_cmd with a positional-only signature.
    if native_params is not None:
        cmd = _build_oneshot_cmd(
            harness,
            engine,
            tag,
            state.wire_result,
            effective_prompt,
            native_params=native_params,
        )
    else:
        cmd = _build_oneshot_cmd(harness, engine, tag, state.wire_result, effective_prompt)
    if cmd is None:
        fail(f"Unknown harness for `ccl run`: {harness}")
        return 1

    env_overlay = _resolve_wire_env(state.wire_result)
    full_env = {**os.environ, **env_overlay}
    try:
        proc = subprocess.run(cmd, env=full_env)
    except FileNotFoundError as exc:
        fail(f"Cannot launch harness: {exc}")
        return 127
    except KeyboardInterrupt:
        return 130
    rc = proc.returncode
    if not bridge_off and harness in _BRIDGE_HARNESSES:
        _bridge_post_run_capture(harness, cwd)
    return rc


def run_serve() -> int:
    """
    Exposed as `ccl serve`. Ensures the llama-server backing the persisted
    wizard state is running. Called by the `cc`/`cx` helper script's
    pre-flight when the /health probe fails, so users don't see
    `ConnectionRefused` when the server died between sessions.

    Idempotent. Silent when the server is already up. When a cold start is
    required, prints a prominent banner so the user knows the multi-second
    model-load delay is expected, not a hang.

    Returns 0 on success (server already up or freshly started), 1 on
    misconfiguration or start failure.
    """
    if not STATE_FILE.exists():
        fail(f"No wizard state found at {STATE_FILE}. Run `ccl setup` first.")
        return 1
    state = WizardState.load()
    if state.primary_engine != "llamacpp":
        # Other engines (Ollama, LM Studio, 9router, OpenRouter) are
        # usually long-lived services managed outside ccl; nothing to
        # do here.
        info(f"Engine '{state.primary_engine or 'unset'}' does not need `ccl serve`.")
        return 0
    if not state.engine_model_tag:
        fail("Wizard state has no model tag — re-run `ccl setup`.")
        return 1

    # Detect cold start *before* calling the ensure helper so the user sees
    # the loading banner before the model-load wait starts, not after.
    info_dict = pb.llamacpp_info()
    if info_dict.get("server_running"):
        # Already up — silent reuse keeps the cc/cx hot path quiet.
        return 0

    # Remote endpoint: `ccl serve` cannot start a server on someone else's
    # box (issue #123). The helper script no longer calls us in this case,
    # but defend against direct `ccl serve` invocation with a clear error
    # rather than attempting a doomed local spawn.
    if info_dict.get("remote"):
        base_url = info_dict.get("base_url") or pb.llamacpp_base_url()
        fail(
            f"Remote llama.cpp server at {base_url} is not reachable. "
            f"`ccl serve` does not manage remote engines — start the "
            f"server on the remote host."
        )
        return 1

    warn(
        "Starting llama-server — first run can take 30s+ while the "
        "model is loaded into VRAM. Subsequent calls will be instant."
    )
    result = _ensure_llamacpp_server_running(state)
    if not result.get("ok"):
        fail(f"Could not start llama-server: {result.get('error')}")
        return 1
    return 0


def _get_engine_health(engine: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Get health status for a given engine."""
    from claude_codex_local.core import (
        LlamaCppAdapter,
        LMStudioAdapter,
        OllamaAdapter,
        OpenRouterAdapter,
        Router9Adapter,
        RuntimeAdapter,
        VLLMAdapter,
    )

    adapters: dict[str, RuntimeAdapter] = {
        "ollama": OllamaAdapter(),
        "lmstudio": LMStudioAdapter(),
        "llamacpp": LlamaCppAdapter(),
        "vllm": VLLMAdapter(),
        "9router": Router9Adapter(),
        "openrouter": OpenRouterAdapter(),
    }

    if engine not in adapters:
        return {"ok": False, "detail": f"Unknown engine: {engine}"}

    return adapters[engine].healthcheck()


# Fence tag -> (harness, engine-kind) so we never have to slice strings (which
# bites on "codex" because plain .replace("o","") strips the o in the harness
# name itself).
_FENCE_TAG_TO_HARNESS_ENGINE: dict[str, tuple[str, str]] = {
    "claude": ("claude", "local"),
    "codex": ("codex", "local"),
    "pi": ("pi", "local"),
    "claude9": ("claude", "9router"),
    "codex9": ("codex", "9router"),
    "pi9": ("pi", "9router"),
    "claudeo": ("claude", "openrouter"),
    "codexo": ("codex", "openrouter"),
    "pio": ("pi", "openrouter"),
}

# Basenames the wizard installs in STATE_DIR/bin. The long aliases
# (claude-local, codex-local, pi-local) are shell aliases that point at the
# short-form helper script, not separate files, so we only track the short
# basenames here.
_BASENAME_TO_FENCE_TAG: dict[str, str] = {
    "cc": "claude",
    "cx": "codex",
    "ccp": "pi",
    "cc9": "claude9",
    "cx9": "codex9",
    "cp9": "pi9",
    "cco": "claudeo",
    "cxo": "codexo",
    "cpo": "pio",
}

# Legacy Pi local basename. `cp` shadowed the POSIX copy command, so the
# canonical helper was renamed to `ccp` in #120. The legacy basename stays
# in the detection table so existing installs are recognized and migrated
# the next time the wizard runs.
_LEGACY_BASENAME_TO_FENCE_TAG: dict[str, str] = {
    "cp": "pi",
}


def _detect_existing_shortcuts() -> dict[str, dict[str, Any]]:
    """
    Detect existing helper scripts in STATE_DIR/bin.

    Returns a dict mapping helper-script basename -> {path, fence_tag, harness,
    engine_kind, legacy}. `engine_kind` is one of "local", "9router",
    "openrouter". `legacy` is True for basenames that have been renamed
    (currently just `cp` → `ccp` per #120); the wizard uses this to migrate
    them on the next install. The concrete engine (ollama/llamacpp/...) for
    local helpers is inferred separately by :func:`_infer_engine_from_script`.
    """
    result: dict[str, dict[str, Any]] = {}
    bin_dir = STATE_DIR / "bin"
    if not bin_dir.exists():
        return result

    for script_path in bin_dir.iterdir():
        if not script_path.is_file() or not os.access(script_path, os.X_OK):
            continue
        basename = script_path.name
        fence_tag = _BASENAME_TO_FENCE_TAG.get(basename)
        legacy = False
        if not fence_tag:
            fence_tag = _LEGACY_BASENAME_TO_FENCE_TAG.get(basename)
            if not fence_tag:
                continue
            legacy = True
        harness, engine_kind = _FENCE_TAG_TO_HARNESS_ENGINE[fence_tag]
        result[basename] = {
            "path": str(script_path),
            "fence_tag": fence_tag,
            "harness": harness,
            "engine_kind": engine_kind,
            "legacy": legacy,
        }
    return result


def _infer_engine_from_script(
    script_path: str,
) -> tuple[str | None, str | None]:
    """
    Infer (engine, model_tag) from a local helper script's content.

    Returns (None, None) when the script doesn't match a known shape.
    Only called for helper scripts whose fence tag maps to engine_kind="local";
    9router / openrouter helpers don't need parsing because the engine and
    model live in router config, not in the script.
    """
    try:
        content = Path(script_path).read_text()
    except Exception:
        return (None, None)

    def _model_from(*patterns: str) -> str | None:
        for pattern in patterns:
            m = re.search(pattern, content)
            if m:
                return m.group(1).strip("'\"")
        return None

    # llamacpp: served on port 8001 with ANTHROPIC_BASE_URL pointing at it.
    if ":8001" in content and "ANTHROPIC_BASE_URL" in content:
        model = _model_from(
            r"--model\s+(\S+)",
            r"ANTHROPIC_CUSTOM_MODEL_OPTION=(\S+)",
        )
        return ("llamacpp", model or "(unknown)")

    # ollama: invoked directly or via the ccl-ollama pi provider.
    if "ccl-ollama" in content or re.search(r"\bollama\b", content):
        model = _model_from(
            r"--provider\s+ccl-ollama\s+--model\s+(\S+)",
            r"--model\s+(\S+)",
            r"MODEL=(\S+)",
        )
        return ("ollama", model or "(unknown)")

    # lmstudio: served on port 1234.
    if ":1234" in content:
        model = _model_from(
            r"--model\s+(\S+)",
            r"ANTHROPIC_CUSTOM_MODEL_OPTION=(\S+)",
        )
        return ("lmstudio", model or "(unknown)")

    # vllm: default port 8000 (or the configured VLLM_BASE_URL).
    if ":8000" in content or pb.VLLM_BASE_URL in content:
        model = _model_from(r"--model\s+(\S+)")
        return ("vllm", model or "(unknown)")

    return (None, None)


def run_status() -> int:
    """Exposed as `ccl status`. Display current setup and shortcut availability."""
    header("status — current ccl setup and shortcut availability")

    if not STATE_FILE.exists():
        warn(f"No wizard state found at {STATE_FILE}.")
        console.print("\nRun [bold]ccl setup[/bold] to configure your ccl setup.")
        console.print("\n[italic]No shortcuts are available until setup is complete.[/italic]")
        return 1

    state = WizardState.load()

    profile = pb.machine_profile()
    presence = profile.get("presence", {})
    installed_engines: list[str] = presence.get("engines", []) or []

    # Helper scripts are the source of truth for what `cc` / `cx` / `ccp`
    # (and the 9/o variants) actually run today. The wizard state captures
    # the user's most recent setup choices but the user may have rerun the
    # wizard for one harness without touching another, so we inspect each
    # harness's helper script independently rather than projecting a single
    # "default engine" across all rows.
    existing_shortcuts = _detect_existing_shortcuts()

    local_script_configs: dict[str, dict[str, str]] = {}
    for basename, info in existing_shortcuts.items():
        if info["engine_kind"] != "local":
            continue
        engine, model = _infer_engine_from_script(info["path"])
        if engine:
            local_script_configs[basename] = {"engine": engine, "model": model or "(unknown)"}

    # Legacy Pi local helper basename — see #120. The current `ccp` install
    # rules take precedence; a legacy `cp` helper only surfaces when the
    # user hasn't re-run setup yet, and even then we mark the row so users
    # know to re-run.
    legacy_pi_basename = "cp"
    has_legacy_pi_script = legacy_pi_basename in existing_shortcuts and existing_shortcuts[
        legacy_pi_basename
    ].get("legacy")

    all_shortcuts: list[dict[str, Any]] = []

    for harness in ["claude", "codex", "pi"]:
        # ---- Primary (local engine) variant: cc / cx / ccp + long alias ----
        local_basename = _helper_script_basename(harness)
        has_local_script = local_basename in existing_shortcuts
        script_cfg = local_script_configs.get(local_basename)
        # Surface a pre-#120 `cp` install on the Pi row so users see it in
        # `ccl status` and know to re-run setup.
        if harness == "pi" and not has_local_script and has_legacy_pi_script:
            has_local_script = True
            script_cfg = local_script_configs.get(legacy_pi_basename)

        if script_cfg:
            primary_engine_name = script_cfg["engine"]
            primary_model_tag = script_cfg["model"]
        elif state.primary_harness == harness and state.primary_engine in (
            "ollama",
            "llamacpp",
            "lmstudio",
            "vllm",
        ):
            # No script for this harness yet, but the wizard state records the
            # user's intended config — show that so the row reflects what
            # `ccl setup` would wire up.
            primary_engine_name = state.primary_engine
            primary_model_tag = state.engine_model_tag or state.model_name or "(not set)"
        else:
            primary_engine_name = "(unset)"
            primary_model_tag = "(not set)"

        primary_fence_tag = _fence_tag_for(harness, primary_engine_name)
        primary_aliases = _alias_names_for(primary_fence_tag)

        primary_engine_installed = primary_engine_name in installed_engines
        primary_engine_status = (
            "[green]on[/green]" if primary_engine_installed else "[red]off[/red]"
        )

        has_primary_model = primary_model_tag not in ("(not set)", "(unknown)")

        # Availability rules (in priority order):
        #   available     — helper script exists, engine installed, model set
        #   unavailable   — helper script exists but engine binary missing or
        #                   model not parseable; running the alias would fail
        #   unconfigured  — no helper script for this harness yet
        if has_local_script and primary_engine_installed and has_primary_model:
            primary_availability = "[green]available[/green]"
        elif has_local_script:
            primary_availability = "[red]unavailable[/red]"
        else:
            primary_availability = "[yellow]unconfigured[/yellow]"

        all_shortcuts.append(
            {
                "aliases": primary_aliases,
                "model": primary_model_tag,
                "engine": primary_engine_name,
                "engine_status": primary_engine_status,
                "availability": primary_availability,
            }
        )

        # ---- Router-backed variants: 9router (cc9/cx9/cp9), openrouter (cco/cxo/cpo) ----
        for fence_suffix, router_engine in (("9", "9router"), ("o", "openrouter")):
            fence_tag = f"{harness}{fence_suffix}"
            aliases = _alias_names_for(fence_tag)
            basename = _helper_script_basename(fence_tag)
            has_script = basename in existing_shortcuts
            engine_installed = router_engine in installed_engines
            engine_status = "[green]on[/green]" if engine_installed else "[red]off[/red]"

            # For router-backed shortcuts, "configured" means the helper
            # script is wired up; "available" further requires the API key
            # to be detected (presence.engines includes the router).
            if has_script and engine_installed:
                availability = "[green]available[/green]"
            elif has_script:
                availability = "[red]unavailable[/red]"
            else:
                availability = "[yellow]unconfigured[/yellow]"

            all_shortcuts.append(
                {
                    "aliases": aliases,
                    "model": "(API key only)",
                    "engine": router_engine,
                    "engine_status": engine_status,
                    "availability": availability,
                }
            )

    # Display shortcuts table
    table = Table(title="Configured shortcuts", show_header=True)
    table.add_column("Shortcut", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Engine", style="magenta")
    table.add_column("Engine Status", style="yellow")
    table.add_column("Availability", style="green")

    for shortcut in all_shortcuts:
        # Format aliases
        aliases_str = ", ".join(shortcut["aliases"])
        row_values: list[str] = [
            aliases_str,
            shortcut["model"],
            shortcut["engine"],
            shortcut["engine_status"],
            shortcut["availability"],
        ]
        table.add_row(*row_values)

    console.print(table)
    console.print()

    # Skills surface — currently only the llamacpp-tuner (issue #124). The
    # tuner targets a *local* llama-server (it kills + restarts the process
    # over a shell), so it must show as `unavailable (remote)` whenever the
    # configured LLAMACPP_BASE_URL points outside loopback. We surface it
    # here so users see why the skill no-ops instead of discovering it only
    # at invocation time.
    skills_table = Table(title="Skills", show_header=True)
    skills_table.add_column("Skill", style="bold")
    skills_table.add_column("Engine", style="magenta")
    skills_table.add_column("Availability", style="green")

    llamacpp_url = pb.llamacpp_base_url()
    if pb._is_local_base_url(llamacpp_url):
        tuner_availability = "[green]available[/green]"
    else:
        tuner_availability = f"[red]unavailable (remote — {llamacpp_url})[/red]"
    skills_table.add_row("llamacpp-tuner", "llamacpp", tuner_availability)

    console.print(skills_table)
    console.print()

    # Overall setup summary
    console.print("[bold]Overall Setup Summary[/bold]")

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("key", style="bold")
    summary_table.add_column("value")

    summary_table.add_row(
        "Engines detected", ", ".join(installed_engines) if installed_engines else "(none)"
    )

    running_engines = []
    for engine in installed_engines:
        engine_health = _get_engine_health(engine, profile)
        if "ok" in engine_health and engine_health["ok"]:
            running_engines.append(engine)

    summary_table.add_row(
        "Engines running", ", ".join(running_engines) if running_engines else "(none)"
    )

    # "Default …" rows reflect what `ccl setup` recorded — not what's wired
    # up on disk. The shortcuts table above already tells the user what each
    # cc/cx/ccp script actually runs, so we don't try to back-fill these rows
    # from helper-script inference (different harnesses can use different
    # engines, and picking one to call "the default" misleads the user).
    if state.primary_harness:
        summary_table.add_row("Default harness", state.primary_harness)
    if state.primary_engine:
        summary_table.add_row("Default engine", state.primary_engine)
    if state.model_name:
        summary_table.add_row("Selected model", state.model_name)

    console.print(summary_table)
    console.print()

    # Hint for unconfigured setup
    has_any_configured = any("[green]available[/green]" in s["availability"] for s in all_shortcuts)
    if not has_any_configured:
        console.print(
            "[info]No shortcuts are configured yet. Run [bold]ccl setup[/bold] to get started.[/info]"
        )

    return 0


def run_find_model_standalone() -> int:
    """Exposed as `ccl find-model` — no setup, just a recommendation."""
    header("find-model — llmfit coding-model recommendation")
    profile = pb.machine_profile()
    if not profile["presence"]["llmfit"]:
        if not _ensure_llmfit():
            return 1
        # Refresh profile after successful install.
        profile = pb.machine_profile()
    engines = profile["presence"]["engines"] or ["ollama"]
    engine = engines[0]
    info(f"Ranking models for engine: {engine}")
    picked = _find_model_interactive(engine, profile)
    if picked:
        console.print(f"\n[bold]You picked:[/bold] {picked['display']}")
        console.print(f"[bold]Engine tag:[/bold] {picked['tag']}")
        return 0
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccl",
        description=(
            "ccl — claude-codex-local. Wire up Claude Code or Codex to a local LLM engine "
            "(Ollama, LM Studio, or llama.cpp). Run without arguments to start the interactive "
            "first-run wizard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ccl                              Run the interactive first-run wizard\n"
            "  ccl --resume                     Resume an interrupted wizard\n"
            "  ccl --non-interactive            Scripted install with defaults\n"
            "  ccl status                       Show current setup and shortcut availability\n"
            "  ccl doctor                       Triage the current install\n"
            "  ccl find-model                   Show a recommended coding model\n"
            "  ccl run                          Launch the configured session interactively\n"
            '  ccl run -p "what is 2+2?"        Launch one-shot for agent automation\n'
            "  ccl run --native-params -- --dangerously-skip-permissions\n"
            "                                   Forward harness-native flags verbatim\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"ccl {__version__}")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (also honors the NO_COLOR env var)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpointed step",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-pick defaults (for CI and scripted installs)",
    )

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    setup = sub.add_parser(
        "setup",
        help="Run the interactive first-run wizard (this is the default)",
        description="Run the interactive first-run wizard to pick a harness, engine, and model.",
    )
    setup.add_argument(
        "--non-interactive",
        action="store_true",
        help="Auto-pick defaults (for CI and scripted installs)",
    )
    setup.add_argument(
        "--harness", choices=("claude", "codex", "pi"), help="Force the primary harness"
    )
    setup.add_argument(
        "--engine",
        choices=("ollama", "lmstudio", "llamacpp", "vllm", "9router", "openrouter"),
        help="Force the primary engine",
    )
    setup.add_argument(
        "--force-scan",
        action="store_true",
        help="Ignore the in-process setup snapshot; selected harness/engine are checked live",
    )
    # Lazy-llmfit flag (issue #79). The hardware capability scan is deferred by
    # default; --run-llmfit refreshes only llmfit recommendations.
    setup.add_argument(
        "--run-llmfit",
        action="store_true",
        help="Run the llmfit hardware capability scan (refresh cached data)",
    )

    sub.add_parser(
        "find-model",
        help="Show an llmfit-driven coding-model recommendation",
        description="Rank local coding models with llmfit and show the best fit for this machine.",
    )
    sub.add_parser(
        "doctor",
        help="Triage: print wizard state and re-run the presence check",
        description="Show the current wizard state and re-check that harness, engine, and model are healthy.",
    )
    session = sub.add_parser(
        "session",
        help="Manage opt-in shared conversation sessions between agents",
        description=(
            "Manage JSONL session files under ~/.claude-codex-local/sessions. "
            "Use `ccl session sync` to copy redacted context between agents."
        ),
    )
    session_sub = session.add_subparsers(dest="session_subcommand", required=True)
    session_sync = session_sub.add_parser(
        "sync", help="Copy redacted messages from one agent to another"
    )
    session_sync.add_argument("--from", dest="from_agent", default="claude", help="Source agent id")
    session_sync.add_argument("--to", dest="to_agent", default="codex", help="Target agent id")
    session_list = session_sub.add_parser("list", help="List known session files")
    session_list.add_argument("--agent", help="Filter by agent id")
    session_show = session_sub.add_parser("show", help="Show messages for an agent")
    session_show.add_argument("agent", help="Agent id to show")
    session_clear = session_sub.add_parser("clear", help="Clear one agent session")
    session_clear.add_argument("agent", help="Agent id to clear")
    session_truncate = session_sub.add_parser("truncate", help="Keep only the last N messages")
    session_truncate.add_argument("agent", help="Agent id to truncate")
    session_truncate.add_argument(
        "--keep",
        type=int,
        required=True,
        help=(
            "Number of messages to keep (required). Use `ccl session clear` "
            "to wipe the session entirely; passing --keep 0 is treated as an "
            "explicit truncate to zero."
        ),
    )
    sub.add_parser(
        "serve",
        help="Ensure the llama-server backing the wizard's chosen model is up",
        description=(
            "Probe the configured llama-server's /health endpoint and "
            "auto-start it (with a model-load banner) when it is down. "
            "Idempotent and silent on the hot path. Used internally by the "
            "cc/cx helper scripts."
        ),
    )
    sub.add_parser(
        "status",
        help="Show current ccl setup and shortcut availability",
        description=(
            "Display the current ccl configuration, including which shortcuts "
            "are available and whether their engines are running."
        ),
    )

    run = sub.add_parser(
        "run",
        help="Launch the configured harness, optionally with an initial prompt",
        description=(
            "Launch the configured Claude Code, Codex, or Pi session. With "
            "-p/--prompt PROMPT, the prompt is submitted as the first user "
            "message and the harness runs in one-shot mode (Claude's `-p`, "
            "Codex's `exec`) — useful when calling CCL from another agent or "
            "CI script. Without -p, behavior matches the cc/cx alias and the "
            "session starts interactively."
        ),
    )
    run.add_argument(
        "-p",
        "--prompt",
        help="Initial prompt to submit; runs the harness in one-shot mode",
    )
    run.add_argument(
        "--no-context",
        action="store_true",
        help=(
            "Skip the cross-agent context bridge (no prefix injection, no "
            "capture). Same effect as CCL_SESSION_BRIDGE=0."
        ),
    )
    # `--native-params` is intentionally NOT declared with argparse: argparse's
    # REMAINDER has a long-standing interaction bug with subparsers plus the
    # `--` token, and a positional collector would shadow ccl's own flags. The
    # flag is sliced out of sys.argv by `_extract_native_params` before
    # argparse runs. We add a no-op argument here purely so it appears in
    # `ccl run --help` per the issue's acceptance criteria.
    run.add_argument(
        "--native-params",
        nargs="*",
        default=None,
        metavar="ARG",
        help=(
            "Forward all following arguments verbatim to the launched harness "
            "(Claude Code, Codex, or Pi). Must be the last flag on the line; "
            "use `--` to separate ccl flags from native ones, e.g. "
            "`ccl run --native-params -- --dangerously-skip-permissions`. "
            "ccl does not validate native params — the harness does. "
            "(Parsed before argparse; the argparse declaration exists only "
            "for `--help` documentation.)"
        ),
    )

    return parser


def _extract_native_params(
    argv: list[str],
) -> tuple[list[str], list[str] | None]:
    """
    Slice ``--native-params ARG…`` (and any leading ``--`` separator) out of
    a raw argv list, returning the argv that argparse should see and the
    captured passthrough list.

    Pre-processing sidesteps two argparse pitfalls: ``argparse.REMAINDER``
    misbehaves under subparsers, and a regular positional collector would
    swallow ``ccl run``'s own flags. Doing the split ourselves keeps the
    passthrough verbatim — exactly the property AC #2 asks for.

    Returns ``(argv_without_native_params, native_params_or_None)``. When the
    flag is absent the second element is ``None`` so callers can preserve the
    pre-existing "flag was never passed" launch path.

    Limitations:
    - Only the first ``--native-params`` occurrence is treated as the
      boundary; later occurrences are forwarded verbatim as harness args.
    - We do not respect option-consuming flags, so the pathological case of
      ``ccl run -p --native-params`` (i.e. the literal string
      ``--native-params`` as a flag value) will be misinterpreted. Not a
      real-world concern but documented for future maintainers.
    """
    if "--native-params" not in argv:
        return list(argv), None
    idx = argv.index("--native-params")
    tail = list(argv[idx + 1 :])
    if tail and tail[0] == "--":
        tail = tail[1:]
    return list(argv[:idx]), tail


def main() -> int:
    parser = _build_parser()
    # Strip `--native-params …` out of argv before argparse sees it; argparse's
    # subparser + REMAINDER + `--` interactions are unreliable, and we need
    # the tail forwarded verbatim per issue #97 AC #2/#4.
    raw_argv = sys.argv[1:]
    cleaned_argv, native_params = _extract_native_params(raw_argv)
    args = parser.parse_args(cleaned_argv)

    # Honor --no-color and NO_COLOR env var for the Rich console.
    if getattr(args, "no_color", False) or os.environ.get("NO_COLOR"):
        console.no_color = True

    cmd = args.cmd or "setup"
    if cmd == "setup":
        return run_wizard(
            resume=getattr(args, "resume", False),
            non_interactive=getattr(args, "non_interactive", False),
            force_harness=getattr(args, "harness", None),
            force_engine=getattr(args, "engine", None),
            force_scan=getattr(args, "force_scan", False),
            run_llmfit_flag=getattr(args, "run_llmfit", False),
        )
    if cmd == "find-model":
        return run_find_model_standalone()
    if cmd == "doctor":
        return run_doctor()
    if cmd == "status":
        return run_status()
    if cmd == "session":
        from claude_codex_local import session as sess
        from claude_codex_local.core import print_payload

        subcmd = getattr(args, "session_subcommand", "")
        if subcmd == "sync":
            print_payload(sess.sync_session(agent_id=args.to_agent, other_agent_id=args.from_agent))
            return 0
        if subcmd == "list":
            summaries = sess.get_all_sessions()
            if getattr(args, "agent", None):
                summaries = [item for item in summaries if item.get("agent_id") == args.agent]
            print_payload({"sessions": summaries})
            return 0
        if subcmd == "show":
            messages = [
                message.to_dict(agent_id=message.agent_id or args.agent)
                for message in sess.load_session(args.agent)
            ]
            print_payload({"agent": args.agent, "messages": messages})
            return 0
        if subcmd == "clear":
            print_payload(sess.clear_session(args.agent))
            return 0
        if subcmd == "truncate":
            print_payload(sess.truncate_session(args.agent, keep_last=args.keep))
            return 0
        parser.print_help()
        return 2
    if cmd == "serve":
        return run_serve()
    if cmd == "run":
        return run_session(
            prompt=getattr(args, "prompt", None),
            no_context=bool(getattr(args, "no_context", False)),
            native_params=native_params,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
