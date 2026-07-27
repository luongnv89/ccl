#!/usr/bin/env python3
"""
Interactive first-run wizard for claude-codex-local.

This file is a **thin re-export facade** over the decomposed wizard modules:

  - ``wizard_state.py`` — ``WireResult``, ``WizardState``, state helpers
  - ``wizard_ui.py`` — banner, header, ok, warn, fail, info
  - ``wizard_discovery.py`` — step_2_1_discover, step_2_2_install_missing, …
  - ``wizard_steps.py`` — step_2_select_harness … step_2_8_generate_guide
  - ``wizard_cli.py`` — run_wizard, run_doctor, run_session, run_serve,
    run_status, run_find_model_standalone, main, _build_parser

All existing imports (``from claude_codex_local import wizard`` or
``from claude_codex_local.wizard import X``) continue working with zero
source changes — the same facade pattern ``core.py`` uses since #161.

Monkeypatch note:
  - Functions that tests monkeypatch via ``monkeypatch.setattr(wizard, ...)``
    are resolved at call time via wrapper functions so that patches on
    ``wizard`` propagate to callers automatically.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Stdlib / third-party re-exports — kept at the top of wizard.py so that
# tests which monkeypatch ``wizard.subprocess``, ``wizard.pb``, etc. still
# work.  These are imported once and re-exported as module-level attributes.
# ---------------------------------------------------------------------------
import argparse as argparse
import os as os
import shutil as shutil
import subprocess as subprocess
import sys as sys
import time as time
from collections.abc import Callable
from pathlib import Path as Path

import questionary as questionary

from claude_codex_local import __version__
from claude_codex_local import core as pb
from claude_codex_local.engines import ALL_ENGINES as _REGISTRY_ENGINES

# ---------------------------------------------------------------------------
# Rich imports — needed for console and Panel
# ---------------------------------------------------------------------------
from rich.console import Console
from rich.panel import Panel as Panel

# ---------------------------------------------------------------------------
# Console — tests patch ``wizard.console`` and expect ``warn`` /
# ``print_welcome_banner`` to use the same console.
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Module-level constants that tests reference directly.
# _ALL_ENGINES is derived from the registry (same as wizard_cli).
# ---------------------------------------------------------------------------
_ALL_ENGINES = list(_REGISTRY_ENGINES)

# ---------------------------------------------------------------------------
# Re-export from wizard_state
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_state import (
    GUIDE_PATH,
    WireResult,
    WizardState,
    _backup_invalid_wizard_state,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Re-export from wizard_ui — but wrap print_welcome_banner so it uses
# this module's ``console`` (tests patch ``wizard.console`` and expect
# ``warn`` / ``print_welcome_banner`` to use the same console).
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_ui import (
    _CCL_BANNER as _UI_BANNER,
    _CCL_TAGLINE as _UI_TAGLINE,
    _CCL_REPO_URL as _UI_REPO_URL,
)  # noqa: E402


def print_welcome_banner() -> None:
    """Print the ASCII 3D CCL banner, tagline, version, and repo URL."""
    console.print(_UI_BANNER, style="bold cyan", highlight=False)
    console.print(f"  [bold white]{_UI_TAGLINE}[/bold white]")
    console.print(f"  [dim]v{__version__}  ·  [link={_UI_REPO_URL}]{_UI_REPO_URL}[/link][/dim]")
    console.print()


def header(title: str) -> None:
    console.print()
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def ok(msg: str) -> None:
    console.print(f"[green]\u2713[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


def fail(msg: str) -> None:
    console.print(f"[red]\u2717[/red] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]\u00b7[/dim] {msg}")


# ---------------------------------------------------------------------------
# Re-export from wizard_discovery
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_discovery import (
    _ensure_llmfit,
    _ensure_tool,
    _is_model_compatible_with_engine,
    _persist_targeted_profile_update,
    _refresh_selected_engine,
    _refresh_selected_harness,
    _show_selected_harness_status,
    _sync_presence_from_tools,
    _try_llmfit_fallback,
    step_2_1_discover,
    step_2_2_install_missing,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Re-export from wizard_steps — but NOT step_3_select_engine.
# step_3_select_engine is defined below because it references
# _prompt_local_or_remote and _refresh_selected_engine which tests
# monkeypatch via ``monkeypatch.setattr(wizard, ...)``.  Keeping the
# original implementation in wizard.py ensures patches propagate
# naturally (the function lives in the same module as the patches).
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_steps import (
    _alias_block,
    _alias_names_for,
    _apply_local_endpoint,
    _apply_remote_endpoint,
    _append_toml_table,
    _build_profile_recommendations,
    _candidate_tag,
    _collect_gguf_variants,
    _codex_config_path,
    _codex_home,
    _codex_provider_config,
    _codex_provider_env_key,
    _codex_provider_for_engine,
    _configure_codex_with_backup,
    _configure_pi_with_backup,
    _default_variant_label,
    _detect_shell_rc,
    _download_gguf_via_hf_cli,
    _ensure_llamacpp_server_running,
    _download_model,
    _lms_model_size_hint,
    _ollama_model_size_hint,
    _env_block,
    _estimate_model_size,
    _fence_tag_for,
    _find_model_auto,
    _find_model_interactive,
    _format_context_length,
    _format_tokens_per_second,
    _helper_script_basename,
    _human_bytes,
    _human_duration,
    _infer_harness_from_legacy_block,
    _install_shell_aliases,
    _largest_gguf_in,
    _llamacpp_models_match,
    _llamacpp_smoke_test,
    _llamacpp_spawn_and_smoke,
    _looks_like_missing_repo,
    _map_to_engine,
    _materialize_pi_api_key_files,
    _materialize_raw_env,
    _model_already_installed,
    _model_known_incompatible_with_claude_code,
    _normalize_model_id,
    _persist_remote_env_to_shell_rc,
    _pi_api_key_for_engine,
    _pi_base_url_for_engine,
    _pi_provider_for_engine,
    _prompt_fuzzy_hf_match,
    _prompt_gguf_variant,
    _profile_choice_label,
    _prompt_local_or_remote,
    _prompt_remote_endpoint,
    _remove_legacy_pi_helper,
    _remove_toml_table,
    _render_llama_server_command,
    _report_smoke_test_speed,
    _rollback_config_backup,
    _speed_verdict,
    _step_4_openrouter_model_browser,
    _step_4_pick_9router_model_interactive,
    _step_4_pick_model_9router,
    _step_4_pick_model_9router_impl,
    _step_4_pick_model_local_impl,
    _step_4_pick_model_openrouter,
    _step_4_pick_model_openrouter_impl,
    _step_4_pick_model_remote,
    _step_4_pick_model_remote_impl,
    _step_4_pick_model_vllm,
    _step_4_pick_model_vllm_impl,
    _toml_quote,
    _upsert_top_level_toml_key,
    _variant_token,
    _wire_claude,
    _wire_codex,
    _wire_pi,
    _write_codex_config,
    _write_helper_script,
    step_2_4_pick_model,
    step_2_5_5_benchmark,
    step_2_5_smoke_test,
    step_2_65_install_aliases,
    step_2_6_wire_harness,
    step_2_7_verify,
    step_2_8_generate_guide,
    step_2_select_harness,
)  # noqa: E402

# ---------------------------------------------------------------------------
# Local-vs-remote engine list — used by step_3_select_engine.
# ---------------------------------------------------------------------------
_LOCAL_OR_REMOTE_ENGINES = ("ollama", "llamacpp", "vllm")

# ---------------------------------------------------------------------------
# Re-export from wizard_cli — data and helpers only (no entry-point functions).
# Entry-point functions are defined below as wrappers so that
# monkeypatches on ``wizard`` propagate correctly.
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_cli import (
    _ALL_ENGINES as _CLI_ALL_ENGINES,
    _BASENAME_TO_FENCE_TAG,
    _LEGACY_BASENAME_TO_FENCE_TAG,
    _build_oneshot_cmd as _cli_build_oneshot_cmd,
    _build_parser as _cli_build_parser,
    _detect_existing_shortcuts,
    _extract_native_params,
    _infer_engine_from_script,
    _get_engine_health,
    _resolve_wire_env as _cli_resolve_wire_env,
)  # noqa: E402

# Also re-export these names so tests can access them directly.
_build_oneshot_cmd = _cli_build_oneshot_cmd
_resolve_wire_env = _cli_resolve_wire_env

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
_CCL_BANNER = r"""
  ██████╗ ██████╗██╗
 ██╔════╝██╔════╝██║
 ██║     ██║     ██║
 ██║     ██║     ██║
 ╚██████╗╚██████╗███████╗
  ╚═════╝ ╚═════╝╚══════╝
"""

_CCL_TAGLINE = "Hit your limit? Need privacy? Just swap the model."
_CCL_REPO_URL = "https://github.com/luongnv89/claude-codex-local"

# ---------------------------------------------------------------------------
# Module-level state references (tests reference these directly).
# ---------------------------------------------------------------------------
STATE_DIR = pb.STATE_DIR
STATE_FILE = STATE_DIR / "wizard-state.json"
# GUIDE_PATH is re-exported from wizard_state.py — tests patch wizard.GUIDE_PATH


# ---------------------------------------------------------------------------
# step_3_select_engine — kept here so that test monkeypatches on
# ``wizard._refresh_selected_engine`` and ``wizard._ensure_tool``
# propagate naturally (the function lives in the same module).
# ---------------------------------------------------------------------------
def _default_engine(engines: list[str], profile: dict[str, Any]) -> str:
    """Pick a sensible default engine."""
    ollama_ready = "ollama" in engines and bool(profile.get("ollama", {}).get("models"))
    lms_data = profile.get("lmstudio", {})
    lms_ready = (
        "lmstudio" in engines
        and lms_data.get("server_running", False)
        and bool(lms_data.get("models"))
    )
    is_apple_silicon = profile.get("host", {}).get("system") == "Darwin" and profile.get(
        "host", {}
    ).get("machine") in ("arm64", "aarch64")
    if is_apple_silicon and lms_ready:
        return "lmstudio"
    if ollama_ready:
        return "ollama"
    if is_apple_silicon and "lmstudio" in engines:
        return "lmstudio"
    if "ollama" in engines:
        return "ollama"
    return engines[0]


def step_3_select_engine(state: WizardState, non_interactive: bool = False) -> bool:
    """Select the primary engine (interactive or non-interactive).

    This is the original implementation from the wizard.py monolith.
    It is kept here rather than delegated to wizard_steps so that
    test monkeypatches on wizard._refresh_selected_engine /
    wizard._ensure_tool propagate correctly.
    """
    header("Step 3 — Select engine")
    presence = _sync_presence_from_tools(state.profile)
    engines = presence["engines"]

    if state.primary_engine:
        choice = state.primary_engine
        if choice not in _ALL_ENGINES:
            fail(f"Unknown engine: {choice}")
            return False
        _refresh_selected_engine(state.profile, choice)
        engines = state.profile["presence"]["engines"]
        if choice not in engines:
            if non_interactive:
                fail(f"Forced engine {choice!r} is not installed or reachable.")
                return False
            if not _ensure_tool(choice):
                fail(f"Forced engine {choice!r} is still not available.")
                return False
            _refresh_selected_engine(state.profile, choice)
            engines = state.profile["presence"]["engines"]
        state.primary_engine = choice
        state.secondary_engines = [e for e in engines if e != choice]
        ok(f"Using forced primary engine: [bold]{state.primary_engine}[/bold]")
    elif non_interactive:
        if not engines:
            fail("No engine installed. Cannot continue in non-interactive mode.")
            return False
        default_engine = _default_engine(engines, state.profile)
        engine_candidates = [default_engine, *(e for e in engines if e != default_engine)]
        live_engine = None
        for candidate in engine_candidates:
            if _refresh_selected_engine(state.profile, candidate):
                live_engine = candidate
                break
        if live_engine is None:
            fail("Cached engine data is stale; no configured engine is currently installed.")
            return False
        engines = state.profile["presence"]["engines"]
        state.primary_engine = live_engine
        state.secondary_engines = [e for e in engines if e != live_engine]
        ok(f"Non-interactive: picking [bold]{state.primary_engine}[/bold] as primary engine")
    else:
        engine_choices = [
            questionary.Choice(
                e if e in engines else f"{e}  [not installed]",
                value=e,
            )
            for e in _ALL_ENGINES
        ]
        default_engine = _default_engine(engines, state.profile) if engines else _ALL_ENGINES[0]
        while True:
            choice = questionary.select(
                "Which engine do you want as primary?",
                choices=engine_choices,
                default=default_engine,
            ).ask()
            if choice is None:
                return False
            # Surface the local-vs-remote choice for local-capable engines
            # BEFORE attempting any install or probe — when remote is picked
            # we want to skip `_ensure_tool(choice)` entirely (the binary is
            # not needed on this host), and we want the probe below to hit
            # the remote URL.
            remote = False
            if choice in _LOCAL_OR_REMOTE_ENGINES:
                remote = _prompt_local_or_remote(state, choice)
            _refresh_selected_engine(state.profile, choice)
            engines = state.profile["presence"]["engines"]
            if choice not in engines:
                if remote:
                    warn(
                        f"Remote {choice} endpoint is not reachable yet. "
                        "Check the URL and that the server is up, then pick "
                        "this engine again."
                    )
                    continue
                if not _ensure_tool(choice):
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
                _refresh_selected_engine(state.profile, choice)
                engines = state.profile["presence"]["engines"]
                if choice not in engines:
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
            state.primary_engine = choice
            state.secondary_engines = [e for e in engines if e != choice]
            break

    ok(f"Selected: [bold]{state.primary_harness}[/bold] + [bold]{state.primary_engine}[/bold]")
    if state.secondary_harnesses or state.secondary_engines:
        info(
            f"Fallbacks: harnesses={state.secondary_harnesses or '-'} engines={state.secondary_engines or '-'}"
        )

    if not _is_model_compatible_with_engine(state, state.primary_engine):
        info("Engine changed. Model selection will be required in next step.")
        state.model_name = ""
        state.model_source = ""
        state.engine_model_tag = ""
        state.model_candidate = {}

    state.mark("3")
    return True


# ---------------------------------------------------------------------------
# STEPS — the ordered list of wizard steps (used by run_wizard and tests).
# ---------------------------------------------------------------------------
STEPS: list[tuple[str, str, Callable[[WizardState, bool], bool]]] = [
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

# ---------------------------------------------------------------------------
# Monkeypatch propagation — set sub-module references to this module's
# values so that ``monkeypatch.setattr(wizard, 'X', ...)`` patches
# propagate to all wizard sub-modules after reload.
# ---------------------------------------------------------------------------
import claude_codex_local.wizard_ui as _wizard_ui  # noqa: E402
import claude_codex_local.wizard_steps as _wizard_steps_mod  # noqa: E402
import claude_codex_local.wizard_cli as _wizard_cli_mod  # noqa: E402
import claude_codex_local.wizard_discovery as _wizard_discovery_mod  # noqa: E402

_wizard_ui.console = console
_wizard_steps_mod.console = console
_wizard_cli_mod.console = console
_wizard_discovery_mod.console = console

_wizard_steps_mod.questionary = questionary
_wizard_cli_mod.questionary = questionary  # type: ignore[attr-defined]
_wizard_discovery_mod.questionary = questionary  # type: ignore[attr-defined]

# Keep private references to objects used by __getattr__.
_q = questionary
_rs_e = _refresh_selected_engine
_et = _ensure_tool

# Keep _refresh_selected_engine and _ensure_tool in __dict__ so that
# wizard_steps.py wrappers can find them via sys.modules["wizard"].__dict__
# and pick up test monkeypatches.  Also keep questionary in __dict__
# because step_3_select_engine references it directly.
# (The __getattr__ / __setattr__ below still exist for backward compat
# with tests that set attributes directly on the wizard module.)

# Make wizard_cli's internal helpers resolve through this module.
_wizard_cli_mod._build_oneshot_cmd = _build_oneshot_cmd
_wizard_cli_mod._resolve_wire_env = _resolve_wire_env

# ---------------------------------------------------------------------------
# Wrapper functions — resolve at call time so test monkeypatches on
# the ``wizard`` module propagate to callers automatically.
# ---------------------------------------------------------------------------


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
    """Delegate to ``wizard_cli.run_wizard`` at call time."""
    return _cli_run_wizard(
        resume=resume,
        non_interactive=non_interactive,
        start_step=start_step,
        force_harness=force_harness,
        force_engine=force_engine,
        force_scan=force_scan,
        run_llmfit_flag=run_llmfit_flag,
    )


def run_doctor() -> int:
    """Delegate to ``wizard_cli.run_doctor`` at call time."""
    return _cli_run_doctor()


def run_session(
    prompt: str | None = None,
    no_context: bool = False,
    native_params: list[str] | None = None,
) -> int:
    """Delegate to ``wizard_cli.run_session`` at call time.

    Monkeypatch propagation: resolve _build_oneshot_cmd and
    _resolve_wire_env from wizard.__dict__ at call time so that
    ``monkeypatch.setattr(wizard, ...)`` patches propagate.
    """
    # Resolve helpers from wizard.__dict__ at call time so monkeypatches
    # on wizard._build_oneshot_cmd / wizard._resolve_wire_env propagate.
    _wizard_cli_mod._build_oneshot_cmd = sys.modules[__name__].__dict__.get(
        "_build_oneshot_cmd", _build_oneshot_cmd
    )
    _wizard_cli_mod._resolve_wire_env = sys.modules[__name__].__dict__.get(
        "_resolve_wire_env", _resolve_wire_env
    )
    return _cli_run_session(
        prompt=prompt,
        no_context=no_context,
        native_params=native_params,
    )


def run_serve() -> int:
    """Delegate to ``wizard_cli.run_serve`` at call time."""
    return _cli_run_serve()


def run_status() -> int:
    """Delegate to ``wizard_cli.run_status`` at call time."""
    return _cli_run_status()


def run_find_model_standalone() -> int:
    """Delegate to ``wizard_cli.run_find_model_standalone`` at call time."""
    return _cli_run_find_model_standalone()


def main() -> int:
    """Delegate to ``wizard_cli.main`` at call time.

    Monkeypatch propagation: resolve run_session from this module at
    call time so that ``monkeypatch.setattr(wizard, 'run_session', ...)``
    patches propagate.
    """
    # Resolve run_session from this module at call time so monkeypatches
    # on wizard.run_session propagate.
    _wizard_cli_mod.run_session = run_session
    return _cli_main()


def _build_parser() -> argparse.ArgumentParser:
    """Delegate to ``wizard_cli._build_parser`` at call time."""
    return _cli_build_parser()


# ---------------------------------------------------------------------------
# Import the actual implementations (used by the wrappers above).
# These are imported AFTER the wrapper definitions to avoid circular
# references, and they are stored as private names so that
# monkeypatches on the public names (``wizard.run_wizard``, etc.)
# propagate through the wrappers.
# ---------------------------------------------------------------------------
from claude_codex_local.wizard_cli import (
    run_wizard as _cli_run_wizard,
    run_doctor as _cli_run_doctor,
    run_session as _cli_run_session,
    run_serve as _cli_run_serve,
    run_status as _cli_run_status,
    run_find_model_standalone as _cli_run_find_model_standalone,
    main as _cli_main,
)  # noqa: E402


def __getattr__(name: str):
    """Lazy attribute access — ensures sub-module references stay in sync."""
    if name == "_refresh_selected_engine":
        val = globals().get("_refresh_selected_engine", _rs_e)
        _wizard_steps_mod._refresh_selected_engine = val
        _wizard_discovery_mod._refresh_selected_engine = val  # type: ignore[attr-defined]
        _wizard_cli_mod._refresh_selected_engine = val  # type: ignore[attr-defined]
        return val
    if name == "_ensure_tool":
        val = globals().get("_ensure_tool", _et)
        _wizard_steps_mod._ensure_tool = val
        _wizard_discovery_mod._ensure_tool = val  # type: ignore[attr-defined]
        _wizard_cli_mod._ensure_tool = val  # type: ignore[attr-defined]
        return val
    if name == "questionary":
        val = globals().get("questionary", _q)
        _wizard_steps_mod.questionary = val
        _wizard_cli_mod.questionary = val  # type: ignore[attr-defined]
        _wizard_discovery_mod.questionary = val  # type: ignore[attr-defined]
        return val
    if name == "_build_oneshot_cmd":
        val = globals().get("_build_oneshot_cmd", _cli_build_oneshot_cmd)
        _wizard_cli_mod._build_oneshot_cmd = val  # type: ignore[attr-defined]
        return val
    if name == "_resolve_wire_env":
        val = globals().get("_resolve_wire_env", _cli_resolve_wire_env)
        _wizard_cli_mod._resolve_wire_env = val  # type: ignore[attr-defined]
        return val
    if name == "subprocess":
        val = globals().get("subprocess", subprocess)
        _wizard_steps_mod.subprocess = val
        _wizard_cli_mod.subprocess = val
        _wizard_discovery_mod.subprocess = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __setattr__(name: str, value):
    """Intercept attribute writes so monkeypatches propagate to sub-modules."""
    if name in (
        "_refresh_selected_engine",
        "_ensure_tool",
        "questionary",
        "_build_oneshot_cmd",
        "_resolve_wire_env",
        "subprocess",
    ):
        object.__setattr__(sys.modules[__name__], name, value)
        if name == "_refresh_selected_engine":
            _wizard_steps_mod._refresh_selected_engine = value
            _wizard_discovery_mod._refresh_selected_engine = value  # type: ignore[attr-defined]
            _wizard_cli_mod._refresh_selected_engine = value  # type: ignore[attr-defined]
        elif name == "_ensure_tool":
            _wizard_steps_mod._ensure_tool = value
            _wizard_discovery_mod._ensure_tool = value  # type: ignore[attr-defined]
            _wizard_cli_mod._ensure_tool = value  # type: ignore[attr-defined]
        elif name == "questionary":
            _wizard_steps_mod.questionary = value
            _wizard_cli_mod.questionary = value  # type: ignore[attr-defined]
            _wizard_discovery_mod.questionary = value  # type: ignore[attr-defined]
        elif name == "_build_oneshot_cmd":
            _wizard_cli_mod._build_oneshot_cmd = value  # type: ignore[attr-defined]
        elif name == "_resolve_wire_env":
            _wizard_cli_mod._resolve_wire_env = value  # type: ignore[attr-defined]
        elif name == "subprocess":
            _wizard_steps_mod.subprocess = value
            _wizard_cli_mod.subprocess = value
            _wizard_discovery_mod.subprocess = value
    else:
        object.__setattr__(sys.modules[__name__], name, value)


if __name__ == "__main__":
    sys.exit(main())
