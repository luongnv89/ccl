"""
Wizard steps — harness selection, engine selection, model picking,
smoke test, harness wiring, alias installation, verification, and guide generation.

Exports all step functions and their helpers.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from claude_codex_local import core as pb
from claude_codex_local.engines import ALL_ENGINES as _REGISTRY_ENGINES
from claude_codex_local.engines.pickers import get_picker
from claude_codex_local.wizard_discovery import (
    INSTALL_HINTS,
    _ALL_ENGINES,
    _ALL_HARNESSES,
    _ensure_llmfit,
    _ensure_tool,
    _is_model_compatible_with_engine,
    _persist_targeted_profile_update,
    _refresh_llmfit_for_profile,
    _refresh_selected_engine,
    _refresh_selected_harness,
    _show_install_hint,
    _show_selected_harness_status,
    _sync_presence_from_tools,
    _try_llmfit_fallback,
)
from claude_codex_local.wizard_state import STATE_DIR, STATE_FILE, WizardState, WireResult
from claude_codex_local.wizard_ui import console, fail, header, info, ok, warn

console = Console()

def step_2_select_harness(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2 — Select harness")
    presence = _sync_presence_from_tools(state.profile)
    harnesses = presence["harnesses"]

    if state.primary_harness:
        choice = state.primary_harness
        if choice not in _ALL_HARNESSES:
            fail(f"Unknown harness: {choice}")
            return False
        _refresh_selected_harness(state.profile, choice)
        harnesses = state.profile["presence"]["harnesses"]
        if choice not in harnesses:
            if non_interactive:
                fail(f"Forced harness {choice!r} is not installed.")
                return False
            if not _ensure_tool(choice):
                fail(f"Forced harness {choice!r} is still not available.")
                return False
            _refresh_selected_harness(state.profile, choice)
            harnesses = state.profile["presence"]["harnesses"]
        state.primary_harness = choice
        state.secondary_harnesses = [h for h in harnesses if h != choice]
        ok(f"Using forced primary harness: [bold]{state.primary_harness}[/bold]")
    elif non_interactive:
        if not harnesses:
            fail("No harness installed. Cannot continue in non-interactive mode.")
            return False
        live_harness = None
        for candidate in harnesses:
            if _refresh_selected_harness(state.profile, candidate):
                live_harness = candidate
                break
        if live_harness is None:
            fail("Cached harness data is stale; no configured harness is currently installed.")
            return False
        harnesses = state.profile["presence"]["harnesses"]
        state.primary_harness = live_harness
        state.secondary_harnesses = [h for h in harnesses if h != live_harness]
        ok(f"Non-interactive: picking [bold]{state.primary_harness}[/bold] as primary harness")
    else:
        harness_choices = [
            questionary.Choice(
                h if h in harnesses else f"{h}  [not installed]",
                value=h,
            )
            for h in _ALL_HARNESSES
        ]
        while True:
            choice = questionary.select(
                "Which harness do you want as primary?",
                choices=harness_choices,
                default=harnesses[0] if harnesses else _ALL_HARNESSES[0],
            ).ask()
            if choice is None:
                return False
            _refresh_selected_harness(state.profile, choice)
            harnesses = state.profile["presence"]["harnesses"]
            if choice not in harnesses:
                if not _ensure_tool(choice):
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
                _refresh_selected_harness(state.profile, choice)
                harnesses = state.profile["presence"]["harnesses"]
                if choice not in harnesses:
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
            state.primary_harness = choice
            state.secondary_harnesses = [h for h in harnesses if h != choice]
            break

    _show_selected_harness_status(state)
    state.mark("2")
    return True


# Engines that can either be run locally or pointed at a remote OpenAI-compatible
# endpoint. The wizard surfaces the local-vs-remote choice for these only —
# lmstudio is local-only, and 9router/openrouter are inherently remote.
_LOCAL_OR_REMOTE_ENGINES = ("ollama", "llamacpp", "vllm")

# Default ports applied automatically when the user enters a hostname or IP
# without an explicit port during the remote endpoint prompt.
_ENGINE_DEFAULT_PORTS: dict[str, int] = {
    "ollama": 11434,
    "llamacpp": 8001,
    "vllm": 8000,
}

# Canonical local base URLs per engine — used to reset in-process state when
# the user picks Local after a prior shell export (e.g. OLLAMA_HOST) seeded
# core.py's snapshotted constants with a remote URL at import time.
_ENGINE_LOCAL_BASE_URLS: dict[str, str] = {
    "ollama": f"http://localhost:{_ENGINE_DEFAULT_PORTS['ollama']}",
    "llamacpp": f"http://localhost:{_ENGINE_DEFAULT_PORTS['llamacpp']}",
    "vllm": f"http://localhost:{_ENGINE_DEFAULT_PORTS['vllm']}",
}


def _remote_env_var_names(engine: str) -> tuple[str, str | None]:
    """
    Return (URL env var name, API-key env var name or None) for the given
    engine. These are the names the existing remote-engine plumbing already
    reads at module load time (see core.py constants) and the names we
    write into the user's shell rc when they opt in to env-var persistence.
    """
    if engine == "ollama":
        return "OLLAMA_HOST", "OLLAMA_API_KEY"
    if engine == "llamacpp":
        return "LLAMACPP_BASE_URL", "LLAMACPP_API_KEY"
    if engine == "vllm":
        return "VLLM_BASE_URL", "VLLM_API_KEY"
    raise ValueError(f"engine {engine!r} does not support local-vs-remote selection")


def _apply_remote_endpoint(engine: str, url: str, api_key: str) -> None:
    """
    Push the user-supplied remote URL + API key into the live module state so
    every downstream code path (probe, wire, helper-script emission) reads
    the new values. The constants in core.py snapshot env vars at import time;
    we mirror the env vars AND the snapshotted constants so both the live
    `os.environ.get(...)` reads (vLLM key path) and the `pb.OLLAMA_BASE_URL`-
    style reads (Claude/Codex wire functions) pick up the choice without a
    module reload.
    """
    url_var, key_var = _remote_env_var_names(engine)
    os.environ[url_var] = url
    if engine == "ollama":
        pb.OLLAMA_BASE_URL = url
        if key_var is not None:
            os.environ[key_var] = api_key
            pb.OLLAMA_API_KEY = api_key
    elif engine == "llamacpp":
        pb.LLAMACPP_BASE_URL = url
        if key_var is not None:
            os.environ[key_var] = api_key
            pb.LLAMACPP_API_KEY = api_key
    elif engine == "vllm":
        pb.VLLM_BASE_URL = url
        if key_var is not None:
            os.environ[key_var] = api_key
            pb.VLLM_API_KEY = api_key


def _apply_local_endpoint(engine: str) -> str | None:
    """
    Reset in-process state to the canonical localhost URL for `engine`, and
    clear the matching env var so subsequent reads in this process don't see
    the previous remote value. Mirrors `_apply_remote_endpoint` for the
    Local branch of the wizard.

    Returns the URL that was previously in effect when it was a *remote*
    address that would have overridden the localhost default; otherwise
    None. Prior values that were themselves local (any host in
    127.0.0.0/8, ::1, or `*.localhost`) are not treated as overrides — a
    user exporting `OLLAMA_HOST=http://127.0.0.1:11434` does not need the
    "remove the export from your rc" warning.
    """
    url_var, key_var = _remote_env_var_names(engine)
    local_url = _ENGINE_LOCAL_BASE_URLS[engine]
    previous_env = os.environ.pop(url_var, None)
    previous_snapshot: str | None = None
    if engine == "ollama":
        previous_snapshot = pb.OLLAMA_BASE_URL
        pb.OLLAMA_BASE_URL = local_url
        if key_var is not None:
            os.environ.pop(key_var, None)
            pb.OLLAMA_API_KEY = ""
    elif engine == "llamacpp":
        previous_snapshot = pb.LLAMACPP_BASE_URL
        pb.LLAMACPP_BASE_URL = local_url
        if key_var is not None:
            os.environ.pop(key_var, None)
            pb.LLAMACPP_API_KEY = ""
    elif engine == "vllm":
        previous_snapshot = pb.VLLM_BASE_URL
        pb.VLLM_BASE_URL = local_url
        if key_var is not None:
            os.environ.pop(key_var, None)
            pb.VLLM_API_KEY = ""
    candidate = previous_env or (
        previous_snapshot if previous_snapshot and previous_snapshot != local_url else None
    )
    if candidate and pb._is_local_base_url(candidate):
        return None
    return candidate


def _env_block(engine: str, url: str, api_key: str) -> str:
    """Build the fenced shell-rc block of `export FOO=bar` lines for a remote engine."""
    url_var, key_var = _remote_env_var_names(engine)
    fence = f"claude-codex-local:remote:{engine}"
    body_lines = [
        f"# >>> {fence} >>>",
        "# Managed by claude-codex-local wizard. Re-run the wizard to update,",
        "# or delete this block to clear the remote endpoint env vars.",
        f"export {url_var}={shlex.quote(url)}",
    ]
    if key_var is not None and api_key:
        body_lines.append(f"export {key_var}={shlex.quote(api_key)}")
    body_lines.append(f"# <<< {fence} <<<")
    return "\n".join(body_lines) + "\n"


def _persist_remote_env_to_shell_rc(engine: str, url: str, api_key: str) -> Path | None:
    """
    Append (or replace) a fenced block of `export FOO=bar` lines in the user's
    shell rc so a fresh shell inherits the remote-endpoint env vars without
    re-running the wizard. Returns the rc path on success, None when no
    supported shell rc could be found.
    """
    block = _env_block(engine, url, api_key)
    rc_path = _detect_shell_rc()
    if rc_path is None:
        warn("Unsupported shell — please add the following to your shell rc manually:")
        console.print(block)
        return None
    fence = f"claude-codex-local:remote:{engine}"
    pattern = re.compile(
        rf"# >>> {re.escape(fence)} >>>.*?# <<< {re.escape(fence)} <<<\n?",
        re.DOTALL,
    )
    existing = rc_path.read_text() if rc_path.exists() else ""
    if pattern.search(existing):
        new_text = pattern.sub(block, existing, count=1)
    else:
        sep = "" if existing.endswith("\n") or not existing else "\n"
        prefix = "\n" if existing else ""
        new_text = existing + sep + prefix + block
    rc_path.write_text(new_text)
    ok(f"Persisted remote env vars into {rc_path}")
    return rc_path


def _prompt_remote_endpoint(engine: str) -> tuple[str, str] | None:
    """
    Interactive prompt for a remote endpoint base URL + optional API key.

    Returns (normalized_url, api_key) on success, or None if the user
    cancelled (Ctrl-C / Esc) at any step. The API key is "" for ollama
    (we do not ask) and for llamacpp/vllm when the user leaves it blank.

    URL validation: the input must be a bare base URL (scheme + host).
    A non-empty path/query/fragment is rejected and the user is re-prompted.
    If no port is specified, the engine-specific default port is applied
    automatically (Ollama: 11434, llama.cpp: 8001, vLLM: 8000).
    """
    while True:
        url_raw = questionary.text(
            f"Remote {engine} base URL (scheme + host, port optional):",
            default="",
        ).ask()
        if url_raw is None:
            return None
        url_raw = url_raw.strip()
        if not url_raw:
            warn("Base URL must not be empty.")
            continue
        # Reject path/query/fragment BEFORE normalization. The normalizer
        # strips them with a warning, but interactive users should get a
        # hard error so they re-type the right value instead of silently
        # losing the segment they intended.
        probe = url_raw
        if not probe.startswith(("http://", "https://")):
            probe = f"http://{probe}"
        parsed = urlparse(probe)
        if parsed.path or parsed.query or parsed.fragment:
            warn(
                "Base URL must not contain a path, query, or fragment "
                "(engine probes append their own paths). Got: "
                f"{url_raw!r}"
            )
            continue
        if not parsed.netloc:
            warn(f"Base URL is missing a host. Got: {url_raw!r}")
            continue
        # Apply engine default port when none is specified.
        if not parsed.port:
            default_port = _ENGINE_DEFAULT_PORTS.get(engine)
            if default_port:
                info(f"No port specified — using default port {default_port} for {engine}.")
                url_raw = f"{parsed.scheme}://{parsed.hostname}:{default_port}"
        url = pb._normalize_base_url(url_raw)
        api_key = ""
        if engine in ("llamacpp", "vllm"):
            # questionary.password masks input; allow empty (open endpoint).
            key_raw = questionary.password(
                f"{engine} API key (leave empty for no auth):",
                default="",
            ).ask()
            if key_raw is None:
                return None
            api_key = key_raw.strip()
        return url, api_key


def _prompt_local_or_remote(state: WizardState, engine: str) -> bool:
    """
    For `ollama`/`llamacpp`/`vllm`, ask whether to run the engine locally or
    point at a remote endpoint. Default Local. On Remote, also prompt for
    the base URL (and API key for llamacpp/vllm), apply the choice to live
    module state, and optionally persist the env vars to the user's shell
    rc.

    Returns True when remote was selected so the caller can skip the
    `_ensure_tool` local-install path; False for local (caller proceeds
    with the existing install flow).
    """
    choice = questionary.select(
        f"Run `{engine}` locally, or use a remote endpoint?",
        choices=["Local", "Remote"],
        default="Local",
    ).ask()
    if choice is None or choice == "Local":
        # Reset in-process state so the snapshotted base URL (which may have
        # been seeded from a shell export like OLLAMA_HOST at import time)
        # does not silently keep pointing at a remote server.
        overridden = _apply_local_endpoint(engine)
        if overridden:
            url_var, _ = _remote_env_var_names(engine)
            warn(
                f"Your shell exports {url_var}={overridden}; using "
                f"{_ENGINE_LOCAL_BASE_URLS[engine]} for this run. "
                f"Remove that export (or re-pick Local) next session to "
                f"avoid hitting the remote endpoint again."
            )
        return False

    prompt_result = _prompt_remote_endpoint(engine)
    if prompt_result is None:
        # User cancelled the URL/key prompt — fall back to local install path.
        return False
    url, api_key = prompt_result

    _apply_remote_endpoint(engine, url, api_key)
    ok(f"Remote {engine} endpoint configured: [bold]{url}[/bold]")

    # Re-probe the engine against the new URL so the engines presence list
    # picks it up (so the outer loop's `choice not in engines` check does
    # not bounce back into `_ensure_tool`).
    _refresh_selected_engine(state.profile, engine)

    # Default No — accidentally mutating the user's rc on a typo would be
    # confusing, and the helper script already encodes the choice anyway.
    persist = questionary.confirm(
        "Also persist these env vars to your shell rc?",
        default=False,
    ).ask()
    if persist:
        _persist_remote_env_to_shell_rc(engine, url, api_key)
    else:
        info("Skipping shell-rc persistence; env vars apply to this wizard run only.")
    return True


def step_3_select_engine(state: WizardState, non_interactive: bool = False) -> bool:
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


def _default_engine(engines: list[str], profile: dict[str, Any]) -> str:
    """
    Pick a sensible default engine.

    Rules:
      1. Prefer an engine that already has a coding model installed *and* is
         ready to serve (ollama server running, lmstudio server running).
      2. On Apple Silicon, prefer lmstudio when it's ready.
      3. Otherwise fall back to ollama, then whatever's first.
    """
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


# ---------------------------------------------------------------------------
# Step 4 — Pick a model (user-first, optional find-model helper)
# ---------------------------------------------------------------------------


_ROUTER9_DEFAULT_MODEL = "kr/claude-sonnet-4.5"
# Lenient model name regex: allows provider-prefixed names like
# "kr/claude-sonnet-4.5", "or/gpt-5", "kr/gpt-4o-mini" etc. The leading
# segment must match a 9router-style provider id; the suffix is free
# enough to admit dotted version numbers.
_ROUTER9_MODEL_RE = re.compile(r"^[a-z0-9_-]+/[A-Za-z0-9._-]+$")


def _step_4_pick_model_9router_impl(state: WizardState, non_interactive: bool = False) -> bool:
    """Step 4 specialisation for engine=9router.

    Skips llmfit/disk/download entirely — 9router routes to cloud models
    that aren't downloaded locally. Asks the user for an API key (or reads
    CCL_9ROUTER_API_KEY from env) and writes it to ROUTER9_KEY_FILE with
    chmod 0o600. Then asks for a model name with default kr/claude-sonnet-4.5.
    """
    pb.ensure_state_dirs()

    env_key = os.environ.get("CCL_9ROUTER_API_KEY", "").strip()
    if non_interactive:
        api_key = env_key
        if not api_key and pb.ROUTER9_KEY_FILE.exists():
            api_key = pb.ROUTER9_KEY_FILE.read_text().strip()
        if not api_key:
            fail(
                "9router API key required. Set CCL_9ROUTER_API_KEY or write "
                f"the key to {pb.ROUTER9_KEY_FILE} (chmod 600) before running "
                "non-interactively."
            )
            return False
    else:
        if env_key:
            api_key = env_key
            ok("Using 9router API key from CCL_9ROUTER_API_KEY env var.")
        else:
            api_key_input = questionary.password(
                "Paste your 9router API key (kept locally, chmod-600):",
            ).ask()
            if not api_key_input:
                fail("No API key provided. Cannot continue.")
                return False
            api_key = api_key_input.strip()

    pb.ROUTER9_KEY_FILE.write_text(api_key + "\n")
    pb.ROUTER9_KEY_FILE.chmod(0o600)
    ok(f"Wrote 9router API key to [bold]{pb.ROUTER9_KEY_FILE}[/bold] (chmod 0600).")

    env_model = os.environ.get("CCL_9ROUTER_MODEL", "").strip()
    if non_interactive:
        model_name = env_model or _ROUTER9_DEFAULT_MODEL
    else:
        interactive_model = _step_4_pick_9router_model_interactive(env_model)
        if interactive_model is None:
            return False
        model_name = interactive_model

    if len(model_name) > 256 or not _ROUTER9_MODEL_RE.match(model_name):
        fail(
            f"Invalid 9router model name: {model_name!r}. Expected "
            "<provider>/<model-id> (e.g. kr/claude-sonnet-4.5)."
        )
        return False

    state.engine_model_tag = model_name
    state.model_name = model_name
    state.model_source = "9router-direct"
    state.model_candidate = {}
    ok(f"Picked 9router model: [bold]{model_name}[/bold]")
    state.mark("4")
    return True


def _step_4_pick_model_9router(state: WizardState, non_interactive: bool = False) -> bool:
    """Backward-compatible wrapper — delegates to the implementation."""
    return _step_4_pick_model_9router_impl(state, non_interactive)


def _step_4_pick_9router_model_interactive(env_model: str) -> str | None:
    """Fetch 9router models for selection, falling back to manual entry."""
    prompt_default = env_model or _ROUTER9_DEFAULT_MODEL

    info("Fetching available models from 9router...")
    result = pb.smoke_test_router9_models()
    model_ids = [m for m in result.get("models", []) if isinstance(m, str) and m]

    if result.get("ok") and model_ids:
        choices: list[questionary.Choice] = [
            questionary.Choice(model_id, value=model_id) for model_id in model_ids
        ]
        choices.append(questionary.Separator())
        choices.append(questionary.Choice("Enter a model name manually instead", value=""))
        picked = questionary.select(
            "Select a 9router model (or choose manual entry):",
            choices=choices,
            default=env_model if env_model in model_ids else model_ids[0],
        ).ask()
        if picked:
            return picked.strip()
    else:
        if result.get("ok"):
            warn("9router returned zero models. Falling back to manual model entry.")
        else:
            warn(
                f"Could not fetch 9router model list: {result.get('error', 'unknown error')}. "
                "Falling back to manual model entry."
            )

    model_input = questionary.text(
        "9router model name:",
        default=prompt_default,
    ).ask()
    if not model_input:
        fail("No model name provided.")
        return None
    return model_input.strip()


_OPENROUTER_DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

# OpenRouter publishes variant IDs of the form `provider/model:variant`
# (e.g. `google/gemma-4-31b-it:free`, `mistralai/mistral-7b-instruct:nitro`).
# The trailing `:variant` suffix is mandatory for the free-tier catalog —
# we admit `:` in the model segment in addition to the 9router charset.
_OPENROUTER_MODEL_RE = re.compile(r"^[a-z0-9_-]+/[A-Za-z0-9._:-]+$")


def _format_context_length(ctx_len: int) -> str:
    """Format a context-length integer as a human-readable string."""
    if ctx_len >= 1_000_000:
        return f"{ctx_len // 1_000_000}M"
    if ctx_len >= 1000:
        return f"{ctx_len // 1000}k"
    return str(ctx_len)


def _step_4_pick_model_openrouter_impl(state: WizardState, non_interactive: bool = False) -> bool:
    """Step 4 specialisation for engine=openrouter.

    Skips llmfit/disk/download entirely — OpenRouter routes to cloud models
    that aren't downloaded locally. Asks the user for an API key (or reads
    CCL_OPENROUTER_API_KEY from env) and writes it to OPENROUTER_KEY_FILE
    with chmod 0o600. Then offers an interactive free-model browser that
    fetches the OpenRouter catalog, filters to free-tier entries, and
    displays a rich table (provider/model ID, context length, capabilities).
    Falls back to the text-input flow if the user declines or the API call
    fails (offline, rate-limited, empty response).
    """
    pb.ensure_state_dirs()

    env_key = os.environ.get("CCL_OPENROUTER_API_KEY", "").strip()
    if non_interactive:
        api_key = env_key
        if not api_key and pb.OPENROUTER_KEY_FILE.exists():
            api_key = pb.OPENROUTER_KEY_FILE.read_text().strip()
        if not api_key:
            fail(
                "OpenRouter API key required. Set CCL_OPENROUTER_API_KEY or "
                f"write the key to {pb.OPENROUTER_KEY_FILE} (chmod 600) "
                "before running non-interactively."
            )
            return False
    else:
        if env_key:
            api_key = env_key
            ok("Using OpenRouter API key from CCL_OPENROUTER_API_KEY env var.")
        else:
            api_key_input = questionary.password(
                "Paste your OpenRouter API key (kept locally, chmod-600):",
            ).ask()
            api_key = (api_key_input or "").strip()
            if not api_key:
                fail("No API key provided. Cannot continue.")
                return False

    pb.OPENROUTER_KEY_FILE.write_text(api_key + "\n")
    pb.OPENROUTER_KEY_FILE.chmod(0o600)
    ok(f"Wrote OpenRouter API key to [bold]{pb.OPENROUTER_KEY_FILE}[/bold] (chmod 0600).")

    env_model = os.environ.get("CCL_OPENROUTER_MODEL", "").strip()
    if non_interactive:
        model_name = env_model or _OPENROUTER_DEFAULT_MODEL
    else:
        browser_result = _step_4_openrouter_model_browser(env_model)
        if browser_result is None:
            prompt_default = env_model or _OPENROUTER_DEFAULT_MODEL
            model_input = questionary.text(
                "OpenRouter model name:",
                default=prompt_default,
            ).ask()
            if not model_input:
                fail("No model name provided.")
                return False
            model_name = model_input.strip()
        else:
            model_name = browser_result

    if len(model_name) > 256 or not _OPENROUTER_MODEL_RE.match(model_name):
        fail(
            f"Invalid OpenRouter model name: {model_name!r}. Expected "
            "<provider>/<model-id>[:variant] (e.g. anthropic/claude-sonnet-4.6 "
            "or google/gemma-4-31b-it:free)."
        )
        return False

    state.engine_model_tag = model_name
    state.model_name = model_name
    state.model_source = "openrouter-direct"
    state.model_candidate = {}
    ok(f"Picked OpenRouter model: [bold]{model_name}[/bold]")
    state.mark("4")
    return True


def _step_4_pick_model_openrouter(state: WizardState, non_interactive: bool = False) -> bool:
    """Backward-compatible wrapper — delegates to the implementation."""
    return _step_4_pick_model_openrouter_impl(state, non_interactive)


def _step_4_openrouter_model_browser(env_model: str) -> str | None:
    """Interactive free-model browser for OpenRouter.

    Asks the user whether they want to browse free models. If yes, fetches
    the catalog, filters to free-tier, displays a rich table, and returns
    the selected model ID. Returns None if the user declines or the fetch
    fails (caller falls back to text input).
    """
    browse = questionary.confirm(
        "Would you like to browse free models from OpenRouter before entering a model name?",
        default=True,
    ).ask()
    if not browse:
        return None

    info("Fetching free models from OpenRouter...")
    result = pb.fetch_openrouter_free_models()

    if not result.get("ok"):
        warn(
            f"Could not fetch free model list: {result.get('error', 'unknown error')}. "
            "Falling back to manual model entry."
        )
        return None

    free_models: list[dict[str, Any]] = result.get("models", [])
    if not free_models:
        warn("OpenRouter returned zero free models. Falling back to manual model entry.")
        return None

    ok(f"Found {len(free_models)} free models.")

    # --- Display table ---
    table = Table(
        title="OpenRouter Free Models",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Provider / Model ID", style="green", width=42)
    table.add_column("Context", style="yellow", width=8)
    table.add_column("Capabilities", style="magenta", width=30)

    for idx, model in enumerate(free_models, 1):
        ctx_str = _format_context_length(model.get("context_length", 0))
        caps = ", ".join(model.get("capabilities", ["text"]))
        table.add_row(str(idx), model["id"], ctx_str, caps)

    console.print(table)
    console.print()

    # --- Build choices ---
    choices: list[questionary.Choice] = []
    for idx, model in enumerate(free_models, 1):
        ctx_str = _format_context_length(model.get("context_length", 0))
        caps = ", ".join(model.get("capabilities", ["text"]))
        label = f"{idx:>3}. {model['id']:<42s}  ctx={ctx_str:<6s}  [{caps}]"
        choices.append(questionary.Choice(label, value=idx - 1))

    choices.append(questionary.Separator())
    choices.append(questionary.Choice("Enter a model name manually instead", value=-1))

    selected = questionary.select(
        "Select a free model (or choose manual entry):",
        choices=choices,
    ).ask()

    if selected is None or selected == -1:
        return None

    return free_models[selected]["id"]


def _step_4_pick_model_vllm_impl(state: WizardState, non_interactive: bool = False) -> bool:
    """Step 4 specialisation for engine=vllm.

    A running vLLM server already has its model loaded (`vllm serve <id>`),
    so there's nothing to download and no llmfit/disk math to do — we just
    confirm which loaded model to talk to. /v1/models is queried via
    vllm_info() and the user picks one.  With a single loaded model (the
    common case) we skip the prompt.
    """
    profile_vllm = state.profile.get("vllm", {})
    if not profile_vllm.get("present"):
        fail(
            f"vLLM not reachable at {pb.VLLM_BASE_URL}. "
            "Start `vllm serve <model>` before re-running the wizard."
        )
        return False

    models = [m.get("name", "") for m in profile_vllm.get("models", []) if m.get("name")]
    env_model = os.environ.get("CCL_VLLM_MODEL", "").strip()

    if env_model:
        if models and env_model not in models:
            warn(
                f"CCL_VLLM_MODEL={env_model!r} is not in the loaded model list "
                f"({', '.join(models) or 'none'}); using it anyway."
            )
        choice = env_model
    elif not models:
        fail(
            f"vLLM at {profile_vllm.get('base_url')} reports zero loaded models. "
            "Restart it with `vllm serve <hf-model-id>` and re-run the wizard."
        )
        return False
    elif len(models) == 1:
        choice = models[0]
        ok(f"Using the single model loaded by vLLM: [bold]{choice}[/bold]")
    elif non_interactive:
        choice = models[0]
        ok(f"Non-interactive: picking [bold]{choice}[/bold] from {len(models)} loaded models")
    else:
        picked = questionary.select(
            "Which vLLM-loaded model do you want to use?",
            choices=models,
            default=models[0],
        ).ask()
        if not picked:
            return False
        choice = picked

    state.engine_model_tag = choice
    state.model_name = choice
    state.model_source = "vllm-loaded"
    state.model_candidate = {}
    ok(f"Picked vLLM model: [bold]{choice}[/bold]")
    state.mark("4")
    return True


def _step_4_pick_model_vllm(state: WizardState, non_interactive: bool = False) -> bool:
    """Backward-compatible wrapper — delegates to the implementation."""
    return _step_4_pick_model_vllm_impl(state, non_interactive)


def _step_4_pick_model_remote_impl(
    engine: str, state: WizardState, non_interactive: bool = False
) -> bool:
    """Step 4 specialisation for engine=ollama or engine=llamacpp with a
    remote endpoint.

    When the user has configured a remote server URL, the model picker
    fetches the available models from that server's API endpoint and shows
    only those as selectable options — static suggestions and llmfit
    profiles are irrelevant because the remote server determines which
    models are actually available.

    Fallback: if the server is unreachable or reports zero models, show an
    error message and allow the user to enter a model name manually.
    """
    info(f"Remote {engine} endpoint detected — fetching available models from the server...")

    profile = state.profile.get(engine, {})
    profile_models = profile.get("models", [])

    remote_models = [
        {"name": m["name"], "local": False}
        for m in profile_models
        if isinstance(m, dict) and m.get("name")
    ]

    if non_interactive:
        if remote_models:
            choice = remote_models[0]["name"]
            ok(f"Non-interactive: picking [bold]{choice}[/bold] from remote server")
        else:
            fail(f"Remote {engine} server unreachable and no model name was explicitly provided.")
            return False
    else:
        if remote_models:
            choices = [
                questionary.Choice(
                    m["name"],
                    value=m["name"],
                )
                for m in remote_models
            ]
            choices.append(questionary.Separator("── Other ──"))
            choices.append(questionary.Choice("I'll type a different model name", value="direct"))
            choices.append(questionary.Choice("Cancel setup", value="cancel"))
            picked = questionary.select(
                f"Which model does the remote {engine} server serve?",
                choices=choices,
            ).ask()
            if picked is None or picked == "cancel":
                fail("Setup cancelled by user.")
                return False
            if picked == "direct":
                name = questionary.text(
                    f"Model name for {engine} (e.g. qwen3-coder:30b):",
                ).ask()
                if not name:
                    fail("No model name provided.")
                    return False
                choice = name.strip()
            else:
                choice = picked
        else:
            base_url = profile.get("base_url", "")
            warn(
                f"Remote {engine} server at {base_url} is unreachable or "
                "reports zero available models."
            )
            name = questionary.text(
                f"Model name for {engine} (manual entry — server unreachable):",
            ).ask()
            if not name:
                fail("No model name provided.")
                return False
            choice = name.strip()

    state.engine_model_tag = choice
    state.model_name = choice
    state.model_source = f"{engine}-remote"
    state.model_candidate = {}
    ok(f"Picked remote {engine} model: [bold]{choice}[/bold]")
    state.mark("4")
    return True


def _step_4_pick_model_remote(
    state: WizardState, engine: str, non_interactive: bool = False
) -> bool:
    """Backward-compatible wrapper — delegates to the implementation."""
    return _step_4_pick_model_remote_impl(engine, state, non_interactive)


def _step_4_pick_model_local_impl(
    state: WizardState,
    non_interactive: bool = False,
) -> bool:
    """Local model picker for ollama / lmstudio / llama.cpp.

    Handles llmfit profiles, merged model lists, running-server detection
    for llama.cpp, and interactive / non-interactive model selection.
    """
    engine = state.primary_engine
    running_llamacpp_model: str | None = None
    if engine == "llamacpp":
        status = pb.llamacpp_info()
        if status.get("server_running") and status.get("model"):
            running_llamacpp_model = status["model"]
            info(
                f"Detected running llama-server on port {status['server_port']} "
                f"serving model [bold]{running_llamacpp_model}[/bold]."
            )

    if non_interactive:
        if running_llamacpp_model:
            state.model_name = running_llamacpp_model
            state.engine_model_tag = running_llamacpp_model
            state.model_source = "running-server"
            state.model_candidate = {}
            ok(
                f"Non-interactive pick: [bold]{state.engine_model_tag}[/bold] (from running llama-server)"
            )
            state.mark("4")
            return True
        candidate = _find_model_auto(engine, state.profile)
        if not candidate:
            fail("Non-interactive find-model failed and no direct model was provided.")
            return False
        state.model_name = candidate["display"]
        state.engine_model_tag = candidate["tag"]
        state.model_source = "find-model"
        state.model_candidate = candidate.get("candidate") or {}
        ok(f"Non-interactive pick: [bold]{state.engine_model_tag}[/bold]")
    else:
        merged_models = pb.merge_models_for_engine(state.profile, engine)
        profile_recommendations = _build_profile_recommendations(engine, state.profile)
        _show_profile_recommendations_preview(profile_recommendations)
        while True:
            choices: list[Any] = []
            items: dict[str, dict[str, Any]] = {}

            if running_llamacpp_model:
                choices.append(questionary.Separator("── Running server ──"))
                choices.append(
                    questionary.Choice(
                        f"Use running llama-server model: {running_llamacpp_model}",
                        value="running",
                    )
                )

            profile_entries: list[questionary.Choice] = []
            for pmode in pb.RECOMMENDATION_MODES:
                rec = profile_recommendations.get(pmode)
                if rec is None:
                    continue
                key = f"profile:{pmode}"
                items[key] = rec
                profile_entries.append(
                    questionary.Choice(
                        _profile_choice_label(pmode, rec),
                        value=key,
                    )
                )
            if profile_entries:
                choices.append(questionary.Separator("── Suggested by llmfit ──"))
                choices.extend(profile_entries)
                info(
                    "Speed/Quality/Balanced profiles come from llmfit's ranking of coding models "
                    f"for your {engine} engine."
                )

            merged_entries: list[questionary.Choice] = []
            for idx, entry in enumerate(merged_models):
                if running_llamacpp_model and entry.get("running"):
                    continue
                key = f"merged:{idx}"
                items[key] = entry
                size_suffix = f"  ({entry.get('size')})" if entry.get("size") else ""
                source = entry.get("source", "installed")
                if source == "installed":
                    label = f"Use installed model: {entry['display']}{size_suffix}"
                else:
                    label = f"Cached — not yet downloaded: {entry['display']}{size_suffix}"
                merged_entries.append(questionary.Choice(label, value=key))
            if merged_entries:
                choices.append(questionary.Separator("── Installed or recommended ──"))
                choices.extend(merged_entries)
            else:
                llmfit_skipped = pb._is_llmfit_skipped(state.profile.get("llmfit_system"))
                if llmfit_skipped:
                    info(
                        "No installed models detected and the llmfit hardware scan "
                        "is deferred. Pick 'Run llmfit now' below or re-run setup "
                        "with --run-llmfit to populate recommendations."
                    )
                else:
                    info(
                        "No installed models detected for this engine. Use 'Help me "
                        "pick' or type a model name directly."
                    )

            llmfit_skipped = pb._is_llmfit_skipped(state.profile.get("llmfit_system"))
            if llmfit_skipped:
                choices.append(questionary.Separator("── Refresh ──"))
                choices.append(
                    questionary.Choice(
                        "Run llmfit now to refresh recommendations",
                        value="refresh-llmfit",
                    )
                )

            choices.append(questionary.Separator("── Other ──"))
            choices.extend(
                [
                    questionary.Choice("I'll type a specific model name", value="direct"),
                    questionary.Choice(
                        "Help me pick (full llmfit ranked list)", value="find-model"
                    ),
                    questionary.Choice("Cancel setup", value="cancel"),
                ]
            )
            mode = questionary.select(
                "How do you want to choose the model?",
                choices=choices,
            ).ask()
            if mode is None or mode == "cancel":
                fail("Setup cancelled by user.")
                return False
            if mode == "refresh-llmfit":
                pb.invalidate_machine_profile_inproc_cache()
                refreshed = pb.machine_profile(run_llmfit=True)
                state.profile = refreshed
                merged_models = pb.merge_models_for_engine(refreshed, engine)
                profile_recommendations = _build_profile_recommendations(engine, refreshed)
                if pb._is_llmfit_skipped(refreshed.get("llmfit_system")):
                    warn(
                        "llmfit could not be run (binary missing). "
                        "Install llmfit or pick a model directly."
                    )
                else:
                    ok("Hardware scan refreshed.")
                    _show_profile_recommendations_preview(profile_recommendations)
                continue
            if mode == "running" and running_llamacpp_model:
                state.model_name = running_llamacpp_model
                state.engine_model_tag = running_llamacpp_model
                state.model_source = "running-server"
                state.model_candidate = {}
                ok(f"Using running llama-server model: [bold]{running_llamacpp_model}[/bold]")
                break
            if mode.startswith("profile:"):
                pmode = mode.split(":", 1)[1]
                rec = items[mode]
                state.model_name = rec.get("name") or rec["engine_tag"]
                state.engine_model_tag = rec["engine_tag"]
                state.model_source = f"profile:{pmode}"
                state.model_candidate = {
                    k: v for k, v in rec.items() if k not in ("engine_tag", "mode")
                }
                ok(
                    f"Picked {pmode} profile: [bold]{state.engine_model_tag}[/bold] "
                    f"(score={rec.get('score')}, ~{rec.get('estimated_tps')} tok/s)"
                )
                if _handle_model_presence(state):
                    break
                continue
            if mode.startswith("merged:"):
                entry = items[mode]
                state.model_name = entry["display"]
                state.engine_model_tag = entry["tag"]
                if entry.get("source") == "cached":
                    state.model_source = "find-model"
                    state.model_candidate = entry.get("candidate") or {}
                    ok(f"Picked cached recommendation: [bold]{state.engine_model_tag}[/bold]")
                else:
                    state.model_source = "installed"
                    state.model_candidate = {}
                    ok(f"Using installed model: [bold]{state.engine_model_tag}[/bold]")
                if _handle_model_presence(state):
                    break
                continue
            if mode == "direct":
                name = questionary.text(
                    f"Model name for engine '{engine}' (e.g. qwen3-coder:30b):",
                ).ask()
                if not name:
                    continue
                state.model_name = name.strip()
                state.engine_model_tag = _map_to_engine(name.strip(), engine) or name.strip()
                state.model_source = "direct"
            else:
                picked = _find_model_interactive(engine, state.profile)
                if not picked:
                    continue
                state.model_name = picked["display"]
                state.engine_model_tag = picked["tag"]
                state.model_source = "find-model"
                state.model_candidate = picked.get("candidate") or {}

            if _handle_model_presence(state):
                break

    state.mark("4")
    return True


def step_2_4_pick_model(
    state: WizardState,
    non_interactive: bool = False,
    run_llmfit_flag: bool = False,
) -> bool:
    header("Step 4 — Pick a model")
    engine = state.primary_engine

    # Remote engine branch: when Ollama or llama.cpp has a remote endpoint
    # configured, fetch models from the server instead of showing static
    # suggestions / llmfit profiles (which are only meaningful for local).
    if engine == "ollama" and pb._is_local_base_url(pb.ollama_base_url()) is False:
        return _step_4_pick_model_remote_impl("ollama", state, non_interactive)

    if engine == "llamacpp" and pb._is_local_base_url(pb.llamacpp_base_url()) is False:
        return _step_4_pick_model_remote_impl("llamacpp", state, non_interactive)

    # Delegate to the engine-specific picker strategy.
    picker = get_picker(engine)
    if picker is not None:
        return picker.pick_model(state, non_interactive)

    fail(f"No model picker registered for engine '{engine}'.")
    return False


def _build_profile_recommendations(
    engine: str, profile: dict[str, Any]
) -> dict[str, dict[str, Any] | None]:
    """
    Return per-mode llmfit recommendations mapped to `engine`.

    Missing llmfit → every mode maps to None (the picker silently omits the
    profile options in that case, avoiding a crash when llmfit is not
    installed). The caller can show a hint suggesting `_ensure_llmfit()`
    through the existing "Help me pick" path.
    """
    llmfit_present = pb.command_version("llmfit").get("present", False)
    out: dict[str, dict[str, Any] | None] = {m: None for m in pb.RECOMMENDATION_MODES}
    if not llmfit_present:
        return out
    for m in pb.RECOMMENDATION_MODES:
        try:
            out[m] = pb.recommend_for_mode(profile, m, engine)
        except Exception:
            out[m] = None
    return out


def _profile_choice_label(mode: str, rec: dict[str, Any]) -> str:
    """Human-readable single-line label for a recommendation profile choice."""
    title = {
        "balanced": "Balanced profile",
        "fast": "Speed profile",
        "quality": "Quality profile",
    }.get(mode, f"{mode.title()} profile")
    tag = rec.get("engine_tag") or rec.get("name") or "?"
    score = rec.get("score")
    tps = rec.get("estimated_tps")
    fit = rec.get("fit_level", "?")
    bits = [f"→ {tag}"]
    if score is not None:
        bits.append(f"score={score}")
    if tps is not None:
        bits.append(f"~{tps} tok/s")
    if fit:
        bits.append(f"fit={fit}")
    return f"{title}  " + "  ".join(bits)


def _show_profile_recommendations_preview(
    recommendations: dict[str, dict[str, Any] | None],
) -> None:
    """
    Print a small table summarising the Speed/Balanced/Quality recommendations
    before the picker menu appears. Each row documents the speed/quality
    tradeoff (issue #35 acceptance criterion).
    """
    if not any(recommendations.values()):
        info(
            "Recommendation profiles require llmfit. Install llmfit or pick "
            "an installed model / type a model name below."
        )
        return
    table = Table(
        title="Recommendation profiles (llmfit + this machine)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Profile", style="bold")
    table.add_column("Recommended model")
    table.add_column("Score", justify="right")
    table.add_column("~tok/s", justify="right")
    table.add_column("Notes", overflow="fold")
    for pmode in pb.RECOMMENDATION_MODES:
        rec = recommendations.get(pmode)
        note = pb.RECOMMENDATION_MODE_DESCRIPTIONS.get(pmode, "")
        if rec is None:
            table.add_row(pmode.title(), "—", "—", "—", note + "  (no match)")
            continue
        tag = rec.get("engine_tag") or rec.get("name") or "?"
        score = rec.get("score")
        tps = rec.get("estimated_tps")
        table.add_row(
            pmode.title(),
            str(tag),
            "—" if score is None else str(score),
            "—" if tps is None else f"{tps}",
            note,
        )
    console.print(table)


def _map_to_engine(user_input: str, engine: str) -> str | None:
    """Map a free-form user model name to the engine's naming scheme."""
    if engine == "ollama":
        # If it already looks like an ollama tag, keep it.
        if ":" in user_input and "/" not in user_input:
            return user_input
        return pb.hf_name_to_ollama_tag(user_input)
    if engine == "lmstudio":
        # LM Studio accepts hub names directly.
        if "/" in user_input:
            return user_input
        return pb.hf_name_to_lms_hub(user_input)
    return user_input


def _find_model_auto(engine: str, profile: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    Non-interactive model pick. Prefers a model that is *already installed* for
    the chosen engine over a new download, because downloads in non-interactive
    mode are almost always unwanted.
    """
    profile = profile or pb.machine_profile()

    # 1. Already-installed model for this engine — most useful default.
    if engine == "ollama":
        installed = [
            m["name"] for m in profile.get("ollama", {}).get("models", []) if m.get("local")
        ]
        # Prefer recognisable coding models first.
        for preferred in (
            "qwen3-coder",
            "qwen2.5-coder",
            "deepseek-coder",
            "codellama",
            "gemma4",
            "qwen3.5",
        ):
            for name in installed:
                if preferred in name.lower():
                    return {
                        "display": name,
                        "tag": name,
                        "score": None,
                        "candidate": {"name": name},
                    }
        if installed:
            return {
                "display": installed[0],
                "tag": installed[0],
                "score": None,
                "candidate": {"name": installed[0]},
            }
    elif engine == "lmstudio":
        lms_models = profile.get("lmstudio", {}).get("models", [])
        for m in lms_models:
            path = m.get("path", "")
            if any(p in path.lower() for p in ("coder", "code")):
                return {"display": path, "tag": path, "score": None, "candidate": {"name": path}}
        if lms_models:
            path = lms_models[0]["path"]
            return {"display": path, "tag": path, "score": None, "candidate": {"name": path}}

    # 2. Fall back to llmfit's top candidate that maps to this engine.
    candidates = pb.llmfit_coding_candidates(ram_gb=pb._available_ram_gb(profile))
    for c in candidates:
        tag = _candidate_tag(c, engine)
        if tag:
            return {"display": c["name"], "tag": tag, "score": c.get("score"), "candidate": c}
    return None


def _find_model_interactive(
    engine: str, profile: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if not _ensure_llmfit():
        return None
    info("Running llmfit to rank coding models for this machine...")
    ram_gb = pb._available_ram_gb(profile) if profile else None
    candidates = pb.llmfit_coding_candidates(ram_gb=ram_gb)
    if not candidates:
        fail("llmfit returned no coding candidates.")
        return None

    choices: list[questionary.Choice] = []
    items: list[dict[str, Any]] = []
    for c in candidates[:15]:
        tag = _candidate_tag(c, engine)
        if not tag:
            continue
        label = (
            f"{c['name']:60s}  score={c.get('score'):>3}  "
            f"fit={c.get('fit_level', '?'):<12s}  ~{c.get('estimated_tps', '?')} tok/s"
        )
        items.append({"display": c["name"], "tag": tag, "score": c.get("score"), "candidate": c})
        choices.append(questionary.Choice(label, value=len(items) - 1))

    if not choices:
        fail(f"No candidates map to engine '{engine}'. Try another engine or a direct model name.")
        return None

    idx = questionary.select(
        f"Pick a model for {engine}:",
        choices=choices,
    ).ask()
    if idx is None:
        return None
    return items[idx]


def _candidate_tag(c: dict[str, Any], engine: str) -> str | None:
    if engine == "ollama":
        return c.get("ollama_tag")
    if engine == "lmstudio":
        return c.get("lms_hub_name") or c.get("lms_mlx_path")
    if engine == "llamacpp":
        # llama.cpp consumes GGUF hf references; fall back to the raw name.
        return c.get("name")
    return c.get("name")


def _handle_model_presence(state: WizardState) -> bool:
    """
    Check whether the chosen model is already on disk, and if not, handle the
    disk-aware download branches. Returns True when the step should move on.
    """
    engine = state.primary_engine
    tag = state.engine_model_tag
    if _model_already_installed(engine, tag, state.profile):
        ok(f"Model [bold]{tag}[/bold] is already installed on this machine.")
        return True

    size_bytes = _estimate_model_size(state)
    free_bytes = state.profile.get("disk", {}).get("free_bytes", 0)
    size_gib = size_bytes / (1024**3) if size_bytes else None
    free_gib = free_bytes / (1024**3) if free_bytes else 0

    if size_gib is not None:
        info(f"Estimated model size: {size_gib:.1f} GiB. Free disk: {free_gib:.1f} GiB.")
    else:
        info(f"Estimated model size: unknown. Free disk: {free_gib:.1f} GiB.")

    fits = size_gib is None or size_gib < free_gib * 0.9

    if not fits:
        warn(
            f"Model does not comfortably fit in free disk space ({size_gib:.1f} GiB needed, {free_gib:.1f} GiB free)."
        )
        cont = questionary.confirm(
            "Free up space and continue with this model?",
            default=False,
        ).ask()
        if not cont:
            return False  # re-ask

    confirm = questionary.confirm(
        f"Download '{tag}' via {engine} now?",
        default=True,
    ).ask()
    if not confirm:
        return False  # re-ask

    return _download_model(state)


def _model_already_installed(engine: str, tag: str, profile: dict[str, Any]) -> bool:
    if engine == "ollama":
        return any(m.get("name") == tag for m in profile.get("ollama", {}).get("models", []))
    if engine == "lmstudio":
        return any(m.get("path") == tag for m in profile.get("lmstudio", {}).get("models", []))
    if engine == "llamacpp":
        # A llamacpp tag picked from the "installed" list is an absolute path
        # to a GGUF file on disk — if the file exists, the model is installed
        # and we must not re-prompt the user to download it.
        if tag and tag.startswith("/") and Path(tag).is_file():
            return True
        # Otherwise fall back to the running-server check. Use the same
        # loose matcher the rest of the wizard uses for HF tag vs. served
        # GGUF basename — exact equality is a false negative for almost
        # every direct-typed tag (the server reports a file basename like
        # `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` while the saved tag is the HF
        # repo id `unsloth/Qwen3.6-35B-A3B-GGUF`).
        status = pb.llamacpp_info()
        if not status.get("server_running"):
            return False
        return _llamacpp_models_match(status.get("model") or "", tag)
    return False


def _estimate_model_size(state: WizardState) -> int | None:
    """
    Best-effort byte estimate for the chosen model, via llmfit.

    Order of preference:
      1. Already-captured llmfit candidate (find-model path)
      2. `llmfit info <model_name>` lookup (direct-input path)
    Returns None when llmfit is unavailable or the lookup is ambiguous.
    """
    if state.model_candidate:
        size = pb.llmfit_estimate_size_bytes(state.model_candidate)
        if size:
            return size
    # Direct input or candidate was missing size fields — try a fresh lookup.
    if state.model_name:
        return pb.llmfit_estimate_size_bytes(state.model_name)
    return None


def _largest_gguf_in(directory: Path) -> str | None:
    """Return the absolute path of the largest .gguf under ``directory``, or None."""
    if not directory.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.rglob("*.gguf"):
        try:
            candidates.append((path.stat().st_size, path))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return str(candidates[0][1])


def _render_llama_server_command(argv: list[str]) -> str:
    """Render an argv as a shell-paste-ready command line."""
    return " ".join(shlex.quote(part) for part in argv)


def _collect_gguf_variants(files: list[str]) -> list[dict[str, str]]:
    """
    Group a repo's GGUF file listing into pickable single-quant download units.

    Each returned entry is one variant the user can choose:
      - top-level file ``Foo-Q4_K_M.gguf``  → {kind: "file", spec: filename}
      - sharded folder ``BF16/model-…``     → {kind: "include", spec: "BF16/*"}

    Sharded folders collapse all of their shards into a single entry so we
    download the whole shard set together (pulling shard 1 of 12 alone is
    useless). Existing index/config files in the same folder are excluded
    from the variant list itself but will be pulled in by the include glob.

    Returns variants sorted alphabetically by label. Empty list when ``files``
    contains no GGUFs.
    """
    top_files: list[str] = []
    sharded_dirs: dict[str, int] = {}
    for f in files:
        if not f.lower().endswith(".gguf"):
            continue
        parent, _, _name = f.rpartition("/")
        if parent:
            sharded_dirs[parent] = sharded_dirs.get(parent, 0) + 1
        else:
            top_files.append(f)
    variants: list[dict[str, str]] = []
    for f in top_files:
        variants.append({"label": f, "kind": "file", "spec": f})
    for d, n in sharded_dirs.items():
        variants.append(
            {
                "label": f"{d}/ (sharded, {n} shards)",
                "kind": "include",
                "spec": f"{d}/*",
            }
        )
    variants.sort(key=lambda v: v["label"].lower())
    return variants


# Preferred quant when the picker auto-selects a default — Q4_K_M is the
# canonical "good balance" quant for most consumer hardware. Falls back to
# the first variant alphabetically when no Q4_K_M is present.
_DEFAULT_QUANT_HINT = "q4_k_m"


def _default_variant_label(variants: list[dict[str, str]]) -> str | None:
    """Pick a sensible default label for the quant picker, or None if empty."""
    if not variants:
        return None
    for v in variants:
        if _DEFAULT_QUANT_HINT in v["label"].lower():
            return v["label"]
    return variants[0]["label"]


def _prompt_gguf_variant(repo_id: str, variants: list[dict[str, str]]) -> dict[str, str] | None:
    """Show a picker over GGUF variants in ``repo_id``; return the chosen one or None on cancel."""
    console.print(
        f"\n[cyan]'{repo_id}' contains {len(variants)} GGUF variants.[/cyan] "
        f"Pick one — downloading the whole repo would pull every quant ({len(variants)} files / folders)."
    )
    choices: list[Any] = [questionary.Choice(v["label"], value=v["label"]) for v in variants]
    choices.append(questionary.Choice("Cancel", value="__cancel__"))
    pick = questionary.select(
        "Select a quant to download:",
        choices=choices,
        default=_default_variant_label(variants),
    ).ask()
    if pick is None or pick == "__cancel__":
        return None
    for v in variants:
        if v["label"] == pick:
            return v
    return None


def _download_gguf_via_hf_cli(repo_id: str) -> dict:
    """
    Download a GGUF model from Hugging Face Hub using the HuggingFace CLI.

    repo_id may be:
      - A bare repo like "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
        (downloads entire repo; the CLI picks the right files)
      - A repo + filename like "org/repo filename.gguf"
        (downloads the specific file)

    Shows the HF CLI's native progress bar (bytes / speed / ETA) by inheriting
    stdout. On completion prints a summary with total bytes and elapsed time
    (issue #39). When the repo cannot be found, falls back to a fuzzy search
    of the Hub and lets the user pick from up to 3 close matches or re-enter a
    different name (issue #38).

    Returns {"ok": bool, "path": str|None, "repo_id": str|None}. ``repo_id`` is
    the resolved repo ID that was actually downloaded — it may differ from the
    caller's input when the user picked a fuzzy-search suggestion.
    """
    if not pb.huggingface_cli_detect().get("present"):
        warn("HuggingFace CLI (hf / huggingface-cli) is not installed.")
        _show_install_hint("huggingface-cli")
        install = questionary.confirm(
            "Install huggingface_hub[cli] now via pip?",
            default=True,
        ).ask()
        if install:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                fail(f"pip install failed: {exc}")
                return {"ok": False, "path": None, "repo_id": None}
            # pip installs the CLI binary into the same scripts directory as
            # the running Python interpreter.  That directory may not be on
            # the current process PATH yet (the user hasn't sourced their
            # shell profile), so we add it explicitly before re-checking.
            scripts_dir = sysconfig.get_path("scripts")
            if scripts_dir:
                path_env = os.environ.get("PATH", "")
                if scripts_dir not in path_env.split(os.pathsep):
                    os.environ["PATH"] = scripts_dir + os.pathsep + path_env
        if not pb.huggingface_cli_detect().get("present"):
            warn(
                "HuggingFace CLI still not found after install attempt.\n"
                "Re-run the wizard with --resume once it is available."
            )
            return {"ok": False, "path": None, "repo_id": None}

    current = repo_id
    # Cap fuzzy-search re-entries so a pathological input cannot loop forever.
    for _attempt in range(5):
        # Split "repo_id filename.gguf" if the caller passed both in one string.
        parts = current.split(None, 1)
        hf_repo = parts[0]
        filename = parts[1] if len(parts) > 1 else None
        include: str | None = None

        # Pre-flight (#58, #60): when the user is fetching a whole repo,
        # enumerate the repo's files first. Two failure modes to head off:
        #   (a) MLX-only / safetensors-only repo with zero GGUFs — catastrophic
        #       80+ GiB of unusable files (#58). Redirect via fuzzy search.
        #   (b) GGUF mirror with many quants (e.g. unsloth's 30+ variants) —
        #       a bare `hf download <repo>` mirrors *all* of them, easily 1+ TB
        #       (#60). Show a picker so the user grabs one quant.
        # When the user already pinned a specific filename we trust them.
        # When the listing is empty (network blip / private repo / API change)
        # we treat it as ambiguous and proceed without a filter — better to
        # let the CLI fail downstream than block a real download on an HF
        # outage. The user can always Ctrl-C if it goes sideways.
        if filename is None:
            repo_files = pb.huggingface_list_repo_files(hf_repo)
            variants = _collect_gguf_variants(repo_files) if repo_files else []
            if repo_files and not variants:
                warn(
                    f"Repo '{hf_repo}' contains no .gguf files — "
                    f"llama.cpp can only load GGUF models.\n"
                    f"  Try a GGUF mirror such as 'bartowski/<model>-GGUF' "
                    f"or 'unsloth/<model>-GGUF'."
                )
                next_repo = _prompt_fuzzy_hf_match(hf_repo)
                if next_repo is None:
                    return {
                        "ok": False,
                        "path": None,
                        "repo_id": current,
                        "error": f"repo '{hf_repo}' contains no .gguf files",
                    }
                current = next_repo
                continue
            if len(variants) == 1:
                only = variants[0]
                if only["kind"] == "file":
                    filename = only["spec"]
                else:
                    include = only["spec"]
                info(f"Repo contains a single GGUF variant: {only['label']}")
            elif len(variants) > 1:
                picked = _prompt_gguf_variant(hf_repo, variants)
                if picked is None:
                    return {
                        "ok": False,
                        "path": None,
                        "repo_id": current,
                        "error": "user cancelled quant selection",
                    }
                if picked["kind"] == "file":
                    filename = picked["spec"]
                else:
                    include = picked["spec"]

        # Route every download under STATE_DIR/models/<slug> so we can recover
        # the resolved .gguf path after the streaming HF CLI finishes (issue #53).
        local_dir = STATE_DIR / "models" / pb.safe_repo_slug(hf_repo)
        local_dir.mkdir(parents=True, exist_ok=True)

        download_label = filename or include or "<full repo>"
        console.print(
            f"\n[cyan]Downloading {hf_repo} ({download_label}) from Hugging Face Hub...[/cyan]"
        )
        result = pb.huggingface_download_gguf(
            hf_repo,
            filename=filename,
            local_dir=str(local_dir),
            include=include,
            stream=True,
        )
        if result.get("ok"):
            summary_bits: list[str] = []
            size = result.get("bytes_downloaded")
            if isinstance(size, int) and size > 0:
                summary_bits.append(_human_bytes(size))
            elapsed = result.get("elapsed_seconds")
            if isinstance(elapsed, int | float) and elapsed > 0:
                summary_bits.append(f"in {_human_duration(float(elapsed))}")
            summary = f" ({' '.join(summary_bits)})" if summary_bits else ""
            ok(f"Downloaded {current}{summary}")
            # When the streaming HF CLI couldn't return a precise path (no
            # filename pinned, only local_dir set), scan the local dir for
            # the largest .gguf we just pulled so Step 5 has a concrete path.
            resolved_path = result.get("path")
            if not resolved_path:
                resolved_path = _largest_gguf_in(local_dir)
            if resolved_path:
                info(f"Path: {resolved_path}")
            return {
                "ok": True,
                "path": resolved_path,
                "repo_id": current,
                "bytes_downloaded": size,
                "elapsed_seconds": elapsed,
            }

        err = result.get("error") or "unknown error"
        fail(f"Hugging Face download failed: {err}")
        # Even with streamed output we can sometimes tell a repo is missing —
        # the Popen return code is non-zero but HF also prints "404 Client
        # Error" to the inherited stderr, which we can't read. As a pragmatic
        # signal, treat any failure that looks not-found OR any first failure
        # on an unrecognised repo as a trigger for the fuzzy-search fallback.
        looks_missing = bool(result.get("not_found")) or _looks_like_missing_repo(hf_repo, err)
        if not looks_missing:
            return {"ok": False, "path": None, "repo_id": current, "error": err}

        # Offer fuzzy-search suggestions (#38).
        next_repo = _prompt_fuzzy_hf_match(hf_repo)
        if next_repo is None:
            return {"ok": False, "path": None, "repo_id": current, "error": err}
        # Re-attempt with the user's picked / re-entered repo. If they typed
        # "org/repo filename.gguf" pass the filename through untouched.
        current = next_repo

    warn("Too many download attempts — giving up.")
    return {"ok": False, "path": None, "repo_id": current, "error": "max attempts"}


def _looks_like_missing_repo(hf_repo: str, err: str) -> bool:
    """
    Heuristic: does ``err`` look like HF couldn't find the repo?

    We can't always tell from the wrapped error string (streamed runs only
    surface "exited with status N"), so we also treat an unreachable repo as
    missing when the HF search API has **zero** exact hits for it — this is a
    strong signal the user's spelling is off.
    """
    if pb._looks_like_not_found(err):
        return True
    if "exited with status" not in err.lower():
        return False
    # Streamed failure — probe the HF search API. Exact-case hit means the
    # repo exists and the failure was something else (auth, quota, network).
    # Critical: if the search API itself fails (network down, HF outage),
    # we must NOT treat that as "repo missing" — otherwise we trigger a
    # fuzzy fallback that will find nothing and mask the real download
    # failure. Propagate the original error instead by returning False.
    try:
        hits = pb.huggingface_search_models(hf_repo, limit=10, raise_on_error=True)
    except Exception:
        return False
    hits_lower = {h.lower() for h in hits}
    return hf_repo.lower() not in hits_lower


def _prompt_fuzzy_hf_match(query: str) -> str | None:
    """
    Fuzzy-search Hugging Face for up to 3 models similar to ``query``, present
    them as a numbered picker, and let the user re-enter a different name.

    Returns the chosen repo ID (optionally including "<repo> <filename>" for
    targeted downloads) or None when the user cancels.
    """
    info("Searching Hugging Face Hub for similar model names...")
    matches = pb.huggingface_fuzzy_find(query, max_results=3)
    if matches:
        console.print("[cyan]Closest matches on Hugging Face:[/cyan]")
        choices: list[Any] = []
        for i, mid in enumerate(matches, 1):
            choices.append(questionary.Choice(f"{i}. {mid}", value=mid))
        choices.append(questionary.Choice("Enter a different model name...", value="__reenter__"))
        choices.append(questionary.Choice("Cancel", value="__cancel__"))
        pick = questionary.select(
            "Pick a suggested model or re-enter the name:",
            choices=choices,
        ).ask()
        if pick is None or pick == "__cancel__":
            return None
        if pick != "__reenter__":
            return pick
    else:
        warn(f"No similar models found on Hugging Face for '{query}'.")

    retry = questionary.text(
        "Enter a different Hugging Face repo (e.g. 'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF')"
        " or leave blank to cancel:",
    ).ask()
    if not retry or not retry.strip():
        return None
    return retry.strip()


def _human_bytes(n: int) -> str:
    """Format a byte count as the largest unit that keeps the number readable."""
    if n < 0:
        return str(n)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} {units[-1]}"


def _human_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '3.2s', '1m 42s', '1h 03m 20s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _download_model(state: WizardState) -> bool:
    import time

    engine = state.primary_engine
    tag = state.engine_model_tag
    llamacpp_model_path: str | None = None
    llamacpp_bytes: int | None = None
    llamacpp_elapsed: float | None = None
    # Stream sub-command stdout/stderr straight to the user's terminal so the
    # engines' own progress bars (ollama "pulling manifest...", lms download
    # spinner, hf CLI tqdm) are visible. We bracket with time.monotonic() so
    # we can always print a summary line on success — addresses issue #39.
    console.print(f"\n[cyan]Downloading {tag} via {engine}...[/cyan]")
    start = time.monotonic()
    try:
        if engine == "ollama":
            subprocess.run(["ollama", "pull", tag], check=True)
        elif engine == "lmstudio":
            lms = pb.lms_binary()
            if not lms:
                fail("lms CLI not found")
                return False
            subprocess.run([lms, "get", tag, "-y"], check=True)
        elif engine == "llamacpp":
            hf_result = _download_gguf_via_hf_cli(tag)
            if not hf_result.get("ok"):
                return False
            llamacpp_model_path = hf_result.get("path")
            llamacpp_bytes = hf_result.get("bytes_downloaded")
            llamacpp_elapsed = hf_result.get("elapsed_seconds")
            # A fuzzy-search re-pick returned a different repo ID than the
            # one we started with — persist it so step 6 wires the harness
            # to the model the user actually downloaded (#38).
            resolved_repo = hf_result.get("repo_id")
            if resolved_repo and resolved_repo != tag:
                state.model_name = resolved_repo
                state.engine_model_tag = resolved_repo
                tag = resolved_repo
                info(f"Updated model selection to: [bold]{resolved_repo}[/bold]")
    except KeyboardInterrupt:
        fail("Download interrupted by user.")
        return False
    except subprocess.CalledProcessError as exc:
        fail(f"Download failed: {exc}")
        return False
    elapsed = time.monotonic() - start
    # Per-engine summary line — the body of work for issue #39's acceptance
    # criteria ("success line with final size and elapsed time"). For engines
    # without a reliable size hook we still show elapsed time.
    if engine == "llamacpp":
        # _download_gguf_via_hf_cli already printed its own summary; avoid a
        # duplicate line here.
        if llamacpp_elapsed is None and elapsed > 0:
            info(f"Total wizard time for download: {_human_duration(elapsed)}")
    else:
        size_hint: str | None = None
        if engine == "ollama":
            size_hint = _ollama_model_size_hint(tag)
        elif engine == "lmstudio":
            size_hint = _lms_model_size_hint(tag)
        bits = []
        if size_hint:
            bits.append(size_hint)
        bits.append(f"in {_human_duration(elapsed)}")
        ok(f"Downloaded {tag} ({' '.join(bits)})")
    # Refresh profile so step 5 sees the new model; preserve llamacpp_model_path
    # since machine_profile() never returns that key.
    state.profile = pb.machine_profile()
    if engine == "llamacpp" and llamacpp_model_path:
        state.profile["llamacpp_model_path"] = llamacpp_model_path
        if llamacpp_bytes:
            state.profile.setdefault("llamacpp", {})["model_bytes"] = llamacpp_bytes
    return True


def _ollama_model_size_hint(tag: str) -> str | None:
    """Return the ollama-reported size for ``tag`` (e.g. '19 GB') or None."""
    try:
        for entry in pb.parse_ollama_list():
            if entry.get("name") == tag and entry.get("size"):
                return str(entry["size"])
    except Exception:
        pass
    return None


def _lms_model_size_hint(tag: str) -> str | None:
    """Return the lmstudio-reported size for ``tag`` (bytes → human) or None."""
    try:
        info_out = pb.lms_info()
        for m in info_out.get("models", []) or []:
            if m.get("path") == tag and isinstance(m.get("size"), int):
                return _human_bytes(int(m["size"]))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Step 5 — Smoke test engine + model
# ---------------------------------------------------------------------------


def _llamacpp_spawn_and_smoke(state: WizardState, *, tag: str, model_path: str) -> dict[str, Any]:
    """
    Verify a GGUF path, auto-start llama-server with sensible defaults, then
    run the smoke test against the fresh server.

    Returns the standard ``{"ok": bool, ...}`` dict; on failure the dict
    carries a ``manual_command`` field so the caller can echo it.
    """
    # Phase 2 — no server running. We need a real GGUF path to spawn one.
    if not model_path or not Path(model_path).is_file():
        warn(
            "No GGUF model path is recorded in the wizard state — re-run "
            "Step 4 (model download) so the file path is captured."
        )
        return {
            "ok": False,
            "error": (f"no resolved GGUF path for model '{tag}'; re-run wizard step 4"),
        }

    # Phase 3 — auto-start llama-server with sensible defaults.
    info(
        f"Auto-starting llama-server on "
        f"{pb.LLAMACPP_SERVER_HOST}:{pb.LLAMACPP_SERVER_PORT} "
        f"with model {Path(model_path).name}..."
    )
    start_result = pb.llamacpp_start_server(
        model_path=model_path,
        profile=state.profile,
        port=pb.LLAMACPP_SERVER_PORT,
        host=pb.LLAMACPP_SERVER_HOST,
    )
    argv = start_result.get("argv") or []
    manual_cmd = _render_llama_server_command(argv) if argv else ""
    if not start_result.get("ok"):
        err = start_result.get("error") or "unknown error"
        log_path = start_result.get("log_path") or ""
        hint = start_result.get("hint")
        fail(f"Auto-start failed: {err}")
        if hint:
            warn(hint)
        if manual_cmd:
            warn("To start llama-server manually, run:")
            console.print(f"  [bold]{manual_cmd}[/bold]")
        if log_path:
            info(f"Server log: {log_path}")
        return {
            "ok": False,
            "error": err,
            "hint": hint,
            "manual_command": manual_cmd,
            "log_path": log_path,
        }

    handle = start_result["handle"]
    ok(
        f"llama-server is ready (pid {handle.pid}, port {handle.port}). "
        f"args: {' '.join(handle.argv[1:])}"
    )
    mtp = start_result.get("mtp") or {}
    if mtp.get("enabled"):
        info(
            f"MTP speculative decoding enabled "
            f"(--spec-draft-n-max {mtp.get('spec_draft_n_max')}, "
            f"source: {mtp.get('source')})"
        )
    elif mtp.get("warning"):
        warn(mtp["warning"])
    for note in mtp.get("notes") or []:
        warn(note)
    info(f"Server log: {handle.log_path}")

    smoke = pb.smoke_test_llamacpp_model(tag)
    if not smoke.get("ok"):
        smoke = dict(smoke)
        smoke.setdefault("manual_command", manual_cmd)
        smoke.setdefault("log_path", handle.log_path)
    return smoke


def _llamacpp_smoke_test(state: WizardState, *, non_interactive: bool) -> dict[str, Any]:
    """
    Auto-start llama-server with the just-downloaded GGUF (issue #53), wait
    for the /health endpoint, then run the actual smoke test against
    /v1/chat/completions.

    Returns the standard ``{"ok": bool, ...}`` dict that
    ``step_2_5_smoke_test`` already understands. On failure, the dict carries
    a ``manual_command`` field so the caller can echo it.
    """
    tag = state.engine_model_tag
    model_path = state.profile.get("llamacpp_model_path") or ""

    # Remote endpoint: skip every local-spawn code path (issue #123). The
    # smoke test just hits the remote /v1/chat/completions; if the remote
    # /health probe failed we cannot recover by spawning a local server.
    info_dict = pb.llamacpp_info()
    if info_dict.get("remote"):
        base_url = info_dict.get("base_url") or pb.llamacpp_base_url()
        if info_dict.get("server_running"):
            ok(f"Using remote llama.cpp server at {base_url}")
            return pb.smoke_test_llamacpp_model(tag)
        return {
            "ok": False,
            "error": (
                f"remote llama.cpp server at {base_url} is not reachable "
                f"(GET /health failed); check the remote host and "
                f"LLAMACPP_BASE_URL before re-running"
            ),
        }

    # Phase 1 — handle a server that is already running on our port.
    if info_dict.get("server_running"):
        running_model = (info_dict.get("model") or "").strip()
        if _llamacpp_models_match(running_model, tag):
            ok(
                f"Reusing running llama-server on "
                f"{pb.LLAMACPP_SERVER_HOST}:{info_dict['server_port']} "
                f"(model: {running_model or tag})"
            )
            return pb.smoke_test_llamacpp_model(tag)

        # Different model on our port. We can stop+restart with the wanted
        # model, but only if WE started the running server (pid-file gated).
        warn(
            f"A different llama.cpp model is already loaded on port "
            f"{info_dict['server_port']}: '{running_model or 'unknown'}'.\n"
            f"  Wanted: '{tag}'."
        )

        def _switch() -> dict[str, Any]:
            stop_res = pb.llamacpp_stop_server_by_port(info_dict["server_port"])
            if not stop_res.get("ok"):
                return {
                    "ok": False,
                    "error": (
                        f"could not stop the running llama-server on port "
                        f"{info_dict['server_port']}: {stop_res.get('error')}; "
                        f"stop it manually and re-run with --resume"
                    ),
                }
            info(
                f"Stopped the previous llama-server on port "
                f"{info_dict['server_port']} (was '{running_model or 'unknown'}')."
            )
            return _llamacpp_spawn_and_smoke(state, tag=tag, model_path=model_path)

        if non_interactive:
            return _switch()
        try:
            switch = questionary.confirm(
                f"Stop it and start '{tag}' instead?",
                default=True,
            ).ask()
        except KeyboardInterrupt:
            return {"ok": False, "error": "user cancelled at server-conflict prompt"}
        if switch:
            return _switch()

        try:
            use_running = questionary.confirm(
                f"Use the already-running model ({running_model or 'unknown'}) for the smoke test?",
                default=True,
            ).ask()
        except KeyboardInterrupt:
            return {"ok": False, "error": "user cancelled at server-conflict prompt"}
        if not use_running:
            return {
                "ok": False,
                "error": (
                    f"port {info_dict['server_port']} is busy with another model; "
                    f"stop it (kill the running llama-server) and re-run with --resume"
                ),
            }
        # User opted in: run the smoke test against whatever is loaded.
        # Do NOT mutate state.engine_model_tag here — Step 6 wires the
        # harness from the persisted HF repo id; clobbering it with the
        # running server's basename would break that downstream config.
        # llama-server ignores the `model` field and uses whatever is
        # loaded, so passing ``running_model`` only affects this request.
        return pb.smoke_test_llamacpp_model(running_model or tag)

    return _llamacpp_spawn_and_smoke(state, tag=tag, model_path=model_path)


def _ensure_llamacpp_server_running(state: WizardState) -> dict[str, Any]:
    """
    Make sure a llama-server is up and serving the wizard's chosen model.

    Step 5 already auto-starts the server before its smoke test, but the
    server can die between steps (OOM, user kill, machine sleep). Step 7's
    verify-launch test then fails with ``ConnectionRefused`` and leaves the
    user stuck. Steps that need the server should call this first.

    Returns ``{"ok": True}`` when the server is healthy (already running with
    the right model, or freshly started). Returns ``{"ok": False, "error": …}``
    on failure — caller decides how to surface it.
    """
    tag = state.engine_model_tag
    info_dict = pb.llamacpp_info()
    # Remote endpoint: a local spawn is never the right answer (issue #123).
    # If the remote /health probe says the server is up, we're done; if not,
    # surface a clear error so the caller does not try to recover by
    # launching `llama-server` on the wrong machine.
    if info_dict.get("remote"):
        base_url = info_dict.get("base_url") or pb.llamacpp_base_url()
        if info_dict.get("server_running"):
            return {"ok": True, "reused": True, "remote": True}
        return {
            "ok": False,
            "error": (
                f"remote llama.cpp server at {base_url} is not reachable "
                f"(GET /health failed); start it on the remote host or "
                f"point LLAMACPP_BASE_URL elsewhere"
            ),
            "remote": True,
        }
    if info_dict.get("server_running"):
        running_model = (info_dict.get("model") or "").strip()
        if not running_model or _llamacpp_models_match(running_model, tag):
            return {"ok": True, "reused": True}
        # Different model on our port — try a ccl-managed stop so we can
        # restart with the wanted model. If the running server was not
        # started by ccl (no pid file), the stop fails and we surface the
        # original "different model" error rather than killing it.
        stop_res = pb.llamacpp_stop_server_by_port(info_dict["server_port"])
        if not stop_res.get("ok"):
            return {
                "ok": False,
                "error": (
                    f"port {info_dict['server_port']} is serving a different model "
                    f"('{running_model}'); wanted '{tag}'"
                ),
            }
        info(
            f"Stopped the previous llama-server on port "
            f"{info_dict['server_port']} (was '{running_model}')."
        )
        # Fall through to the spawn block below to restart with the wanted model.
    model_path = state.profile.get("llamacpp_model_path") or ""
    if not model_path or not Path(model_path).is_file():
        return {
            "ok": False,
            "error": (
                f"no llama-server running and no resolved GGUF path for '{tag}' — "
                f"re-run wizard step 4 to capture the file path"
            ),
        }
    tuning = state.profile.get("llamacpp_tuning") or {}
    host = str(tuning.get("host") or pb.LLAMACPP_SERVER_HOST)
    info(
        f"Restarting llama-server on "
        f"{host}:{pb.LLAMACPP_SERVER_PORT} "
        f"with model {Path(model_path).name}..."
    )
    start_result = pb.llamacpp_start_server(
        model_path=model_path,
        profile=state.profile,
        port=pb.LLAMACPP_SERVER_PORT,
        host=host,
        ctx_size=int(tuning.get("ctx_size") or pb.LLAMACPP_CTX_SIZE),
        extra_argv=list(tuning.get("extra_argv") or []),
    )
    if not start_result.get("ok"):
        return {
            "ok": False,
            "error": start_result.get("error") or "auto-start failed",
            "log_path": start_result.get("log_path"),
        }
    handle = start_result["handle"]
    ok(f"llama-server is ready (pid {handle.pid}, port {handle.port}).")
    mtp = start_result.get("mtp") or {}
    if mtp.get("enabled"):
        info(
            f"MTP speculative decoding enabled "
            f"(--spec-draft-n-max {mtp.get('spec_draft_n_max')}, "
            f"source: {mtp.get('source')})"
        )
    elif mtp.get("warning"):
        warn(mtp["warning"])
    for note in mtp.get("notes") or []:
        warn(note)
    return {"ok": True, "reused": False}


_MIN_MODEL_MATCH_LEN = 12

# Mutually-exclusive variant tokens: a 'base' GGUF is not interchangeable with
# an 'instruct'/'chat' GGUF even when the rest of the family/size matches.
_VARIANT_TOKENS = ("instruct", "chat", "base", "it")


def _variant_token(normalized: str) -> str | None:
    """Return the variant token present in ``normalized`` (already lowercased), if any."""
    for tok in _VARIANT_TOKENS:
        # Match as a hyphen-delimited token so 'baseline' doesn't read as 'base'.
        if (
            normalized == tok
            or normalized.startswith(f"{tok}-")
            or normalized.endswith(f"-{tok}")
            or f"-{tok}-" in normalized
        ):
            return tok
    return None


def _llamacpp_models_match(running: str, wanted: str) -> bool:
    """
    Loose match between the model llama-server reports on /v1/models and the
    HF repo id / file path the wizard wants. ``running`` is often the GGUF
    file basename, while ``wanted`` is typically an HF repo id like
    ``org/repo``. We require a substring overlap of at least
    ``_MIN_MODEL_MATCH_LEN`` characters so different sizes/quants of the same
    family (e.g. ``...-1.5B`` vs ``...-7B``) do not collapse to a match.

    Additionally, if both sides carry a variant token (``base``/``instruct``/
    ``chat``/``it``) and the tokens differ, refuse the match — a base model
    is not a drop-in replacement for an instruct/chat model.
    """
    if not running or not wanted:
        return False
    a = _normalize_model_id(running)
    b = _normalize_model_id(wanted)
    if a == b:
        return True
    short = a if len(a) <= len(b) else b
    long_ = b if short is a else a
    if len(short) < _MIN_MODEL_MATCH_LEN or short not in long_:
        return False
    a_variant = _variant_token(a)
    b_variant = _variant_token(b)
    return not (a_variant and b_variant and a_variant != b_variant)


def _normalize_model_id(value: str) -> str:
    """Lowercase + strip extension, dir prefix, and `-gguf` suffix for matching."""
    raw = value.strip().lower()
    # Take the basename without the parent dir.
    raw = raw.rsplit("/", 1)[-1]
    # Strip a trailing `.gguf` (or any final extension) only when present.
    if raw.endswith(".gguf"):
        raw = raw[: -len(".gguf")]
    # HF repos are commonly suffixed `-gguf` to mark the quantized variant.
    if raw.endswith("-gguf"):
        raw = raw[: -len("-gguf")]
    return raw


def _run_engine_lifecycle(
    engine: str,
    action: str,
    *,
    model: str = "",
    dry_run: bool = True,
    profile: dict[str, Any] | None = None,
    non_interactive: bool = True,
) -> dict[str, Any]:
    from claude_codex_local.engines import run_engine_action
    from claude_codex_local.engines.registry import EngineLifecycleError

    try:
        return run_engine_action(
            engine,
            action,
            model=model,
            dry_run=dry_run,
            profile=profile,
            non_interactive=non_interactive,
        )
    except EngineLifecycleError as exc:
        return {
            "engine": engine,
            "action": action,
            "ok": False,
            "detail": str(exc),
            "data": {"ok": False, "error": str(exc)},
        }


def step_2_5_smoke_test(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 5 — Smoke test engine + model")
    engine = state.primary_engine
    tag = state.engine_model_tag
    info(f"Running minimal prompt through {engine} / {tag}...")

    lifecycle = _run_engine_lifecycle(
        engine,
        "test",
        model=tag,
        dry_run=False,
        profile=state.profile,
        non_interactive=non_interactive,
    )
    result = lifecycle.get("data") or {
        "ok": lifecycle.get("ok", False),
        "response": lifecycle.get("detail", ""),
        "error": None if lifecycle.get("ok") else lifecycle.get("detail", ""),
    }

    state.smoke_test_result = result
    if not result.get("ok"):
        fail(f"Smoke test failed: {result.get('error') or result.get('response')}")
        return False

    ok(f"Smoke test passed: {str(result.get('response', ''))[:80]}")

    # Report throughput (tokens/second) and let the user react if it's slow.
    if not _report_smoke_test_speed(result, non_interactive=non_interactive):
        return False

    state.mark("5")
    return True


def _format_tokens_per_second(tps: float) -> str:
    """Human-readable tokens/second string (e.g. '~15.3 tok/s')."""
    return f"~{tps:.1f} tok/s"


def _speed_verdict(tps: float) -> tuple[str, Callable[[str], None]]:
    """
    Classify a tokens/second value and return a label + printer function.

    Thresholds:
      - < 10 tok/s  → slow
      - 10–30 tok/s → acceptable
      - > 30 tok/s  → fast
    """
    if tps < 10:
        return ("slow — may feel sluggish for interactive use", warn)
    if tps < 30:
        return ("acceptable for most interactive coding tasks", info)
    return ("fast — should feel snappy", ok)


def _report_smoke_test_speed(result: dict[str, Any], non_interactive: bool = False) -> bool:
    """
    Display the measured throughput and offer to re-pick the model when it's slow.

    Returns True if the wizard should keep this model and continue, or False
    if the user wants to go back and pick a different model (interactive only).
    """
    tps = result.get("tokens_per_second")
    completion_tokens = result.get("completion_tokens")
    duration_seconds = result.get("duration_seconds")

    if not isinstance(tps, int | float) or tps <= 0:
        # No measurement available (e.g. Ollama CLI fallback) — do not block.
        if duration_seconds is not None:
            info(f"Inference duration: ~{float(duration_seconds):.2f}s (throughput unavailable)")
        return True

    verdict, printer = _speed_verdict(float(tps))
    detail_bits = [_format_tokens_per_second(float(tps))]
    if isinstance(completion_tokens, int) and completion_tokens > 0:
        detail_bits.append(f"{completion_tokens} tokens")
    if isinstance(duration_seconds, int | float) and duration_seconds > 0:
        detail_bits.append(f"in {float(duration_seconds):.2f}s")
    printer(f"Model speed: {' | '.join(detail_bits)} — {verdict}")
    info("Speed guide: <10 tok/s slow · 10–30 acceptable · 30+ fast")

    if float(tps) < 10:
        if non_interactive:
            warn("Speed is below 10 tok/s but continuing (non-interactive mode).")
            return True
        keep_going = questionary.confirm(
            "Model throughput is below 10 tok/s. Keep this model and continue anyway?",
            default=True,
        ).ask()
        if keep_going is False:
            info("Go back and pick a different model, then re-run the wizard.")
            return False
    return True


# ---------------------------------------------------------------------------
# Step 5.5 — Optional benchmark
# ---------------------------------------------------------------------------


def step_2_5_5_benchmark(state: WizardState, non_interactive: bool = False) -> bool:
    """Offer an optional lightweight benchmark of the selected model."""
    from claude_codex_local import bench

    header("Step 5.5 — Optional lightweight benchmark")

    engine = state.primary_engine
    model_tag = state.engine_model_tag

    if not engine or not model_tag:
        info("No model selected. Skipping benchmark.")
        state.mark("5.5")
        return True

    if non_interactive:
        info("Benchmark skipped in non-interactive mode.")
        state.mark("5.5")
        return True

    run_bench = questionary.confirm(
        f"Run a lightweight benchmark of {engine} / {model_tag}?\n"
        "(Measures first-token latency and throughput on a coding prompt. Takes ~30-60s.)",
        default=False,
    ).ask()

    if run_bench is None or not run_bench:
        info("Benchmark skipped.")
        state.mark("5.5")
        return True

    console.print()
    info(f"Benchmarking {engine} / {model_tag}...")
    info("(Running 3 trials of the benchmark prompt)\n")

    try:
        summary = bench.benchmark_model(
            engine=engine,
            model=model_tag,
            num_trials=3,
            timeout=120,
        )

        if summary.num_trials == 0:
            warn("Benchmark failed: no successful trials. Continuing setup.")
            state.mark("5.5")
            return True

        # Display results
        console.print()
        ok(f"Benchmark completed: {summary.num_trials} trials")
        console.print()

        results_table = Table(title="Benchmark Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Range", style="yellow")

        results_table.add_row(
            "First Token Latency",
            f"{summary.avg_first_token_ms:.1f} ms",
            f"{summary.min_first_token_ms:.1f}–{summary.max_first_token_ms:.1f} ms",
        )
        results_table.add_row(
            "Throughput",
            f"{summary.avg_tokens_per_second:.2f} tok/s",
            f"{summary.min_tokens_per_second:.2f}–{summary.max_tokens_per_second:.2f} tok/s",
        )
        results_table.add_row(
            "Total Generation Time",
            f"{summary.avg_total_time_ms:.1f} ms",
            "(average across trials)",
        )

        console.print(results_table)
        console.print()

        # Save report
        report_dir = STATE_DIR / "benchmarks"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = (
            report_dir / f"benchmark-{engine}-{model_tag.replace(':', '-')}-{timestamp}.md"
        )
        bench.save_benchmark_report(summary, report_path)

        ok(f"Benchmark report saved: {report_path}")
        info("View the full report to see detailed metrics and interpretation.")
        console.print()

        state.mark("5.5")
        return True

    except Exception as exc:
        fail(f"Benchmark error: {exc}")
        console.print()
        state.mark("5.5")
        return True  # Don't block wizard on benchmark failure


# ---------------------------------------------------------------------------
# Direct Codex/Pi settings mutation
# ---------------------------------------------------------------------------


def _backup_config_file(path: Path, target: str) -> dict[str, Any]:
    """Copy a config file before mutation and return rollback metadata."""
    backup_dir = STATE_DIR / "backups" / target
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{path.name}.{stamp}.{os.getpid()}.bak"
    existed = path.exists()
    if existed:
        shutil.copy2(path, backup_path)
    else:
        backup_path.write_text("")
        backup_path.chmod(0o600)
    return {
        "target": target,
        "path": str(path),
        "backup_path": str(backup_path),
        "existed": existed,
    }


def _rollback_config_backup(backup: dict[str, Any]) -> None:
    """Best-effort restore for a config mutation that failed mid-write."""
    path = Path(str(backup["path"]))
    backup_path = Path(str(backup["backup_path"]))
    if backup.get("existed"):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, path)
    else:
        path.unlink(missing_ok=True)


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def _codex_config_path() -> Path:
    return _codex_home() / "config.toml"


def _toml_quote(value: str) -> str:
    # JSON string syntax is valid TOML basic string syntax.
    return json.dumps(value)


def _upsert_top_level_toml_key(text: str, key: str, value: str) -> str:
    rendered = f"{key} = {_toml_quote(value)}"
    lines = text.splitlines()
    first_table_idx = len(lines)
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            first_table_idx = idx
            break
        if stripped.startswith("#"):
            continue
        if key_re.match(line):
            lines[idx] = rendered
            return "\n".join(lines).rstrip() + "\n"

    lines.insert(first_table_idx, rendered)
    return "\n".join(lines).rstrip() + "\n"


def _remove_toml_table(text: str, table: str) -> str:
    lines = text.splitlines()
    header = f"[{table}]"
    start: int | None = None
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start = idx
            break
    if start is None:
        return text.rstrip() + ("\n" if text else "")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break
    del lines[start:end]
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def _append_toml_table(text: str, table: str, values: dict[str, Any]) -> str:
    body = [f"[{table}]"]
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = _toml_quote(str(value))
        body.append(f"{key} = {rendered}")
    prefix = text.rstrip()
    sep = "\n\n" if prefix else ""
    return prefix + sep + "\n".join(body) + "\n"


def _codex_provider_for_engine(engine: str) -> str:
    return f"ccl-{engine}"


def _codex_provider_env_key(engine: str) -> str | None:
    if engine == "ollama" and pb.OLLAMA_API_KEY:
        return "OLLAMA_API_KEY"
    if engine == "llamacpp" and pb.LLAMACPP_API_KEY:
        return "LLAMACPP_API_KEY"
    if engine == "vllm":
        if os.environ.get("VLLM_API_KEY") or pb.VLLM_KEY_FILE.exists():
            return "VLLM_API_KEY"
        return None
    if engine == "9router":
        return "CCL_9ROUTER_API_KEY"
    if engine == "openrouter":
        return "CCL_OPENROUTER_API_KEY"
    return None


def _codex_provider_config(engine: str) -> dict[str, Any] | None:
    base_url = _pi_base_url_for_engine(engine)
    if base_url is None:
        return None
    config: dict[str, Any] = {
        "name": f"CCL {engine}",
        "base_url": base_url,
        "requires_openai_auth": False,
    }
    env_key = _codex_provider_env_key(engine)
    if env_key:
        config["env_key"] = env_key
        config["env_key_instructions"] = f"Export {env_key} before launching codex."
    return config


def _write_codex_config(engine: str, tag: str) -> Path:
    """
    Update Codex's normal config so launching `codex` directly uses this model.

    For local Ollama/LM Studio, prefer Codex's built-in OSS provider knobs.
    Other OpenAI-compatible endpoints are written as CCL-managed providers.
    """
    config_path = _codex_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = config_path.read_text() if config_path.exists() else ""
    text = _upsert_top_level_toml_key(text, "model", tag)

    if engine == "ollama" and pb._is_local_base_url(pb.ollama_base_url()):
        text = _upsert_top_level_toml_key(text, "model_provider", "oss")
        text = _upsert_top_level_toml_key(text, "oss_provider", "ollama")
    elif engine == "lmstudio":
        text = _upsert_top_level_toml_key(text, "model_provider", "oss")
        text = _upsert_top_level_toml_key(text, "oss_provider", "lmstudio")
    else:
        provider = _codex_provider_for_engine(engine)
        provider_config = _codex_provider_config(engine)
        if provider_config is None:
            raise ValueError(f"Unknown engine for Codex config: {engine!r}")
        text = _upsert_top_level_toml_key(text, "model_provider", provider)
        text = _remove_toml_table(text, f"model_providers.{provider}")
        text = _append_toml_table(text, f"model_providers.{provider}", provider_config)

    config_path.write_text(text)
    config_path.chmod(0o600)
    return config_path


def _configure_codex_with_backup(engine: str, tag: str) -> tuple[Path, dict[str, Any]]:
    config_path = _codex_config_path()
    backup = _backup_config_file(config_path, "codex")
    try:
        # Resolve via the wizard module so that test monkeypatches on
        # ``wizard._write_codex_config`` take effect (the original
        # monolithic wizard.py had everything in one namespace).
        import claude_codex_local.wizard as _wizard_mod

        written = _wizard_mod._write_codex_config(engine, tag)
    except Exception:
        _rollback_config_backup(backup)
        raise
    return written, backup


def _configure_pi_with_backup(engine: str, tag: str) -> tuple[Path, dict[str, Any]]:
    config_path = _pi_agent_dir() / "models.json"
    backup = _backup_config_file(config_path, "pi")
    try:
        written = _write_pi_models_config(engine, tag)
    except Exception:
        _rollback_config_backup(backup)
        raise
    return written, backup


# ---------------------------------------------------------------------------
# Step 6 — Wire up harness with isolated helper and direct tool settings
# ---------------------------------------------------------------------------


def step_2_6_wire_harness(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 6 — Wire up harness")
    harness = state.primary_harness
    engine = state.primary_engine
    tag = state.engine_model_tag

    if not _is_model_compatible_with_engine(state, engine):
        warn(f"Model '{tag}' was selected for a different engine. Please select a new model.")
        if non_interactive:
            fail("Model/engine mismatch in non-interactive mode.")
            return False
        ok("Returning to model selection to pick a compatible model.")
        return False

    pb.ensure_state_dirs()

    if harness == "claude":
        result = _wire_claude(engine, tag)
    elif harness == "codex":
        # Write Codex config with backup/rollback, then build the wire result.
        # Codex's /model list is controlled by Codex itself; the helper
        # script/alias launches the selected model with the right provider.
        try:
            config_path, backup = _configure_codex_with_backup(engine, tag)
        except Exception as exc:
            fail(f"Cannot update Codex config: {exc}")
            return False
        state.config_backups["codex"] = backup
        ok(f"Updated Codex config: {config_path}")
        result = _wire_codex(engine, tag)
    elif harness == "pi":
        _materialize_pi_api_key_files(engine)
        try:
            config_path, backup = _configure_pi_with_backup(engine, tag)
        except Exception as exc:
            fail(f"Cannot update Pi config: {exc}")
            return False
        state.config_backups["pi"] = backup
        ok(f"Updated Pi config: {config_path}")
        result = _wire_pi(engine, tag, configure=False)
    else:
        fail(f"Unknown harness: {harness}")
        return False

    if result is None:
        return False

    state.wire_result = {
        "argv": result.argv,
        "env": result.env,
        "effective_tag": result.effective_tag,
        "raw_env": result.raw_env,
    }
    state.engine_model_tag = result.effective_tag
    fence_tag = _fence_tag_for(harness, engine)
    alias_short = _alias_names_for(fence_tag)[0]
    state.launch_command = [alias_short]
    ok(f"Harness wired. argv: [bold]{' '.join(shlex.quote(x) for x in result.argv)}[/bold]")
    state.mark("6")
    return True


def _model_known_incompatible_with_claude_code(tag: str) -> bool:
    t = tag.lower()
    return "qwen3" in t


def _materialize_remote_api_key(key_file: Path, key_value: str) -> str:
    # Write the env-supplied API key to a chmod-600 file under STATE_DIR and
    # return the shell expression the helper script substitutes at exec time.
    # Mirrors the 9router/openrouter/vllm keyfile pattern so the literal
    # secret never lives in the generated cc*/cx*/ccp* scripts. The file is
    # re-written on each wizard run so env-as-source-of-truth wins; user-
    # managed key files (e.g. pre-existing VLLM_KEY_FILE) are handled by
    # caller precedence and not overwritten here.
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key_value + "\n")
    key_file.chmod(0o600)
    return f'"$(cat {shlex.quote(str(key_file))})"'


def _wire_claude(engine: str, tag: str) -> WireResult | None:
    """
    Build a WireResult for the Claude harness against the chosen engine.

    For Ollama we delegate to `ollama launch claude` which sets the right env
    vars internally and execs the user's real `claude` binary against the
    user's real `~/.claude`. For LM Studio / llama.cpp we set the inline env
    explicitly because `ollama launch` does not apply.
    """
    if _model_known_incompatible_with_claude_code(tag):
        warn(
            f"Model '{tag}' is known to misbehave with Claude Code. Recommended\n"
            "alternatives: gemma3:27b, qwen2.5-coder:32b."
        )

    if engine == "ollama":
        if not pb._is_local_base_url(pb.ollama_base_url()):
            base_url = pb.ollama_openai_base_url()
            env = {
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
                "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"Remote Ollama {tag}",
                "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                    f"Remote model served by Ollama at {base_url}"
                ),
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            raw_env: dict[str, str] = {}
            if pb.OLLAMA_API_KEY:
                key_expr = _materialize_remote_api_key(pb.OLLAMA_KEY_FILE, pb.OLLAMA_API_KEY)
                raw_env["ANTHROPIC_API_KEY"] = key_expr
                raw_env["ANTHROPIC_AUTH_TOKEN"] = key_expr
            else:
                env["ANTHROPIC_API_KEY"] = "ollama"  # pragma: allowlist secret
                env["ANTHROPIC_AUTH_TOKEN"] = "ollama"  # pragma: allowlist secret
            return WireResult(
                argv=["claude", "--model", tag], env=env, effective_tag=tag, raw_env=raw_env
            )
        # Trailing "--" is important: the helper script appends "$@" after
        # this argv, and `ollama launch` would otherwise eat any user flag
        # (e.g. `cc -p "hi"` -> `ollama launch` rejects `-p`). The `--`
        # tells `ollama launch` to forward everything after it to `claude`.
        return WireResult(
            argv=["ollama", "launch", "claude", "--model", tag, "--"],
            env={},
            effective_tag=tag,
        )
    if engine == "lmstudio":
        env = {
            "ANTHROPIC_BASE_URL": f"http://localhost:{pb.LMS_SERVER_PORT}",
            "ANTHROPIC_API_KEY": "lmstudio",  # pragma: allowlist secret
            "ANTHROPIC_AUTH_TOKEN": "lmstudio",  # pragma: allowlist secret
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"Local (lmstudio) {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Local model served by lmstudio at http://localhost:{pb.LMS_SERVER_PORT}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        return WireResult(argv=["claude", "--model", tag], env=env, effective_tag=tag)
    if engine == "llamacpp":
        base_url = pb.llamacpp_base_url()
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"Local (llamacpp) {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Local model served by llamacpp at {base_url}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        raw_env = {}
        if pb.LLAMACPP_API_KEY:
            key_expr = _materialize_remote_api_key(pb.LLAMACPP_KEY_FILE, pb.LLAMACPP_API_KEY)
            raw_env["ANTHROPIC_API_KEY"] = key_expr
            raw_env["ANTHROPIC_AUTH_TOKEN"] = key_expr
        else:
            env["ANTHROPIC_API_KEY"] = "sk-local"  # pragma: allowlist secret
            env["ANTHROPIC_AUTH_TOKEN"] = "sk-local"  # pragma: allowlist secret
        return WireResult(
            argv=["claude", "--model", tag], env=env, effective_tag=tag, raw_env=raw_env
        )
    if engine == "vllm":
        # vLLM exposes an OpenAI-compatible API. Auth is off by default but
        # vllm supports `--api-key`; if a key was written to VLLM_KEY_FILE
        # we read it at exec-time the same way 9router does (chmod-600
        # keyfile, $(cat …) expression — never literal in the script).
        base_url = pb.VLLM_BASE_URL
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"vLLM {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Local model served by vLLM at {base_url}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        raw_env = {}
        # If the user pre-populated VLLM_KEY_FILE (user-managed path), prefer
        # it; otherwise, when VLLM_API_KEY is set in the env, materialize it
        # into the same chmod-600 file so the helper script reads via
        # $(cat …) and the secret never lands inline.
        env_key = os.environ.get("VLLM_API_KEY", "")
        if pb.VLLM_KEY_FILE.exists():
            key_expr = f'"$(cat {shlex.quote(str(pb.VLLM_KEY_FILE))})"'
            raw_env["ANTHROPIC_AUTH_TOKEN"] = key_expr
            raw_env["ANTHROPIC_API_KEY"] = key_expr
        elif env_key:
            key_expr = _materialize_remote_api_key(pb.VLLM_KEY_FILE, env_key)
            raw_env["ANTHROPIC_AUTH_TOKEN"] = key_expr
            raw_env["ANTHROPIC_API_KEY"] = key_expr
        else:
            env["ANTHROPIC_API_KEY"] = "sk-local"  # pragma: allowlist secret
            env["ANTHROPIC_AUTH_TOKEN"] = "sk-local"  # pragma: allowlist secret
        return WireResult(
            argv=["claude", "--model", tag], env=env, effective_tag=tag, raw_env=raw_env
        )
    if engine == "9router":
        # 9router exposes an OpenAI-compatible API and requires a paid
        # cloud API key. We deliberately keep the key OUT of the helper
        # script and out of the wizard state file: the script reads it
        # at exec-time from a chmod-600 file in STATE_DIR. See
        # WireResult.raw_env for the security boundary.
        base_url = pb.ROUTER9_BASE_URL
        key_file = pb.ROUTER9_KEY_FILE
        key_expr = f'"$(cat {shlex.quote(str(key_file))})"'
        raw_env = {
            "ANTHROPIC_AUTH_TOKEN": key_expr,
            "ANTHROPIC_API_KEY": key_expr,
        }
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"9router {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Cloud-routed via 9router at {base_url}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        return WireResult(
            argv=["claude", "--model", tag], env=env, effective_tag=tag, raw_env=raw_env
        )
    if engine == "openrouter":
        # OpenRouter is hosted SaaS at https://openrouter.ai/api/v1 with
        # an OpenAI-compatible API. Same key-on-disk boundary as 9router:
        # the helper script reads the API key at exec-time from a chmod-600
        # file, never embedding it. HTTP_REFERER and X_TITLE are
        # OpenRouter's optional attribution headers; harmless when ignored
        # by the harness, useful in OpenRouter's dashboard analytics.
        # Env-var names use underscores (HTTP_REFERER, X_TITLE) because
        # POSIX env-var names cannot contain hyphens; the harness is
        # responsible for translating them into the hyphenated HTTP
        # headers (HTTP-Referer, X-Title) on the wire.
        base_url = pb.OPENROUTER_BASE_URL
        key_file = pb.OPENROUTER_KEY_FILE
        key_expr = f'"$(cat {shlex.quote(str(key_file))})"'
        raw_env = {
            "ANTHROPIC_AUTH_TOKEN": key_expr,
            "ANTHROPIC_API_KEY": key_expr,
        }
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"OpenRouter {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Cloud-routed via OpenRouter at {base_url}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "HTTP_REFERER": "https://github.com/luongnv89/ccl",
            "X_TITLE": "claude-codex-local",
        }
        return WireResult(
            argv=["claude", "--model", tag], env=env, effective_tag=tag, raw_env=raw_env
        )
    fail(f"Unknown engine for Claude wire-up: {engine}")
    return None


def _wire_codex(engine: str, tag: str) -> WireResult | None:
    if engine == "ollama":
        if not pb._is_local_base_url(pb.ollama_base_url()):
            env = {"OPENAI_BASE_URL": pb.ollama_openai_base_url()}
            raw_env: dict[str, str] = {}
            if pb.OLLAMA_API_KEY:
                key_expr = _materialize_remote_api_key(pb.OLLAMA_KEY_FILE, pb.OLLAMA_API_KEY)
                raw_env["OPENAI_API_KEY"] = key_expr
            else:
                env["OPENAI_API_KEY"] = "ollama"  # pragma: allowlist secret
            return WireResult(
                argv=["codex", "-m", tag], env=env, effective_tag=tag, raw_env=raw_env
            )
        # Known limitation: `--oss --local-provider=ollama` are codex
        # subcommand options, not top-level options. They work in
        # interactive mode (no subcommand), which is the common case.
        # `cx exec "<prompt>"` (one-shot) will hit a ChatGPT-account
        # error because the flags land before the `exec` subcommand.
        # Workaround for one-shot use: run
        #   ollama launch codex --model <tag> -- exec --oss \
        #     --local-provider=ollama --skip-git-repo-check "<prompt>"
        # directly instead of via the alias.
        return WireResult(
            argv=[
                "ollama",
                "launch",
                "codex",
                "--model",
                tag,
                "--",
                "--oss",
                "--local-provider=ollama",
            ],
            env={},
            effective_tag=tag,
        )
    if engine == "lmstudio":
        env = {
            "OPENAI_BASE_URL": f"http://localhost:{pb.LMS_SERVER_PORT}/v1",
            "OPENAI_API_KEY": "lmstudio",  # pragma: allowlist secret
        }
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag)
    if engine == "llamacpp":
        env = {"OPENAI_BASE_URL": f"{pb.llamacpp_base_url()}/v1"}
        raw_env = {}
        if pb.LLAMACPP_API_KEY:
            key_expr = _materialize_remote_api_key(pb.LLAMACPP_KEY_FILE, pb.LLAMACPP_API_KEY)
            raw_env["OPENAI_API_KEY"] = key_expr
        else:
            env["OPENAI_API_KEY"] = "sk-local"  # pragma: allowlist secret
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag, raw_env=raw_env)
    if engine == "vllm":
        # Same pattern as _wire_claude(engine="vllm"): if the user wrote a
        # key file we read it at exec-time; otherwise a placeholder is fine
        # (vLLM doesn't validate keys unless `--api-key` was passed).
        base_url = pb.VLLM_BASE_URL.rstrip("/")
        env = {"OPENAI_BASE_URL": f"{base_url}/v1"}
        raw_env = {}
        env_key = os.environ.get("VLLM_API_KEY", "")
        if pb.VLLM_KEY_FILE.exists():
            key_expr = f'"$(cat {shlex.quote(str(pb.VLLM_KEY_FILE))})"'
            raw_env["OPENAI_API_KEY"] = key_expr
        elif env_key:
            key_expr = _materialize_remote_api_key(pb.VLLM_KEY_FILE, env_key)
            raw_env["OPENAI_API_KEY"] = key_expr
        else:
            env["OPENAI_API_KEY"] = "sk-local"  # pragma: allowlist secret
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag, raw_env=raw_env)
    if engine == "9router":
        # See _wire_claude(engine="9router") for the rationale: the API
        # key is read at exec-time from a chmod-600 file, never embedded
        # in the helper script body.
        base_url = pb.ROUTER9_BASE_URL
        key_file = pb.ROUTER9_KEY_FILE
        key_expr = f'"$(cat {shlex.quote(str(key_file))})"'
        raw_env = {"OPENAI_API_KEY": key_expr}
        env = {"OPENAI_BASE_URL": base_url}
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag, raw_env=raw_env)
    if engine == "openrouter":
        # See _wire_claude(engine="openrouter"): the API key is read at
        # exec-time from a chmod-600 file, never embedded in the helper
        # script body. HTTP_REFERER / X_TITLE are decorative attribution
        # headers OpenRouter accepts.
        base_url = pb.OPENROUTER_BASE_URL
        key_file = pb.OPENROUTER_KEY_FILE
        key_expr = f'"$(cat {shlex.quote(str(key_file))})"'
        raw_env = {"OPENAI_API_KEY": key_expr}
        env = {
            "OPENAI_BASE_URL": base_url,
            "HTTP_REFERER": "https://github.com/luongnv89/ccl",
            "X_TITLE": "claude-codex-local",
        }
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag, raw_env=raw_env)
    fail(f"Unknown engine for Codex wire-up: {engine}")
    return None


def _pi_agent_dir() -> Path:
    """Return the Pi config dir CCL should augment.

    CCL intentionally writes Pi's local-provider entry into the same config
    directory that a normal ``pi`` invocation uses.  Earlier versions pointed
    ``PI_CODING_AGENT_DIR`` at an isolated CCL directory, which made the
    generated ``ccp``/``pi-local`` helper lose the user's installed Pi packages,
    extensions, skills, prompts, themes, settings, and auth state.
    """
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".pi" / "agent"


def _pi_provider_for_engine(engine: str) -> str:
    return f"ccl-{engine}"


def _pi_base_url_for_engine(engine: str) -> str | None:
    if engine == "ollama":
        return pb.ollama_openai_base_url()
    if engine == "lmstudio":
        return f"http://localhost:{pb.LMS_SERVER_PORT}/v1"
    if engine == "llamacpp":
        return f"{pb.llamacpp_base_url()}/v1"
    if engine == "vllm":
        return f"{pb.VLLM_BASE_URL.rstrip('/')}/v1"
    if engine == "9router":
        return pb.ROUTER9_BASE_URL
    if engine == "openrouter":
        return pb.OPENROUTER_BASE_URL
    return None


def _pi_api_key_for_engine(engine: str) -> str:
    # Pi reads provider keys from models.json. For engines where the wizard
    # has materialized the env-supplied key to a chmod-600 file, emit a
    # `!cat <file>` reference so the literal secret never lands in
    # models.json — matching the 9router/openrouter/vLLM-keyfile pattern.
    # Materialization is done by the caller (_wire_pi) before this fires.
    if engine == "ollama":
        if pb.OLLAMA_KEY_FILE.exists():
            return f"!cat {shlex.quote(str(pb.OLLAMA_KEY_FILE))}"
        return "ollama"
    if engine == "lmstudio":
        return "lmstudio"
    if engine == "llamacpp":
        if pb.LLAMACPP_KEY_FILE.exists():
            return f"!cat {shlex.quote(str(pb.LLAMACPP_KEY_FILE))}"
        return "sk-local"  # pragma: allowlist secret
    if engine == "vllm":
        if pb.VLLM_KEY_FILE.exists():
            return f"!cat {shlex.quote(str(pb.VLLM_KEY_FILE))}"
        return "sk-local"  # pragma: allowlist secret
    if engine == "9router":
        return f"!cat {shlex.quote(str(pb.ROUTER9_KEY_FILE))}"
    if engine == "openrouter":
        return f"!cat {shlex.quote(str(pb.OPENROUTER_KEY_FILE))}"
    return "sk-local"  # pragma: allowlist secret


def _write_pi_models_config(engine: str, tag: str) -> Path:
    """
    Write the Pi custom-provider entry used by the CCL helper.

    Pi reads OpenAI-compatible local backends from models.json rather than
    OPENAI_BASE_URL-style env vars. Write only CCL's provider entry into the
    normal Pi config dir (or the user's existing PI_CODING_AGENT_DIR override)
    so ccp/pi-local keeps the same extensions, skills, prompts, themes,
    settings, and auth state as official pi.
    """
    base_url = _pi_base_url_for_engine(engine)
    if base_url is None:
        raise ValueError(f"Unknown engine for Pi wire-up: {engine!r}")

    agent_dir = _pi_agent_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)
    models_path = agent_dir / "models.json"
    provider = _pi_provider_for_engine(engine)
    data: dict[str, Any] = {"providers": {}}
    if models_path.exists():
        try:
            loaded = json.loads(models_path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Cannot update {models_path}: invalid JSON. "
                "Fix or back up the file before re-running ccl setup."
            ) from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"Cannot update {models_path}: expected a JSON object at the top level."
            )
        data = loaded
    providers = data.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise RuntimeError(
            f"Cannot update {models_path}: expected 'providers' to be a JSON object."
        )
    providers[provider] = {
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": _pi_api_key_for_engine(engine),
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        },
        "models": [{"id": tag, "name": f"CCL {engine} {tag}"}],
    }
    models_path.write_text(json.dumps(data, indent=2) + "\n")
    return models_path


def _materialize_pi_api_key_files(engine: str) -> None:
    """Materialize env-supplied API keys before writing Pi models.json."""
    # vLLM intentionally honors a pre-existing user-managed VLLM_KEY_FILE.
    if engine == "ollama" and pb.OLLAMA_API_KEY:
        _materialize_remote_api_key(pb.OLLAMA_KEY_FILE, pb.OLLAMA_API_KEY)
    elif engine == "llamacpp" and pb.LLAMACPP_API_KEY:
        _materialize_remote_api_key(pb.LLAMACPP_KEY_FILE, pb.LLAMACPP_API_KEY)
    elif engine == "vllm" and not pb.VLLM_KEY_FILE.exists():
        env_key = os.environ.get("VLLM_API_KEY", "")
        if env_key:
            _materialize_remote_api_key(pb.VLLM_KEY_FILE, env_key)


def _wire_pi(engine: str, tag: str, configure: bool = True) -> WireResult | None:
    """Build a WireResult for Pi against a CCL-supported local/provider engine."""
    # Ensure direct callers that still use configure=True keep the historical
    # write-before-launch behavior and emit `!cat <keyfile>` references.
    _materialize_pi_api_key_files(engine)
    if configure:
        try:
            _configure_pi_with_backup(engine, tag)
        except ValueError:
            fail(f"Unknown engine for Pi wire-up: {engine}")
            return None
        except RuntimeError as exc:
            fail(str(exc))
            return None
    provider = _pi_provider_for_engine(engine)
    return WireResult(
        argv=["pi", "--provider", provider, "--model", tag],
        env={"PI_CODING_AGENT_DIR": str(_pi_agent_dir())},
        effective_tag=tag,
    )


# ---------------------------------------------------------------------------
# Step 6.5 — Helper script + shell aliases
# ---------------------------------------------------------------------------


# Legacy fence (pre-#16) used a single shared block for whichever harness
# was set up last. Kept for one-shot migration to the per-harness format.
_LEGACY_ALIAS_BLOCK_RE = re.compile(
    r"^# >>> claude-codex-local >>>.*?^# <<< claude-codex-local <<<\n?",
    re.DOTALL | re.MULTILINE,
)


def _harness_alias_block_re(harness: str) -> re.Pattern[str]:
    """
    Per-harness fenced block regex. Each harness owns its own block so
    installing cx does not overwrite a previously installed cc block
    (and vice versa). See issue #16.
    """
    tag = re.escape(harness)
    return re.compile(
        rf"^# >>> claude-codex-local:{tag} >>>.*?^# <<< claude-codex-local:{tag} <<<\n?",
        re.DOTALL | re.MULTILINE,
    )


def _infer_harness_from_legacy_block(block_text: str) -> str:
    """
    Guess which harness owns a legacy (pre-#16) alias block by inspecting its
    contents. Returns "claude" or "codex". Defaults to "claude" when the block
    is ambiguous — the caller is about to rewrite the block anyway, so the
    worst case is that an ambiguous legacy block is replaced with a fresh
    claude block for the current install (no data loss).
    """
    if "alias cx=" in block_text or "alias codex-local=" in block_text:
        return "codex"
    return "claude"


def _migrate_legacy_alias_block(existing: str) -> str:
    """
    If the rc file still contains a pre-#16 unified alias block, rewrap it in
    the per-harness fence so a subsequent per-harness replace/append leaves it
    alone when it belongs to a different harness. Idempotent.
    """
    match = _LEGACY_ALIAS_BLOCK_RE.search(existing)
    if not match:
        return existing
    legacy = match.group(0)
    harness = _infer_harness_from_legacy_block(legacy)
    # Rewrap: swap the top/bottom fence lines, preserve everything in between.
    migrated = legacy.replace(
        "# >>> claude-codex-local >>>",
        f"# >>> claude-codex-local:{harness} >>>",
        1,
    ).replace(
        "# <<< claude-codex-local <<<",
        f"# <<< claude-codex-local:{harness} <<<",
        1,
    )
    return existing[: match.start()] + migrated + existing[match.end() :]


def _remove_legacy_pi_helper(state_dir: Path) -> bool:
    """
    Delete the pre-#120 `cp` helper script if it exists.

    Returns True when a legacy helper was removed. The current `ccp` helper
    is left untouched. Safe to call when no legacy helper exists.
    """
    legacy_path = state_dir / "bin" / "cp"
    if legacy_path.is_file():
        try:
            legacy_path.unlink()
            return True
        except OSError:
            return False
    return False


def _fence_tag_for(harness: str, engine: str) -> str:
    """
    Derive the per-install fence-tag from semantic state.

    `state.primary_harness` stays "claude" / "codex" (semantic). The fence
    tag — used as the helper-script filename, the alias short name, and
    the ~/.zshrc fence label — is `claude` / `codex` for the local engines,
    `claude9` / `codex9` for 9router, and `claudeo` / `codexo` for
    OpenRouter. This keeps step 6/7/8 and the codex-limitation guard
    branching on `state.primary_harness` unchanged.
    """
    if engine == "9router":
        return f"{harness}9"
    if engine == "openrouter":
        return f"{harness}o"
    return harness


def _helper_script_basename(harness: str) -> str:
    """
    Map a fence tag to the helper-script filename.

    Valid harness values are the fence tags supported by the install:
    "claude" / "codex" / "pi" (existing local harnesses), their
    "claude9" / "codex9" / "pi9" 9router variants, and the OpenRouter
    variants "claudeo" / "codexo" / "pio". The script names must stay
    distinct so the local (cc/cx/ccp), 9router (cc9/cx9/cp9), and
    OpenRouter (cco/cxo/cpo) install paths can all coexist on the same
    machine. The Pi local short name is `ccp` (not `cp`) because `cp`
    is the standard POSIX copy command — see issue #120.
    """
    mapping = {
        "claude": "cc",
        "codex": "cx",
        "pi": "ccp",
        "claude9": "cc9",
        "codex9": "cx9",
        "pi9": "cp9",
        "claudeo": "cco",
        "codexo": "cxo",
        "pio": "cpo",
    }
    if harness not in mapping:
        raise ValueError(f"Unknown harness fence tag: {harness!r}")
    return mapping[harness]


def _alias_names_for(harness: str) -> list[str]:
    """
    Map a fence tag to the alias names installed in the user's shell rc.

    The 9router and OpenRouter variants intentionally expose ONLY the
    short alias (cc9/cx9/cp9 and cco/cxo/cpo). The long forms
    (claude-local / codex-local / pi-local) are reserved for the
    original local-only paths so existing shell aliases keep pointing
    where users expect. The Pi local short name is `ccp` (not `cp`)
    so the alias does not shadow the standard POSIX copy command —
    see issue #120.
    """
    mapping = {
        "claude": ["cc", "claude-local"],
        "codex": ["cx", "codex-local"],
        "pi": ["ccp", "pi-local"],
        "claude9": ["cc9"],
        "codex9": ["cx9"],
        "pi9": ["cp9"],
        "claudeo": ["cco"],
        "codexo": ["cxo"],
        "pio": ["cpo"],
    }
    if harness not in mapping:
        raise ValueError(f"Unknown harness fence tag: {harness!r}")
    return list(mapping[harness])


def _write_helper_script(harness: str, result: WireResult, *, engine: str | None = None) -> Path:
    """
    Write a small bash helper that exports any inline env and execs the
    wire-result argv. Returns the absolute path to the helper.

    `harness` is a fence tag — one of "claude", "codex", "pi",
    "claude9", "codex9", "pi9", "claudeo", "codexo", "pio" — and
    selects the helper-script filename.

    When `engine == "llamacpp"`, the helper grows a pre-flight stanza that
    probes the configured llama-server `/health` endpoint and runs
    `ccl serve` to auto-start it (with the model-load banner) when the
    server is down. This stops `ConnectionRefused` from killing every
    `cc`/`cx` invocation after a reboot, OOM, or manual kill.
    """
    pb.ensure_state_dirs()
    name = _helper_script_basename(harness)
    path = pb.STATE_DIR / "bin" / name

    lines = [
        "#!/usr/bin/env bash",
        "# Managed by claude-codex-local wizard. Re-run the wizard to update.",
        "set -e",
    ]

    if engine == "llamacpp" and pb._is_local_base_url(pb.llamacpp_base_url()):
        # Hot path = single curl probe (~10ms when server is up). Cold path
        # shells out to `ccl serve`, which prints a clear "loading model
        # into VRAM" banner and waits for /health. Absolute path to the
        # `ccl` binary is captured at install time so the alias works even
        # when the user's interactive shell PATH differs from login PATH.
        #
        # Skipped entirely for remote LLAMACPP_BASE_URL (issue #123): the
        # local helper script must not spawn (or attempt to spawn) a remote
        # server, and the local `llama-server` binary is irrelevant when the
        # harness will talk to a GPU box over the network. The harness env
        # vars below already point at the remote base URL.
        ccl_bin = shutil.which("ccl") or "ccl"
        health_url = f"http://{pb.LLAMACPP_SERVER_HOST}:{pb.LLAMACPP_SERVER_PORT}/health"
        lines.extend(
            [
                "",
                "# llama.cpp pre-flight: ensure the backing server is up.",
                f"__CCL_HEALTH_URL={shlex.quote(health_url)}",
                f"__CCL_BIN={shlex.quote(ccl_bin)}",
                'if ! curl -fsS --max-time 1 -o /dev/null "$__CCL_HEALTH_URL" 2>/dev/null; then',
                '    "$__CCL_BIN" serve || {',
                '        echo "ccl: failed to start llama-server. '
                "Run 'ccl serve' to investigate.\" >&2",
                "        exit 1",
                "    }",
                "fi",
                "",
            ]
        )

    if result.env:
        for key, value in result.env.items():
            lines.append(f"export {key}={shlex.quote(value)}")
    if result.raw_env:
        for key, value in result.raw_env.items():
            # raw_env values are shell expressions evaluated at exec-time;
            # do NOT shlex.quote them, or they become literal strings.
            # See WireResult.raw_env docstring for the security boundary.
            lines.append(f"export {key}={value}")
    quoted_argv = " ".join(shlex.quote(part) for part in result.argv)
    lines.append(f'exec {quoted_argv} "$@"')
    body = "\n".join(lines) + "\n"
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _alias_block(script_path: Path, harness: str) -> tuple[str, list[str]]:
    """Build the fenced rc-block for `harness` (a 4-way fence tag)."""
    quoted_path = shlex.quote(str(script_path))
    names = _alias_names_for(harness)
    body_lines = [
        f"# >>> claude-codex-local:{harness} >>>",
        "# Managed by claude-codex-local wizard. Re-run the wizard to update,",
        "# or delete this block to remove the aliases.",
    ]
    for n in names:
        body_lines.append(f"alias {n}={quoted_path}")
    body_lines.append(f"# <<< claude-codex-local:{harness} <<<")
    return "\n".join(body_lines) + "\n", names


def _detect_shell_rc() -> Path | None:
    shell = os.environ.get("SHELL", "")
    home = Path.home()
    if shell.endswith("zsh") or "zsh" in shell:
        rc = home / ".zshrc"
        if not rc.exists():
            rc.touch()
        return rc
    if shell.endswith("bash") or "bash" in shell:
        rc = home / ".bashrc"
        if rc.exists():
            return rc
        bp = home / ".bash_profile"
        if bp.exists():
            return bp
        rc.touch()
        return rc
    return None


def _install_shell_aliases(
    script_path: Path, harness: str, non_interactive: bool
) -> tuple[Path | None, list[str]]:
    block, names = _alias_block(script_path, harness)
    rc_path = _detect_shell_rc()
    if rc_path is None:
        warn("Unsupported shell — please add the following to your shell rc manually:")
        console.print(block)
        return None, names

    if not non_interactive:
        proceed = questionary.confirm(f"Install aliases into {rc_path}?", default=True).ask()
        if not proceed:
            info("Skipped alias install. Add this block manually to enable the aliases:")
            console.print(block)
            return None, names

    existing = rc_path.read_text() if rc_path.exists() else ""
    # One-shot migration: rewrap any legacy unified block in its per-harness
    # fence so the replace/append logic below only touches the current
    # harness's block (fixes #16).
    existing = _migrate_legacy_alias_block(existing)
    harness_re = _harness_alias_block_re(harness)
    if harness_re.search(existing):
        new_text = harness_re.sub(block, existing, count=1)
    else:
        sep = "" if existing.endswith("\n") or not existing else "\n"
        prefix = "\n" if existing else ""
        new_text = existing + sep + prefix + block
    # Pre-#120 Pi installs left an `alias cp=` line inside the `:pi` fence
    # block. If the user re-runs setup for a Pi router variant (pi9 / pio)
    # without ever re-installing the local engine, that legacy block sticks
    # around and continues to shadow POSIX `cp`. Force-rewrite it to the
    # new `:pi` block whenever any Pi-family install runs.
    if harness in ("pi9", "pio"):
        pi_re = _harness_alias_block_re("pi")
        pi_match = pi_re.search(new_text)
        if pi_match and "alias cp=" in pi_match.group(0):
            existing_ccp = pb.STATE_DIR / "bin" / "ccp"
            if existing_ccp.is_file():
                refreshed_block, _ = _alias_block(existing_ccp, "pi")
                new_text = pi_re.sub(refreshed_block, new_text, count=1)
            else:
                # No fresh ccp helper to point at — drop the stale block
                # entirely so `alias cp=` stops shadowing POSIX cp.
                new_text = pi_re.sub("", new_text, count=1)
    rc_path.write_text(new_text)
    ok(f"Installed aliases into {rc_path}: {', '.join(names)}")
    # Clean up the orphaned pre-#120 `cp` Pi helper binary. The rc block
    # was overwritten above (either by the per-harness regex for harness=="pi"
    # or by the cross-fence rewrite for pi9/pio), so the binary is now
    # unreachable from any alias.
    if harness.startswith("pi") and _remove_legacy_pi_helper(pb.STATE_DIR):
        warn(
            "Removed legacy `cp` helper — the Pi local shortcut is now `ccp` (it no "
            "longer shadows the POSIX copy command). See #120."
        )
    return rc_path, names


def step_2_65_install_aliases(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 7 — Install helper script + shell aliases")
    if not state.wire_result:
        fail("No wire result on state — run step 6 first.")
        return False
    result = WireResult(
        argv=list(state.wire_result.get("argv", [])),
        env=dict(state.wire_result.get("env", {})),
        effective_tag=state.wire_result.get("effective_tag", ""),
        raw_env=dict(state.wire_result.get("raw_env", {})),
    )
    # state.primary_harness stays "claude"/"codex"; the fence tag is a
    # presentation concern derived from harness + engine. See
    # _fence_tag_for for the rationale.
    fence_tag = _fence_tag_for(state.primary_harness, state.primary_engine)
    script_path = _write_helper_script(fence_tag, result, engine=state.primary_engine)
    state.helper_script_path = str(script_path)
    ok(f"Wrote helper script: [bold]{script_path}[/bold]")

    # Import locally so test monkeypatches on ``wizard._install_shell_aliases``
    # take effect (the original monolithic wizard.py had everything in one namespace).
    import claude_codex_local.wizard as _wiz_mod

    rc_path, names = _wiz_mod._install_shell_aliases(script_path, fence_tag, non_interactive)
    state.alias_names = names
    state.shell_rc_path = str(rc_path) if rc_path else ""
    state.mark("6.5")
    return True


# ---------------------------------------------------------------------------
# Step 7 — Verify launch command end-to-end
# ---------------------------------------------------------------------------


def _materialize_raw_env(raw_env: dict[str, str]) -> dict[str, str]:
    """Resolve trusted key-file raw env expressions for verify subprocesses."""
    resolved: dict[str, str] = {}
    for key, expr in raw_env.items():
        match = re.fullmatch(r'"\$\(cat (.+)\)"', expr)
        if not match:
            continue
        try:
            parts = shlex.split(f"cat {match.group(1)}")
        except ValueError:
            continue
        if len(parts) != 2 or parts[0] != "cat":
            continue
        try:
            resolved[key] = Path(parts[1]).read_text().strip()
        except OSError:
            continue
    return resolved


def step_2_7_verify(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 8 — Verify launch command end-to-end")
    harness = state.primary_harness
    engine = state.primary_engine
    tag = state.engine_model_tag
    if not state.wire_result:
        fail("No wire result on state — run step 6 first.")
        return False

    # CRITICAL 9router branch — never call /chat/completions for 9router,
    # that would burn paid cloud quota. We verify reachability via
    # /v1/models instead and skip the chat test entirely. The launch
    # command would otherwise issue a paid `claude --model … -p READY`
    # call against kr/claude-sonnet-4.5 (or whichever paid model the user
    # picked).
    if engine == "9router":
        result = pb.smoke_test_router9_models()
        state.verify_result = {
            "ok": bool(result.get("ok")),
            "via": "9router-models-endpoint",
            "skipped_chat": True,
            "detail": result.get("response") or result.get("error", ""),
        }
        state.save()
        if not result.get("ok"):
            fail(f"9router not reachable: {result.get('error')}")
            return False
        ok("Verify (9router): /v1/models reachable, skipping chat call to avoid quota burn.")
        state.mark("7")
        return True

    # CRITICAL OpenRouter branch — same paid-cloud-quota constraint as
    # 9router. Probe /models on the hosted endpoint and skip the chat
    # test entirely; otherwise the verify command would issue a paid
    # `claude --model anthropic/claude-sonnet-4.6 -p READY` call.
    if engine == "openrouter":
        result = pb.smoke_test_openrouter_models()
        state.verify_result = {
            "ok": bool(result.get("ok")),
            "via": "openrouter-models-endpoint",
            "skipped_chat": True,
            "detail": result.get("response") or result.get("error", ""),
        }
        state.save()
        if not result.get("ok"):
            fail(f"OpenRouter not reachable: {result.get('error')}")
            return False
        ok("Verify (OpenRouter): /models reachable, skipping chat call to avoid quota burn.")
        state.mark("7")
        return True

    wire_env: dict[str, str] = dict(state.wire_result.get("env", {}))
    wire_env.update(_materialize_raw_env(dict(state.wire_result.get("raw_env", {}))))

    # The verify command talks to llama-server over HTTP. If the server is
    # gone (OOM, killed, machine slept since Step 5), the harness errors with
    # `ConnectionRefused` and Step 7 fails for an unrelated reason. Restart
    # it transparently before running the verify command.
    if engine == "llamacpp":
        ensure = _ensure_llamacpp_server_running(state)
        if not ensure.get("ok"):
            fail(f"Cannot run verify: {ensure.get('error')}")
            return False

    if harness == "claude":
        if engine == "ollama" and pb._is_local_base_url(pb.ollama_base_url()):
            cmd = [
                "ollama",
                "launch",
                "claude",
                "--model",
                tag,
                "--",
                "-p",
                "Reply with exactly READY",
                "--model",
                tag,
            ]
        else:
            cmd = list(state.wire_result["argv"]) + ["-p", "Reply with exactly READY"]
    elif harness == "codex":
        if engine == "ollama" and pb._is_local_base_url(pb.ollama_base_url()):
            cmd = [
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
                "Reply with exactly READY",
            ]
        else:
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "-m",
                tag,
                "Reply with exactly READY",
            ]
    elif harness == "pi":
        cmd = list(state.wire_result["argv"]) + ["-p", "Reply with exactly READY"]
    else:
        fail(f"Unknown harness: {harness}")
        return False

    info(f"Running: {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, **wire_env},
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        fail("Verify command timed out after 5 minutes.")
        return False

    output = (proc.stdout or "") + (proc.stderr or "")
    ready = "READY" in output.upper()
    state.verify_result = {
        "ok": ready,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }
    state.save()
    if not ready:
        fail(f"Verify failed (rc={proc.returncode}). See wizard-state.json for details.")
        if proc.stderr:
            console.print(f"[dim]{proc.stderr[-400:]}[/dim]")
        if proc.stdout:
            console.print(f"[dim]{proc.stdout[-400:]}[/dim]")
        # Targeted hint for the most common llama.cpp failure: the auto-started
        # server's --ctx-size is smaller than the harness's system prompt
        # (Claude Code is ~26k tokens). Suggest the env-var override and the
        # exact restart sequence rather than leaving the user to puzzle it out.
        if engine == "llamacpp" and "context size" in output.lower():
            warn(
                "The running llama-server's --ctx-size is too small for the harness's "
                "system prompt. Stop it, set a larger context, and re-run:"
            )
            console.print(
                "  [bold]pkill -f llama-server && LLAMACPP_CTX_SIZE=131072 ccl --resume[/bold]"
            )
        return False
    ok("End-to-end verify succeeded (got READY).")
    state.mark("7")
    return True


# ---------------------------------------------------------------------------
# Step 8 — Generate personalized guide.md
# ---------------------------------------------------------------------------

GUIDE_TEMPLATE = """\
# Local coding guide (generated)

This file was generated by `ccl` on your machine.

## What was set up

- **Harness**: `{harness}`
- **Engine**: `{engine}`
- **Model**: `{model}`
{alias_summary}
- **Helper script**: `{helper_script}`

## Daily use

{daily_use}

{launch_explanation}. Claude/Codex helpers either run `ollama launch {harness}`
(Ollama path) or export the engine env vars (LM Studio / llama.cpp / vLLM).
Pi helpers set `PI_CODING_AGENT_DIR` to Pi's normal config directory and
launch `pi --provider ccl-{engine} --model {model}`.

{settings_note}

You can still pass extra args: `{launch_command} -p "what does foo.py do?"`.
{codex_limitation}
## Troubleshooting

{alias_troubleshooting}
- **Engine not responding?** Re-run the wizard smoke test:
  ```bash
  ccl doctor
  ```
- **Want to switch models?** Re-run the wizard:
  ```bash
  ccl setup --resume
  ```

## Return to official mode

Run `claude`, `codex`, or `pi` directly (without `cc`/`cx`/`ccp`) to use the
tool normally. If this setup targeted Pi, the selected model/provider is also
present in Pi's normal settings file.

## Rollback

Each install (claude / codex / pi / claude9 / codex9 / pi9 / claudeo /
codexo / pio) has its own fenced block, so you can remove just this one
without touching any other install.

To wipe only this install:

1. Delete the fenced block for `{fence_tag}` from `{shell_rc}` (between the
   `# >>> claude-codex-local:{fence_tag} >>>` and
   `# <<< claude-codex-local:{fence_tag} <<<` markers).
2. `rm -f {helper_script}`
3. `rm -f {guide_path}`
4. For Pi installs, optionally remove this install's `ccl-*` provider from
   Pi's normal `models.json`.

To wipe every ccl install (all fence-tagged blocks):

1. Delete every `# >>> claude-codex-local:<fence-tag> >>>` block from
   `{shell_rc}`.
2. `rm -rf {state_dir}`
3. `rm -f {guide_path}`
4. For Pi installs, optionally remove all `ccl-*` providers from Pi's normal
   `models.json`.
"""


def step_2_8_generate_guide(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 9 — Generate personalized guide.md")
    fence_tag = _fence_tag_for(state.primary_harness, state.primary_engine)
    alias_names = state.alias_names or _alias_names_for(fence_tag)
    alias_short = alias_names[0]
    # claude9 / codex9 / claudeo / codexo only have the short alias; reuse it as the long form.
    alias_long = alias_names[1] if len(alias_names) > 1 else alias_names[0]
    alias_installed = bool(state.shell_rc_path)
    helper_script = state.helper_script_path or "(helper script)"
    launch_command = alias_short if alias_installed else helper_script
    if alias_installed:
        alias_summary = (
            f"- **Aliases**: `{alias_short}`, `{alias_long}` (installed in `{state.shell_rc_path}`)"
        )
        daily_use = (
            "> **First time after setup?** Reload your shell so the new alias is on\n"
            f"> your `PATH` — run `source {state.shell_rc_path}` or open a new terminal. You only\n"
            "> need to do this once per shell session.\n\n"
            "Then run:\n\n"
            "```bash\n"
            f"{alias_short}\n"
            "```"
        )
        launch_explanation = f"That's it. The alias execs `{helper_script}`"
        alias_troubleshooting = (
            f"- **`{alias_short}: command not found`?** Open a new terminal or run\n"
            f"  `source {state.shell_rc_path}`."
        )
    else:
        alias_summary = (
            f"- **Aliases**: `{alias_short}`, `{alias_long}` (not installed; helper script written)"
        )
        daily_use = (
            "> Alias installation was skipped, so no shell reload is required.\n\n"
            "Run the helper script directly:\n\n"
            "```bash\n"
            f"{helper_script}\n"
            "```\n\n"
            f"To enable `{alias_short}` later, re-run `ccl setup --resume` and accept "
            "alias installation, or add the alias block printed by setup to your shell rc."
        )
        launch_explanation = (
            f"That helper script execs the selected harness command from `{helper_script}`"
        )
        alias_troubleshooting = (
            f"- **Want the short `{alias_short}` alias later?** Re-run `ccl setup --resume` "
            "and accept alias installation, or add the printed alias block to your shell rc."
        )

    if state.primary_harness == "codex":
        settings_note = (
            "Codex's normal config is updated so the selected model/provider "
            "is available from the helper script/alias. Your interactive `/model` "
            "list is controlled by Codex itself, so the custom model may not "
            "appear there. Use the helper script/alias for the CCL-configured model "
            "and run `codex` directly to switch back to your usual model."
        )
    elif state.primary_harness == "pi":
        settings_note = (
            "For Pi targets, CCL adds/updates only its `ccl-*` provider in the normal "
            "`models.json`, so installed Pi extensions, packages, skills, prompts, "
            f"themes, settings, and auth stay available from `{launch_command}`."
        )
    else:
        settings_note = "Your real `~/.claude`, `~/.codex`, and Pi config are used as-is."

    codex_limitation = ""
    if state.primary_harness == "codex":
        tag = state.engine_model_tag
        codex_limitation = (
            f"\n"
            f"> **Codex model list note**: CCL launches Codex with `{tag}` via\n"
            f"> `{launch_command}`. Codex owns the interactive `/model` picker, so\n"
            f"> custom launch-time models may not appear there. Use `{launch_command}`\n"
            f"> to keep starting Codex on the selected CCL model/provider.\n"
        )
        if state.primary_engine == "ollama":
            codex_limitation += (
                f"\n"
                f'> **Known limitation**: `{launch_command} exec "prompt"` does not work\n'
                f"> for one-shot runs. The `--oss --local-provider=ollama` flags in the\n"
                f"> helper are top-level options and land before the `exec` subcommand,\n"
                f"> which Codex rejects with a ChatGPT-account error. Interactive\n"
                f"> `{launch_command}` works fine. For one-shot use, run directly:\n"
                f"> ```bash\n"
                f"> ollama launch codex --model {tag} -- exec --oss "
                f'--local-provider=ollama --skip-git-repo-check "<prompt>"\n'
                f"> ```\n"
            )
    # Resolve GUIDE_PATH via the wizard module so that conftest monkeypatches
    # (which redirect GUIDE_PATH under tmp_path) take effect.
    import claude_codex_local.wizard as _wiz

    guide_path = _wiz.GUIDE_PATH
    content = GUIDE_TEMPLATE.format(
        harness=state.primary_harness,
        engine=state.primary_engine,
        model=state.engine_model_tag,
        alias_short=alias_short,
        alias_long=alias_long,
        shell_rc=state.shell_rc_path or "(your shell rc)",
        helper_script=helper_script,
        alias_summary=alias_summary,
        daily_use=daily_use,
        launch_explanation=launch_explanation,
        launch_command=launch_command,
        settings_note=settings_note,
        alias_troubleshooting=alias_troubleshooting,
        state_dir=pb.STATE_DIR,
        guide_path=guide_path,
        codex_limitation=codex_limitation,
        fence_tag=fence_tag,
    )
    guide_path.write_text(content)
    ok(f"Wrote [bold]{guide_path}[/bold]")
    state.mark("8")
    return True


# ---------------------------------------------------------------------------
# Wizard driver
# ---------------------------------------------------------------------------

