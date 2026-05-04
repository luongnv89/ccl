#!/usr/bin/env python3
"""
Interactive first-run wizard for claude-codex-local.

Implements the 8-step flow from PRD v1.2 §4.1:

  1 Discover environment (harnesses, engines, llmfit, disk)
  2 Install missing components (guided sub-process)
  3 Pick preferences (primary harness + engine)
  4 Pick a model (user-first, optional find-model helper)
  5 Smoke test engine + model
  6 Wire up harness (isolated settings.json / launch config)
  7 Verify launch command end-to-end
  8 Generate personalized guide.md

The wizard is idempotent and resumable: state is checkpointed to
`.claude-codex-local/wizard-state.json` after every completed step.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from claude_codex_local import __version__
from claude_codex_local import core as pb

console = Console()

STATE_DIR = pb.STATE_DIR
STATE_FILE = STATE_DIR / "wizard-state.json"
GUIDE_PATH = Path.cwd() / "guide.md"


# ---------------------------------------------------------------------------
# WizardState — the single source of truth for wizard progress
# ---------------------------------------------------------------------------


@dataclass
class WireResult:
    argv: list[str]
    env: dict[str, str]
    effective_tag: str
    raw_env: dict[str, str] = field(default_factory=dict)
    """
    Env-var entries whose VALUES are shell expressions to be expanded at
    exec-time (e.g. `"$(cat /path/to/key)"`). Use ONLY for shell expressions
    originating in this codebase, NEVER user input. Emitted unquoted by
    `_write_helper_script` so the shell can evaluate them at exec time.
    """


@dataclass
class WizardState:
    # which steps have completed successfully
    completed_steps: list[str] = field(default_factory=list)
    # full machine profile from last discover pass
    profile: dict[str, Any] = field(default_factory=dict)
    # user's primary + secondary selections
    primary_harness: str = ""  # "claude" | "codex"
    secondary_harnesses: list[str] = field(default_factory=list)
    primary_engine: str = ""  # "ollama" | "lmstudio" | "llamacpp"
    secondary_engines: list[str] = field(default_factory=list)
    # model pick
    model_name: str = ""  # raw user input or find-model selection
    model_source: str = ""  # "direct" | "find-model"
    engine_model_tag: str = ""  # engine-specific tag (e.g. qwen3-coder:30b)
    model_candidate: dict[str, Any] = field(
        default_factory=dict
    )  # llmfit candidate metadata when available
    # launch command the wizard wired up
    launch_command: list[str] = field(default_factory=list)
    # serialized WireResult: {"argv": [...], "env": {...}, "effective_tag": "..."}
    wire_result: dict[str, Any] | None = None
    # alias install metadata
    helper_script_path: str = ""
    shell_rc_path: str = ""
    alias_names: list[str] = field(default_factory=list)
    # smoke test + verify outputs
    smoke_test_result: dict[str, Any] = field(default_factory=dict)
    verify_result: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls) -> WizardState:
        if not STATE_FILE.exists():
            return cls()
        try:
            data = json.loads(STATE_FILE.read_text())
            # Migrate pre-rename step IDs (2.1–2.8, 2.65) to the new sequential scheme.
            legacy_to_new = {
                "2.1": "1",
                "2.2": "2",
                "2.3": "3",
                "2.4": "4",
                "2.5": "5",
                "2.6": "6",
                "2.65": "6.5",
                "2.7": "7",
                "2.8": "8",
            }
            if "completed_steps" in data:
                data["completed_steps"] = [legacy_to_new.get(s, s) for s in data["completed_steps"]]
            return cls(**data)
        except Exception:
            return cls()

    def mark(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.save()


# ---------------------------------------------------------------------------
# Welcome banner
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


def print_welcome_banner() -> None:
    """Print the ASCII 3D CCL banner, tagline, version, and repo URL."""
    console.print(_CCL_BANNER, style="bold cyan", highlight=False)
    console.print(f"  [bold white]{_CCL_TAGLINE}[/bold white]")
    console.print(f"  [dim]v{__version__}  ·  [link={_CCL_REPO_URL}]{_CCL_REPO_URL}[/link][/dim]")
    console.print()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def header(title: str) -> None:
    console.print()
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def ok(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


def fail(msg: str) -> None:
    console.print(f"[red]✗[/red] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]·[/dim] {msg}")


# ---------------------------------------------------------------------------
# Step 1 — Discover environment
# ---------------------------------------------------------------------------


def step_2_1_discover(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 1 — Discover environment")
    profile = pb.machine_profile()
    state.profile = profile

    tools = profile["tools"]
    presence = profile["presence"]
    disk = profile.get("disk", {})
    llmfit_sys = profile.get("llmfit_system", {})

    table = Table(show_header=True, header_style="bold")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    def row(name: str, info: dict[str, Any]) -> None:
        if info.get("present"):
            table.add_row(name, "[green]found[/green]", info.get("version", "") or "-")
        else:
            table.add_row(name, "[red]missing[/red]", info.get("error", "") or "-")

    row("claude (harness)", tools["claude"])
    row("codex (harness)", tools["codex"])
    row("ollama (engine)", tools["ollama"])
    row("lmstudio (engine)", tools["lmstudio"])
    row("llama.cpp (engine)", tools["llamacpp"])
    row("hf / huggingface-cli (model downloader)", tools.get("huggingface_cli", {}))
    console.print(table)

    # Machine specs table
    console.print()
    console.print("[bold]Machine Specifications[/bold]")
    spec_table = Table(show_header=True, header_style="bold blue")
    spec_table.add_column("Specification", style="cyan")
    spec_table.add_column("Value", style="green")

    if llmfit_sys:
        sys_info = llmfit_sys.get("system", llmfit_sys)
        cpu_name = sys_info.get("cpu_name", "Unknown")
        cpu_cores = sys_info.get("cpu_cores", "Unknown")
        total_ram = sys_info.get("total_ram_gb", "?")
        available_ram = sys_info.get("available_ram_gb", "?")
        has_gpu = sys_info.get("has_gpu", False)
        gpu_name = sys_info.get("gpu_name", "N/A") if has_gpu else "N/A"
        gpu_vram = sys_info.get("gpu_vram_gb", 0) if has_gpu else 0

        spec_table.add_row("CPU", f"{cpu_name} ({cpu_cores} cores)")
        spec_table.add_row("RAM", f"{total_ram} GB (Available: {available_ram} GB)")
        if has_gpu:
            spec_table.add_row("GPU", f"{gpu_name} ({gpu_vram} GB VRAM)")
        spec_table.add_row(
            "Platform",
            f"{platform.system()} / {platform.machine()}",
        )
    else:
        spec_table.add_row("CPU", "Not available (llmfit not installed)")
        spec_table.add_row("RAM", "Not available (llmfit not installed)")
        spec_table.add_row("GPU", "Not available (llmfit not installed)")

    console.print(spec_table)

    free_gib = disk.get("free_gib", "?")
    total_gib = disk.get("total_gib", "?")
    info(f"Free disk on state dir: {free_gib} GiB of {total_gib} GiB")

    if presence["has_minimum"]:
        ok(f"Found harnesses: {', '.join(presence['harnesses'])}")
        ok(f"Found engines: {', '.join(presence['engines'])}")
        state.mark("1")
        return True

    # Missing pieces — fall through to step 2
    if not presence["harnesses"]:
        warn("No harness found (need claude or codex)")
    if not presence["engines"]:
        warn("No engine found (need ollama, lmstudio, or llama.cpp)")
    return False


# ---------------------------------------------------------------------------
# Step 2 — Install missing components
# ---------------------------------------------------------------------------

INSTALL_HINTS: dict[str, dict[str, str]] = {
    "claude": {
        "name": "Claude Code CLI",
        "cmd": "npm install -g @anthropic-ai/claude-code",
        "url": "https://docs.claude.com/claude-code",
    },
    "codex": {
        "name": "Codex CLI",
        "cmd": "npm install -g @openai/codex",
        "url": "https://github.com/openai/codex",
    },
    "ollama": {
        "name": "Ollama",
        "cmd": "curl -fsSL https://ollama.com/install.sh | sh",
        "url": "https://ollama.com",
    },
    "lmstudio": {
        "name": "LM Studio",
        "cmd": "# Download from https://lmstudio.ai, then: npx lmstudio install-cli",
        "url": "https://lmstudio.ai",
    },
    "llamacpp": {
        "name": "llama.cpp",
        "cmd": "brew install llama.cpp   # or build from https://github.com/ggml-org/llama.cpp",
        "url": "https://github.com/ggml-org/llama.cpp",
    },
    "huggingface-cli": {
        "name": "Hugging Face CLI",
        "cmd": "pip install 'huggingface_hub[cli]'",
        "url": "https://huggingface.co/docs/huggingface_hub/guides/cli",
    },
    "llmfit": {
        "name": "llmfit",
        "cmd": "See docs/poc-bootstrap.md for the install script",
        "url": "https://github.com/AlexsJones/llmfit",
    },
}


def step_2_2_install_missing(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2 — Install missing components")
    presence = state.profile.get("presence", {})

    missing: list[str] = []
    if not presence.get("harnesses"):
        missing.append("HARNESS (claude or codex)")
    if not presence.get("engines"):
        missing.append("ENGINE (ollama, lmstudio, or llamacpp)")

    if not missing:
        info("Nothing missing.")
        state.mark("2")
        return True

    console.print(f"Missing categories: [red]{', '.join(missing)}[/red]")

    # Offer install hints for each missing category
    if not presence.get("harnesses"):
        _show_install_hint("claude")
        _show_install_hint("codex")
    if not presence.get("engines"):
        _show_install_hint("ollama")
        _show_install_hint("lmstudio")
        _show_install_hint("llamacpp")

    if non_interactive:
        fail("Cannot install missing components in --non-interactive mode.")
        return False

    console.print()
    proceed = questionary.confirm(
        "After installing the missing pieces in another terminal, ready to re-probe?",
        default=True,
    ).ask()
    if not proceed:
        fail("Setup cancelled by user.")
        return False

    # Re-probe
    state.profile = pb.machine_profile()
    if state.profile["presence"]["has_minimum"]:
        ok("Minimum requirements now satisfied.")
        state.mark("2")
        return True
    fail("Still missing required components. Install them and re-run the wizard.")
    return False


def _show_install_hint(key: str) -> None:
    hint = INSTALL_HINTS.get(key)
    if not hint:
        return
    console.print(f"\n[bold]{hint['name']}[/bold] → {hint['url']}")
    console.print(f"    [cyan]{hint['cmd']}[/cyan]")


_LLMFIT_INSTALL_SCRIPT = """\
REPO='AlexsJones/llmfit'
BINARY='llmfit'
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
  Linux) OS='unknown-linux-musl' ;;
  Darwin) OS='apple-darwin' ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac
case "$ARCH" in
  x86_64|amd64) ARCH='x86_64' ;;
  aarch64|arm64) ARCH='aarch64' ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac
PLATFORM="${ARCH}-${OS}"
TAG=$(curl -fsSI "https://github.com/${REPO}/releases/latest" | grep -i '^location:' | head -1 | sed 's|.*/tag/||' | tr -d '\\r\\n')
ASSET="${BINARY}-${TAG}-${PLATFORM}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
curl -fsSL "$URL" -o "$TMPDIR/$ASSET"
if curl -fsSL --max-time 10 "${URL}.sha256" -o "$TMPDIR/${ASSET}.sha256"; then
  (cd "$TMPDIR" && sha256sum -c "${ASSET}.sha256")
fi
tar -xzf "$TMPDIR/$ASSET" -C "$TMPDIR"
install -d "$HOME/.local/bin"
install -m 0755 "$TMPDIR/$BINARY" "$HOME/.local/bin/$BINARY"
export PATH="$HOME/.local/bin:$PATH"
llmfit --version
"""


def _ensure_tool(key: str) -> bool:
    """
    Offer to install a tool by key (matching INSTALL_HINTS).
    For tools with a runnable install command (ollama, llamacpp, claude, codex,
    huggingface-cli) the command is executed directly.
    For tools requiring manual steps (lmstudio) the hint is shown and the user
    is asked to confirm when done, then the profile is re-probed.
    Returns True when the tool is detected as present after the attempt.
    """
    detect_cmd = {
        "claude": "claude",
        "codex": "codex",
        "ollama": "ollama",
        "lmstudio": "lms",
        "llamacpp": "llama-server",
    }.get(key, key)

    if pb.command_version(detect_cmd).get("present"):
        return True

    _show_install_hint(key)

    # lmstudio requires a manual GUI download — can't script it.
    if key == "lmstudio":
        proceed = questionary.confirm(
            "Install LM Studio manually (see link above), then confirm when ready to re-probe?",
            default=True,
        ).ask()
        if not proceed:
            return False
        return pb.command_version(detect_cmd).get("present", False)

    # All other tools have a runnable one-liner.
    hint = INSTALL_HINTS.get(key, {})
    cmd_str = hint.get("cmd", "")
    install = questionary.confirm(
        f"Run install command now?  [{cmd_str}]",
        default=True,
    ).ask()
    if not install:
        return False

    try:
        subprocess.run(["bash", "-c", cmd_str], check=True)
    except subprocess.CalledProcessError as exc:
        fail(f"Install failed: {exc}")
        return False

    if not pb.command_version(detect_cmd).get("present"):
        warn(
            f"{key} still not found after install. "
            "You may need to open a new terminal or add its bin directory to PATH."
        )
        return False

    ok(f"{key} installed successfully.")
    return True


def _ensure_llmfit() -> bool:
    """
    Check if llmfit is present. If not, offer to install it via the official
    bootstrap script. Returns True if llmfit is available after the check/install.
    """
    if pb.command_version("llmfit").get("present"):
        return True

    warn("llmfit is not installed.")
    _show_install_hint("llmfit")
    install = questionary.confirm(
        "Install llmfit now via the official bootstrap script?",
        default=True,
    ).ask()
    if not install:
        return False

    try:
        subprocess.run(["bash", "-c", _LLMFIT_INSTALL_SCRIPT], check=True)
    except subprocess.CalledProcessError as exc:
        fail(f"llmfit install failed: {exc}")
        return False

    # Add ~/.local/bin to PATH for this process so the re-check finds it.
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")

    if not pb.command_version("llmfit").get("present"):
        warn(
            "llmfit still not found after install. "
            "Ensure ~/.local/bin is on your PATH, then re-run the wizard with --resume."
        )
        return False

    ok("llmfit installed successfully.")
    return True


# ---------------------------------------------------------------------------
# Step 3 — Pick preferences
# ---------------------------------------------------------------------------


_ALL_HARNESSES = ["claude", "codex"]
_ALL_ENGINES = ["ollama", "lmstudio", "llamacpp"]


def step_2_3_pick_preferences(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 3 — Pick preferences")
    presence = state.profile["presence"]
    harnesses = presence["harnesses"]
    engines = presence["engines"]

    # Harness pick
    if non_interactive:
        if not harnesses:
            fail("No harness installed. Cannot continue in non-interactive mode.")
            return False
        state.primary_harness = harnesses[0]
        state.secondary_harnesses = harnesses[1:]
        ok(f"Non-interactive: picking [bold]{state.primary_harness}[/bold] as primary harness")
    else:
        # Show all known harnesses; mark uninstalled ones.
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
            if choice not in harnesses:
                if not _ensure_tool(choice):
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
                # Refresh presence after install.
                state.profile = pb.machine_profile()
                harnesses = state.profile["presence"]["harnesses"]
            state.primary_harness = choice
            state.secondary_harnesses = [h for h in harnesses if h != choice]
            break

    # Engine pick
    if non_interactive:
        if not engines:
            fail("No engine installed. Cannot continue in non-interactive mode.")
            return False
        default_engine = _default_engine(engines, state.profile)
        state.primary_engine = default_engine
        state.secondary_engines = [e for e in engines if e != default_engine]
        ok(f"Non-interactive: picking [bold]{state.primary_engine}[/bold] as primary engine")
    else:
        # Show all known engines; mark uninstalled ones.
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
            if choice not in engines:
                if not _ensure_tool(choice):
                    warn(
                        f"{choice} is still not available. Please pick another or install it first."
                    )
                    continue
                # Refresh presence after install.
                state.profile = pb.machine_profile()
                engines = state.profile["presence"]["engines"]
            state.primary_engine = choice
            state.secondary_engines = [e for e in engines if e != choice]
            break

    ok(f"Primary: [bold]{state.primary_harness}[/bold] + [bold]{state.primary_engine}[/bold]")
    if state.secondary_harnesses or state.secondary_engines:
        info(
            f"Fallbacks: harnesses={state.secondary_harnesses or '-'} engines={state.secondary_engines or '-'}"
        )
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


def step_2_4_pick_model(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 4 — Pick a model")
    engine = state.primary_engine

    # If llamacpp is primary and a server is already running with a model loaded,
    # offer to use that model directly — the user clearly already has it set up.
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
        # Non-interactive: go straight through find-model (prefers installed models).
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
        # Pre-populate discovered local models for the chosen engine (issue #36)
        # and per-mode llmfit recommendations (issue #35). Both read from the
        # cached profile captured in step 1 — we never re-probe here.
        installed_models = pb.installed_models_for_engine(state.profile, engine)
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

            # --- Recommendation profiles (Speed / Balanced / Quality) ---
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

            # --- Installed local models for the chosen engine ---
            installed_entries: list[questionary.Choice] = []
            for idx, entry in enumerate(installed_models):
                if running_llamacpp_model and entry.get("running"):
                    # Already surfaced as the top "running llama-server" choice.
                    continue
                key = f"installed:{idx}"
                items[key] = entry
                size_suffix = f"  ({entry.get('size')})" if entry.get("size") else ""
                installed_entries.append(
                    questionary.Choice(
                        f"Use installed {entry['source']} model: {entry['display']}{size_suffix}",
                        value=key,
                    )
                )
            if installed_entries:
                choices.append(questionary.Separator("── Installed on this machine ──"))
                choices.extend(installed_entries)

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
            if mode.startswith("installed:"):
                entry = items[mode]
                state.model_name = entry["display"]
                state.engine_model_tag = entry["tag"]
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
        # If the llama-server is already running and serving this model alias,
        # treat it as installed — the user has it set up and we shouldn't
        # prompt for a download.
        status = pb.llamacpp_info()
        return bool(status.get("server_running") and status.get("model") == tag)
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

        console.print(f"\n[cyan]Downloading {current} from Hugging Face Hub...[/cyan]")
        result = pb.huggingface_download_gguf(hf_repo, filename=filename, stream=True)
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
            if result.get("path"):
                info(f"Path: {result['path']}")
            return {
                "ok": True,
                "path": result.get("path"),
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


def step_2_5_smoke_test(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 5 — Smoke test engine + model")
    engine = state.primary_engine
    tag = state.engine_model_tag
    info(f"Running minimal prompt through {engine} / {tag}...")

    if engine == "ollama":
        result = pb.smoke_test_ollama_model(tag)
    elif engine == "lmstudio":
        # Ensure server is up + model loaded
        if not pb.lms_info().get("server_running"):
            info("Starting LM Studio server...")
            pb.lms_start_server()
        pb.lms_load_model(tag)
        result = pb.smoke_test_lmstudio_model(tag)
    elif engine == "llamacpp":
        # llama.cpp server must be started manually by the user with the GGUF model loaded.
        llamacpp_status = pb.llamacpp_info()
        if not llamacpp_status.get("server_running"):
            model_path = state.profile.get("llamacpp_model_path", "<path/to/model.gguf>")
            warn(
                f"llama.cpp server is not running on port {llamacpp_status['server_port']}. "
                f"Start it with: llama-server --port {llamacpp_status['server_port']} "
                f"--model {model_path}"
            )
            result = {"ok": False, "error": "llama.cpp server not running"}
        else:
            result = pb.smoke_test_llamacpp_model(tag)
    else:
        warn(f"Smoke test for engine '{engine}' not implemented — skipping.")
        result = {"ok": True, "response": "(skipped)"}

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
# Step 6 — Wire up harness with isolated settings
# ---------------------------------------------------------------------------


def step_2_6_wire_harness(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 6 — Wire up harness")
    harness = state.primary_harness
    engine = state.primary_engine
    tag = state.engine_model_tag

    pb.ensure_state_dirs()

    if harness == "claude":
        result = _wire_claude(engine, tag)
    elif harness == "codex":
        result = _wire_codex(engine, tag)
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
    alias_short = "cc" if harness == "claude" else "cx"
    state.launch_command = [alias_short]
    ok(f"Harness wired. argv: [bold]{' '.join(shlex.quote(x) for x in result.argv)}[/bold]")
    state.mark("6")
    return True


def _model_known_incompatible_with_claude_code(tag: str) -> bool:
    t = tag.lower()
    return "qwen3" in t


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
        base_url = f"http://localhost:{pb.LLAMACPP_SERVER_PORT}"
        env = {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": "sk-local",  # pragma: allowlist secret
            "ANTHROPIC_AUTH_TOKEN": "sk-local",  # pragma: allowlist secret
            "ANTHROPIC_CUSTOM_MODEL_OPTION": tag,
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": f"Local (llamacpp) {tag}",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION": (
                f"Local model served by llamacpp at {base_url}"
            ),
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        return WireResult(argv=["claude", "--model", tag], env=env, effective_tag=tag)
    fail(f"Unknown engine for Claude wire-up: {engine}")
    return None


def _wire_codex(engine: str, tag: str) -> WireResult | None:
    if engine == "ollama":
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
        env = {
            "OPENAI_BASE_URL": f"http://localhost:{pb.LLAMACPP_SERVER_PORT}/v1",
            "OPENAI_API_KEY": "sk-local",  # pragma: allowlist secret
        }
        return WireResult(argv=["codex", "-m", tag], env=env, effective_tag=tag)
    fail(f"Unknown engine for Codex wire-up: {engine}")
    return None


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


def _helper_script_basename(harness: str) -> str:
    """
    Map a fence tag to the helper-script filename.

    Valid harness values are the four fence tags supported by the
    install: "claude" / "codex" (existing harnesses) and "claude9" /
    "codex9" (their 9router variants from issue #51). The script names
    must stay distinct so the cc/cx (local) and cc9/cx9 (9router)
    install paths can coexist on the same machine.
    """
    mapping = {
        "claude": "cc",
        "codex": "cx",
        "claude9": "cc9",
        "codex9": "cx9",
    }
    if harness not in mapping:
        raise ValueError(f"Unknown harness fence tag: {harness!r}")
    return mapping[harness]


def _alias_names_for(harness: str) -> list[str]:
    """
    Map a fence tag to the alias names installed in the user's shell rc.

    The 9router variants intentionally expose ONLY the short alias
    (cc9 / cx9). The long forms (claude-local / codex-local) are reserved
    for the original local-only paths so existing shell aliases keep
    pointing where users expect.
    """
    mapping = {
        "claude": ["cc", "claude-local"],
        "codex": ["cx", "codex-local"],
        "claude9": ["cc9"],
        "codex9": ["cx9"],
    }
    if harness not in mapping:
        raise ValueError(f"Unknown harness fence tag: {harness!r}")
    return list(mapping[harness])


def _write_helper_script(harness: str, result: WireResult) -> Path:
    """
    Write a small bash helper that exports any inline env and execs the
    wire-result argv. Returns the absolute path to the helper.

    `harness` is a fence tag — one of "claude", "codex", "claude9",
    "codex9" — and selects the helper-script filename.
    """
    pb.ensure_state_dirs()
    name = _helper_script_basename(harness)
    path = pb.STATE_DIR / "bin" / name

    lines = [
        "#!/usr/bin/env bash",
        "# Managed by claude-codex-local wizard. Re-run the wizard to update.",
        "set -e",
    ]
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
    rc_path.write_text(new_text)
    ok(f"Installed aliases into {rc_path}: {', '.join(names)}")
    return rc_path, names


def step_2_65_install_aliases(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 6.5 — Install helper script + shell aliases")
    if not state.wire_result:
        fail("No wire result on state — run step 6 first.")
        return False
    result = WireResult(
        argv=list(state.wire_result.get("argv", [])),
        env=dict(state.wire_result.get("env", {})),
        effective_tag=state.wire_result.get("effective_tag", ""),
        raw_env=dict(state.wire_result.get("raw_env", {})),
    )
    harness = state.primary_harness
    script_path = _write_helper_script(harness, result)
    state.helper_script_path = str(script_path)
    ok(f"Wrote helper script: [bold]{script_path}[/bold]")

    rc_path, names = _install_shell_aliases(script_path, harness, non_interactive)
    state.alias_names = names
    state.shell_rc_path = str(rc_path) if rc_path else ""
    state.mark("6.5")
    return True


# ---------------------------------------------------------------------------
# Step 7 — Verify launch command end-to-end
# ---------------------------------------------------------------------------


def step_2_7_verify(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 7 — Verify launch command end-to-end")
    harness = state.primary_harness
    engine = state.primary_engine
    tag = state.engine_model_tag
    if not state.wire_result:
        fail("No wire result on state — run step 6 first.")
        return False
    wire_env: dict[str, str] = dict(state.wire_result.get("env", {}))

    if harness == "claude":
        if engine == "ollama":
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
        if engine == "ollama":
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
- **Aliases**: `{alias_short}`, `{alias_long}` (installed in `{shell_rc}`)
- **Helper script**: `{helper_script}`

## Daily use

> **First time after setup?** Reload your shell so the new alias is on
> your `PATH` — run `source {shell_rc}` or open a new terminal. You only
> need to do this once per shell session.

Then run:

```bash
{alias_short}
```

That's it. The alias execs `{helper_script}`, which either runs
`ollama launch {harness}` (Ollama path) or sets the right env vars and
execs `{harness}` directly (LM Studio / llama.cpp path).

Your real `~/.claude` and `~/.codex` are used as-is, so all your skills,
statusline, agents, plugins, and MCP servers keep working.

You can still pass extra args: `{alias_short} -p "what does foo.py do?"`.
{codex_limitation}
## Troubleshooting

- **`{alias_short}: command not found`?** Open a new terminal or run
  `source {shell_rc}`.
- **Engine not responding?** Re-run the wizard smoke test:
  ```bash
  ccl doctor
  ```
- **Want to switch models?** Re-run the wizard:
  ```bash
  ccl setup --resume
  ```

## Return to official mode

Your global `~/.claude` and `~/.codex` are unchanged. Run `claude` or
`codex` directly (without `{alias_short}`) to use the cloud backend.

## Rollback

Each harness (claude / codex) has its own fenced block, so you can remove
just this one without touching any other harness you may have set up.

To wipe only this harness:

1. Delete the fenced block for `{harness}` from `{shell_rc}` (between the
   `# >>> claude-codex-local:{harness} >>>` and
   `# <<< claude-codex-local:{harness} <<<` markers).
2. `rm -f {helper_script}`
3. `rm -f {guide_path}`

To wipe the local ccl setup entirely (both harnesses, if installed):

1. Delete every `# >>> claude-codex-local:<harness> >>>` block from
   `{shell_rc}`.
2. `rm -rf {state_dir}`
3. `rm -f {guide_path}`
"""


def step_2_8_generate_guide(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 8 — Generate personalized guide.md")
    alias_names = state.alias_names or (
        ["cc", "claude-local"] if state.primary_harness == "claude" else ["cx", "codex-local"]
    )
    alias_short = alias_names[0]
    alias_long = alias_names[1] if len(alias_names) > 1 else alias_names[0]
    codex_limitation = ""
    if state.primary_harness == "codex" and state.primary_engine == "ollama":
        tag = state.engine_model_tag
        codex_limitation = (
            f"\n"
            f'> **Known limitation**: `{alias_short} exec "prompt"` does not work\n'
            f"> for one-shot runs. The `--oss --local-provider=ollama` flags in the\n"
            f"> alias are top-level options and land before the `exec` subcommand,\n"
            f"> which Codex rejects with a ChatGPT-account error. Interactive\n"
            f"> `{alias_short}` works fine. For one-shot use, run directly:\n"
            f"> ```bash\n"
            f"> ollama launch codex --model {tag} -- exec --oss "
            f'--local-provider=ollama --skip-git-repo-check "<prompt>"\n'
            f"> ```\n"
        )
    content = GUIDE_TEMPLATE.format(
        harness=state.primary_harness,
        engine=state.primary_engine,
        model=state.engine_model_tag,
        alias_short=alias_short,
        alias_long=alias_long,
        shell_rc=state.shell_rc_path or "(your shell rc)",
        helper_script=state.helper_script_path or "(helper script)",
        state_dir=pb.STATE_DIR,
        guide_path=GUIDE_PATH,
        codex_limitation=codex_limitation,
    )
    GUIDE_PATH.write_text(content)
    ok(f"Wrote [bold]{GUIDE_PATH}[/bold]")
    state.mark("8")
    return True


# ---------------------------------------------------------------------------
# Wizard driver
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, str, Callable[[WizardState, bool], bool]]] = [
    ("1", "Discover environment", step_2_1_discover),
    ("2", "Install missing components", step_2_2_install_missing),
    ("3", "Pick preferences", step_2_3_pick_preferences),
    ("4", "Pick a model", step_2_4_pick_model),
    ("5", "Smoke test engine + model", step_2_5_smoke_test),
    ("6", "Wire up harness", step_2_6_wire_harness),
    ("6.5", "Install helper script + shell aliases", step_2_65_install_aliases),
    ("7", "Verify launch command", step_2_7_verify),
    ("8", "Generate guide.md", step_2_8_generate_guide),
]


def run_wizard(
    *,
    resume: bool = False,
    non_interactive: bool = False,
    start_step: str | None = None,
    force_harness: str | None = None,
    force_engine: str | None = None,
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

    for step_id, title, fn in STEPS:
        if resume and step_id in state.completed_steps and step_id != start_step:
            continue
        # Honor forced harness/engine by skipping the picker.
        if step_id == "3" and state.primary_harness and state.primary_engine:
            ok(
                f"Using forced picks: harness=[bold]{state.primary_harness}[/bold] engine=[bold]{state.primary_engine}[/bold]"
            )
            state.mark("3")
            continue
        # Step 2 is conditional: only run if step 1 failed presence check.
        if step_id == "2" and state.profile.get("presence", {}).get("has_minimum"):
            continue
        ok_step = fn(state, non_interactive)
        if not ok_step:
            fail(f"Step {step_id} ({title}) did not complete. Re-run with --resume to continue.")
            return 1

    alias_short = (
        state.alias_names[0]
        if state.alias_names
        else ("cc" if state.primary_harness == "claude" else "cx")
    )
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Reload your shell so the new alias is picked up:\n"
            f"  [cyan]source ~/.zshrc[/cyan]  (or [cyan]~/.bashrc[/cyan], or open a new terminal)\n\n"
            f"Then run: [cyan]{alias_short}[/cyan]\n\n"
            f"See [bold]{GUIDE_PATH}[/bold] for the full guide.",
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

    # Helper script (cc / cx)
    if state.helper_script_path:
        script_path = Path(state.helper_script_path)
        add_row(
            "helper script",
            state.helper_script_path,
            script_path.exists(),
            "present" if script_path.exists() else "missing — re-run step 6.5",
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
            "  ccl doctor                       Triage the current install\n"
            "  ccl find-model                   Show a recommended coding model\n"
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
    setup.add_argument("--harness", choices=("claude", "codex"), help="Force the primary harness")
    setup.add_argument(
        "--engine",
        choices=("ollama", "lmstudio", "llamacpp"),
        help="Force the primary engine",
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

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

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
        )
    if cmd == "find-model":
        return run_find_model_standalone()
    if cmd == "doctor":
        return run_doctor()
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
