#!/usr/bin/env python3
"""
Interactive first-run wizard for claude-codex-local.

Implements the 8-step flow from PRD v1.2 §4.1:

  1 Discover environment (harnesses, engines, llmfit, disk)
  2 Defer install prompts until selected component checks
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
import contextlib
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
    primary_engine: str = ""  # "ollama" | "lmstudio" | "llamacpp" | "vllm" | "9router"
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


def step_2_1_discover(
    state: WizardState,
    non_interactive: bool = False,
    force_scan: bool = False,
    run_llmfit_flag: bool = False,
) -> bool:
    header("Step 1 — Discover environment")

    if force_scan:
        pb.invalidate_machine_profile_inproc_cache()

    profile = pb.machine_profile(run_llmfit=run_llmfit_flag)
    state.profile = profile

    if run_llmfit_flag:
        _refresh_llmfit_for_profile(profile)

    llmfit_sys = profile.get("llmfit_system")
    llmfit_skipped = pb._is_llmfit_skipped(llmfit_sys)
    disk = profile.get("disk", {})

    console.print("[bold]Machine Specifications[/bold]")
    spec_table = Table(show_header=True, header_style="bold blue")
    spec_table.add_column("Specification", style="cyan")
    spec_table.add_column("Value", style="green")

    if llmfit_skipped:
        spec_table.add_row("CPU", "(scan deferred)")
        spec_table.add_row("RAM", "(scan deferred)")
        spec_table.add_row("GPU", "(scan deferred)")
        spec_table.add_row(
            "Platform",
            f"{platform.system()} / {platform.machine()}",
        )
    elif llmfit_sys:
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

    info(
        "Harness and engine status will be checked when you select them in Step 3. "
        "Use --run-llmfit to refresh hardware recommendations."
    )

    state.mark("1")
    return True


# ---------------------------------------------------------------------------
# Step 2 — Defer install prompts
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
    "vllm": {
        "name": "vLLM",
        "cmd": (
            "pip install vllm  &&  "
            "vllm serve <hf-model-id> --host 0.0.0.0 --port 8000   "
            "# expects an OpenAI-compatible API at $VLLM_BASE_URL (default http://localhost:8000)"
        ),
        "url": "https://docs.vllm.ai/",
    },
    "9router": {
        "name": "9router",
        "cmd": "npm install -g 9router  # OpenAI-compatible API at http://localhost:20128/v1",
        "url": "https://github.com/decolua/9router",
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
    header("Step 2 — Defer install prompts")
    info("Install prompts are tied to the harness and engine you choose in Step 3.")
    state.mark("2")
    return True


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
    For 9router the npm package is installed, but the long-running daemon
    must be started manually by the user.  For tools requiring manual steps
    (lmstudio, vllm) the hint is shown
    and the user is asked to confirm when done, then the profile is re-probed.
    Returns True when the tool is detected as present after the attempt.
    """
    # 9router ships as an npm package (`npm install -g 9router`) but the
    # detection check is HTTP-based — the server has to be running too.
    # We can install for the user, but the long-running `9router` daemon
    # has to be started manually in another terminal.
    if key == "9router":
        if pb.Router9Adapter().detect().get("present"):
            return True
        _show_install_hint(key)
        if not shutil.which("9router"):
            if not shutil.which("npm"):
                warn(
                    "npm not found. Install Node.js (https://nodejs.org) first, "
                    "then re-run the wizard so it can install 9router."
                )
                return False
            install = questionary.confirm(
                "Install 9router globally now via [npm install -g 9router]?",
                default=True,
            ).ask()
            if not install:
                warn(
                    f"9router not installed. Once you install and start it on "
                    f"{pb.ROUTER9_BASE_URL}, re-run the wizard."
                )
                return False
            try:
                subprocess.run(["npm", "install", "-g", "9router"], check=True)
            except subprocess.CalledProcessError as exc:
                fail(f"npm install -g 9router failed: {exc}")
                return False
            ok("9router installed.")
        # The dashboard server is long-running — we don't fork-spawn it from
        # the wizard. Ask the user to start it in another terminal, then probe.
        console.print(
            "\n[bold]Start 9router in another terminal:[/bold]\n"
            "    [cyan]9router[/cyan]\n"
            "Then sign in and add provider keys at the dashboard "
            "(http://localhost:20128)."
        )
        proceed = questionary.confirm(
            f"Confirm when 9router is running and reachable at {pb.ROUTER9_BASE_URL}.",
            default=True,
        ).ask()
        if not proceed:
            return False
        if pb.Router9Adapter().detect().get("present"):
            # Tool was just installed — invalidate the in-process profile
            # cache so the next discover step sees the freshly-installed
            # engine without forcing a fresh disk scan.
            pb.invalidate_machine_profile_inproc_cache()
            ok("9router is reachable.")
            return True
        warn(f"9router still not reachable at {pb.ROUTER9_BASE_URL}.")
        return False

    # vLLM: CLI must be installed (via pip install vllm), and the server must be
    # running (via vllm serve) to serve models. The CLI check comes first.
    if key == "vllm":
        if pb.vllm_info().get("present"):
            return True
        _show_install_hint(key)
        warn(
            f"vLLM not reachable at {pb.VLLM_BASE_URL}. "
            "Install vLLM and start `vllm serve <model>` then re-run the wizard."
        )
        return False

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
        present = pb.command_version(detect_cmd).get("present", False)
        if present:
            # Manual install completed — invalidate the in-process cache so
            # subsequent discover/picker calls see the new engine.
            pb.invalidate_machine_profile_inproc_cache()
        return present

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

    # Successful install + re-detect — invalidate in-process profile cache
    # so subsequent discover/picker calls see the freshly-installed tool.
    pb.invalidate_machine_profile_inproc_cache()
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
_ALL_ENGINES = ["ollama", "lmstudio", "llamacpp", "vllm", "9router"]


def _persist_targeted_profile_update(profile: dict[str, Any]) -> None:
    """Best-effort persistence after a selected-component live probe."""
    try:
        fingerprint = pb._compute_machine_fingerprint(profile)
        profile["_fingerprint"] = fingerprint
        pb._save_machine_profile_cache(profile, fingerprint)
        pb._set_machine_profile_in_process_cache(profile)
    except Exception:
        # Cache writes are an optimization. A failed targeted refresh must not
        # block setup; the live probe result remains in WizardState.
        pass


def _sync_presence_from_tools(profile: dict[str, Any]) -> dict[str, Any]:
    tools = profile.setdefault("tools", {})
    presence = profile.setdefault("presence", {})
    # Some tests and older wizard-state snapshots carry only the compact
    # presence block. Seed equivalent cached tool entries so a targeted refresh
    # only changes the selected component instead of erasing the rest.
    for harness in presence.get("harnesses", []) or []:
        tools.setdefault(harness, {"present": True, "version": ""})
    for engine in presence.get("engines", []) or []:
        tools.setdefault(engine, {"present": True, "version": ""})
    if presence.get("llmfit"):
        tools.setdefault("llmfit", {"present": True, "version": ""})
    harnesses = [h for h in _ALL_HARNESSES if tools.get(h, {}).get("present")]
    engines = [e for e in _ALL_ENGINES if tools.get(e, {}).get("present")]
    presence.update(
        {
            "harnesses": harnesses,
            "engines": engines,
            "llmfit": bool(tools.get("llmfit", {}).get("present")),
            "has_minimum": bool(harnesses) and bool(engines),
        }
    )
    return presence


def _refresh_llmfit_for_profile(profile: dict[str, Any]) -> bool:
    """
    Refresh only the llmfit system block.

    This is intentionally narrower than rebuilding machine_profile(): --run-llmfit
    means "refresh recommendations", not "probe every supported backend".
    """
    profile.setdefault("tools", {})["llmfit"] = pb.command_version("llmfit")
    llmfit_sys = pb.llmfit_system()
    if llmfit_sys:
        profile["llmfit_system"] = llmfit_sys
    else:
        profile.pop("llmfit_system", None)
    _sync_presence_from_tools(profile)
    _persist_targeted_profile_update(profile)
    return bool(llmfit_sys)


def _refresh_selected_harness(profile: dict[str, Any], harness: str) -> bool:
    """Live-check only the selected harness and update the cached snapshot."""
    if harness not in _ALL_HARNESSES:
        return False
    profile.setdefault("tools", {})[harness] = pb.command_version(harness)
    _sync_presence_from_tools(profile)
    _persist_targeted_profile_update(profile)
    return bool(profile["tools"][harness].get("present"))


def _refresh_selected_engine(profile: dict[str, Any], engine: str) -> bool:
    """Live-check only the selected engine and update the cached snapshot."""
    tools = profile.setdefault("tools", {})
    if engine == "ollama":
        info = pb.command_version("ollama")
        tools["ollama"] = info
        profile["ollama"] = {"models": pb.parse_ollama_list() if info.get("present") else []}
    elif engine == "lmstudio":
        lms = pb.lms_info()
        profile["lmstudio"] = lms
        tools["lmstudio"] = {
            "present": bool(lms.get("present")),
            "version": pb.command_version("lms")["version"] if lms.get("present") else "",
            "error": lms.get("error", ""),
        }
    elif engine == "llamacpp":
        info = pb.llamacpp_detect()
        tools["llamacpp"] = info
        profile["llamacpp"] = info
    elif engine == "vllm":
        vllm = pb.vllm_info()
        profile["vllm"] = vllm
        tools["vllm"] = {
            "present": bool(vllm.get("present")),
            "version": vllm.get("version", ""),
            "base_url": vllm.get("base_url", pb.VLLM_BASE_URL),
        }
    elif engine == "9router":
        adapter = pb.Router9Adapter()
        info = adapter.detect()
        health = (
            adapter.healthcheck()
            if info.get("present")
            else {"ok": False, "detail": "9router endpoint not reachable"}
        )
        tools["9router"] = {
            "present": bool(info.get("present")),
            "version": info.get("version", ""),
            "base_url": pb.ROUTER9_BASE_URL,
        }
        profile["9router"] = {
            "present": bool(info.get("present")),
            "base_url": pb.ROUTER9_BASE_URL,
            "healthcheck": health,
        }
    else:
        return False

    _sync_presence_from_tools(profile)
    _persist_targeted_profile_update(profile)
    return engine in profile["presence"].get("engines", [])


def _show_selected_harness_status(state: WizardState) -> None:
    """Show the live CLI status and any existing local helper wiring."""
    harness = state.primary_harness
    if not harness:
        return

    tool_info = state.profile.get("tools", {}).get(harness, {})
    if tool_info.get("present"):
        ok(f"{harness} CLI detected: {tool_info.get('version') or 'found'}")
    else:
        warn(f"{harness} CLI is not currently detected.")

    if state.helper_script_path:
        script_path = Path(state.helper_script_path)
        if script_path.exists():
            ok(f"Existing local helper present: {script_path}")
        else:
            warn(f"Previous local helper missing: {script_path}; setup will recreate it later.")
    elif state.wire_result:
        info("Existing wire configuration found; helper aliases are not installed yet.")
    else:
        info("No existing local helper configuration recorded; setup will wire it later.")


def _is_model_compatible_with_engine(state: WizardState, engine: str) -> bool:
    """Check if the selected model is compatible with the given engine."""
    if not state.engine_model_tag or not state.model_source:
        return False

    if state.model_source == "9router-direct":
        return engine == "9router"
    if state.model_source == "vllm-loaded":
        return engine == "vllm"
    if state.model_source == "running-server":
        return engine == "llamacpp"
    return engine in ("ollama", "lmstudio", "llamacpp")


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
        state.secondary_harnesses = harnesses[1:]
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
            _refresh_selected_engine(state.profile, choice)
            engines = state.profile["presence"]["engines"]
            if choice not in engines:
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


def _step_4_pick_model_9router(state: WizardState, non_interactive: bool = False) -> bool:
    """Step 4 specialisation for engine=9router.

    Skips llmfit/disk/download entirely — 9router routes to cloud models
    that aren't downloaded locally. Asks the user for an API key (or reads
    CCL_9ROUTER_API_KEY from env) and writes it to ROUTER9_KEY_FILE with
    chmod 0o600. Then asks for a model name with default kr/claude-sonnet-4.5.
    """
    pb.ensure_state_dirs()

    # --- API key ---
    env_key = os.environ.get("CCL_9ROUTER_API_KEY", "").strip()
    if non_interactive:
        api_key = env_key
        if not api_key and pb.ROUTER9_KEY_FILE.exists():
            # In non-interactive mode, accept a previously written key file.
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

    # --- Model name ---
    env_model = os.environ.get("CCL_9ROUTER_MODEL", "").strip()
    if non_interactive:
        model_name = env_model or _ROUTER9_DEFAULT_MODEL
    else:
        prompt_default = env_model or _ROUTER9_DEFAULT_MODEL
        model_input = questionary.text(
            "9router model name:",
            default=prompt_default,
        ).ask()
        if not model_input:
            fail("No model name provided.")
            return False
        model_name = model_input.strip()

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


def _step_4_pick_model_vllm(state: WizardState, non_interactive: bool = False) -> bool:
    """Step 4 specialisation for engine=vllm.

    A running vLLM server already has its model loaded (`vllm serve <id>`),
    so there's nothing to download and no llmfit/disk math to do — we just
    confirm which loaded model to talk to. /v1/models is queried via
    vllm_info() and the user picks one. With a single loaded model (the
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


def step_2_4_pick_model(
    state: WizardState,
    non_interactive: bool = False,
    run_llmfit_flag: bool = False,
) -> bool:
    header("Step 4 — Pick a model")
    engine = state.primary_engine

    # 9router is a cloud-routing engine — no local models, no llmfit, no
    # disk-based size checks. Branch to a dedicated picker.
    if engine == "9router":
        return _step_4_pick_model_9router(state, non_interactive)

    # vLLM hosts exactly one model per `vllm serve` process; we read it
    # from /v1/models rather than llmfit / disk.
    if engine == "vllm":
        return _step_4_pick_model_vllm(state, non_interactive)

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
        # Pre-populate the merged installed-or-cached model list for the chosen
        # engine (issue #79). Live entries win over cached duplicates. Per-mode
        # llmfit recommendations (issue #35) still come from the cached profile
        # — we never re-probe unless the user explicitly chooses "refresh".
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

            # --- Merged model list (installed + cached recommendations) ---
            merged_entries: list[questionary.Choice] = []
            for idx, entry in enumerate(merged_models):
                if running_llamacpp_model and entry.get("running"):
                    # Already surfaced as the top "running llama-server" choice.
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
                # Show installed-first ordering (already done by merge_models_for_engine
                # via installed_models_for_engine's coder-first sort).
                choices.append(questionary.Separator("── Installed or recommended ──"))
                choices.extend(merged_entries)
            else:
                # Empty state — no installed models, llmfit either absent or skipped.
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

            # --- Refresh recommendations on demand (issue #79) ---
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
                # User asked for a fresh llmfit scan. Drop the in-process
                # cache so machine_profile() goes back to the disk cache and
                # detects the skip-sentinel, then re-runs llmfit.
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
                # Map to the existing model_source vocabulary that downstream
                # steps (5/6/7) and run_doctor already understand:
                #   installed model → "installed"
                #   cached llmfit recommendation needing download → "find-model"
                # (semantically identical to the "Help me pick" llmfit path —
                # we still went through llmfit ranking, the user just picked
                # from the merged inline list).
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

    # Phase 1 — handle a server that is already running on our port.
    info_dict = pb.llamacpp_info()
    if info_dict.get("server_running"):
        running_model = (info_dict.get("model") or "").strip()
        if _llamacpp_models_match(running_model, tag):
            ok(
                f"Reusing running llama-server on "
                f"{pb.LLAMACPP_SERVER_HOST}:{info_dict['server_port']} "
                f"(model: {running_model or tag})"
            )
            return pb.smoke_test_llamacpp_model(tag)

        # Different model on our port — refuse to kill it. Ask interactively.
        warn(
            f"A different llama.cpp model is already loaded on port "
            f"{info_dict['server_port']}: '{running_model or 'unknown'}'.\n"
            f"  Wanted: '{tag}'."
        )
        if non_interactive:
            return {
                "ok": False,
                "error": (
                    f"port {info_dict['server_port']} is serving a different model "
                    f"('{running_model or 'unknown'}'); aborting in non-interactive mode"
                ),
            }
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
        smoke_target = running_model or tag
        return pb.smoke_test_llamacpp_model(smoke_target)

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
    info(f"Server log: {handle.log_path}")

    smoke = pb.smoke_test_llamacpp_model(tag)
    if not smoke.get("ok"):
        smoke = dict(smoke)
        smoke.setdefault("manual_command", manual_cmd)
        smoke.setdefault("log_path", handle.log_path)
    return smoke


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
    if info_dict.get("server_running"):
        running_model = (info_dict.get("model") or "").strip()
        if not running_model or _llamacpp_models_match(running_model, tag):
            return {"ok": True, "reused": True}
        return {
            "ok": False,
            "error": (
                f"port {info_dict['server_port']} is serving a different model "
                f"('{running_model}'); wanted '{tag}'"
            ),
        }
    model_path = state.profile.get("llamacpp_model_path") or ""
    if not model_path or not Path(model_path).is_file():
        return {
            "ok": False,
            "error": (
                f"no llama-server running and no resolved GGUF path for '{tag}' — "
                f"re-run wizard step 4 to capture the file path"
            ),
        }
    info(
        f"Restarting llama-server on "
        f"{pb.LLAMACPP_SERVER_HOST}:{pb.LLAMACPP_SERVER_PORT} "
        f"with model {Path(model_path).name}..."
    )
    start_result = pb.llamacpp_start_server(
        model_path=model_path,
        profile=state.profile,
        port=pb.LLAMACPP_SERVER_PORT,
        host=pb.LLAMACPP_SERVER_HOST,
    )
    if not start_result.get("ok"):
        return {
            "ok": False,
            "error": start_result.get("error") or "auto-start failed",
            "log_path": start_result.get("log_path"),
        }
    handle = start_result["handle"]
    ok(f"llama-server is ready (pid {handle.pid}, port {handle.port}).")
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
        # Auto-start llama-server with the just-downloaded GGUF model so the
        # user doesn't have to launch it by hand (issue #53).
        result = _llamacpp_smoke_test(state, non_interactive=non_interactive)
    elif engine == "vllm":
        # vLLM is user-managed (Python venv + GPU drivers); the wizard never
        # starts the server. Hit the OpenAI-compatible chat endpoint directly.
        base_url = state.profile.get("vllm", {}).get("base_url") or pb.VLLM_BASE_URL
        api_key = ""
        if pb.VLLM_KEY_FILE.exists():
            api_key = pb.VLLM_KEY_FILE.read_text().strip()
        if not api_key:
            api_key = os.environ.get("VLLM_API_KEY", "")
        result = pb.smoke_test_vllm_model(tag, base_url=base_url, api_key=api_key)
    elif engine == "9router":
        # CRITICAL: never call /chat/completions for 9router — that's paid
        # cloud quota. We verify reachability by re-checking /v1/models.
        result = pb.smoke_test_router9_models()
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
    fence_tag = _fence_tag_for(harness, engine)
    alias_short = _alias_names_for(fence_tag)[0]
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
        raw_env: dict[str, str] = {}
        if pb.VLLM_KEY_FILE.exists():
            key_expr = f'"$(cat {shlex.quote(str(pb.VLLM_KEY_FILE))})"'
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
    if engine == "vllm":
        # Same pattern as _wire_claude(engine="vllm"): if the user wrote a
        # key file we read it at exec-time; otherwise a placeholder is fine
        # (vLLM doesn't validate keys unless `--api-key` was passed).
        base_url = pb.VLLM_BASE_URL.rstrip("/")
        env = {"OPENAI_BASE_URL": f"{base_url}/v1"}
        raw_env: dict[str, str] = {}
        if pb.VLLM_KEY_FILE.exists():
            key_expr = f'"$(cat {shlex.quote(str(pb.VLLM_KEY_FILE))})"'
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


def _fence_tag_for(harness: str, engine: str) -> str:
    """
    Derive the per-install fence-tag from semantic state.

    `state.primary_harness` stays "claude" / "codex" (semantic). The fence
    tag — used as the helper-script filename, the alias short name, and
    the ~/.zshrc fence label — is `claude` / `codex` for the local engines
    and `claude9` / `codex9` for 9router. This keeps step 6/7/8 and the
    codex-limitation guard branching on `state.primary_harness` unchanged.
    """
    if engine == "9router":
        return f"{harness}9"
    return harness


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


def _write_helper_script(harness: str, result: WireResult, *, engine: str | None = None) -> Path:
    """
    Write a small bash helper that exports any inline env and execs the
    wire-result argv. Returns the absolute path to the helper.

    `harness` is a fence tag — one of "claude", "codex", "claude9",
    "codex9" — and selects the helper-script filename.

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

    if engine == "llamacpp":
        # Hot path = single curl probe (~10ms when server is up). Cold path
        # shells out to `ccl serve`, which prints a clear "loading model
        # into VRAM" banner and waits for /health. Absolute path to the
        # `ccl` binary is captured at install time so the alias works even
        # when the user's interactive shell PATH differs from login PATH.
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
    rc_path.write_text(new_text)
    ok(f"Installed aliases into {rc_path}: {', '.join(names)}")
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

    rc_path, names = _install_shell_aliases(script_path, fence_tag, non_interactive)
    state.alias_names = names
    state.shell_rc_path = str(rc_path) if rc_path else ""
    state.mark("6.5")
    return True


# ---------------------------------------------------------------------------
# Step 7 — Verify launch command end-to-end
# ---------------------------------------------------------------------------


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

    wire_env: dict[str, str] = dict(state.wire_result.get("env", {}))

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

Each install (claude / codex / claude9 / codex9) has its own fenced block,
so you can remove just this one without touching any other install.

To wipe only this install:

1. Delete the fenced block for `{fence_tag}` from `{shell_rc}` (between the
   `# >>> claude-codex-local:{fence_tag} >>>` and
   `# <<< claude-codex-local:{fence_tag} <<<` markers).
2. `rm -f {helper_script}`
3. `rm -f {guide_path}`

To wipe every ccl install (all fence-tagged blocks):

1. Delete every `# >>> claude-codex-local:<fence-tag> >>>` block from
   `{shell_rc}`.
2. `rm -rf {state_dir}`
3. `rm -f {guide_path}`
"""


def step_2_8_generate_guide(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 9 — Generate personalized guide.md")
    fence_tag = _fence_tag_for(state.primary_harness, state.primary_engine)
    alias_names = state.alias_names or _alias_names_for(fence_tag)
    alias_short = alias_names[0]
    # claude9 / codex9 only have the short alias; reuse it as the long form.
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
        fence_tag=fence_tag,
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
    ("2", "Select harness", step_2_select_harness),
    ("3", "Select engine", step_3_select_engine),
    ("4", "Pick a model", step_2_4_pick_model),
    ("5", "Smoke test engine + model", step_2_5_smoke_test),
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

    for step_id, title, fn in STEPS:
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

    # Helper script (cc / cx / cc9 / cx9)
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
) -> list[str] | None:
    """
    Build the harness/engine-specific argv for a one-shot session driven by
    `ccl run -p PROMPT`.

    Mirrors the verify step's per-backend argv shape (see step_2_7_verify) so
    automation drivers get the same dispatch path the wizard already exercises.
    The Codex+Ollama branch sidesteps the documented top-level-flag limitation
    by placing `--oss --local-provider=ollama` AFTER the `exec` subcommand.
    """
    if harness == "claude":
        if engine == "ollama":
            return [
                "ollama",
                "launch",
                "claude",
                "--model",
                tag,
                "--",
                "-p",
                prompt,
                "--model",
                tag,
            ]
        return list(wire_result.get("argv", [])) + ["-p", prompt]
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
                prompt,
            ]
        return ["codex", "exec", "--skip-git-repo-check", "-m", tag, prompt]
    return None


def _resolve_wire_env(wire_result: dict[str, Any]) -> dict[str, str]:
    """
    Materialize the env dict the harness should run under.

    `wire_result.env` values are literal strings. `wire_result.raw_env` values
    are bash expressions evaluated at exec-time by the helper script (e.g.
    `"$(cat /path/to/key)"` for the 9router/vllm key files — see WireResult).
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


def run_session(prompt: str | None = None) -> int:
    """
    Exposed as `ccl run [-p PROMPT]`. Launch the configured harness with an
    optional initial prompt so external agents can drive CCL non-interactively.

    With `-p PROMPT`, the harness runs in one-shot mode (Claude Code's `-p` /
    Codex's `exec` subcommand) and exits when the response is complete — the
    common automation case. Without `-p`, behavior is identical to invoking
    the alias (`cc` / `cx` / `cc9` / `cx9`): the helper script execs the
    wired argv and the user gets an interactive session.

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

    if prompt is None:
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
        try:
            proc = subprocess.run([str(helper)])
        except KeyboardInterrupt:
            return 130
        return proc.returncode

    cmd = _build_oneshot_cmd(harness, engine, tag, state.wire_result, prompt)
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
    return proc.returncode


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
        # Other engines (Ollama, LM Studio, 9router) are usually long-lived
        # services managed outside ccl; nothing to do here.
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

    warn(
        "Starting llama-server — first run can take 30s+ while the "
        "model is loaded into VRAM. Subsequent calls will be instant."
    )
    result = _ensure_llamacpp_server_running(state)
    if not result.get("ok"):
        fail(f"Could not start llama-server: {result.get('error')}")
        return 1
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
            "  ccl run                          Launch the configured session interactively\n"
            '  ccl run -p "what is 2+2?"        Launch one-shot for agent automation\n'
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
        choices=("ollama", "lmstudio", "llamacpp", "9router"),
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

    run = sub.add_parser(
        "run",
        help="Launch the configured harness, optionally with an initial prompt",
        description=(
            "Launch the configured Claude Code or Codex session. With "
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
            force_scan=getattr(args, "force_scan", False),
            run_llmfit_flag=getattr(args, "run_llmfit", False),
        )
    if cmd == "find-model":
        return run_find_model_standalone()
    if cmd == "doctor":
        return run_doctor()
    if cmd == "serve":
        return run_serve()
    if cmd == "run":
        return run_session(prompt=getattr(args, "prompt", None))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
