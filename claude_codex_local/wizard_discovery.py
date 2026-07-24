"""
Wizard discovery — Step 1 environment discovery, tool installation, and
profile management.

Exports:
    INSTALL_HINTS — mapping of tool keys to install hints
    step_2_1_discover — Step 1: discover environment
    step_2_2_install_missing — Step 2: defer install prompts
    _ensure_tool — offer to install a tool by key
    _ensure_llmfit — check/install llmfit
    _persist_targeted_profile_update — persist profile after live probe
    _sync_presence_from_tools — sync presence from tools dict
    _refresh_llmfit_for_profile — refresh llmfit system block
    _try_llmfit_fallback — opportunistic llmfit invocation
    _refresh_selected_harness — live-check a harness
    _refresh_selected_engine — live-check an engine
    _show_selected_harness_status — show harness CLI status
    _is_model_compatible_with_engine — check model/engine compatibility
    _ALL_HARNESSES, _ALL_ENGINES — available harness/engine lists
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.table import Table

from claude_codex_local import core as pb
from claude_codex_local.engines import ALL_ENGINES as _REGISTRY_ENGINES
from claude_codex_local.wizard_state import WizardState
from claude_codex_local.wizard_ui import console, fail, info, ok, warn

# Forward reference — _run_engine_lifecycle is defined in wizard_steps
# to avoid circular imports at module load time.

# ---------------------------------------------------------------------------
# Step 1 — Discover environment
# ---------------------------------------------------------------------------


def step_2_1_discover(
    state: WizardState,
    non_interactive: bool = False,
    force_scan: bool = False,
    run_llmfit_flag: bool = False,
) -> bool:
    from claude_codex_local.wizard_ui import header

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

    # Issue #95: when the cached scan was deferred, try llmfit opportunistically.
    # On success, render real values; on failure (missing/error/timeout), keep
    # the "(scan deferred)" rendering below intact.
    if llmfit_skipped:
        fallback = _try_llmfit_fallback(profile)
        if fallback is not None:
            llmfit_sys = fallback
            llmfit_skipped = False

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
    "pi": {
        "name": "Pi coding agent",
        "cmd": "npm install -g @earendil-works/pi-coding-agent",
        "url": "https://pi.dev/",
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
    "openrouter": {
        "name": "OpenRouter",
        "cmd": "Get an API key at https://openrouter.ai/keys (no install required — hosted SaaS)",
        "url": "https://openrouter.ai/docs",
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
    from claude_codex_local.wizard_ui import header

    header("Step 2 — Defer install prompts")
    info("Install prompts are tied to the harness and engine you choose in Step 3.")
    state.mark("2")
    return True


def _run_engine_lifecycle(key: str, action: str) -> dict[str, Any]:
    """Run an engine lifecycle action (e.g. 'install') and return structured result."""
    from claude_codex_local.wizard_steps import _run_engine_lifecycle as _rel

    return _rel(key, action)


def _show_install_hint(key: str) -> None:
    lifecycle = _run_engine_lifecycle(key, "install")
    if lifecycle.get("ok"):
        engine_hint = INSTALL_HINTS.get(key, {})
        title = engine_hint.get("name", key)
        url = engine_hint.get("url")
        console.print(f"\n[bold]{title}[/bold]" + (f" → {url}" if url else ""))
        for cmd in lifecycle.get("commands", []):
            console.print(f"    [cyan]{cmd}[/cyan]")
        detail = lifecycle.get("detail")
        if detail:
            info(str(detail))
        return

    fallback_hint = INSTALL_HINTS.get(key)
    if not fallback_hint:
        return
    console.print(f"\n[bold]{fallback_hint['name']}[/bold] → {fallback_hint['url']}")
    console.print(f"    [cyan]{fallback_hint['cmd']}[/cyan]")


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
    pi, huggingface-cli) the command is executed directly.
    For 9router the npm package is installed, but the long-running daemon
    must be started manually by the user.  For tools requiring manual steps
    (lmstudio, vllm) the hint is shown
    and the user is asked to confirm when done, then the profile is re-probed.
    OpenRouter is hosted SaaS — nothing to install, just an API key in step 4.
    Returns True when the tool is detected as present after the attempt.
    """
    # OpenRouter is hosted SaaS at https://openrouter.ai/api/v1 — there is
    # no binary to install and no daemon to start. We only inform the user
    # that the API key prompt happens in step 4 and return True so the
    # wizard does not block here on offline machines (the key check
    # happens later regardless).
    if key == "openrouter":
        if pb.OpenRouterAdapter().detect().get("present"):
            ok("OpenRouter endpoint reachable.")
            pb.invalidate_machine_profile_inproc_cache()
            return True
        info(
            "OpenRouter is hosted at https://openrouter.ai. Make sure your "
            "network can reach it and that you have an API key "
            "(https://openrouter.ai/keys). The wizard will prompt for the "
            "key in step 4."
        )
        return True

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
        "pi": "pi",
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
# Profile management helpers
# ---------------------------------------------------------------------------


_ALL_HARNESSES = ["claude", "codex", "pi"]
# Derive the engine list from the registry so adding a new engine package
# automatically surfaces it in the wizard without touching this file.
_ALL_ENGINES = list(_REGISTRY_ENGINES)


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


def _try_llmfit_fallback(profile: dict[str, Any]) -> dict[str, Any] | None:
    """
    Opportunistic llmfit invocation when the cached scan was deferred (#95).

    Returns the llmfit system dict on success, or None when llmfit is not
    installed, errors, or times out — preserving the deferred-scan rendering.

    `pb.llmfit_system()` already returns None when llmfit is absent or any
    subprocess / JSON-parse error occurs, so AC2 and AC3 are covered by the
    inner contract. The outer try/except is defense-in-depth.
    """
    try:
        result = pb.llmfit_system()
    except Exception:
        return None
    if not result:
        return None
    profile["llmfit_system"] = result
    profile.setdefault("tools", {})["llmfit"] = pb.command_version("llmfit")
    try:
        _sync_presence_from_tools(profile)
        _persist_targeted_profile_update(profile)
    except Exception:
        # Cache-write side effects must never block the wizard.
        pass
    return result


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
        info = pb.ollama_info()
        tools["ollama"] = {
            "present": bool(info.get("present")),
            "version": info.get("version", ""),
            "base_url": info.get("base_url", pb.ollama_base_url()),
            "error": info.get("error", ""),
        }
        profile["ollama"] = {"models": info.get("models", []), **info}
    elif engine == "lmstudio":
        lms = pb.lms_info()
        profile["lmstudio"] = lms
        tools["lmstudio"] = {
            "present": bool(lms.get("present")),
            "version": pb.command_version("lms")["version"] if lms.get("present") else "",
            "error": lms.get("error", ""),
        }
    elif engine == "llamacpp":
        info = pb.llamacpp_info()
        tools["llamacpp"] = {
            "present": bool(info.get("present")),
            "version": info.get("version", ""),
            "base_url": info.get("base_url", pb.llamacpp_base_url()),
            "error": info.get("error", ""),
        }
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
        adapter: pb.Router9Adapter | pb.OpenRouterAdapter = pb.Router9Adapter()
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
    elif engine == "openrouter":
        adapter = pb.OpenRouterAdapter()
        info = adapter.detect()
        health = (
            adapter.healthcheck()
            if info.get("present")
            else {"ok": False, "detail": "OpenRouter endpoint not reachable"}
        )
        tools["openrouter"] = {
            "present": bool(info.get("present")),
            "version": info.get("version", ""),
            "base_url": pb.OPENROUTER_BASE_URL,
        }
        profile["openrouter"] = {
            "present": bool(info.get("present")),
            "base_url": pb.OPENROUTER_BASE_URL,
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

    if state.model_source == "openrouter-direct":
        return engine == "openrouter"
    if state.model_source == "vllm-loaded":
        return engine == "vllm"
    if state.model_source == "running-server":
        return engine == "llamacpp"
    return engine in ("ollama", "lmstudio", "llamacpp")
