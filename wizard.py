#!/usr/bin/env python3
"""
Interactive first-run wizard for claude-codex-local.

Implements the 8-step flow from PRD v1.2 §4.1:

  2.1 Discover environment (harnesses, engines, llmfit, disk)
  2.2 Install missing components (guided sub-process)
  2.3 Pick preferences (primary harness + engine)
  2.4 Pick a model (user-first, optional find-model helper)
  2.5 Smoke test engine + model
  2.6 Wire up harness (isolated settings.json / launch config)
  2.7 Verify launch command end-to-end
  2.8 Generate personalized guide.md

The wizard is idempotent and resumable: state is checkpointed to
`.claude-codex-local/wizard-state.json` after every completed step.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import poc_bridge as pb

console = Console()

ROOT = Path(__file__).resolve().parent
STATE_DIR = pb.STATE_DIR
STATE_FILE = STATE_DIR / "wizard-state.json"
GUIDE_PATH = ROOT / "guide.md"


# ---------------------------------------------------------------------------
# WizardState — the single source of truth for wizard progress
# ---------------------------------------------------------------------------

@dataclass
class WizardState:
    # which steps have completed successfully
    completed_steps: list[str] = field(default_factory=list)
    # full machine profile from last discover pass
    profile: dict[str, Any] = field(default_factory=dict)
    # user's primary + secondary selections
    primary_harness: str = ""           # "claude" | "codex"
    secondary_harnesses: list[str] = field(default_factory=list)
    primary_engine: str = ""            # "ollama" | "lmstudio" | "llamacpp"
    secondary_engines: list[str] = field(default_factory=list)
    # model pick
    model_name: str = ""                # raw user input or find-model selection
    model_source: str = ""              # "direct" | "find-model"
    engine_model_tag: str = ""          # engine-specific tag (e.g. qwen3-coder:30b)
    # launch command the wizard wired up
    launch_command: list[str] = field(default_factory=list)
    # smoke test + verify outputs
    smoke_test_result: dict[str, Any] = field(default_factory=dict)
    verify_result: dict[str, Any] = field(default_factory=dict)

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls) -> "WizardState":
        if not STATE_FILE.exists():
            return cls()
        try:
            data = json.loads(STATE_FILE.read_text())
            return cls(**data)
        except Exception:
            return cls()

    def mark(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.save()


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
# Step 2.1 — Discover environment
# ---------------------------------------------------------------------------

def step_2_1_discover(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.1 — Discover environment")
    profile = pb.machine_profile()
    state.profile = profile

    tools = profile["tools"]
    presence = profile["presence"]
    disk = profile.get("disk", {})

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
    row("llmfit", tools["llmfit"])
    console.print(table)

    free_gib = disk.get("free_gib", "?")
    total_gib = disk.get("total_gib", "?")
    info(f"Free disk on state dir: {free_gib} GiB of {total_gib} GiB")

    if presence["has_minimum"]:
        ok(f"Found harnesses: {', '.join(presence['harnesses'])}")
        ok(f"Found engines: {', '.join(presence['engines'])}")
        ok("llmfit is available")
        state.mark("2.1")
        return True

    # Missing pieces — fall through to step 2.2
    if not presence["harnesses"]:
        warn("No harness found (need claude or codex)")
    if not presence["engines"]:
        warn("No engine found (need ollama, lmstudio, or llama.cpp)")
    if not presence["llmfit"]:
        warn("llmfit is not installed")
    return False


# ---------------------------------------------------------------------------
# Step 2.2 — Install missing components
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
    "llmfit": {
        "name": "llmfit",
        "cmd": "See docs/poc-bootstrap.md for the install script",
        "url": "https://github.com/AlexsJones/llmfit",
    },
}


def step_2_2_install_missing(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.2 — Install missing components")
    presence = state.profile.get("presence", {})

    missing: list[str] = []
    if not presence.get("harnesses"):
        missing.append("HARNESS (claude or codex)")
    if not presence.get("engines"):
        missing.append("ENGINE (ollama, lmstudio, or llamacpp)")
    if not presence.get("llmfit"):
        missing.append("llmfit")

    if not missing:
        info("Nothing missing.")
        state.mark("2.2")
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
    if not presence.get("llmfit"):
        _show_install_hint("llmfit")

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
        state.mark("2.2")
        return True
    fail("Still missing required components. Install them and re-run the wizard.")
    return False


def _show_install_hint(key: str) -> None:
    hint = INSTALL_HINTS.get(key)
    if not hint:
        return
    console.print(f"\n[bold]{hint['name']}[/bold] → {hint['url']}")
    console.print(f"    [cyan]{hint['cmd']}[/cyan]")


# ---------------------------------------------------------------------------
# Step 2.3 — Pick preferences
# ---------------------------------------------------------------------------

def step_2_3_pick_preferences(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.3 — Pick preferences")
    presence = state.profile["presence"]
    harnesses = presence["harnesses"]
    engines = presence["engines"]

    # Harness pick
    if len(harnesses) == 1:
        state.primary_harness = harnesses[0]
        ok(f"Only one harness available: [bold]{state.primary_harness}[/bold]")
    elif non_interactive:
        state.primary_harness = harnesses[0]
        state.secondary_harnesses = harnesses[1:]
        ok(f"Non-interactive: picking [bold]{state.primary_harness}[/bold] as primary harness")
    else:
        choice = questionary.select(
            "Which harness do you want as primary?",
            choices=harnesses,
        ).ask()
        if choice is None:
            return False
        state.primary_harness = choice
        state.secondary_harnesses = [h for h in harnesses if h != choice]

    # Engine pick
    if len(engines) == 1:
        state.primary_engine = engines[0]
        ok(f"Only one engine available: [bold]{state.primary_engine}[/bold]")
    elif non_interactive:
        # Prefer lmstudio on Apple Silicon, else ollama, else whatever's first.
        default_engine = _default_engine(engines, state.profile)
        state.primary_engine = default_engine
        state.secondary_engines = [e for e in engines if e != default_engine]
        ok(f"Non-interactive: picking [bold]{state.primary_engine}[/bold] as primary engine")
    else:
        default_engine = _default_engine(engines, state.profile)
        choice = questionary.select(
            "Which engine do you want as primary?",
            choices=engines,
            default=default_engine,
        ).ask()
        if choice is None:
            return False
        state.primary_engine = choice
        state.secondary_engines = [e for e in engines if e != choice]

    ok(f"Primary: [bold]{state.primary_harness}[/bold] + [bold]{state.primary_engine}[/bold]")
    if state.secondary_harnesses or state.secondary_engines:
        info(f"Fallbacks: harnesses={state.secondary_harnesses or '-'} engines={state.secondary_engines or '-'}")
    state.mark("2.3")
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
    is_apple_silicon = (
        profile.get("host", {}).get("system") == "Darwin"
        and profile.get("host", {}).get("machine") in ("arm64", "aarch64")
    )
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
# Step 2.4 — Pick a model (user-first, optional find-model helper)
# ---------------------------------------------------------------------------

def step_2_4_pick_model(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.4 — Pick a model")
    engine = state.primary_engine

    if non_interactive:
        # Non-interactive: go straight through find-model (prefers installed models).
        candidate = _find_model_auto(engine, state.profile)
        if not candidate:
            fail("Non-interactive find-model failed and no direct model was provided.")
            return False
        state.model_name = candidate["display"]
        state.engine_model_tag = candidate["tag"]
        state.model_source = "find-model"
        ok(f"Non-interactive pick: [bold]{state.engine_model_tag}[/bold]")
    else:
        while True:
            mode = questionary.select(
                "How do you want to choose the model?",
                choices=[
                    questionary.Choice("I'll type a specific model name", value="direct"),
                    questionary.Choice("Help me pick (llmfit recommendation)", value="find-model"),
                    questionary.Choice("Cancel setup", value="cancel"),
                ],
            ).ask()
            if mode is None or mode == "cancel":
                fail("Setup cancelled by user.")
                return False
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
                picked = _find_model_interactive(engine)
                if not picked:
                    continue
                state.model_name = picked["display"]
                state.engine_model_tag = picked["tag"]
                state.model_source = "find-model"

            if _handle_model_presence(state):
                break

    state.mark("2.4")
    return True


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
            m["name"] for m in profile.get("ollama", {}).get("models", [])
            if m.get("local") and pb.NOTHINK_VARIANT_SUFFIX not in m["name"].split(":", 1)[0]
        ]
        # Prefer recognisable coding models first.
        for preferred in ("qwen3-coder", "qwen2.5-coder", "deepseek-coder", "codellama", "gemma4", "qwen3.5"):
            for name in installed:
                if preferred in name.lower():
                    return {"display": name, "tag": name, "score": None, "candidate": {"name": name}}
        if installed:
            return {"display": installed[0], "tag": installed[0], "score": None, "candidate": {"name": installed[0]}}
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
    candidates = pb.llmfit_coding_candidates()
    for c in candidates:
        tag = _candidate_tag(c, engine)
        if tag:
            return {"display": c["name"], "tag": tag, "score": c.get("score"), "candidate": c}
    return None


def _find_model_interactive(engine: str) -> dict[str, Any] | None:
    info("Running llmfit to rank coding models for this machine...")
    candidates = pb.llmfit_coding_candidates()
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
            f"fit={c.get('fit_level','?'):<12s}  ~{c.get('estimated_tps','?')} tok/s"
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
    size_gib = size_bytes / (1024 ** 3) if size_bytes else None
    free_gib = free_bytes / (1024 ** 3) if free_bytes else 0

    if size_gib is not None:
        info(f"Estimated model size: {size_gib:.1f} GiB. Free disk: {free_gib:.1f} GiB.")
    else:
        info(f"Estimated model size: unknown. Free disk: {free_gib:.1f} GiB.")

    fits = size_gib is None or size_gib < free_gib * 0.9

    if not fits:
        warn(f"Model does not comfortably fit in free disk space ({size_gib:.1f} GiB needed, {free_gib:.1f} GiB free).")
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
        for m in profile.get("ollama", {}).get("models", []):
            if m.get("name") == tag:
                return True
        return False
    if engine == "lmstudio":
        for m in profile.get("lmstudio", {}).get("models", []):
            if m.get("path") == tag:
                return True
        return False
    return False  # llama.cpp: treat as not cached — user manages GGUFs manually


def _estimate_model_size(state: WizardState) -> int | None:
    """Return a best-effort byte estimate for the chosen model, or None."""
    # llmfit sometimes exposes size in the candidate metadata via `size_gb`.
    # Fall back to None when unknown — we'll warn but not block.
    return None  # TODO: wire llmfit size if available


def _download_model(state: WizardState) -> bool:
    engine = state.primary_engine
    tag = state.engine_model_tag
    console.print(f"\n[cyan]Downloading {tag} via {engine}...[/cyan]")
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
            warn("llama.cpp does not manage downloads. Fetch the GGUF manually and re-run with --resume.")
            return False
    except subprocess.CalledProcessError as exc:
        fail(f"Download failed: {exc}")
        return False
    ok(f"Downloaded {tag}")
    # Refresh profile so 2.5 sees the new model.
    state.profile = pb.machine_profile()
    return True


# ---------------------------------------------------------------------------
# Step 2.5 — Smoke test engine + model
# ---------------------------------------------------------------------------

def step_2_5_smoke_test(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.5 — Smoke test engine + model")
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
    else:
        warn(f"Smoke test for engine '{engine}' not implemented — skipping.")
        result = {"ok": True, "response": "(skipped)"}

    state.smoke_test_result = result
    if not result.get("ok"):
        fail(f"Smoke test failed: {result.get('error') or result.get('response')}")
        return False

    ok(f"Smoke test passed: {str(result.get('response',''))[:80]}")
    state.mark("2.5")
    return True


# ---------------------------------------------------------------------------
# Step 2.6 — Wire up harness with isolated settings
# ---------------------------------------------------------------------------

def step_2_6_wire_harness(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.6 — Wire up harness")
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
    cmd, effective_tag = result
    if effective_tag != tag:
        info(f"Using patched model tag [bold]{effective_tag}[/bold] (was [dim]{tag}[/dim])")
        state.engine_model_tag = effective_tag
    state.launch_command = cmd
    ok(f"Harness wired. Launch command: [bold]{' '.join(shlex.quote(x) for x in cmd)}[/bold]")
    state.mark("2.6")
    return True


def _wire_claude(engine: str, tag: str) -> tuple[list[str], str] | None:
    """
    Write an isolated Claude Code settings.json under the state HOME pointing at
    the local engine. Critically sets CLAUDE_CODE_ATTRIBUTION_HEADER=0 in
    settings.json (NOT as a shell env var — the shell value is ignored by Claude Code).
    """
    settings_dir = pb.STATE_HOME / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = settings_dir / "settings.json"

    effective_tag = tag

    if engine == "ollama":
        base_url = "http://localhost:11434"
        auth_token = "ollama"
        # Claude Code sends a `thinking` payload that breaks Qwen3 and wastes
        # latency on Gemma4. Bake a derived Ollama model with the no-think /
        # 64K-ctx fix so verify (2.7) and day-to-day use behave sanely.
        info("Checking for Claude-Code-friendly Ollama variant of the model...")
        effective_tag, patch_info = pb.ollama_ensure_nothink_variant(tag)
        if patch_info["patched"]:
            if patch_info.get("reused"):
                ok(f"Reusing existing patched variant [bold]{effective_tag}[/bold]")
            else:
                ok(f"Built patched variant [bold]{effective_tag}[/bold] from {tag}")
        else:
            info(f"No Ollama variant patch applied ({patch_info['reason']}).")
    elif engine == "lmstudio":
        base_url = f"http://localhost:{pb.LMS_SERVER_PORT}"
        auth_token = "lmstudio"
        if _lmstudio_needs_nothink(tag):
            warn(
                "LM Studio + Claude Code is known to 400 on Qwen3 reasoning models "
                "because Claude Code sends a `thinking` payload the server cannot "
                "disable via `--chat-template-kwargs`. If verify (2.7) fails with "
                "`400 thinking.type`, switch to engine=ollama (auto-patched) or run "
                "llama.cpp with `--chat-template-kwargs '{\"enable_thinking\": false}'`."
            )
    elif engine == "llamacpp":
        base_url = "http://localhost:8001"
        auth_token = "sk-local"
    else:
        fail(f"Unknown engine for Claude wire-up: {engine}")
        return None

    settings = {
        "env": {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_AUTH_TOKEN": auth_token,
            "ANTHROPIC_API_KEY": auth_token,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    }
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    info(f"Wrote {settings_file}")

    return ["claude", "--model", effective_tag], effective_tag


def _lmstudio_needs_nothink(tag: str) -> bool:
    t = tag.lower()
    return "qwen3" in t


def _wire_codex(engine: str, tag: str) -> tuple[list[str], str] | None:
    if engine == "ollama":
        pb.configure_ollama_integration("codex", tag)
        return ["codex", "--oss", "-m", tag], tag
    if engine == "lmstudio":
        pb.configure_lmstudio_integration("codex", tag)
        return ["codex", "-m", tag], tag
    if engine == "llamacpp":
        return ["codex", "-m", tag], tag
    fail(f"Unknown engine for Codex wire-up: {engine}")
    return None


# ---------------------------------------------------------------------------
# Step 2.7 — Verify launch command end-to-end
# ---------------------------------------------------------------------------

def step_2_7_verify(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.7 — Verify launch command end-to-end")
    harness = state.primary_harness
    tag = state.engine_model_tag
    env = pb.state_env()
    # Make sure Anthropic env vars from host do NOT leak in
    for leaky in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        env.pop(leaky, None)

    if harness == "claude":
        settings_path = pb.STATE_HOME / ".claude" / "settings.json"
        cmd = [
            "claude",
            "--bare",
            "--settings", str(settings_path),
            "--model", tag,
            "--dangerously-skip-permissions",
            "-p", "Reply with exactly READY",
        ]
    elif harness == "codex":
        cmd = ["codex", "exec", "--skip-git-repo-check", "-m", tag]
        if state.primary_engine == "ollama":
            cmd = ["codex", "exec", "--skip-git-repo-check", "--oss", "-m", tag, "Reply with exactly READY"]
        else:
            cmd.append("Reply with exactly READY")
    else:
        fail(f"Unknown harness: {harness}")
        return False

    info(f"Running: {' '.join(shlex.quote(x) for x in cmd)}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=300,
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
    state.mark("2.7")
    return True


# ---------------------------------------------------------------------------
# Step 2.8 — Generate personalized guide.md
# ---------------------------------------------------------------------------

GUIDE_TEMPLATE = """\
# Local coding guide (generated)

This file was generated by `claude-codex-local` on your machine.

## What was set up

- **Harness**: `{harness}`
- **Engine**: `{engine}`
- **Model**: `{model}`
- **Isolated HOME**: `{state_home}`

## Daily use

Run this single command to start your local coding session:

```bash
{launch_cmd}
```

The wizard wrote isolated config under `{state_home}` so your official
`~/.claude` and `~/.codex` directories are untouched. You can switch back
to cloud mode at any time by running `claude` or `codex` directly (without
the `claude-codex-local` wrapper).

## Troubleshooting

- **Slow second turn in Claude Code?** Check that
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` is set inside
  `{state_home}/.claude/settings.json`. It will not work as a shell env var.
- **Engine not responding?** Re-run the smoke test:
  ```bash
  ./bin/poc-doctor
  ```
- **Model missing?** Re-run the wizard — it will detect the gap and offer to
  re-download: `python3 -m wizard`
- **Switch to a different model?** Run the standalone `find-model` helper:
  ```bash
  python3 -m wizard find-model
  ```

## Return to official mode

The wizard never mutates your global `~/.claude` or `~/.codex` config.
Just run `claude` or `codex` directly (outside the wrapper) and you are
back on cloud.

## Rollback

To wipe all local bridge state:

```bash
rm -rf {state_dir}
rm -f {guide_path}
```
"""


def step_2_8_generate_guide(state: WizardState, non_interactive: bool = False) -> bool:
    header("Step 2.8 — Generate personalized guide.md")
    launch_cmd = " ".join(shlex.quote(x) for x in state.launch_command)
    content = GUIDE_TEMPLATE.format(
        harness=state.primary_harness,
        engine=state.primary_engine,
        model=state.engine_model_tag,
        state_home=pb.STATE_HOME,
        state_dir=pb.STATE_DIR,
        launch_cmd=launch_cmd,
        guide_path=GUIDE_PATH,
    )
    GUIDE_PATH.write_text(content)
    ok(f"Wrote [bold]{GUIDE_PATH}[/bold]")
    state.mark("2.8")
    return True


# ---------------------------------------------------------------------------
# Wizard driver
# ---------------------------------------------------------------------------

STEPS: list[tuple[str, str, Callable[[WizardState, bool], bool]]] = [
    ("2.1", "Discover environment", step_2_1_discover),
    ("2.2", "Install missing components", step_2_2_install_missing),
    ("2.3", "Pick preferences", step_2_3_pick_preferences),
    ("2.4", "Pick a model", step_2_4_pick_model),
    ("2.5", "Smoke test engine + model", step_2_5_smoke_test),
    ("2.6", "Wire up harness", step_2_6_wire_harness),
    ("2.7", "Verify launch command", step_2_7_verify),
    ("2.8", "Generate guide.md", step_2_8_generate_guide),
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
        if step_id == "2.3" and state.primary_harness and state.primary_engine:
            ok(f"Using forced picks: harness=[bold]{state.primary_harness}[/bold] engine=[bold]{state.primary_engine}[/bold]")
            state.mark("2.3")
            continue
        # Step 2.2 is conditional: only run if 2.1 failed presence check.
        if step_id == "2.2":
            if state.profile.get("presence", {}).get("has_minimum"):
                continue
        ok_step = fn(state, non_interactive)
        if not ok_step:
            fail(f"Step {step_id} ({title}) did not complete. Re-run with --resume to continue.")
            return 1

    console.print()
    console.print(Panel.fit(
        f"[bold green]Setup complete![/bold green]\n\n"
        f"Launch your local coding session with:\n  [cyan]{' '.join(shlex.quote(x) for x in state.launch_command)}[/cyan]\n\n"
        f"See [bold]{GUIDE_PATH}[/bold] for the full guide.",
        border_style="green",
    ))
    return 0


def run_find_model_standalone() -> int:
    """Exposed as `claude-codex-local find-model` — no setup, just a recommendation."""
    header("find-model — llmfit coding-model recommendation")
    profile = pb.machine_profile()
    if not profile["presence"]["llmfit"]:
        fail("llmfit is not installed.")
        return 1
    engines = profile["presence"]["engines"] or ["ollama"]
    engine = engines[0]
    info(f"Ranking models for engine: {engine}")
    picked = _find_model_interactive(engine)
    if picked:
        console.print(f"\n[bold]You picked:[/bold] {picked['display']}")
        console.print(f"[bold]Engine tag:[/bold] {picked['tag']}")
        return 0
    return 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="claude-codex-local")
    sub = parser.add_subparsers(dest="cmd")

    setup = sub.add_parser("setup", help="Run the interactive first-run wizard (default)")
    setup.add_argument("--resume", action="store_true", help="Resume from last checkpointed step")
    setup.add_argument("--non-interactive", action="store_true", help="Auto-pick defaults (CI/script use)")
    setup.add_argument("--harness", choices=("claude", "codex"), help="Force primary harness")
    setup.add_argument("--engine", choices=("ollama", "lmstudio", "llamacpp"), help="Force primary engine")

    sub.add_parser("find-model", help="Show an llmfit-driven coding model recommendation")

    args = parser.parse_args()
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
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
