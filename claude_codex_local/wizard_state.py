"""
Wizard state management — the single source of truth for wizard progress.

Exports:
    WireResult — serialised wiring result (argv, env, effective_tag)
    WizardState — dataclass with save/load/mark for checkpointed progress
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from claude_codex_local import core as _pb

STATE_DIR = _pb.STATE_DIR
STATE_FILE = STATE_DIR / "wizard-state.json"
GUIDE_PATH = Path.cwd() / "guide.md"


# ---------------------------------------------------------------------------
# WireResult — the serialised wiring result for a completed harness
# ---------------------------------------------------------------------------


@dataclass
class WireResult:
    argv: list[str]
    env: dict[str, str]
    effective_tag: str
    raw_env: dict[str, str] = field(default_factory=dict)
    """
    Env-var entries whose VALUES are shell expressions to be expanded at
    exec-time (e.g. ``"$(cat /path/to/key)"``). Use ONLY for shell expressions
    originating in this codebase, NEVER user input. Emitted unquoted by
    ``_write_helper_script`` so the shell can evaluate them at exec time.
    """


# ---------------------------------------------------------------------------
# WizardState — the single source of truth for wizard progress
# ---------------------------------------------------------------------------


@dataclass
class WizardState:
    # which steps have completed successfully
    completed_steps: list[str] = field(default_factory=list)
    # full machine profile from last discover pass
    profile: dict = field(default_factory=dict)
    # user's primary + secondary selections
    primary_harness: str = ""  # "claude" | "codex" | "pi"
    secondary_harnesses: list[str] = field(default_factory=list)
    # "ollama" | "lmstudio" | "llamacpp" | "vllm" | "9router" | "openrouter"
    primary_engine: str = ""
    secondary_engines: list[str] = field(default_factory=list)
    # model pick
    model_name: str = ""  # raw user input or find-model selection
    model_source: str = ""  # "direct" | "find-model"
    engine_model_tag: str = ""  # engine-specific tag (e.g. qwen3-coder:30b)
    model_candidate: dict = field(
        default_factory=dict
    )  # llmfit candidate metadata when available
    # launch command the wizard wired up
    launch_command: list[str] = field(default_factory=list)
    # serialized WireResult: {"argv": [...], "env": {...}, "effective_tag": "..."}
    wire_result: dict = field(default_factory=dict)
    # alias install metadata
    helper_script_path: str = ""
    shell_rc_path: str = ""
    alias_names: list[str] = field(default_factory=list)
    # smoke test + verify outputs
    smoke_test_result: dict = field(default_factory=dict)
    verify_result: dict = field(default_factory=dict)
    # direct tool config backups created before mutating Codex/Pi settings
    config_backups: dict = field(default_factory=dict)

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls) -> WizardState:
        if not STATE_FILE.exists():
            return cls()
        raw_state = STATE_FILE.read_bytes()
        try:
            data = json.loads(raw_state.decode())
            # Migrate pre-rename step IDs (2.1-2.8, 2.65) to the new sequential scheme.
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
            # Clear pre-#120 Pi alias cache so the next install step
            # repopulates it from the new `ccp` mapping.
            if "cp" in data.get("alias_names", []):
                data["alias_names"] = []
            if data.get("launch_command") == ["cp"]:
                data["launch_command"] = []
            return cls(**data)
        except Exception as exc:
            backup_path = _backup_invalid_wizard_state(raw_state)
            from claude_codex_local.wizard_ui import warn as _warn

            _warn(
                "Invalid wizard state could not be loaded; starting with a blank state. "
                f"Previous content was backed up to {backup_path} for recovery. "
                f"Reason: {exc}"
            )
            return cls()

    def mark(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.save()


def _backup_invalid_wizard_state(raw_state: bytes) -> Path:
    """Preserve unreadable wizard state so users can inspect or recover it."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = STATE_FILE.with_name(f"{STATE_FILE.name}.invalid-{timestamp}.bak")
    counter = 1
    while backup_path.exists():
        backup_path = STATE_FILE.with_name(f"{STATE_FILE.name}.invalid-{timestamp}-{counter}.bak")
        counter += 1
    backup_path.write_bytes(raw_state)
    return backup_path
