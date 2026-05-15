"""
Shared fixtures for the claude-codex-local test suite.

Key ideas:
  * `isolated_state` — reroute STATE_DIR/STATE_HOME/GUIDE_PATH under tmp_path and
    reload the modules so every test gets a fresh filesystem world. Without this,
    tests would clobber the real `.claude-codex-local/` next to the source.
  * `fake_bin` — a tmp directory populated with shell-script stubs for ollama,
    lms, claude, codex, llmfit. Placed first on PATH so the code exercises its
    real subprocess plumbing without ever touching the real tools.
  * `local_tool` — marker-aware skip helper for the `@pytest.mark.local` tier.
"""

from __future__ import annotations

import importlib
import shutil
import stat
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# State isolation — every test gets its own STATE_DIR under tmp_path.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """
    Point STATE_DIR at tmp_path/state and reload core + wizard so their
    module-level STATE_DIR constants pick up the override. Also reroutes
    Path.home() to tmp_path so shell-alias installs land in the sandbox.
    Returns (core_module, wizard_module, state_dir).
    """
    state_dir = tmp_path / "state"
    guide_root = tmp_path / "repo"
    guide_root.mkdir()

    monkeypatch.setenv("CLAUDE_CODEX_LOCAL_STATE_DIR", str(state_dir))
    # HOME is what core.ORIG_HOME reads at import time — give it a clean
    # one so ensure_path() doesn't prepend the real ~/.lmstudio/bin.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    # Deterministic shell detection for the alias-installer tests.
    monkeypatch.setenv("SHELL", "/bin/zsh")
    # Make Path.home() resolve to the sandbox so shell-rc edits stay confined.
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    # Reload core first, then wizard (wizard imports core).
    import claude_codex_local.core as pb_mod

    pb_mod = importlib.reload(pb_mod)
    import claude_codex_local.wizard as wiz_mod

    wiz_mod = importlib.reload(wiz_mod)

    # Redirect the wizard's guide.md so it doesn't splatter the real repo.
    monkeypatch.setattr(wiz_mod, "GUIDE_PATH", guide_root / "guide.md")

    return pb_mod, wiz_mod, state_dir


# ---------------------------------------------------------------------------
# Fake bin directory — stub executables for ollama/lms/claude/codex/pi/llmfit.
# ---------------------------------------------------------------------------


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """
    Create a fake bin/ dir and prepend it to PATH so subprocess calls in the
    real code hit predictable stubs instead of the host's ollama/claude/etc.

    Yields (bin_dir, put_stub) where put_stub(name, script_body) installs or
    overrides an individual tool.
    """
    bdir = tmp_path / "fake-bin"
    bdir.mkdir()

    # Default stubs — most are trivial no-ops that succeed. Individual tests
    # override these with richer responses via put_stub().
    _write_stub(
        bdir / "ollama",
        'case "$1" in --version) echo "ollama version 0.1.99";; list) printf "NAME\\tID\\tSIZE\\tMODIFIED\\n";; *) exit 0;; esac',
    )
    _write_stub(
        bdir / "lms",
        'case "$1" in --version) echo "lms 0.2.0";; ls) echo "LLM";; "server") [[ "$2" == "status" ]] && echo "running on port 1234";; ps) echo "IDENTIFIER";; *) exit 0;; esac',
    )
    _write_stub(bdir / "claude", 'echo "claude 1.0.0"')
    _write_stub(bdir / "codex", 'echo "codex 0.1.0"')
    _write_stub(bdir / "pi", 'echo "pi 0.1.0"')
    _write_stub(
        bdir / "llmfit",
        """case "$1" in
  --version) echo "llmfit 1.2.3" ;;
  system) echo '{"system": {"ram_gb": 32, "gpu": "apple-m2"}}' ;;
  fit) echo '{"models": []}' ;;
  info) echo '{"models": []}' ;;
  coding) echo '{"models": []}' ;;
  *) exit 0 ;;
esac""",
    )

    # Keep /usr/bin + /bin so `bash` itself still resolves inside run_shell.
    base = "/usr/bin:/bin"
    monkeypatch.setenv("PATH", f"{bdir}:{base}")

    def put_stub(name: str, body: str) -> None:
        _write_stub(bdir / name, body)

    return bdir, put_stub


# ---------------------------------------------------------------------------
# Local-tier helper — skip when the real tool isn't installed.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.local items whose `needs_tool` param is missing."""
    for item in items:
        mark = item.get_closest_marker("local")
        if not mark:
            continue
        tools = mark.kwargs.get("needs", [])
        missing = [t for t in tools if shutil.which(t) is None]
        if missing:
            item.add_marker(pytest.mark.skip(reason=f"local tool(s) missing: {', '.join(missing)}"))
