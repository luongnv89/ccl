"""
Local-tier tests that exercise real binaries when they happen to be installed.

These run on a developer's machine where ollama/lms/claude/codex/llmfit may be
present. On CI or on a clean machine they auto-skip via the `local` marker in
conftest.py (see `pytest_collection_modifyitems`).

They are intentionally read-only — no model downloads, no writes to official
config, no long-running smoke tests. The goal is to make sure the real CLI
wrapping logic still speaks the current version of each tool.

Mark a test with `@pytest.mark.local(needs=["ollama"])` to gate it on one
or more binaries being present in PATH.
"""

from __future__ import annotations

import pytest

import claude_codex_local.bridge as pb
import claude_codex_local.wizard as wiz

# ---------------------------------------------------------------------------
# Tool version sniffing — cheap, fast, safe.
# ---------------------------------------------------------------------------


@pytest.mark.local(needs=["ollama"])
def test_real_ollama_version_is_detected():
    info = pb.command_version("ollama")
    assert info["present"] is True
    assert info["version"]  # non-empty


@pytest.mark.local(needs=["ollama"])
def test_real_parse_ollama_list_returns_list():
    models = pb.parse_ollama_list()
    assert isinstance(models, list)
    for m in models:
        assert "name" in m
        assert "local" in m


@pytest.mark.local(needs=["claude"])
def test_real_claude_cli_reachable():
    assert pb.command_version("claude")["present"] is True


@pytest.mark.local(needs=["codex"])
def test_real_codex_cli_reachable():
    assert pb.command_version("codex")["present"] is True


@pytest.mark.local(needs=["llmfit"])
def test_real_llmfit_system_returns_dict():
    sys_info = pb.llmfit_system()
    # Either returns a dict with keys or None if llmfit is installed but not
    # configured — both are acceptable for this smoke check.
    assert sys_info is None or isinstance(sys_info, dict)


@pytest.mark.local(needs=["llmfit"])
def test_real_llmfit_coding_candidates_shape():
    candidates = pb.llmfit_coding_candidates()
    assert isinstance(candidates, list)
    for c in candidates[:5]:  # only sanity-check the top 5
        assert "name" in c
        # Each candidate should at least attempt engine-tag mapping.
        assert "ollama_tag" in c or "lms_hub_name" in c or "lms_mlx_path" in c


# ---------------------------------------------------------------------------
# Full machine_profile() — expensive but safe; no mutations.
# ---------------------------------------------------------------------------


@pytest.mark.local(needs=["ollama"])
def test_real_machine_profile_has_minimum_shape():
    profile = pb.machine_profile()
    assert "tools" in profile
    assert "presence" in profile
    assert "ollama" in profile
    assert "disk" in profile
    assert profile["tools"]["ollama"]["present"] is True


# ---------------------------------------------------------------------------
# LM Studio-specific checks. Only meaningful when the user has lms installed.
# ---------------------------------------------------------------------------


@pytest.mark.local(needs=["lms"])
def test_real_lms_info_returns_expected_keys():
    info = pb.lms_info()
    assert info["present"] is True
    assert "server_running" in info
    assert "server_port" in info
    assert "models" in info


# ---------------------------------------------------------------------------
# Wizard doctor — safe to run whenever there's a real wizard state on disk.
# Skips gracefully when no state file exists.
# ---------------------------------------------------------------------------


@pytest.mark.local(needs=["ollama"])
def test_real_wizard_doctor_runs_against_real_state(capsys):
    if not wiz.STATE_FILE.exists():
        pytest.skip("no local wizard state to exercise")
    rc = wiz.run_doctor()
    assert rc in (0, 1)  # either clean or flags regressions — both valid
