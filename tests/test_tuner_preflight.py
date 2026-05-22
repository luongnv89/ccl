"""
Tests for the llamacpp-tuner skill's Step 0 pre-flight gate
(skills/llamacpp-tuner/scripts/tuner_preflight.py).

AC #1 of issue #124: invoking the tuner with a remote LLAMACPP_BASE_URL
must exit cleanly (exit 0) with a specific skip message. With a local URL
the gate must be silent and exit 0 so the rest of the workflow runs.

The script is exercised as a subprocess (driven via --base) so we cover
the real argparse / import / stdout path the SKILL.md workflow uses, not
just the internal main() function. That also keeps the test independent
of any process-wide env mutation other tests might still be holding.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_SCRIPT = REPO_ROOT / "skills" / "llamacpp-tuner" / "scripts" / "tuner_preflight.py"
EXPECTED_PREFIX = "llamacpp is configured as a remote endpoint"
EXPECTED_SUFFIX = "The tuner targets local `llama-server` instances only — skipping."


def _run_preflight(base: str | None) -> subprocess.CompletedProcess[str]:
    """
    Run the preflight script with ``--base`` (when provided) and return the
    completed process. PYTHONPATH is forced to the repo root so the script
    can `from claude_codex_local.core import …` regardless of the test
    runner's cwd.
    """
    cmd: list[str] = [sys.executable, str(PREFLIGHT_SCRIPT)]
    if base is not None:
        cmd += ["--base", base]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
        check=False,
        timeout=30,
    )


class TestTunerPreflightRemote:
    """AC #1: remote URL ⇒ clean exit + skip message."""

    def test_remote_host_prints_skip_message_and_exits_zero(self):
        proc = _run_preflight("http://gpu-box.lan:8001")
        assert proc.returncode == 0, proc.stderr
        # The message format is part of the contract — both the URL and
        # the trailing "skipping" tail are explicit in issue #124's AC.
        assert EXPECTED_PREFIX in proc.stdout
        assert "http://gpu-box.lan:8001" in proc.stdout
        assert EXPECTED_SUFFIX in proc.stdout

    def test_remote_ip_prints_skip_message(self):
        """RFC1918 IPs are remote — only 127/8 + ::1 count as local."""
        proc = _run_preflight("http://10.0.0.5:8001")
        assert proc.returncode == 0, proc.stderr
        assert EXPECTED_PREFIX in proc.stdout
        assert "10.0.0.5" in proc.stdout

    def test_public_https_endpoint_treated_as_remote(self):
        proc = _run_preflight("https://llama.example.com")
        assert proc.returncode == 0, proc.stderr
        assert EXPECTED_PREFIX in proc.stdout
        assert "llama.example.com" in proc.stdout


class TestTunerPreflightLocal:
    """AC #3: local URL ⇒ silent passthrough (no skip message)."""

    def test_localhost_is_silent_and_exits_zero(self):
        proc = _run_preflight("http://localhost:8001")
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == "" or EXPECTED_PREFIX not in proc.stdout
        # Belt-and-braces: skipping marker must be entirely absent.
        assert "skipping" not in proc.stdout

    def test_loopback_ipv4_is_silent(self):
        proc = _run_preflight("http://127.0.0.1:8001")
        assert proc.returncode == 0, proc.stderr
        assert "skipping" not in proc.stdout

    def test_loopback_ipv6_is_silent(self):
        proc = _run_preflight("http://[::1]:8001")
        assert proc.returncode == 0, proc.stderr
        assert "skipping" not in proc.stdout


class TestTunerPreflightCli:
    """Surface checks — help, exit codes, basic ergonomics."""

    def test_help_flag_works(self):
        proc = _run_preflight(None)  # baseline (uses env default = localhost)
        # Even with no override the script must complete without raising.
        assert proc.returncode == 0, proc.stderr

    def test_explicit_help(self):
        proc = subprocess.run(
            [sys.executable, str(PREFLIGHT_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
            check=False,
            timeout=10,
        )
        assert proc.returncode == 0
        assert "Gate the llamacpp-tuner skill" in proc.stdout
