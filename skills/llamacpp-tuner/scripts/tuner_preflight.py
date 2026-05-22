#!/usr/bin/env python3
"""Remote pre-flight gate for the llamacpp-tuner skill.

The tuner benchmarks and tunes a *local* ``llama-server`` process — it kills,
restarts, and re-probes it on the same host. None of that is meaningful (or
safe) against a remote llama.cpp endpoint reached over the network, where the
local machine has no shell, no `llama-server` binary, and no permission to
restart someone else's server.

This script runs as Step 0 of the SKILL.md workflow. When
``LLAMACPP_BASE_URL`` resolves to a remote host (anything outside
loopback / ``localhost``), it prints the exact skip message named in the
issue's acceptance criteria and exits ``0`` so the workflow stops cleanly
without a stack trace or a non-zero status that would look like a failure.

When the base URL is local, the script exits ``0`` silently so the rest of
the workflow proceeds exactly as today.

Usage:
    python3 tuner_preflight.py            # honor LLAMACPP_BASE_URL env
    python3 tuner_preflight.py --base URL # override (used by tests)

Exit codes:
    0  — local URL (continue) **or** remote URL (skip — message printed).
         Both are "non-error" by design; AC #1 requires the skip path to
         be a clean exit, not a failure.
    2  — unexpected internal error (e.g. core module unimportable).
"""

from __future__ import annotations

import argparse
import sys


def _resolve_base_url(cli_base: str | None) -> tuple[str, bool]:
    """
    Return ``(base_url, is_local)``. ``cli_base`` (when given) wins over the
    env var because tests need to drive the script with explicit URLs without
    mutating process-wide environment.

    Imports happen lazily so ``--help`` still works without the package on
    ``sys.path`` (the script ships inside ``skills/llamacpp-tuner/scripts/``
    and is sometimes invoked directly from a checked-out repo).
    """
    from claude_codex_local.core import _is_local_base_url, llamacpp_base_url

    base_url = cli_base if cli_base is not None else llamacpp_base_url()
    return base_url, _is_local_base_url(base_url)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tuner_preflight.py",
        description=(
            "Gate the llamacpp-tuner skill: exit 0 with a skip message when "
            "LLAMACPP_BASE_URL points at a remote host, exit 0 silently when "
            "it points at localhost."
        ),
    )
    ap.add_argument(
        "--base",
        default=None,
        help="Override LLAMACPP_BASE_URL (test hook); defaults to the env value.",
    )
    args = ap.parse_args(argv)

    try:
        base_url, is_local = _resolve_base_url(args.base)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"tuner_preflight: internal error: {exc}", file=sys.stderr)
        return 2

    if is_local:
        # Silent on the happy path — Step 0 should not add noise when the
        # workflow is going to continue. SKILL.md's Step Completion Report
        # is the user-visible "pass" signal.
        return 0

    # Remote endpoint: print the exact message named in issue #124 AC #1.
    # Plain stdout (not stderr) because this is the user-facing skip notice,
    # not an error, and Claude Code captures stdout for the step report.
    print(
        f"llamacpp is configured as a remote endpoint ({base_url}). "
        "The tuner targets local `llama-server` instances only — skipping."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
