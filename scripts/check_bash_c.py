#!/usr/bin/env python3
"""Gate: fail on new ["bash", "-c", ...] shell-outs outside the allowlist.

bandit keys on shell=True, so a subprocess call like
``run(["bash", "-c", command])`` stays invisible to B605/B602-style checks at
any scan scope. This repo-local grep gate closes that gap: any Python file
under the scanned tree containing a ``bash`` + ``-c`` argument pair must be
listed in ALLOWLIST below, otherwise this script exits non-zero.

To accept a new site, have it reviewed, then add the repo-relative path to
ALLOWLIST together with a comment explaining why piping the command through a
shell is required.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_DIR = REPO_ROOT / "claude_codex_local"

# Repo-relative paths with reviewed bash -c constructions.
ALLOWLIST: set[str] = {
    # Exports env vars through bash and re-captures them to snapshot the
    # harness environment; the command text is built from fixed keys.
    "claude_codex_local/wizard_cli.py",
    # Runs the vendored llmfit install script verbatim.
    "claude_codex_local/wizard_discovery.py",
}

# Matches ("bash", "-c" / ['bash', '-c' openings inside list/tuple literals.
BASH_C_PATTERN = re.compile(r"""[\[(]\s*(['"])bash\1\s*,\s*(['"])-c\2""")


def find_violations(paths: list[Path]) -> tuple[list[str], list[str]]:
    """Return (violations, allowlisted_hits) as formatted strings."""
    violations: list[str] = []
    allowlisted_hits: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not BASH_C_PATTERN.search(line):
                continue
            try:
                rel = str(path.resolve().relative_to(REPO_ROOT))
            except ValueError:
                rel = str(path)
            entry = f"{rel}:{lineno}: {line.strip()}"
            if rel in ALLOWLIST:
                allowlisted_hits.append(entry)
            else:
                violations.append(entry)
    return violations, allowlisted_hits


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [DEFAULT_SCAN_DIR]
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(sorted(root.rglob("*.py")))
        else:
            paths.append(root)

    violations, allowlisted_hits = find_violations(paths)

    for hit in allowlisted_hits:
        print(f"○ allowlisted: {hit}")
    if violations:
        for hit in violations:
            print(f"✗ unreviewed bash -c shell-out: {hit}", file=sys.stderr)
        print(
            "\nAdd the file to ALLOWLIST in scripts/check_bash_c.py only after "
            "reviewing why the command needs a shell.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
