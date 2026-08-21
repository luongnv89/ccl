from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "9router",
        "install",
        detail="Install the 9router CLI and start the dashboard/service manually.",
        # Issue #194: npm installs name a pinned version.
        commands=["npm install -g 9router@0.5.55", "9router"],
    )
