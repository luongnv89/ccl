from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "lmstudio",
        "install",
        detail="Install LM Studio, then install its lms CLI.",
        commands=["open https://lmstudio.ai", "npx lmstudio install-cli", "lms --version"],
    )
