from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "openrouter",
        "install",
        detail="OpenRouter is hosted SaaS; create an API key rather than installing a daemon.",
        commands=["open https://openrouter.ai/keys"],
    )
