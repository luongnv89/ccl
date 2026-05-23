from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "openrouter",
        "benchmark",
        detail="OpenRouter benchmark is not run by default because it may call paid hosted models.",
        data={"skipped_chat": True, "reason": "avoid hosted-model quota burn"},
    )
