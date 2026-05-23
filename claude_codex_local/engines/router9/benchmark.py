from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "9router",
        "benchmark",
        detail="9router benchmark is intentionally not run by CCL because chat calls route to paid cloud providers.",
        ok=True,
        data={"skipped_chat": True, "reason": "avoid paid quota burn"},
    )
