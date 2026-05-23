from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "openrouter",
        "optimize",
        detail="Optimize by selecting a hosted model with the desired context, price, and capability.",
        commands=["ccl setup --engine openrouter"],
        data={
            "strategy": "prefer free-tier models during setup unless the user chooses a paid model"
        },
    )
