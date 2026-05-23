from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "9router",
        "optimize",
        detail="Optimize by selecting provider/model routing in 9router; CCL avoids paid chat probes.",
        commands=["9router", "ccl setup --engine 9router"],
        data={"safety": "metadata probes use /v1/models only"},
    )
