from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(dry_run: bool = True, **_: object) -> dict[str, object]:
    if dry_run:
        return lifecycle_result(
            "9router",
            "test",
            detail="Would probe 9router /v1/models without calling paid chat completions.",
            data={"dry_run": True},
        )
    result = pb.smoke_test_router9_models()
    return lifecycle_result(
        "9router",
        "test",
        ok=bool(result.get("ok")),
        detail=result.get("error") or result.get("response") or "",
        data=result,
    )
