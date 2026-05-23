from __future__ import annotations

import os

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(model: str = "", dry_run: bool = True, **_: object) -> dict[str, object]:
    if dry_run:
        return lifecycle_result(
            "openrouter",
            "test",
            detail=f"Would smoke-test OpenRouter model {model!r}",
            data={"dry_run": True, "model": model},
        )
    api_key = pb.OPENROUTER_KEY_FILE.read_text().strip() if pb.OPENROUTER_KEY_FILE.exists() else ""
    if not api_key:
        api_key = os.environ.get("CCL_OPENROUTER_API_KEY", "")
    result = pb.smoke_test_openrouter_model(model, api_key=api_key)
    return lifecycle_result(
        "openrouter",
        "test",
        ok=bool(result.get("ok")),
        detail=result.get("error") or result.get("response") or "",
        data=result,
    )
