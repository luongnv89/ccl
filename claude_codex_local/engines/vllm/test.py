from __future__ import annotations

import os
from typing import Any

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(
    model: str = "",
    dry_run: bool = True,
    profile: dict[str, Any] | None = None,
    **_: object,
) -> dict[str, object]:
    if dry_run:
        return lifecycle_result(
            "vllm",
            "test",
            detail=f"Would smoke-test vLLM model {model!r}",
            data={"dry_run": True, "model": model},
        )
    api_key = pb.VLLM_KEY_FILE.read_text().strip() if pb.VLLM_KEY_FILE.exists() else ""
    if not api_key:
        api_key = os.environ.get("VLLM_API_KEY", "")
    base_url = (profile or {}).get("vllm", {}).get("base_url") or pb.VLLM_BASE_URL
    result = pb.smoke_test_vllm_model(model, base_url=base_url, api_key=api_key)
    return lifecycle_result(
        "vllm",
        "test",
        ok=bool(result.get("ok")),
        detail=result.get("error") or result.get("response") or "",
        data=result,
    )
