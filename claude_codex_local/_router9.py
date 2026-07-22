from __future__ import annotations

from typing import Any

from claude_codex_local._config import _probe_openai_models_endpoint, ROUTER9_BASE_URL


def smoke_test_router9_models(base_url: str | None = None) -> dict[str, Any]:
    return _probe_openai_models_endpoint(
        base_url or ROUTER9_BASE_URL,
        service_name="9router",
        timeout=15,
        headers={"Content-Type": "application/json"},
    )
