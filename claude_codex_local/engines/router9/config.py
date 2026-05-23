from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "9router",
        "config",
        detail="9router uses a local OpenAI-compatible endpoint and a chmod-600 API key file.",
        env={"CCL_9ROUTER_BASE_URL": pb.ROUTER9_BASE_URL, "CCL_9ROUTER_API_KEY": "<optional>"},
        files=[str(pb.ROUTER9_KEY_FILE)],
        data={"base_url": pb.ROUTER9_BASE_URL},
    )
