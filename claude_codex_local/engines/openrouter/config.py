from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "openrouter",
        "config",
        detail="OpenRouter uses its hosted OpenAI-compatible endpoint and a chmod-600 key file.",
        env={
            "CCL_OPENROUTER_BASE_URL": pb.OPENROUTER_BASE_URL,
            "CCL_OPENROUTER_API_KEY": "<optional>",
        },
        files=[str(pb.OPENROUTER_KEY_FILE)],
        data={"base_url": pb.OPENROUTER_BASE_URL},
    )
