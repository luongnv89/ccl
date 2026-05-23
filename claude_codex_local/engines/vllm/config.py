from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "vllm",
        "config",
        detail="vLLM is configured through its OpenAI-compatible endpoint and optional API key.",
        env={
            "VLLM_BASE_URL": pb.VLLM_BASE_URL,
            "VLLM_API_KEY": "<optional>",
            "VLLM_TIMEOUT": "60",
            "VLLM_MAX_TOKENS": "2048",
        },
        files=[str(pb.VLLM_KEY_FILE)],
        data={"base_url": pb.VLLM_BASE_URL},
    )
