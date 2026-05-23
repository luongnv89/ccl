from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "vllm",
        "optimize",
        detail="Tune vLLM at server startup; CCL only records the endpoint contract.",
        env={"VLLM_MAX_TOKENS": "2048", "VLLM_TIMEOUT": "60"},
        commands=["vllm serve <hf-model-id> --max-model-len <tokens> --gpu-memory-utilization 0.9"],
    )
