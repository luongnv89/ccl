from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "vllm",
        "install",
        detail="Install vLLM in a Python environment and start a vllm serve process.",
        commands=[
            "pip install vllm",
            "vllm serve <hf-model-id> --host 0.0.0.0 --port 8000",
        ],
    )
