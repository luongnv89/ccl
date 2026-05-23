from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "lmstudio",
        "optimize",
        detail="Use LM Studio's quantized MLX/GGUF variants selected by lms get.",
        commands=["lms get <hub-model> -y", "lms load <model> -y"],
        data={"strategy": "prefer MLX 4-bit on Apple Silicon and coding-tuned models"},
    )
