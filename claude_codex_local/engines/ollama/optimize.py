from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "ollama",
        "optimize",
        detail="Use model selection for Ollama optimization; runtime flags stay owned by Ollama.",
        commands=["ollama list", "ollama pull <recommended-model>"],
        data={
            "strategy": "choose the smallest coding model that satisfies quality and memory goals"
        },
    )
