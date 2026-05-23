from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "ollama",
        "config",
        detail="Ollama configuration is expressed through OLLAMA_HOST and optional OLLAMA_API_KEY.",
        env={"OLLAMA_HOST": pb.ollama_base_url(), "OLLAMA_API_KEY": "<optional>"},
        files=[str(pb.OLLAMA_KEY_FILE)],
        data={"base_url": pb.ollama_base_url(), "openai_base_url": pb.ollama_openai_base_url()},
    )
