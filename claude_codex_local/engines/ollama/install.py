from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "ollama",
        "install",
        detail="Install Ollama and verify the CLI is on PATH.",
        commands=["curl -fsSL https://ollama.com/install.sh | sh", "ollama --version"],
        files=[],
    )
