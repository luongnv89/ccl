from __future__ import annotations

from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "llamacpp",
        "install",
        detail="Install a llama-server binary and make it available on PATH.",
        commands=[
            "brew install llama.cpp",
            "llama-server --version",
            "pip install 'huggingface_hub[cli]'",
        ],
    )
