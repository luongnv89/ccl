from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import smoke_result


def run(model: str = "", dry_run: bool = True, **_: object) -> dict[str, object]:
    return smoke_result("ollama", model, pb.smoke_test_ollama_model, dry_run=dry_run)
