from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import benchmark_smoke


def run(model: str = "", dry_run: bool = True, **_: object) -> dict[str, object]:
    return benchmark_smoke("vllm", model, pb.smoke_test_vllm_model, dry_run=dry_run)
