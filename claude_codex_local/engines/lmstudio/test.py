from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import smoke_result


def run(model: str = "", dry_run: bool = True, **_: object) -> dict[str, object]:
    if not dry_run:
        if not pb.lms_info().get("server_running"):
            pb.lms_start_server()
        pb.lms_load_model(model)
    return smoke_result("lmstudio", model, pb.smoke_test_lmstudio_model, dry_run=dry_run)
