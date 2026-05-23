from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "lmstudio",
        "config",
        detail="LM Studio exposes an OpenAI-compatible API on the configured local port.",
        env={"LMS_SERVER_PORT": str(pb.LMS_SERVER_PORT)},
        data={"base_url": f"http://localhost:{pb.LMS_SERVER_PORT}/v1"},
    )
