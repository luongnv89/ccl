from __future__ import annotations

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(**_: object) -> dict[str, object]:
    return lifecycle_result(
        "llamacpp",
        "config",
        detail="llama.cpp configuration is driven by endpoint and server tuning environment variables.",
        env={
            "LLAMACPP_BASE_URL": pb.llamacpp_base_url(),
            "LLAMACPP_CTX_SIZE": str(pb.LLAMACPP_CTX_SIZE),
            "LLAMACPP_N_GPU_LAYERS": "<optional>",
            "LLAMACPP_THREADS": "<optional>",
        },
        files=[str(pb.LLAMACPP_KEY_FILE), str(pb.LLAMACPP_LOG_DIR), str(pb.LLAMACPP_PID_DIR)],
        data={"base_url": pb.llamacpp_base_url(), "port": pb.LLAMACPP_SERVER_PORT},
    )
