from __future__ import annotations

from typing import Any

from claude_codex_local import core as pb
from claude_codex_local.engines._shared import lifecycle_result


def run(profile: dict[str, Any] | None = None, **_: object) -> dict[str, object]:
    offload = pb.detect_llamacpp_gpu_offload(profile)
    threads = pb.detect_llamacpp_threads(profile)
    return lifecycle_result(
        "llamacpp",
        "optimize",
        detail="Computed llama.cpp server tuning from profile and environment overrides.",
        env={
            "LLAMACPP_N_GPU_LAYERS": str(offload["n_gpu_layers"]),
            "LLAMACPP_THREADS": str(threads),
            "LLAMACPP_CTX_SIZE": str(pb.LLAMACPP_CTX_SIZE),
        },
        data={"gpu_offload": offload, "threads": threads},
    )
