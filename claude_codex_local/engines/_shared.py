from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def lifecycle_result(
    engine: str,
    action: str,
    *,
    ok: bool = True,
    detail: str = "",
    commands: list[str] | None = None,
    files: list[str] | None = None,
    env: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "engine": engine,
        "action": action,
        "ok": ok,
        "detail": detail,
        "commands": commands or [],
        "files": files or [],
        "env": env or {},
        "data": data or {},
    }


def dry_run_result(engine: str, action: str, detail: str, **data: Any) -> dict[str, Any]:
    return lifecycle_result(engine, action, ok=True, detail=detail, data={"dry_run": True, **data})


def smoke_result(
    engine: str,
    model: str,
    smoke: Callable[[str], dict[str, Any]],
    *,
    action: str = "test",
    dry_run: bool = True,
) -> dict[str, Any]:
    if dry_run:
        return dry_run_result(
            engine, action, f"Would smoke-test {engine} model {model!r}", model=model
        )
    if not model:
        return lifecycle_result(
            engine, action, ok=False, detail="A model is required for this test"
        )
    result = smoke(model)
    return lifecycle_result(
        engine,
        action,
        ok=bool(result.get("ok")),
        detail=result.get("error") or result.get("response") or "",
        data=result,
    )


def benchmark_smoke(
    engine: str,
    model: str,
    smoke: Callable[[str], dict[str, Any]],
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    if dry_run:
        return dry_run_result(
            engine,
            "benchmark",
            f"Would benchmark {engine} model {model!r} with the engine smoke probe",
            model=model,
        )
    start = time.time()
    result = smoke_result(engine, model, smoke, action="benchmark", dry_run=False)
    result["data"]["wall_seconds"] = max(time.time() - start, 1e-6)
    return result
