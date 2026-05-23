from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Any

ACTIONS: tuple[str, ...] = ("install", "config", "optimize", "test", "benchmark")
_PACKAGE = __package__ or "claude_codex_local.engines"


class EngineLifecycleError(ValueError):
    pass


@lru_cache(maxsize=1)
def _engine_packages() -> dict[str, str]:
    root = importlib.import_module(_PACKAGE)
    packages: dict[str, str] = {}
    for item in pkgutil.iter_modules(root.__path__):
        if not item.ispkg or item.name.startswith("_"):
            continue
        package_name = f"{_PACKAGE}.{item.name}"
        package = importlib.import_module(package_name)
        engine_name = getattr(package, "ENGINE_NAME", item.name)
        packages[str(engine_name)] = package_name
    return dict(sorted(packages.items()))


def engine_names() -> tuple[str, ...]:
    return tuple(_engine_packages())


def _engine_package(engine: str) -> str:
    try:
        return _engine_packages()[engine]
    except KeyError as exc:
        available = ", ".join(engine_names())
        raise EngineLifecycleError(f"Unknown engine {engine!r}. Available: {available}") from exc


def load_engine_action(engine: str, action: str) -> ModuleType:
    if action not in ACTIONS:
        raise EngineLifecycleError(
            f"Unknown action {action!r}. Expected one of: {', '.join(ACTIONS)}"
        )
    module_name = f"{_engine_package(engine)}.{action}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise EngineLifecycleError(f"Engine {engine!r} does not provide action {action!r}") from exc
    if not hasattr(module, "run"):
        raise EngineLifecycleError(f"Engine action module {module_name} has no run() function")
    return module


def run_engine_action(engine: str, action: str, **kwargs: Any) -> dict[str, Any]:
    module = load_engine_action(engine, action)
    return module.run(**kwargs)


def engine_action_matrix() -> dict[str, list[str]]:
    matrix: dict[str, list[str]] = {}
    for engine in engine_names():
        matrix[engine] = []
        for action in ACTIONS:
            try:
                load_engine_action(engine, action)
            except EngineLifecycleError:
                continue
            matrix[engine].append(action)
    return matrix
