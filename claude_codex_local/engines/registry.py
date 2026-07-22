from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from types import ModuleType
from typing import Any

ACTIONS: tuple[str, ...] = ("install", "config", "optimize", "test", "benchmark")
_PACKAGE = __package__ or "claude_codex_local.engines"

# Canonical list of every known engine.  New engine packages are discovered
# automatically by ``engine_names()``, but user-facing UIs (wizard, CLI help)
# still need a stable, importable list so they can render choices and help
# text without side-effects.  Keep this in sync with the packages under
# ``claude_codex_local/engines/``.
ALL_ENGINES: tuple[str, ...] = (
    "9router",
    "llamacpp",
    "lmstudio",
    "ollama",
    "openrouter",
    "vllm",
)


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
    """Return every engine package discovered under the engines directory.

    This is the *authoritative* source of engine discovery — any engine
    package that exports an ``ENGINE_NAME`` (or uses its directory name) is
    automatically included.  ``ALL_ENGINES`` is the stable, importable list
    used by user-facing code.
    """
    return tuple(_engine_packages())


def engine_capabilities(engine: str) -> dict[str, Any]:
    """Return the set of supported actions for a given engine.

    Returns a dict with keys from ``ACTIONS`` mapped to booleans indicating
    whether the engine provides that action module.
    """
    capabilities: dict[str, Any] = {}
    for action in ACTIONS:
        try:
            load_engine_action(engine, action)
            capabilities[action] = True
        except EngineLifecycleError:
            capabilities[action] = False
    return capabilities


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
