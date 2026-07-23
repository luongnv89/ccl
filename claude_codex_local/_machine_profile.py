from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_codex_local._config import (
    LLAMACPP_API_KEY,
    MACHINE_PROFILE_CACHE_FILE,
    MACHINE_PROFILE_TTL_SECONDS,
    OLLAMA_API_KEY,
    OPENROUTER_BASE_URL,
    ROUTER9_BASE_URL,
    STATE_DIR,
    VLLM_API_KEY,
    VLLM_BASE_URL,
)
from claude_codex_local._shell import command_version as _command_version_from_shell
from claude_codex_local._shell import llamacpp_base_url, ollama_base_url

# Re-export command_version so that tests patching _machine_profile.command_version
# can target it (tests that patch _mp_mod.command_version expect it to exist).
command_version = _command_version_from_shell


def _compute_machine_fingerprint(profile: dict) -> str:
    host = profile.get("host", {})
    keys = [
        host.get("system", ""),
        host.get("machine", ""),
        host.get("release", ""),
        host.get("platform", ""),
    ]
    sys_block = profile.get("llmfit_system", {}).get("system", {})
    if sys_block:
        keys.append(str(sys_block.get("available_ram_gb", "")))
        keys.append(str(sys_block.get("cpu_model", "")))
    fingerprint_input = "|".join(str(k) for k in keys if k)
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MachineProfileCache:
    path: Path = MACHINE_PROFILE_CACHE_FILE
    ttl_seconds: int = MACHINE_PROFILE_TTL_SECONDS

    def load(self) -> dict[str, Any] | None:
        try:
            if not self.path.exists():
                return None
            with open(self.path) as f:
                data = json.load(f)
            cached_ts = data.get("_cached_at", 0)
            if time.time() - cached_ts > self.ttl_seconds:
                return None
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, profile: dict[str, Any], fingerprint: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                **profile,
                "_cached_at": time.time(),
                "_fingerprint": fingerprint,
            }
            with open(self.path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except OSError:
            pass


def _machine_profile_cache_service() -> MachineProfileCache:
    return MachineProfileCache(MACHINE_PROFILE_CACHE_FILE, MACHINE_PROFILE_TTL_SECONDS)


def _load_machine_profile_cache() -> dict | None:
    return _machine_profile_cache_service().load()


def _save_machine_profile_cache(profile: dict, fingerprint: str) -> None:
    _machine_profile_cache_service().save(profile, fingerprint)


def _machine_profile_in_process_cache() -> dict | None:
    cache_key = "_inproc_cache"
    if not hasattr(_machine_profile_in_process_cache, cache_key):
        setattr(_machine_profile_in_process_cache, cache_key, {"timestamp": 0, "data": None})
    cache = getattr(_machine_profile_in_process_cache, cache_key)
    if time.time() - cache["timestamp"] < 30:
        return cache["data"]
    return None


def _set_machine_profile_in_process_cache(data: dict) -> None:
    cache_key = "_inproc_cache"
    setattr(_machine_profile_in_process_cache, cache_key, {"timestamp": time.time(), "data": data})


def invalidate_machine_profile_inproc_cache() -> None:
    cache_key = "_inproc_cache"
    setattr(_machine_profile_in_process_cache, cache_key, {"timestamp": 0, "data": None})


LLMFIT_SKIPPED_SENTINEL: dict[str, Any] = {"_skipped": True}


def _is_llmfit_skipped(value: Any) -> bool:
    return isinstance(value, dict) and value.get("_skipped") is True


def _endpoint_config_signature() -> dict[str, Any]:
    return {
        "ollama_base_url": ollama_base_url(),
        "ollama_api_key_set": bool(OLLAMA_API_KEY),
        "llamacpp_base_url": llamacpp_base_url(),
        "llamacpp_api_key_set": bool(LLAMACPP_API_KEY),
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_api_key_set": bool(VLLM_API_KEY),
    }


def _default_endpoint_config_signature() -> dict[str, Any]:
    return {
        "ollama_base_url": "http://localhost:11434",
        "ollama_api_key_set": False,
        "llamacpp_base_url": "http://localhost:8001",
        "llamacpp_api_key_set": False,
        "vllm_base_url": "http://localhost:8000",
        "vllm_api_key_set": False,
    }


def _endpoint_config_matches(cache: dict[str, Any], current: dict[str, Any]) -> bool:
    cached = cache.get("_endpoint_config")
    if cached is None:
        return current == _default_endpoint_config_signature()
    return cached == current


@dataclass(frozen=True)
class MachineProfileProbeResults:
    llmfit_system: dict[str, Any] | None
    lms: dict[str, Any]
    llamacpp: dict[str, Any]
    hf_cli: dict[str, Any]
    vllm: dict[str, Any]
    ollama: dict[str, Any]
    claude: dict[str, Any]
    codex: dict[str, Any]
    pi: dict[str, Any]
    llmfit: dict[str, Any]
    router9_info: dict[str, Any]
    router9_health: dict[str, Any]
    openrouter_info: dict[str, Any]
    openrouter_health: dict[str, Any]


def _probe_machine_profile_inputs(run_llmfit: bool) -> MachineProfileProbeResults:
    from claude_codex_local._adapters import OpenRouterAdapter, Router9Adapter
    from claude_codex_local._hf_api import huggingface_cli_detect
    from claude_codex_local._llamacpp_lifecycle import llamacpp_info
    from claude_codex_local._llmfit import llmfit_system
    from claude_codex_local._vllm import vllm_info

    llmfit_sys = llmfit_system() if run_llmfit else None
    import claude_codex_local.core as _core

    lms = _core.lms_info()
    llamacpp = llamacpp_info()
    # Import core at call time so test monkeypatches on core.command_version
    # take effect (tests patch the re-export on the core facade).
    # (already imported above)
    # lms_info is also resolved via _core so monkeypatches on core.lms_info apply.

    hf_cli = huggingface_cli_detect()
    vllm = vllm_info()
    ollama = _core.ollama_info()
    claude_info = _core.command_version("claude")
    codex_info = _core.command_version("codex")
    pi_info = _core.command_version("pi")
    llmfit_info = _core.command_version("llmfit")

    router9_adapter = Router9Adapter()
    router9_info = router9_adapter.detect()
    router9_health = (
        router9_adapter.healthcheck()
        if router9_info.get("present")
        else {"ok": False, "detail": router9_info.get("error", "9router endpoint not reachable")}
    )
    openrouter_adapter = OpenRouterAdapter()
    openrouter_info = openrouter_adapter.detect()
    openrouter_health = (
        openrouter_adapter.healthcheck()
        if openrouter_info.get("present")
        else {
            "ok": False,
            "detail": openrouter_info.get("error", "OpenRouter endpoint not reachable"),
        }
    )
    return MachineProfileProbeResults(
        llmfit_system=llmfit_sys,
        lms=lms,
        llamacpp=llamacpp,
        hf_cli=hf_cli,
        vllm=vllm,
        ollama=ollama,
        claude=claude_info,
        codex=codex_info,
        pi=pi_info,
        llmfit=llmfit_info,
        router9_info=router9_info,
        router9_health=router9_health,
        openrouter_info=openrouter_info,
        openrouter_health=openrouter_health,
    )


def _assemble_machine_profile(
    probes: MachineProfileProbeResults, *, run_llmfit: bool
) -> dict[str, Any]:
    from claude_codex_local._hf_api import disk_usage_for

    harnesses_present = [
        name
        for name, info in (("claude", probes.claude), ("codex", probes.codex), ("pi", probes.pi))
        if info.get("present")
    ]
    engines_present = []
    if probes.ollama.get("present"):
        engines_present.append("ollama")
    if probes.lms.get("present"):
        engines_present.append("lmstudio")
    if probes.llamacpp.get("present"):
        engines_present.append("llamacpp")
    if probes.vllm.get("present"):
        engines_present.append("vllm")
    if probes.router9_info.get("present"):
        engines_present.append("9router")
    if probes.openrouter_info.get("present"):
        engines_present.append("openrouter")

    profile: dict[str, Any] = {
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": {
            "ollama": {
                "present": bool(probes.ollama.get("present")),
                "version": probes.ollama.get("version", ""),
                "base_url": probes.ollama.get("base_url", ollama_base_url()),
                "error": probes.ollama.get("error", ""),
            },
            "lmstudio": {
                "present": probes.lms["present"],
                "version": probes.lms.get("version", ""),
            },
            "llamacpp": probes.llamacpp,
            "vllm": {
                "present": probes.vllm["present"],
                "version": probes.vllm.get("version", ""),
                "base_url": probes.vllm["base_url"],
                "error": probes.vllm.get("error", ""),
                "error_type": probes.vllm.get("error_type", ""),
            },
            "huggingface_cli": probes.hf_cli,
            "claude": probes.claude,
            "codex": probes.codex,
            "pi": probes.pi,
            "llmfit": probes.llmfit,
            "9router": {
                "present": bool(probes.router9_info.get("present")),
                "version": probes.router9_info.get("version", ""),
                "base_url": ROUTER9_BASE_URL,
                "error": probes.router9_info.get("error", ""),
                "error_type": probes.router9_info.get("error_type", ""),
            },
            "openrouter": {
                "present": bool(probes.openrouter_info.get("present")),
                "version": probes.openrouter_info.get("version", ""),
                "base_url": OPENROUTER_BASE_URL,
                "error": probes.openrouter_info.get("error", ""),
                "error_type": probes.openrouter_info.get("error_type", ""),
            },
        },
        "presence": {
            "harnesses": harnesses_present,
            "engines": engines_present,
            "llmfit": probes.llmfit.get("present", False),
            "has_minimum": bool(harnesses_present) and bool(engines_present),
        },
        "ollama": probes.ollama,
        "lmstudio": probes.lms,
        "llamacpp": probes.llamacpp,
        "vllm": probes.vllm,
        "9router": {
            "present": bool(probes.router9_info.get("present")),
            "base_url": ROUTER9_BASE_URL,
            "healthcheck": probes.router9_health,
            "diagnostic": probes.router9_info.get("error_type", ""),
        },
        "openrouter": {
            "present": bool(probes.openrouter_info.get("present")),
            "base_url": OPENROUTER_BASE_URL,
            "healthcheck": probes.openrouter_health,
            "diagnostic": probes.openrouter_info.get("error_type", ""),
        },
        "disk": disk_usage_for(STATE_DIR),
        "state_dir": str(STATE_DIR),
    }
    if probes.llmfit_system:
        profile["llmfit_system"] = probes.llmfit_system
    elif not run_llmfit:
        profile["llmfit_system"] = LLMFIT_SKIPPED_SENTINEL
    return profile


def machine_profile(run_llmfit: bool = True) -> dict[str, Any]:
    endpoint_signature = _endpoint_config_signature()

    cached = _machine_profile_in_process_cache()
    if cached is not None and _endpoint_config_matches(cached, endpoint_signature):
        needs_deferred_llmfit_refresh = run_llmfit and _is_llmfit_skipped(
            cached.get("llmfit_system")
        )
        if not needs_deferred_llmfit_refresh:
            return cached

    file_cache = _load_machine_profile_cache()
    if file_cache is not None and _endpoint_config_matches(file_cache, endpoint_signature):
        cached_llmfit = file_cache.get("llmfit_system")
        if run_llmfit and _is_llmfit_skipped(cached_llmfit):
            from claude_codex_local._llmfit import llmfit_system

            llmfit_sys = llmfit_system()
            if llmfit_sys:
                file_cache["llmfit_system"] = llmfit_sys
                fingerprint = _compute_machine_fingerprint(file_cache)
                file_cache["_fingerprint"] = fingerprint
                _save_machine_profile_cache(file_cache, fingerprint)
        _set_machine_profile_in_process_cache(file_cache)
        return file_cache

    probes = _probe_machine_profile_inputs(run_llmfit)
    profile = _assemble_machine_profile(probes, run_llmfit=run_llmfit)

    fingerprint = _compute_machine_fingerprint(profile)
    profile["_fingerprint"] = fingerprint
    profile["_endpoint_config"] = endpoint_signature
    _save_machine_profile_cache(profile, fingerprint)
    _set_machine_profile_in_process_cache(profile)

    return profile
