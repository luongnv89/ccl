#!/usr/bin/env python3
"""
claude-codex-local core — decomposed into focused modules under
``claude_codex_local/_*.py`` with this file acting as a backward-compatible
re-export facade so that all existing imports (``from claude_codex_local import
core as pb`` → ``pb.*``) continue working with zero source changes.

Monkeypatch-propagation note:
  - Direct imports (e.g. STATE_DIR, machine_profile, select_best_model) are
    bound at module-load time and work for all callers.
  - Functions that tests monkeypatch via ``_patch_run`` / ``_patch_command_version``
    are resolved at call time via ``__getattr__`` or wrapper functions so that
    patches on the underlying sub-modules (``_ollama``, ``_llmfit``, ``_hf_api``,
    ``_lmstudio``) propagate through ``core`` automatically.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stdlib re-exports — shared module objects so test monkeypatches on
# ``pb.shutil``, ``pb.os``, etc. propagate to all internal call sites.
# ---------------------------------------------------------------------------
import os as os
import shutil as shutil
import subprocess as subprocess
import time as time

# Re-export sub-modules for monkeypatch propagation.
import claude_codex_local._hf_api as _hf_api_mod  # noqa: F401
import claude_codex_local._llmfit as _llmfit_mod  # noqa: F401
import claude_codex_local._lmstudio as _lmstudio_mod  # noqa: F401
import claude_codex_local._ollama as _ollama_mod  # noqa: F401
import claude_codex_local._shell as _shell_mod  # noqa: F401

# ---------------------------------------------------------------------------
# Sub-module imports — sorted alphabetically by module name for ruff I001.
# ---------------------------------------------------------------------------
from claude_codex_local._adapters import (
    _ADAPTER_PREFERENCE_ORDER,
    _ENGINE_ADAPTER_MAP,
    ALL_ADAPTERS,
    LlamaCppAdapter,
    LMStudioAdapter,
    OllamaAdapter,
    OpenRouterAdapter,
    Router9Adapter,
    RuntimeAdapter,
    VLLMAdapter,
    _build_adapters,
)
from claude_codex_local._config import (
    HF_TO_LMS_HUB,
    HF_TO_OLLAMA,
    LLAMACPP_API_KEY,
    LLAMACPP_BASE_URL,
    LLAMACPP_CTX_SIZE,
    LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX,
    LLAMACPP_KEY_FILE,
    LLAMACPP_LOG_DIR,
    LLAMACPP_MTP_ENABLED,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_PID_DIR,
    LLAMACPP_SERVER_HOST,
    LLAMACPP_SERVER_PORT,
    LLAMACPP_SPEC_DRAFT_N_MAX,
    LLAMACPP_THREADS,
    LMS_SERVER_PORT,
    MACHINE_PROFILE_CACHE_FILE,
    MACHINE_PROFILE_TTL_SECONDS,
    MLX_QUANT_RANK,
    MLX_QUANT_SUFFIX,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_KEY_FILE,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_FILE,
    ORIG_HOME,
    ROUTER9_BASE_URL,
    ROUTER9_KEY_FILE,
    STATE_DIR,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_KEY_FILE,
    _is_local_base_url,
    _normalize_base_url,
    _probe_openai_models_endpoint,
)
from claude_codex_local._doctor import (
    doctor,
    main,
    print_payload,
    smoke_test_codex,
)
from claude_codex_local._hf_api import (
    _GGUF_MIRROR_AUTHORS,
    _GGUF_MIRROR_CACHE,
    _HF_NOT_FOUND_MARKERS,
    _candidate_base_name,
    _dir_size_bytes,
    _looks_like_not_found,
    disk_usage_for,
)
from claude_codex_local._llamacpp_lifecycle import (
    LlamaServerConfig,
    LlamaServerHandle,
    _cleanup_pid_file,
    _pid_gone,
    _signal_process,
    build_llamacpp_server_args,
    detect_llamacpp_gpu_offload,
    detect_llamacpp_mtp,
    detect_llamacpp_threads,
    diagnose_llama_server_log,
    llamacpp_detect,
    llamacpp_info,
    llamacpp_start_server,
    llamacpp_stop_server,
    llamacpp_stop_server_by_port,
    llamacpp_wait_until_ready,
    probe_gguf_is_mtp,
    safe_repo_slug,
    smoke_test_llamacpp_model,
)
from claude_codex_local._llmfit import (
    _canonical_key,
    _derive_lms_mlx_path,
    hf_name_to_lms_hub,
    hf_name_to_ollama_tag,
    llmfit_estimate_size_bytes,
    llmfit_info,
    llmfit_system,
)
from claude_codex_local._llmfit import (
    llmfit_coding_candidates as _llmfit_coding_candidates,
)
from claude_codex_local._lmstudio import (
    lms_binary as _lms_binary,
)
from claude_codex_local._lmstudio import (
    lms_download_model,
    lms_info,
    lms_load_model,
    lms_responses_api_ok,
    lms_running_models,
    lms_start_server,
    smoke_test_lmstudio_model,
)
from claude_codex_local._machine_profile import (
    LLMFIT_SKIPPED_SENTINEL,
    MachineProfileCache,
    MachineProfileProbeResults,
    _assemble_machine_profile,
    _compute_machine_fingerprint,
    _default_endpoint_config_signature,
    _endpoint_config_matches,
    _endpoint_config_signature,
    _is_llmfit_skipped,
    _load_machine_profile_cache,
    _machine_profile_cache_service,
    _machine_profile_in_process_cache,
    _probe_machine_profile_inputs,
    _save_machine_profile_cache,
    _set_machine_profile_in_process_cache,
    invalidate_machine_profile_inproc_cache,
    machine_profile,
)
from claude_codex_local._model_selection import (
    MODE_CHOICES,
    RECOMMENDATION_MODE_DESCRIPTIONS,
    RECOMMENDATION_MODES,
    _available_ram_gb,
    _candidate_tag_for_engine,
    _gguf_quant_from_filename,
    installed_models_for_engine,
    merge_models_for_engine,
    rank_candidates_for_mode,
    recommend_for_mode,
    scan_huggingface_gguf_cache,
    scan_state_dir_gguf_models,
    select_best_model,
    select_model_decision,
)
from claude_codex_local._ollama import (
    _ollama_http_models,
    _parse_ollama_list_cli,
    _smoke_test_ollama_cli,
    ollama_info,
    parse_ollama_list,
)
from claude_codex_local._openrouter import (
    fetch_openrouter_free_models,
    smoke_test_openrouter_model,
    smoke_test_openrouter_models,
)
from claude_codex_local._router9 import (
    smoke_test_router9_models,
)
from claude_codex_local._shell import (
    _auth_headers,
    command_version,
    ensure_path,
    ensure_state_dirs,
    llamacpp_base_url,
    ollama_base_url,
    ollama_openai_base_url,
    require,
    run_shell,
    state_env,
)
from claude_codex_local._shell import (
    run as _run_from_shell,
)
from claude_codex_local._vllm import (
    smoke_test_vllm_model,
    vllm_info,
)

# ---------------------------------------------------------------------------
# Wrapper functions — resolve at call time so test monkeypatches on the
# underlying sub-modules propagate through the ``core`` facade.
# ---------------------------------------------------------------------------


def run(*args, **kwargs):
    # If _ollama.run has been monkeypatched, use it; otherwise fall back to
    # _shell.run.  This covers both _patch_run (patches _ollama.run) and
    # direct core.run patches.
    if _ollama_mod.run is not _run_from_shell:  # noqa: SIM108
        return _ollama_mod.run(*args, **kwargs)
    return _shell_mod.run(*args, **kwargs)


def llmfit_coding_candidates(*a, **k):
    """Delegate to ``_llmfit.llmfit_coding_candidates`` at call time so that
    test monkeypatches on ``_llmfit.llmfit_coding_candidates`` propagate."""
    return _llmfit_mod.llmfit_coding_candidates(*a, **k)


def huggingface_cli_detect(*a, **k):
    """Delegate to ``_hf_api.huggingface_cli_detect`` at call time so that
    test monkeypatches on ``_hf_api.huggingface_cli_detect`` propagate."""
    return _hf_api_mod.huggingface_cli_detect(*a, **k)


def smoke_test_ollama_model(*a, **k):
    """Delegate to ``_ollama.smoke_test_ollama_model`` at call time so that
    test monkeypatches on ``_ollama.smoke_test_ollama_model`` propagate."""
    return _ollama_mod.smoke_test_ollama_model(*a, **k)


# ---------------------------------------------------------------------------
# Lazy attribute access (__getattr__) — resolves at call time so that
# monkeypatches on sub-module attributes propagate through core automatically.
#
# This is needed for functions that tests patch on the sub-module directly
# (e.g. ``monkeypatch.setattr(_hf_api, "huggingface_search_models", mock)``)
# and expect ``core.huggingface_search_models`` to reflect the patch.
# ---------------------------------------------------------------------------

_hf_lazy_names = (
    "huggingface_repo_has_gguf",
    "huggingface_search_models",
    "huggingface_download_gguf",
    "huggingface_fuzzy_find",
    "huggingface_list_repo_files",
    "resolve_gguf_mirror",
)


def __getattr__(name):
    """Lazy attribute access for sub-module names that must reflect monkeypatches."""
    if name in _hf_lazy_names:
        return getattr(_hf_api_mod, name)
    if name == "lms_binary":
        return _lmstudio_mod.lms_binary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
