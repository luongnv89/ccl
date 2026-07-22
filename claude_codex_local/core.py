#!/usr/bin/env python3
"""
claude-codex-local core — decomposed into focused modules under
``claude_codex_local/_*.py`` with this file acting as a backward-compatible
re-export facade so that all existing imports (``from claude_codex_local import
core as pb`` → ``pb.*``) continue working with zero source changes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-export everything from the sub-modules in dependency order so that
# module-level imports in downstream code resolve without circular errors.
#
# Also re-export stdlib modules that sub-modules import, so that tests
# monkeypatching ``pb.shutil``, ``pb.os``, etc. propagate correctly to all
# internal call sites (every sub-module shares the same stdlib module objects).
# ---------------------------------------------------------------------------

import os as os
import shutil as shutil
import subprocess as subprocess
import time as time

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

from claude_codex_local._shell import (
    _auth_headers,
    command_version,
    ensure_path,
    ensure_state_dirs,
    llamacpp_base_url,
    ollama_base_url,
    ollama_openai_base_url,
    require,
    run as _run_from_shell,
    run_shell,
    state_env,
)

# Re-export _shell module so monkeypatches on _shell.run propagate through
# core._shell.run (tests that patch _shell.run or the _shell module directly).
import claude_codex_local._shell as _shell_mod

# Re-export _ollama module so monkeypatches on _ollama.run propagate through
# core._ollama.run (tests that use _patch_run patch _ollama.run).
import claude_codex_local._ollama as _ollama_mod

# Wrapper that resolves run at call time, checking _ollama.run first
# (for tests that use _patch_run to patch _ollama.run), then falling back
# to _shell.run (for tests that patch core.run directly).
def run(*args, **kwargs):
    # If _ollama.run has been monkeypatched, use it.
    if _ollama_mod.run is not _run_from_shell:
        return _ollama_mod.run(*args, **kwargs)
    return _shell_mod.run(*args, **kwargs)

from claude_codex_local._ollama import (
    _ollama_http_models,
    _parse_ollama_list_cli,
    _smoke_test_ollama_cli,
    ollama_info,
    parse_ollama_list,
    smoke_test_ollama_model,
)

from claude_codex_local._lmstudio import (
    lms_binary as _lms_binary,
    lms_download_model,
    lms_info,
    lms_load_model,
    lms_responses_api_ok,
    lms_running_models,
    lms_start_server,
    smoke_test_lmstudio_model,
)

# Re-export _lmstudio module so monkeypatches on _lmstudio.lms_binary propagate.
import claude_codex_local._lmstudio as _lmstudio_mod

from claude_codex_local._router9 import (
    smoke_test_router9_models,
)

from claude_codex_local._openrouter import (
    fetch_openrouter_free_models,
    smoke_test_openrouter_model,
    smoke_test_openrouter_models,
)

from claude_codex_local._llmfit import (
    _canonical_key,
    _derive_lms_mlx_path,
    hf_name_to_lms_hub,
    hf_name_to_ollama_tag,
    llmfit_coding_candidates as _llmfit_coding_candidates,
    llmfit_estimate_size_bytes,
    llmfit_info,
    llmfit_system,
)

# Re-export the _llmfit module so that monkeypatches on
# _llmfit.llmfit_coding_candidates propagate through core._llmfit.
import claude_codex_local._llmfit as _llmfit_mod


# Wrapper that always delegates to _llmfit.llmfit_coding_candidates at call
# time so that test monkeypatches on _llmfit.llmfit_coding_candidates also
# propagate through core.llmfit_coding_candidates.
def llmfit_coding_candidates(*a, **k):
    return _llmfit_mod.llmfit_coding_candidates(*a, **k)

from claude_codex_local._hf_api import (
    _GGUF_MIRROR_AUTHORS,
    _GGUF_MIRROR_CACHE,
    _HF_NOT_FOUND_MARKERS,
    _candidate_base_name,
    _dir_size_bytes,
    _looks_like_not_found,
    disk_usage_for,
    huggingface_cli_detect,
)

# Re-export _hf_api module so monkeypatches on _hf_api propagate through
# core._hf_api.<function> and also via lazy __getattr__ for direct names.
import claude_codex_local._hf_api as _hf_api_mod


# Lazy attribute access so that monkeypatches on _hf_api.<function> propagate
# through core.<function> automatically (tests that set _hf_api.X = core.X
# expect them to be the same object after patching).
_hf_lazy_names = (
    "huggingface_repo_has_gguf",
    "huggingface_search_models",
    "huggingface_download_gguf",
    "huggingface_fuzzy_find",
    "huggingface_list_repo_files",
    "resolve_gguf_mirror",
)

# Lazy access for _lmstudio.lms_binary so that monkeypatches propagate.
def __getattr__(name):
    if name in _hf_lazy_names:
        return getattr(_hf_api_mod, name)
    if name == "lms_binary":
        return getattr(_lmstudio_mod, "lms_binary")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

from claude_codex_local._vllm import (
    smoke_test_vllm_model,
    vllm_info,
)

from claude_codex_local._adapters import (
    ALL_ADAPTERS,
    LMStudioAdapter,
    LlamaCppAdapter,
    OllamaAdapter,
    OpenRouterAdapter,
    Router9Adapter,
    RuntimeAdapter,
    VLLMAdapter,
    _ADAPTER_PREFERENCE_ORDER,
    _ENGINE_ADAPTER_MAP,
    _build_adapters,
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

from claude_codex_local._doctor import (
    doctor,
    main,
    print_payload,
    smoke_test_codex,
)


if __name__ == "__main__":
    main()
