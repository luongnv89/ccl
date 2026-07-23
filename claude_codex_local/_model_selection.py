from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_codex_local._config import STATE_DIR
from claude_codex_local._machine_profile import _is_llmfit_skipped

RECOMMENDATION_MODES: tuple[str, ...] = ("balanced", "fast", "quality")

MODE_CHOICES = ["balanced", "fast", "quality"]

RECOMMENDATION_MODE_DESCRIPTIONS: dict[str, str] = {
    "balanced": "Best score within comfortable memory headroom — good default for most machines.",
    "fast": "Prioritises tokens/second — smallest model that still fits, snappiest replies.",
    "quality": "Highest llmfit score regardless of size — best output, may be slower.",
}


def _available_ram_gb(profile: dict[str, Any]) -> float | None:
    llmfit_block = profile.get("llmfit_system")
    if _is_llmfit_skipped(llmfit_block):
        return None
    sys_block = (llmfit_block or {}).get("system") or llmfit_block or {}
    val = sys_block.get("available_ram_gb")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def rank_candidates_for_mode(candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    mode = mode if mode in RECOMMENDATION_MODES else "balanced"
    if not candidates:
        return []
    if mode == "fast":
        return sorted(
            candidates,
            key=lambda c: (-(c.get("estimated_tps") or 0), -(c.get("score") or 0)),
        )
    if mode == "quality":
        return sorted(candidates, key=lambda c: -(c.get("score") or 0))
    return list(candidates)


def recommend_for_mode(profile: dict[str, Any], mode: str, engine: str) -> dict[str, Any] | None:
    import claude_codex_local.core as _core

    if engine not in ("ollama", "lmstudio", "llamacpp"):
        return None

    candidates = _core.llmfit_coding_candidates(ram_gb=_available_ram_gb(profile))
    ranked = rank_candidates_for_mode(candidates, mode)
    if not ranked:
        return None

    for c in ranked:
        tag = _candidate_tag_for_engine(c, engine)
        if tag:
            return {**c, "engine_tag": tag, "mode": mode}
    return None


def _candidate_tag_for_engine(c: dict[str, Any], engine: str) -> str | None:
    if engine == "ollama":
        return c.get("ollama_tag")
    if engine == "lmstudio":
        return c.get("lms_hub_name") or c.get("lms_mlx_path")
    if engine == "llamacpp":
        from claude_codex_local._hf_api import resolve_gguf_mirror

        name = c.get("name")
        return resolve_gguf_mirror(name) if name else None
    return None


def scan_huggingface_gguf_cache() -> list[dict[str, Any]]:
    cache_key = "_gguf_cache"
    if not hasattr(scan_huggingface_gguf_cache, cache_key):
        setattr(scan_huggingface_gguf_cache, cache_key, {"timestamp": 0, "models": []})

    cache = getattr(scan_huggingface_gguf_cache, cache_key)
    now = time.time()
    if now - cache["timestamp"] < 300:
        return cache["models"]

    hf_home = os.getenv("HF_HOME")
    cache_dir = Path(hf_home) / "hub" if hf_home else Path.home() / ".cache" / "huggingface" / "hub"

    models: list[dict[str, Any]] = []

    try:
        if not cache_dir.exists():
            cache["timestamp"] = now
            cache["models"] = []
            return []

        for model_dir in cache_dir.glob("models--*"):
            if not model_dir.is_dir():
                continue

            dir_name = model_dir.name
            if not dir_name.startswith("models--"):
                continue

            parts = dir_name[8:].split("--", 1)
            if len(parts) != 2:
                continue

            org, repo = parts
            base_name = f"{org}/{repo}"

            snapshots_dir = model_dir / "snapshots"
            if not snapshots_dir.exists():
                continue

            for snapshot_dir in snapshots_dir.iterdir():
                if not snapshot_dir.is_dir():
                    continue

                for gguf_file in snapshot_dir.glob("*.gguf"):
                    try:
                        resolved = gguf_file.resolve()
                        if not resolved.exists():
                            continue

                        size_bytes = resolved.stat().st_size
                        size_gb = size_bytes / (1024**3)

                        quant = _gguf_quant_from_filename(gguf_file.stem)

                        if quant:
                            display_name = f"{base_name}-{quant} ({size_gb:.1f} GB)"
                        else:
                            display_name = f"{base_name} ({size_gb:.1f} GB)"

                        models.append(
                            {
                                "path": str(resolved),
                                "display": display_name,
                                "size_gb": size_gb,
                            }
                        )
                    except (OSError, ValueError) as exc:
                        logging.debug(f"Skipping {gguf_file}: {exc}")
                        continue

    except PermissionError:
        logging.debug(f"Permission denied reading HF cache: {cache_dir}")
    except Exception as exc:
        logging.debug(f"Error scanning HF cache: {exc}")

    cache["timestamp"] = now
    cache["models"] = models

    return models


def _gguf_quant_from_filename(stem: str) -> str:
    if "-" not in stem:
        return ""
    potential = stem.split("-")[-1]
    if potential.startswith("Q") and any(c.isdigit() for c in potential):
        return potential
    return ""


def scan_state_dir_gguf_models() -> list[dict[str, Any]]:
    cache_key = "_state_dir_gguf_cache"
    if not hasattr(scan_state_dir_gguf_models, cache_key):
        setattr(scan_state_dir_gguf_models, cache_key, {"timestamp": 0, "models": []})

    cache = getattr(scan_state_dir_gguf_models, cache_key)
    now = time.time()
    if now - cache["timestamp"] < 300:
        return cache["models"]

    models_root = STATE_DIR / "models"
    models: list[dict[str, Any]] = []

    try:
        if not models_root.exists():
            cache["timestamp"] = now
            cache["models"] = []
            return []

        for gguf_file in models_root.rglob("*.gguf"):
            try:
                resolved = gguf_file.resolve()
                if not resolved.is_file():
                    continue
                size_bytes = resolved.stat().st_size
                size_gb = size_bytes / (1024**3)

                rel = gguf_file.relative_to(models_root)
                slug = rel.parts[0] if rel.parts else gguf_file.stem
                quant = _gguf_quant_from_filename(gguf_file.stem)
                if quant:
                    display_name = f"{slug}-{quant} ({size_gb:.1f} GB)"
                else:
                    display_name = f"{slug}/{gguf_file.name} ({size_gb:.1f} GB)"

                models.append(
                    {
                        "path": str(resolved),
                        "display": display_name,
                        "size_gb": size_gb,
                    }
                )
            except (OSError, ValueError) as exc:
                logging.debug(f"Skipping {gguf_file}: {exc}")
                continue
    except PermissionError:
        logging.debug(f"Permission denied reading STATE_DIR/models: {models_root}")
    except Exception as exc:
        logging.debug(f"Error scanning STATE_DIR/models: {exc}")

    cache["timestamp"] = now
    cache["models"] = models
    return models


def installed_models_for_engine(profile: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    coder_keywords = (
        "qwen3-coder",
        "qwen2.5-coder",
        "deepseek-coder",
        "codellama",
        "starcoder",
        "granite-code",
        "wizardcoder",
        "coder",
        "code",
    )

    excluded_keywords = (
        "embed",
        "embedding",
        "reranker",
        "rerank",
    )

    def _is_coder(text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in coder_keywords)

    def _is_excluded(text: str) -> bool:
        lower = text.lower()
        return any(k in lower for k in excluded_keywords)

    entries: list[dict[str, Any]] = []
    if engine == "ollama":
        for m in profile.get("ollama", {}).get("models", []) or []:
            if not m.get("local"):
                continue
            name = m.get("name")
            if not name or _is_excluded(name):
                continue
            entries.append(
                {
                    "tag": name,
                    "display": name,
                    "source": "ollama",
                    "size": m.get("size"),
                }
            )
    elif engine == "lmstudio":
        for m in profile.get("lmstudio", {}).get("models", []) or []:
            path = m.get("path")
            if not path or _is_excluded(path):
                continue
            entries.append(
                {
                    "tag": path,
                    "display": path,
                    "source": "lmstudio",
                    "format": m.get("format"),
                }
            )
    elif engine == "llamacpp":
        seen_paths: set[str] = set()
        for m in scan_huggingface_gguf_cache() + scan_state_dir_gguf_models():
            path = m["path"]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            entries.append(
                {
                    "tag": path,
                    "display": m["display"],
                    "source": "llamacpp",
                    "size_gb": m["size_gb"],
                }
            )

        status = profile.get("llamacpp") or {}
        if status.get("server_running") and status.get("model"):
            running_model = status["model"]
            if not any(e["tag"] == running_model for e in entries):
                entries.append(
                    {
                        "tag": running_model,
                        "display": f"{running_model} (running on port {status.get('server_port')})",
                        "source": "llamacpp",
                        "running": True,
                    }
                )

    entries.sort(key=lambda e: (0 if _is_coder(e["display"]) else 1, e["display"]))
    return entries


def merge_models_for_engine(profile: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    installed = installed_models_for_engine(profile, engine)
    merged: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for entry in installed:
        tag = entry.get("tag")
        if not tag:
            continue
        seen_tags.add(tag)
        merged.append({**entry, "source": "installed", "engine_source": entry.get("source")})

    llmfit_block = profile.get("llmfit_system")
    if not llmfit_block or _is_llmfit_skipped(llmfit_block):
        return merged

    from claude_codex_local._llmfit import llmfit_coding_candidates

    candidates = llmfit_coding_candidates(ram_gb=_available_ram_gb(profile))
    for candidate in candidates:
        tag = _candidate_tag_for_engine(candidate, engine)
        if not tag or tag in seen_tags:
            continue
        seen_tags.add(tag)
        score = candidate.get("score")
        size_hint = candidate.get("ram_gb") or candidate.get("size_gb")
        size_label = f"{size_hint} GB" if size_hint else None
        merged.append(
            {
                "tag": tag,
                "display": candidate.get("name") or tag,
                "source": "cached",
                "engine_source": engine,
                "size": size_label,
                "score": score,
                "candidate": candidate,
            }
        )
    return merged


def select_model_decision(
    profile: dict[str, Any],
    mode: str = "balanced",
    *,
    candidates: list[dict[str, Any]] | None = None,
    lms_responses_api_ok: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    # Import core at call time so that test monkeypatches on
    # core._llmfit.llmfit_coding_candidates take effect (tests patch the
    # _llmfit module directly, and core exposes it as _llmfit).
    import claude_codex_local.core as _core

    mode = mode if mode in ("balanced", "fast", "quality") else "balanced"

    if candidates is None:
        candidates = _core.llmfit_coding_candidates(ram_gb=_available_ram_gb(profile))

    if mode == "fast" and candidates:
        candidates = sorted(
            candidates,
            key=lambda c: (-(c.get("estimated_tps") or 0), -(c.get("score") or 0)),
        )
    elif mode == "quality" and candidates:
        candidates = sorted(candidates, key=lambda c: -(c.get("score") or 0))

    rationale: list[str] = []
    caveats: list[str] = []
    next_steps: list[str] = []
    selected_candidate: dict[str, Any] | None = None
    runtime = "ollama"
    status = "ready"
    selected_tag: str = ""
    model_source: str = "fallback"

    ollama_installed: dict[str, dict[str, Any]] = {
        m["name"]: m for m in profile.get("ollama", {}).get("models", []) if m.get("local")
    }
    lms_data: dict[str, Any] = profile.get("lmstudio", {})
    lms_present = lms_data.get("present", False)
    lms_installed: dict[str, dict[str, Any]] = {m["path"]: m for m in lms_data.get("models", [])}
    lms_usable = lms_present

    if lms_present:
        for c in candidates:
            lms_path = c.get("lms_mlx_path")
            lms_hub = c.get("lms_hub_name")
            matched_key = None
            if lms_path and lms_path in lms_installed:
                matched_key = lms_path
            elif lms_hub and lms_hub in lms_installed:
                matched_key = lms_hub
            if matched_key:
                server_up = lms_data.get("server_running", False)

                if not server_up:
                    caveats.append(
                        f"LM Studio has '{matched_key}' installed but the server is not running. "
                        "Falling back to Ollama. Start LM Studio server with: lms server start"
                    )
                    lms_usable = False
                    break

                from claude_codex_local._lmstudio import (
                    lms_responses_api_ok as _lms_responses_api_ok,
                )

                api_ok_fn = lms_responses_api_ok or _lms_responses_api_ok
                if not api_ok_fn(matched_key):
                    caveats.append(
                        f"LM Studio server is running but its /v1/responses streaming endpoint "
                        f"returned no data for '{matched_key}'. "
                        "Falling back to Ollama. Upgrade LM Studio or use Ollama."
                    )
                    lms_usable = False
                    break

                selected_candidate = c
                selected_tag = matched_key
                runtime = "lmstudio"
                model_source = "llmfit"
                rationale.append(
                    f"LM Studio is installed and '{matched_key}' is already on disk — "
                    f"using it (score={c.get('score')}, fit={c.get('fit_level')}, "
                    f"~{c.get('estimated_tps')} tok/s, MLX)."
                )
                break

    if not selected_tag:
        for c in candidates:
            tag = c.get("ollama_tag")
            if tag and tag in ollama_installed:
                selected_candidate = c
                selected_tag = tag
                runtime = "ollama"
                model_source = "llmfit"
                rationale.append(
                    f"llmfit ranked '{c['name']}' as the best-fit coding model "
                    f"(score={c.get('score')}, fit={c.get('fit_level')}, ~{c.get('estimated_tps')} tok/s). "
                    f"Ollama tag '{tag}' is already installed."
                )
                break

    if not selected_tag and ollama_installed:

        def _ollama_size_key(name: str) -> float:
            m = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
            return float(m.group(1)) if m else 0.0

        best_installed = max(ollama_installed.keys(), key=_ollama_size_key)
        selected_tag = best_installed
        runtime = "ollama"
        model_source = "installed-fallback"
        rationale.append(
            f"No llmfit coding model is installed in Ollama. "
            f"Using the largest installed model '{best_installed}' as a best-effort fallback."
        )

    if not selected_tag and lms_usable and candidates:
        best = candidates[0]
        lms_hub = best.get("lms_hub_name")
        lms_path = best.get("lms_mlx_path")
        if lms_hub or lms_path:
            status = "download-required"
            selected_candidate = best
            selected_tag = lms_path or lms_hub or best["name"]
            runtime = "lmstudio"
            model_source = "llmfit"
            rationale.append(
                f"LM Studio is installed. llmfit recommends '{best['name']}' "
                f"(score={best.get('score')}, fit={best.get('fit_level')}, "
                f"mem={best.get('memory_required_gb')}GB, ~{best.get('estimated_tps')} tok/s, MLX)."
            )
            rationale.append(
                "MLX runs natively on Apple Silicon — faster and lower power than GGUF/Ollama."
            )
            dl_cmd = f"lms get {lms_hub} -y" if lms_hub else f"lms get {lms_path} -y"
            next_steps.append(dl_cmd)
            next_steps.append("lms server start")
            caveats.append(
                "Download the model above, then re-run this command to confirm readiness."
            )

    if not selected_tag and candidates:
        best = candidates[0]
        tag = best.get("ollama_tag")
        if tag:
            status = "download-required"
            selected_candidate = best
            selected_tag = tag
            runtime = "ollama"
            model_source = "llmfit"
            rationale.append(
                f"llmfit recommends '{best['name']}' as the best coding model for this hardware "
                f"(score={best.get('score')}, fit={best.get('fit_level')}, "
                f"mem={best.get('memory_required_gb')}GB, ~{best.get('estimated_tps')} tok/s)."
            )
            next_steps.append(f"ollama pull {tag}")
            next_steps.append("./bin/codex-local")
            caveats.append(
                "Run `ollama pull` above, then re-run this command to confirm readiness."
            )

    if not selected_tag:
        status = "download-required"
        selected_tag = "qwen2.5-coder:7b"
        runtime = "ollama"
        rationale.append(
            "llmfit returned no candidates. Defaulting to qwen2.5-coder:7b as a safe fallback."
        )
        next_steps.append(f"ollama pull {selected_tag}")

    modes: dict[str, str | None] = {
        "balanced": selected_tag,
        "fast": selected_tag,
        "quality": selected_tag
        if (selected_candidate and selected_candidate.get("fit_level") in ("Perfect", "Good"))
        else None,
    }

    return {
        "runtime": runtime,
        "mode": mode,
        "status": status,
        "selected_model": selected_tag,
        "selected_candidate": selected_candidate,
        "modes": modes,
        "rationale": rationale,
        "caveats": list(dict.fromkeys(caveats)),
        "next_steps": next_steps,
        "model_source": model_source,
        "candidates_evaluated": len(candidates),
    }


def select_best_model(profile: dict[str, Any], mode: str = "balanced") -> dict[str, Any]:
    mode = mode if mode in ("balanced", "fast", "quality") else "balanced"

    decision = select_model_decision(profile, mode)

    rationale: list[str] = list(decision["rationale"])
    caveats: list[str] = list(decision["caveats"])
    next_steps: list[str] = list(decision["next_steps"])
    smoke: dict[str, Any] | None = None
    selected_candidate = decision.get("selected_candidate")
    selected_tag = decision["selected_model"]
    runtime = decision["runtime"]
    status = decision["status"]

    if status == "ready" and selected_tag:
        if runtime == "lmstudio":
            from claude_codex_local._lmstudio import lms_load_model, smoke_test_lmstudio_model

            load_result = lms_load_model(selected_tag)
            if load_result.get("ok"):
                smoke = smoke_test_lmstudio_model(selected_tag)
                if smoke.get("ok"):
                    rationale.append("LM Studio server smoke test passed.")
                else:
                    caveats.append(
                        f"LM Studio smoke test failed: {smoke.get('error') or smoke.get('response', '')}"
                    )
            else:
                caveats.append(f"Could not load model in LM Studio: {load_result.get('error', '')}")
        elif runtime == "ollama":
            import claude_codex_local.core as _core

            smoke = _core.smoke_test_ollama_model(selected_tag)
            assert smoke is not None
            if smoke.get("ok"):
                rationale.append("Live ollama smoke test passed.")
            else:
                caveats.append(
                    f"Ollama smoke test failed: {smoke.get('error') or smoke.get('response', '')}"
                )

    modes: dict[str, str | None] = {
        "balanced": selected_tag,
        "fast": selected_tag,
        "quality": selected_tag
        if (selected_candidate and selected_candidate.get("fit_level") in ("Perfect", "Good"))
        else None,
    }

    return {
        "runtime": runtime,
        "mode": mode,
        "status": status,
        "selected_model": selected_tag,
        "modes": modes,
        "rationale": rationale,
        "caveats": list(dict.fromkeys(caveats)),
        "next_steps": next_steps,
        "smoke_test": smoke,
        "llmfit": {
            "score": selected_candidate.get("score") if selected_candidate else None,
            "fit_level": selected_candidate.get("fit_level") if selected_candidate else None,
            "estimated_tps": selected_candidate.get("estimated_tps")
            if selected_candidate
            else None,
            "memory_required_gb": selected_candidate.get("memory_required_gb")
            if selected_candidate
            else None,
            "hf_name": selected_candidate.get("name") if selected_candidate else None,
            "best_quant": selected_candidate.get("best_quant") if selected_candidate else None,
            "candidates_evaluated": decision["candidates_evaluated"],
        },
        "state_dir": str(STATE_DIR),
    }
