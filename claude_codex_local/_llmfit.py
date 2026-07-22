from __future__ import annotations

import json
import os
import re
from typing import Any

from claude_codex_local._config import HF_TO_OLLAMA, HF_TO_LMS_HUB, MLX_QUANT_RANK, MLX_QUANT_SUFFIX
from claude_codex_local._shell import command_version, run


def llmfit_system() -> dict[str, Any] | None:
    if not command_version("llmfit").get("present"):
        return None
    try:
        cp = run(["llmfit", "system", "--json"])
        return json.loads(cp.stdout)
    except Exception:
        return None


def llmfit_info(model_name: str) -> dict[str, Any] | None:
    if not command_version("llmfit").get("present"):
        return None
    try:
        cp = run(["llmfit", "info", model_name, "--json"])
    except Exception:
        return None
    try:
        data = json.loads(cp.stdout)
    except Exception:
        return None
    models = data.get("models") or []
    if len(models) != 1:
        return None
    return models[0]


def llmfit_estimate_size_bytes(candidate_or_name: dict[str, Any] | str) -> int | None:
    if isinstance(candidate_or_name, str):
        candidate = llmfit_info(candidate_or_name)
        if candidate is None:
            return None
    else:
        candidate = candidate_or_name

    gb = candidate.get("total_memory_gb") or candidate.get("memory_required_gb")
    if not gb:
        params_b = candidate.get("params_b")
        quant = (candidate.get("best_quant") or "").lower()
        bits_per_param = {
            "mlx-4bit": 4,
            "q4_k_m": 4,
            "q4_0": 4,
            "q4_1": 4,
            "mlx-5bit": 5,
            "q5_k_m": 5,
            "q5_0": 5,
            "mlx-6bit": 6,
            "q6_k": 6,
            "mlx-8bit": 8,
            "q8_0": 8,
        }.get(quant)
        if params_b and bits_per_param:
            gb = params_b * bits_per_param / 8.0
    if not gb:
        return None
    return int(gb * (1024**3))


def hf_name_to_ollama_tag(hf_name: str) -> str | None:
    for pattern, tag in HF_TO_OLLAMA:
        if pattern.search(hf_name):
            return tag
    return None


def hf_name_to_lms_hub(hf_name: str) -> str | None:
    for pattern, hub in HF_TO_LMS_HUB:
        if pattern.search(hf_name):
            return hub
    return None


def llmfit_coding_candidates(ram_gb: float | None = None) -> list[dict[str, Any]]:
    if not command_version("llmfit").get("present"):
        return []
    cmd: list[str] = ["llmfit"]
    if ram_gb is not None and ram_gb > 0:
        cmd.extend(["--ram", f"{ram_gb:.2f}G"])
    cmd.extend(["fit", "--json"])
    try:
        cp = run(cmd)
        data = json.loads(cp.stdout)
    except Exception:
        return []

    all_models: list[dict[str, Any]] = data.get("models", [])
    coding = [m for m in all_models if m.get("category", "").lower() in ("coding", "code")]

    groups: dict[str, dict[str, Any]] = {}

    for m in coding:
        ollama_tag = hf_name_to_ollama_tag(m["name"])
        lms_mlx_path = _derive_lms_mlx_path(m)
        lms_hub_name = hf_name_to_lms_hub(m["name"])

        key = _canonical_key(m["name"])

        existing = groups.get(key)
        if existing is None:
            groups[key] = {
                **m,
                "ollama_tag": ollama_tag,
                "lms_mlx_path": lms_mlx_path,
                "lms_hub_name": lms_hub_name,
            }
        else:
            cur_rank = MLX_QUANT_RANK.get(m.get("best_quant", ""), 99)
            ex_rank = MLX_QUANT_RANK.get(existing.get("best_quant", ""), 99)
            cur_score = m.get("score", 0)
            ex_score = existing.get("score", 0)
            if cur_score > ex_score or (cur_score == ex_score and cur_rank < ex_rank):
                groups[key] = {
                    **m,
                    "ollama_tag": ollama_tag,
                    "lms_mlx_path": lms_mlx_path,
                    "lms_hub_name": lms_hub_name,
                }

    return sorted(groups.values(), key=lambda m: m.get("score", 0), reverse=True)


def _canonical_key(name: str) -> str:
    base = name.split("/", 1)[-1]
    base = re.sub(r"[-_](MLX[-_]\w+|FP\d+)$", "", base, flags=re.IGNORECASE)
    return base.lower()


def _derive_lms_mlx_path(m: dict[str, Any]) -> str | None:
    name: str = m.get("name", "")
    best_quant: str = m.get("best_quant", "")

    if name.startswith("lmstudio-community/") and "MLX" in name:
        return name

    if not best_quant or best_quant not in MLX_QUANT_SUFFIX:
        return None

    basename = name.split("/", 1)[-1]
    basename = re.sub(r"[-_](MLX[-_]\w+|FP\d+)$", "", basename, flags=re.IGNORECASE)
    suffix = MLX_QUANT_SUFFIX[best_quant]
    return f"lmstudio-community/{basename}-{suffix}"
