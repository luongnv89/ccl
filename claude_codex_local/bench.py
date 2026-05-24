"""
Lightweight coding-model benchmark for post-installation validation.

Measures first-token latency, throughput (tokens/sec), and end-to-end latency
on a coding-focused prompt, reports results in markdown.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from claude_codex_local import core as pb


@dataclass
class BenchmarkResult:
    """Single benchmark trial result."""

    model: str
    engine: str
    prompt_length: int
    completion_tokens: int
    first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    status: str  # "ok" | "timeout" | "error"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Aggregate metrics from one or more trials."""

    model: str
    engine: str
    num_trials: int
    avg_first_token_ms: float
    avg_total_time_ms: float
    avg_tokens_per_second: float
    min_first_token_ms: float
    max_first_token_ms: float
    min_tokens_per_second: float
    max_tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Coding-focused benchmark prompt
BENCHMARK_PROMPT = """\
Write a Python function that takes a list of integers and returns the sum of the squares of all even numbers. Include error handling.

def sum_of_squares_even(numbers):
"""

# For compute-bound perf, we use a deterministic prompt suffix to ensure
# token count stays consistent across runs.
BENCHMARK_PROMPT_SUFFIX = " Answer with Python code only, no explanation."


def _measure_first_token(
    engine: str,
    model: str,
    prompt: str,
    timeout: int = 60,
) -> tuple[float, dict[str, Any]]:
    """
    Call the engine-specific probe and extract first-token latency.

    Returns (first_token_ms, full_result_dict). `timeout` is honored only
    by engines whose smoke_test_* signature accepts it (currently vLLM);
    the other probes use their own defaults.
    """
    if engine == "ollama":
        result = pb.smoke_test_ollama_model(model)
    elif engine == "lmstudio":
        result = pb.smoke_test_lmstudio_model(model)
    elif engine == "llamacpp":
        result = pb.smoke_test_llamacpp_model(model)
    elif engine == "vllm":
        result = pb.smoke_test_vllm_model(model, timeout=timeout)
    else:
        return 0.0, {"ok": False, "error": f"unsupported engine: {engine}"}

    # If the smoke test succeeded, we have tokens_per_second and completion_tokens.
    # First-token latency (FTL) = total_duration / (1 + completion_tokens)
    # (the "+1" is for the prompt encoding pass before streaming starts)
    tps = result.get("tokens_per_second")
    comp_tokens = result.get("completion_tokens")
    if tps and comp_tokens and comp_tokens > 0:
        total_ms = (result.get("duration_seconds", 0) or 0) * 1000
        first_token_ms = total_ms / (1 + comp_tokens) if total_ms > 0 else 0.0
    else:
        first_token_ms = 0.0

    return first_token_ms, result


def benchmark_model(
    engine: str,
    model: str,
    num_trials: int = 3,
    timeout: int = 120,
) -> BenchmarkSummary:
    """
    Run multiple trials of the benchmark prompt against engine + model.

    Returns aggregate BenchmarkSummary with min/max/avg metrics.
    """
    prompt = BENCHMARK_PROMPT + BENCHMARK_PROMPT_SUFFIX
    results: list[BenchmarkResult] = []

    for _ in range(num_trials):
        start = time.time()
        try:
            first_token_ms, probe_result = _measure_first_token(
                engine, model, prompt, timeout=timeout
            )
            wall_seconds = time.time() - start
            wall_ms = wall_seconds * 1000

            if not probe_result.get("ok"):
                results.append(
                    BenchmarkResult(
                        model=model,
                        engine=engine,
                        prompt_length=len(prompt),
                        completion_tokens=0,
                        first_token_ms=0.0,
                        total_time_ms=0.0,
                        tokens_per_second=0.0,
                        status="error",
                        error=probe_result.get("error", "unknown"),
                    )
                )
                continue

            comp_tokens = probe_result.get("completion_tokens") or 0
            tps = probe_result.get("tokens_per_second") or 0.0

            results.append(
                BenchmarkResult(
                    model=model,
                    engine=engine,
                    prompt_length=len(prompt),
                    completion_tokens=comp_tokens,
                    first_token_ms=first_token_ms,
                    total_time_ms=wall_ms,
                    tokens_per_second=tps,
                    status="ok",
                )
            )
        except Exception as exc:
            results.append(
                BenchmarkResult(
                    model=model,
                    engine=engine,
                    prompt_length=len(prompt),
                    completion_tokens=0,
                    first_token_ms=0.0,
                    total_time_ms=0.0,
                    tokens_per_second=0.0,
                    status="error",
                    error=str(exc),
                )
            )

    # Aggregate successful results
    successful = [r for r in results if r.status == "ok"]
    if not successful:
        # All failed — return a summary with zeros
        return BenchmarkSummary(
            model=model,
            engine=engine,
            num_trials=num_trials,
            avg_first_token_ms=0.0,
            avg_total_time_ms=0.0,
            avg_tokens_per_second=0.0,
            min_first_token_ms=0.0,
            max_first_token_ms=0.0,
            min_tokens_per_second=0.0,
            max_tokens_per_second=0.0,
        )

    first_token_ms_list = [r.first_token_ms for r in successful]
    total_time_ms_list = [r.total_time_ms for r in successful]
    tps_list = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]

    return BenchmarkSummary(
        model=model,
        engine=engine,
        num_trials=len(successful),
        avg_first_token_ms=sum(first_token_ms_list) / len(first_token_ms_list),
        avg_total_time_ms=sum(total_time_ms_list) / len(total_time_ms_list),
        avg_tokens_per_second=sum(tps_list) / len(tps_list) if tps_list else 0.0,
        min_first_token_ms=min(first_token_ms_list),
        max_first_token_ms=max(first_token_ms_list),
        min_tokens_per_second=min(tps_list) if tps_list else 0.0,
        max_tokens_per_second=max(tps_list) if tps_list else 0.0,
    )


def generate_benchmark_markdown(
    summary: BenchmarkSummary,
    machine_spec: dict[str, Any] | None = None,
) -> str:
    """
    Generate a markdown report from benchmark results.
    """
    lines = []
    lines.append("# Benchmark Report\n")

    if machine_spec:
        lines.append("## Machine Specifications\n")
        cpu = machine_spec.get("cpu_name", "Unknown")
        cores = machine_spec.get("cpu_cores", "?")
        ram_gb = machine_spec.get("total_ram_gb", "?")
        gpu = machine_spec.get("gpu_name", "None")
        gpu_vram = machine_spec.get("gpu_vram_gb", "?")
        lines.append(f"- **CPU**: {cpu} ({cores} cores)")
        lines.append(f"- **RAM**: {ram_gb} GB")
        if gpu and gpu != "None":
            lines.append(f"- **GPU**: {gpu} ({gpu_vram} GB VRAM)")
        lines.append("")

    lines.append("## Model & Engine\n")
    lines.append(f"- **Model**: {summary.model}")
    lines.append(f"- **Engine**: {summary.engine}")
    lines.append(f"- **Trials**: {summary.num_trials}")
    lines.append("")

    lines.append("## Results\n")
    lines.append("| Metric | Value | Notes |")
    lines.append("|--------|-------|-------|")

    # First-token latency
    lines.append(
        f"| First Token Latency | {summary.avg_first_token_ms:.1f} ms | "
        f"(min: {summary.min_first_token_ms:.1f}, max: {summary.max_first_token_ms:.1f}) |"
    )

    # Total time
    lines.append(
        f"| Total Generation Time | {summary.avg_total_time_ms:.1f} ms | "
        f"Average across {summary.num_trials} trials |"
    )

    # Throughput
    lines.append(
        f"| Throughput | {summary.avg_tokens_per_second:.2f} tokens/sec | "
        f"(min: {summary.min_tokens_per_second:.2f}, max: {summary.max_tokens_per_second:.2f}) |"
    )

    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "- **First Token Latency**: Lower is better; impacts perceived responsiveness in interactive agents."
    )
    lines.append(
        "- **Throughput**: Higher is better; impacts speed of multi-token code generation."
    )
    lines.append("")

    lines.append("## Benchmark Setup\n")
    lines.append("```")
    lines.append("Prompt:")
    lines.append(BENCHMARK_PROMPT[:100] + "...")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def save_benchmark_report(summary: BenchmarkSummary, output_path: Path) -> Path:
    """
    Save benchmark summary and markdown report to files.

    Returns the path to the markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    # Save markdown report
    machine_spec = pb.machine_profile().get("llmfit_system")
    markdown = generate_benchmark_markdown(summary, machine_spec)
    with open(output_path, "w") as f:
        f.write(markdown)

    return output_path
