from __future__ import annotations

from pathlib import Path

import pytest

from claude_codex_local import bench


def test_benchmark_model_returns_zero_summary_when_all_trials_fail(monkeypatch):
    """All probe calls return ok=False -> aggregate is a zero summary, never crashes."""

    def fake_measure(engine, model, prompt, timeout=60):
        return 0.0, {"ok": False, "error": "boom"}

    monkeypatch.setattr(bench, "_measure_first_token", fake_measure)

    summary = bench.benchmark_model(engine="ollama", model="fake:1b", num_trials=3)

    assert isinstance(summary, bench.BenchmarkSummary)
    assert summary.num_trials == 0  # no successful trials are reported
    assert summary.avg_first_token_ms == 0.0
    assert summary.avg_tokens_per_second == 0.0
    assert summary.max_tokens_per_second == 0.0


def test_measure_first_token_uses_benchmark_prompt(monkeypatch):
    """Benchmark probes pass the coding prompt through instead of READY smoke text."""
    calls = []

    def fake_smoke(model, **kwargs):
        calls.append((model, kwargs))
        return {
            "ok": True,
            "completion_tokens": 4,
            "tokens_per_second": 20.0,
            "duration_seconds": 0.5,
        }

    monkeypatch.setattr(bench.pb, "smoke_test_ollama_model", fake_smoke)

    first_token_ms, result = bench._measure_first_token(
        "ollama",
        "fake:1b",
        "write benchmark code",
        timeout=12,
    )

    assert result["ok"] is True
    assert first_token_ms == pytest.approx(100.0)
    assert calls == [
        (
            "fake:1b",
            {
                "prompt": "write benchmark code",
                "expected": None,
                "max_tokens": 256,
            },
        )
    ]


def test_benchmark_model_aggregates_successful_trials(monkeypatch):
    """Two ok trials -> avg/min/max derived from those trials only."""
    probe_results = [
        {"ok": True, "completion_tokens": 49, "tokens_per_second": 20.0, "duration_seconds": 2.5},
        {"ok": True, "completion_tokens": 49, "tokens_per_second": 30.0, "duration_seconds": 1.5},
    ]
    calls = {"n": 0}

    def fake_measure(engine, model, prompt, timeout=60):
        idx = calls["n"]
        calls["n"] += 1
        res = probe_results[idx]
        # mirror the FTL math the real helper does, so the test exercises the
        # aggregation path with realistic numbers
        ftl = (res["duration_seconds"] * 1000) / (1 + res["completion_tokens"])
        return ftl, res

    monkeypatch.setattr(bench, "_measure_first_token", fake_measure)

    summary = bench.benchmark_model(engine="ollama", model="fake:1b", num_trials=2)

    assert summary.num_trials == 2
    assert summary.min_tokens_per_second == 20.0
    assert summary.max_tokens_per_second == 30.0
    assert summary.avg_tokens_per_second == pytest.approx(25.0)


def test_generate_benchmark_markdown_without_machine_spec():
    summary = bench.BenchmarkSummary(
        model="qwen3-coder:30b",
        engine="ollama",
        num_trials=3,
        avg_first_token_ms=42.0,
        avg_total_time_ms=1500.0,
        avg_tokens_per_second=22.5,
        min_first_token_ms=40.0,
        max_first_token_ms=45.0,
        min_tokens_per_second=20.0,
        max_tokens_per_second=25.0,
    )

    md = bench.generate_benchmark_markdown(summary, machine_spec=None)

    assert "# Benchmark Report" in md
    assert "qwen3-coder:30b" in md
    assert "ollama" in md
    assert "42.0 ms" in md
    assert "22.50 tokens/sec" in md
    assert "## Machine Specifications" not in md


def test_generate_benchmark_markdown_with_machine_spec():
    summary = bench.BenchmarkSummary(
        model="fake:1b",
        engine="ollama",
        num_trials=1,
        avg_first_token_ms=10.0,
        avg_total_time_ms=100.0,
        avg_tokens_per_second=15.0,
        min_first_token_ms=10.0,
        max_first_token_ms=10.0,
        min_tokens_per_second=15.0,
        max_tokens_per_second=15.0,
    )

    md = bench.generate_benchmark_markdown(
        summary,
        machine_spec={
            "cpu_name": "Apple M2 Pro",
            "cpu_cores": 12,
            "total_ram_gb": 32,
            "gpu_name": "Apple M2 Pro GPU",
            "gpu_vram_gb": 19,
        },
    )

    assert "## Machine Specifications" in md
    assert "Apple M2 Pro" in md
    assert "12 cores" in md
    assert "32 GB" in md
    assert "19 GB VRAM" in md


def test_save_benchmark_report_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    """save_benchmark_report writes both .md and .json next to each other."""
    monkeypatch.setattr(bench.pb, "machine_profile", lambda *a, **kw: {"llmfit_system": None})

    summary = bench.BenchmarkSummary(
        model="fake:1b",
        engine="ollama",
        num_trials=1,
        avg_first_token_ms=5.0,
        avg_total_time_ms=50.0,
        avg_tokens_per_second=10.0,
        min_first_token_ms=5.0,
        max_first_token_ms=5.0,
        min_tokens_per_second=10.0,
        max_tokens_per_second=10.0,
    )

    md_path = tmp_path / "out" / "bench.md"
    returned = bench.save_benchmark_report(summary, md_path)

    assert returned == md_path
    assert md_path.exists()
    assert md_path.with_suffix(".json").exists()
    assert "fake:1b" in md_path.read_text()
