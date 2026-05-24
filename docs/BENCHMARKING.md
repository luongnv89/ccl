# Benchmarking Feature

## Overview

The new **Step 5.5 — Optional lightweight benchmark** provides users with detailed performance metrics for their selected model and engine combination right after the smoke test during setup.

## What's New

### Added Files

1. **`claude_codex_local/bench.py`** — Benchmarking module with:
   - `BenchmarkResult` and `BenchmarkSummary` dataclasses
   - `benchmark_model()` function for running 3-trial benchmarks
   - `generate_benchmark_markdown()` for creating markdown reports
   - `save_benchmark_report()` for persisting results to disk

2. **Step 5.5 in `wizard.py`** — New optional wizard step that:
   - Asks user if they want to benchmark their model (off by default)
   - Runs 3 trials of a coding-focused prompt
   - Displays results in an interactive table
   - Saves detailed reports to `~/.claude-codex-local/benchmarks/`

### Integration Point

The benchmark step is positioned **after the smoke test (Step 5)** and **before the harness wiring (Step 6)**:

```
Step 5: Smoke test engine + model ✓
        ↓
Step 5.5: Optional benchmark ← NEW
        ↓
Step 6: Wire up harness
```

## User Flow

1. **Smoke test completes** → User is prompted:
   ```
   Run a lightweight benchmark of <engine> / <model>?
   (Measures first-token latency and throughput on a coding prompt. Takes ~30-60s.)
   ```

2. **User chooses "Y"** → Benchmark runs:
   ```
   Benchmarking ollama / qwen2.5-coder:7b...
   (Running 3 trials of the benchmark prompt)
   ```

3. **Results display**:
   ```
   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
   ┃ Metric                   ┃ Value           ┃ Range              ┃
   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
   │ First Token Latency      │ 45.2 ms         │ 42.1–48.3 ms      │
   │ Throughput               │ 22.15 tok/s     │ 20.08–24.31 tok/s │
   │ Total Generation Time    │ 1250.5 ms       │ (average)          │
   └──────────────────────────┴─────────────────┴────────────────────┘

   ✓ Benchmark completed: 3 trials
   ✓ Benchmark report saved: ~/.claude-codex-local/benchmarks/benchmark-ollama-qwen2.5-coder:7b-20260523-145032.md
   · View the full report to see detailed metrics and interpretation.
   ```

4. **Setup continues** → Wizard proceeds to Step 6 regardless of benchmark result

## Metrics

### First Token Latency
- **What**: Time until the first token is generated
- **Why**: Affects perceived responsiveness in interactive agents
- **Target**: Lower is better (<50ms for snappy interaction)

### Throughput
- **What**: Tokens per second during generation
- **Why**: Impacts speed of multi-token code generation
- **Target**: Higher is better (>15 tok/s for acceptable, >30+ for fast)

### Total Generation Time
- **What**: End-to-end time for the full benchmark prompt
- **Why**: Shows real-world generation speed
- **Target**: Context-dependent; shorter is better

## Benchmark Prompt

The benchmark uses a lightweight coding-focused prompt:

```python
Write a Python function that takes a list of integers and returns the sum
of the squares of all even numbers. Include error handling.

def sum_of_squares_even(numbers):
```

This prompt is short enough to run quickly (30–60s across 3 trials) while
remaining representative of coding tasks.

## Report Files

After benchmarking, two files are saved to `~/.claude-codex-local/benchmarks/`:

### `benchmark-<engine>-<model>-<timestamp>.json`
Structured data for programmatic analysis:
```json
{
  "model": "qwen2.5-coder:7b",
  "engine": "ollama",
  "num_trials": 3,
  "avg_first_token_ms": 45.2,
  "avg_total_time_ms": 1250.5,
  "avg_tokens_per_second": 22.15,
  "min_first_token_ms": 42.1,
  "max_first_token_ms": 48.3,
  "min_tokens_per_second": 20.08,
  "max_tokens_per_second": 24.31
}
```

### `benchmark-<engine>-<model>-<timestamp>.md`
Human-readable markdown report with:
- Machine specifications (CPU, RAM, GPU)
- Model and engine info
- Results table with metrics and ranges
- Interpretation guide
- Benchmark setup details

## Behavior in Different Modes

### Interactive Mode
- User is prompted whether to run benchmark (default: No)
- Results displayed in rich table format
- User can review before continuing

### Non-Interactive Mode
- Benchmark is automatically skipped
- No prompts; wizard continues without delay

### On Failure
- Benchmark errors don't block wizard
- Step marks as complete, wizard continues
- User can re-run `ccl setup --resume` if needed

## Implementation Details

### Trial Structure
```python
benchmark_model(engine, model, num_trials=3, timeout=120)
```

- **3 trials** ensures statistical stability without excessive time
- **120s timeout** per trial prevents hanging on slow models
- Results aggregated into min/max/avg metrics

### Engine Support
Works with all local engines:
- **ollama** — HTTP API with throughput metrics
- **lmstudio** — HTTP endpoint integration
- **llamacpp** — Direct server probing
- **vllm** — OpenAI-compatible interface

### Data Sourcing
Benchmarks reuse existing **smoke test infrastructure** (`pb.smoke_test_*` functions)
to maintain consistency and avoid duplication.

## Future Enhancements

Potential improvements for future versions:

1. **Multi-prompt benchmark** — Test on different code styles
2. **Comparison mode** — Benchmark multiple models back-to-back
3. **Automated recommendations** — Auto-suggest models based on hardware
4. **Performance tracking** — Compare against previous benchmark runs
5. **Configuration tuning hints** — Suggest llamacpp-tuner or vLLM settings
6. **GPU vs CPU comparison** — Measure acceleration impact

## Testing

Run the wizard and accept the benchmark prompt:

```bash
ccl setup
# ... complete steps 1-5 ...
# At Step 5.5, choose "Yes" to benchmark
```

View results:
```bash
ls -la ~/.claude-codex-local/benchmarks/
cat ~/.claude-codex-local/benchmarks/benchmark-*.md
```

## Design Decisions

### Why Optional?
Benchmarking adds 30–60s to setup. Making it optional respects user time and
intent; users who want immediate setup can skip it.

### Why After Smoke Test?
Ensures the model actually works before benchmarking. A failed model test is
more critical than performance metrics.

### Why Step 5.5 (Not Step 6)?
Maintains logical order: validate → measure → wire. Fractional step ID allows
future insertion of other validation steps without renumbering.

### Why Always Skip in Non-Interactive?
CI/scripted setups need predictable, minimal runtime. Benchmarking is an
interactive introspection tool, not a mandatory build step.

### Why Persist Reports?
Users may want to compare benchmark runs over time, share results, or analyze
trends. Local persistence is low-cost and user-friendly.
