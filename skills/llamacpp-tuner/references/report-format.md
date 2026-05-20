# Final report format

The final report is what the user actually sees and acts on. Lead with the result, follow with the evidence, end with what to try next.

## Required sections

### 1. Machine + model summary (one paragraph)

One sentence on the machine (CPU, GPU, memory model), one on the model (params total/active, quant, ctx capability). Set the stage so the numbers below are interpretable.

Example:

> GB10 unified-memory system (10× Cortex-X925 + 10× A725, 119 GB LPDDR5X, NVIDIA GB10 with Blackwell native FP4) running unsloth/Qwen3.6-35B-A3B-MTP at UD-Q4_K_M (~3B active params, ctx_train 262144).

### 2. Config diff table

Three columns: flag, before → after, one-line rationale. Only include flags that changed.

```
| flag                  | before          | after          | rationale                                            |
|-----------------------|-----------------|----------------|------------------------------------------------------|
| --ctx-size            | 65536           | 131072         | agent loops cross 64k; 109 GB free CUDA, room to grow|
| --ubatch-size         | 512 (default)   | 2048           | prefill not saturated; +18% at 2k context            |
| --cache-ram           | 8192 (default)  | 32768          | re-used project context across turns; biggest win    |
| --parallel            | -1 (auto = 4)   | 1              | single-user agent, dedicates cache to one convo      |
| --cache-type-k        | f16 (default)   | q8_0           | halves KV memory at no measurable quality cost       |
| --cache-type-v        | f16 (default)   | q8_0           | same                                                 |
| --flash-attn          | auto            | on             | explicit; required for KV q8_0 path                  |
| --threads             | 16              | 10             | pinned to 10 X925 perf cores via taskset             |
| --spec-draft-n-max    | 5               | 6              | code is more predictable than prose                  |
| (host) ollama runner  | 58 GB held idle | stopped        | freed unified memory → 109 GB available to llama     |
```

### 3. Benchmark table

Both cold and warm columns, plus a delta column for each metric the user cares about. Always include TTFT (time to first token) and decode tok/s — these are what the agent feels.

```
| case      | prompt | cold ttft (before → after)   | cold decode (before → after) | warm ttft (before → after)   | warm decode (before → after) |
|-----------|--------|------------------------------|------------------------------|------------------------------|------------------------------|
| 256 / 128 |    327 | 0.35s → 0.25s (-28%)         | 50.6 → 65.0 tk/s (+28%)      | 0.20s → 0.042s (-79%)        | 50.9 → 62.5 tk/s (+23%)      |
| 2k / 256  |   2156 | 1.5s → 1.0s   (-33%)         | 49.3 → 56.5 tk/s (+15%)      | 0.18s → 0.044s (-76%)        | 49.2 → 54.7 tk/s (+11%)      |
| 8k / 256  |   8425 | 6.0s → 3.6s   (-40%)         | 38.4 → 44.2 tk/s (+15%)      | 0.20s → 0.047s (-77%)        | 38.0 → 44.2 tk/s (+16%)      |
| 32k / 256 |  33502 | 22.5s → 15.0s (-33%)         | 45.0 → 52.0 tk/s (+16%)      | 0.31s → 0.063s (-80%)        | 44.5 → 51.5 tk/s (+16%)      |
| 64k / 256 |  66940 | 48.0s → 33.9s (-29%)         | 27.0 → 30.4 tk/s (+13%)      | 0.40s → 0.085s (-79%)        | 27.1 → 30.9 tk/s (+14%)      |
```

If a metric got *worse*, show it. Don't filter. Negative results inform the next iteration.

### 4. Verdict (one or two lines)

Where the agent will actually feel the change. Lead with the dominant effect.

Example:

> Warm-turn TTFT is now ~50 ms across all prompt sizes (down from 200-400 ms). Cold prefill at long context is 30% faster but still costs 30+ seconds at 64k — pre-warm the cache at agent startup for best UX.

### 5. Caveats

What the numbers don't tell you. Common ones:
- Workload-dependent tunings (MTP acceptance, KV-quant quality on this user's prompts).
- Single-user assumptions (parallel=1 is wrong if the user multi-agents).
- Measurements with the bench prompt — real prompts may behave differently.

### 6. Next experiments (optional, with commands)

One-flag changes the user can try. For each: what it does, the single-line command to restart with it, the metric to look at.

Example:

```
1. KV q4_0 (long-ctx decode): +15-30% at 64k+, quality risk.
   --cache-type-k q4_0 --cache-type-v q4_0
   Look for: warm decode at 32k+ ↑. Spot-check answer fidelity.

2. ubatch sweep at 4096:
   --ubatch-size 4096
   Look for: cold prefill at 32k/64k.

3. MTP n_max sweep:
   --spec-draft-n-max 8
   Look for: warm decode across all sizes; check log for acceptance rate.
```

## Style rules

- **Numbers, not adjectives**. "23% faster" is useful. "Much faster" is not.
- **Both directions**. Show wins and losses; don't cherry-pick.
- **No emojis** unless the user uses them first.
- **No tables with single rows**. Use a sentence.
- **Don't recommend more than 3 next experiments** — the user will skip past 4+.
