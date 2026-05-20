# Agent-tuning knobs reference

Every flag this skill might recommend, what it does, what it costs, and when to use it. Read this before composing the proposed config.

## High-impact, low-risk (apply unconditionally for a coding agent)

### `--parallel 1`

What: serves only one request at a time, dedicates the prompt cache to a single conversation.

Why for agents: a coding agent runs sequential turns from one user. Default auto-allocates ~4 slots which fragments the prompt cache and adds bookkeeping overhead. With `kv_unified=true` the saving is small but real.

Cost: cannot serve concurrent users. If the user has multiple agent instances pointing at the same server, this is wrong — keep auto.

### `--cache-ram N` (megabytes)

What: maximum size of the server-side prompt cache. Default 8192 (8 GB).

Why for agents: **this is the biggest single agent win**. The agent re-sends the same project context every turn. With a generous prompt cache, the cached prefix is reused and warm-turn TTFT drops from seconds to tens of milliseconds.

Recommended: 32768 (32 GB). Larger if you have memory and your project contexts run >100k tokens.

Cost: consumes unified memory / system RAM. Worthless if your agent never re-sends overlapping context.

### `--flash-attn on`

What: explicit FA enable.

Why: the default is `auto`, which can silently disable FA on some model loads. Make it explicit so behavior is reproducible across restarts.

Cost: requires FA-supported attention shape (almost all modern models). Required upstream of KV q8_0 / q4_0.

### `--cache-type-k q8_0 --cache-type-v q8_0`

What: quantize K and V cache to 8-bit.

Why: KV cache at fp16 is ~10–20 GB at 131k context. Halving it frees memory and improves cache locality, with no measurable quality loss on Q4 models.

Cost: marginally lower fidelity. For coding workloads where the model must recall specific identifiers, q8_0 is safe; q4_0 is not (see below).

### `taskset -c <perf-core-ids>` + `--threads N`

What: pin the server to the high-frequency cores only, match `--threads` to that count.

Why for ARM big.LITTLE (GB10 has 10× Cortex-X925 @ 4 GHz + 10× Cortex-A725 @ 2.86 GHz): mixed threading drags decode latency down to the slowest thread. Pinning to perf cores eliminates the slow-core straggler.

Cost: gives up the efficiency cores for other work. On a dedicated inference box this is fine; on a shared machine, leave headroom.

How to find the perf cores:
```bash
for i in $(seq 0 $(($(nproc)-1))); do
  f=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>/dev/null)
  echo "cpu$i max_khz=$f"
done | sort -t= -k2 -nr
```
Top-N (where N = perf-core count) are your targets.

### `--ctx-size N`

What: maximum context window. The model's `n_ctx_train` is the ceiling.

Why for agents: sized to what the agent actually needs. Going larger than necessary costs KV memory linearly with no upside. Going smaller forces the agent to truncate.

Recommended starting points:
- Short-loop agents (CLI assistants): 32768
- Standard coding agents (Claude Code, Codex, Aider): 131072
- Long-context loops (multi-repo, big monolith): 196608 or up to `n_ctx_train`

Cost: linear KV memory growth. At 128k ctx with q8_0 KV on a ~35B MoE: ~12 GB. At 256k: ~24 GB.

## Medium-impact (sweep before locking)

### `--ubatch-size N`

What: physical micro-batch size for prefill. Default 512.

Why: prefill throughput often climbs with bigger ubatch until GPU is saturated. On a GB10 with Q4_K_M MoE, 1024 is consistently better than 512; 2048 helps further at long prompts; 4096 should be tested.

Recommended: 2048 as a default. Sweep 1024 / 2048 / 4096 if cold prefill matters.

Cost: peak GPU compute-buffer memory grows with ubatch. If the server fails to start with "out of memory" or similar, halve it.

### `--batch-size N`

What: logical batch ceiling (a multiple of ubatch). Default 2048.

Why: should be ≥ ubatch. Setting it well above ubatch costs nothing.

Recommended: 4096.

### `--spec-draft-n-max N` (when MTP is enabled)

What: maximum number of speculative tokens to draft per round.

Why: with high acceptance, longer drafts pay off (fewer round-trips to the target model). With low acceptance, longer drafts waste cycles on rejected tokens.

How to tune: measure draft acceptance rate from the server log (`grep "draft acceptance rate" <log>`). Rule of thumb:
- Acceptance > 60% → try 8–10
- Acceptance 40–60% → 5–6 (most code workloads land here)
- Acceptance < 30% → try 2–3 or disable MTP entirely

Cost: rejected drafts cost the draft model's forward time. The cost-benefit shifts with acceptance rate, so re-tune per workload.

## Experimental (require quality spot-check)

### `--cache-type-k q4_0 --cache-type-v q4_0`

What: quantize KV cache to 4-bit.

Why: halves KV memory again vs q8_0. Long-context decode can lift 15–30% because attention is bandwidth-bound at scale.

Cost: measurable quality loss on tasks that require recalling specific tokens from far back in the context (identifier names, exact code references). For a coding agent, **always spot-check** before keeping.

How to spot-check: run a few realistic prompts before and after; compare the answers manually. If the model invents identifier names or misremembers literals, revert.

### `--prio 1` or `--prio 2` (medium / high)

What: process priority for the server.

Why: on a contended system, the server competes with other processes for CPU. Boosting priority reduces tail latency.

Cost: starves other processes. Don't use on a desktop machine where you also browse / run an IDE; fine on dedicated inference boxes.

## Useless / harmful for unified-memory systems

These are sometimes recommended in generic guides — they do not apply to GB10, Grace-Hopper, Apple Silicon, or any unified-memory device:

- `-cmoe` / `--cpu-moe` — moves MoE experts to "CPU memory". On unified memory, CPU memory IS GPU memory. No transfer is avoided, only overhead is added.
- `-ncmoe N` / `--n-cpu-moe N` — same reasoning, partial form.
- `--no-mmap` — mmap on unified memory costs nothing. Disabling it forces a full read into RSS, doubling memory transiently during load.

## Useless for the BEFORE/AFTER benchmark workflow

These don't affect benchmark numbers; skip unless the user explicitly asks:

- `--threads-http` — concurrent HTTP request handling; benchmark is single-client.
- `--cont-batching` — already the default; turning it off hurts.
- `--no-warmup` — saves a few seconds at startup, costs first-request quality. Don't toggle for a benchmark.

## Quick reference: typical agent-tuned launch

For a GB10-class unified-memory system running a 35B-A3B MoE at Q4_K_M with MTP:

```bash
taskset -c 5-9,15-19 \
llama-server \
  --model <path>.gguf \
  --host 127.0.0.1 --port 8001 \
  --ctx-size 131072 \
  --n-gpu-layers -1 \
  --threads 10 --threads-batch 10 \
  --parallel 1 \
  --batch-size 4096 --ubatch-size 2048 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 32768 \
  --spec-type draft-mtp --spec-draft-n-max 6
```

Adjust:
- `taskset` core list to the perf-core ids of your CPU (see discovery step).
- `--threads N` to the number of perf cores.
- `--ctx-size` to your agent's working context (32k / 64k / 128k / 256k).
- `--ubatch-size` after a sweep at your typical prompt length.
- Remove `--spec-*` lines if MTP isn't supported by your GGUF.
