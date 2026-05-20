---
name: llamacpp-tuner
description: "Profile a running llama.cpp server, propose a coding-agent-tuned config (ctx, prompt cache, MTP), run before/after benchmarks, report deltas. Use to tune llama-server for local coding agents. Don't use for vLLM, ollama, or non-llama.cpp backends."
effort: high
metadata:
  version: 1.0.0
  author: "Luong NGUYEN <luongnv89@gmail.com>"
---

# llamacpp-tuner

## Overview

Inspect a running `llama-server`, identify the bottlenecks that matter for a *coding agent* workload (long context, heavy prompt reuse, latency-sensitive tool calls), propose a tuned configuration, measure before/after, and report concrete deltas. The result is an actionable optimization with numbers behind it, not generic advice.

## When to use

Use this skill when the user asks to **tune, optimize, or speed up llama.cpp / llama-server** for a local coding agent — phrases like "make my local model faster", "optimize llamacpp for Claude Code / Codex / Aider", "best llama-server config", or invokes `/llamacpp-tuner`.

Do **not** use this skill for: vLLM, Ollama (other than as a memory-pressure suspect to free), text-generation-webui, or proprietary inference engines. The benchmark and the knobs are specific to `llama-server`.

## Workflow

The skill runs as a single sequential workflow. Each step gates the next.

```
1. discover machine    →  2. discover running server  →  3. research model
        ↓                          ↓                           ↓
        └──────────────────────────┴───────────────────────────┘
                                   ↓
                          4. propose config
                                   ↓
                          5. user approval gate ← STOP if user rejects
                                   ↓
                          6. benchmark BEFORE
                                   ↓
                          7. apply config (kill+restart)
                                   ↓
                          8. benchmark AFTER
                                   ↓
                          9. final report
```

### Step 1 — Discover machine

Gather what platform you're targeting. The right answer for a GB10 unified-memory system is very different from a workstation with a discrete GPU + system RAM.

Run, then read carefully:

```bash
lscpu | grep -E "Model name|Architecture|^CPU\(s\)|Thread|Core|Socket|NUMA|MHz"
free -h
numactl -H 2>/dev/null || echo "no numactl"
nvidia-smi 2>&1 | head -30
# Per-core max frequency reveals ARM big.LITTLE asymmetry
for i in $(seq 0 $(($(nproc)-1))); do
  f=$(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>/dev/null)
  echo "cpu$i max_khz=$f"
done | sort -t= -k2 -nr
```

Record:
- **Arch** (x86_64 vs aarch64). aarch64 may be big.LITTLE — note the perf-core CPU ids for `taskset`.
- **GPU model** and whether memory is **unified** (GB10, Grace-Hopper, M-series Mac equivalents) or **discrete**.
- **Free RAM and free GPU memory** *right now*. If unified memory and another LLM process (ollama, vLLM, ...) is hoarding it, **flag this** — it's the single biggest performance killer.
- **NUMA topology**. Multi-socket x86 needs `numactl --interleave=all` or per-socket pinning.

### Step 2 — Discover the running server

The current `llama-server` invocation tells you what was already chosen. Don't replace flags you don't understand.

```bash
pgrep -fa llama-server         # full cmdline
ps -o pid,rss -p <pid>         # memory footprint
ss -tlnp | grep <port>         # confirm port
curl -s http://<host>:<port>/health
curl -s http://<host>:<port>/v1/models
```

Find the server log (usually a `/tmp/llama-server*.log` or wherever the user redirected stdout) and grep for the load-time configuration:

```bash
grep -iE "CUDA0|n_threads|n_parallel|n_ctx|kv_unified|flash|cache-ram|prompt cache|GRAPHS|FP4|model loaded|listening" <log>
grep -iE "draft acceptance rate|statistics draft-mtp" <log> | tail -20
```

Record:
- Current ctx, parallel, batch/ubatch, threads, KV cache type, FA on/off, prompt-cache size.
- **Free GPU memory at server start** (vs total) — exposes memory pressure.
- **MTP draft acceptance rate** if speculative decoding is on. <30% = poorly matched, >50% = excellent.

### Step 3 — Research the model

Identify the model from the loaded GGUF filename. Pull the HF repo metadata for ground truth (the GGUF filename and the user's mental model can disagree):

```bash
curl -s "https://huggingface.co/api/models/<org>/<repo>" \
  | python3 -c "import json,sys;d=json.load(sys.stdin);
print('id:',d['id']);
print('tags:',d.get('tags'));
[print(' ',s['rfilename']) for s in d.get('siblings',[]) if s['rfilename'].endswith('.gguf')]"
```

Or via the model's own context if `llama-server` exposes it:

```bash
curl -s http://<host>:<port>/props 2>/dev/null | python3 -m json.tool | head -40
```

Record:
- **Total vs active params** (MoE: e.g. 35B-A3B = 35B total, ~3B active per token). Tells you whether decode is bandwidth-bound or compute-bound.
- **n_ctx_train** (native max context). Is the current `--ctx-size` leaving headroom on the table, or already at the ceiling?
- **MTP / multi-token prediction support** — the GGUF metadata or repo name will tell you.
- **Quant**: Q4_K_M, Q5_K_M, UD-Q*, IQ*, Q8_0. Decode speed and memory cost scale with this.

### Step 4 — Propose the optimized configuration

Read `references/agent-tuning-knobs.md` for the full knob-by-knob rationale. Apply the priorities in this order:

1. **Free unified memory first** (if applicable). `ollama stop <model>` or kill idle LLM runners. On unified-memory systems, every GB held by another process is one less for the GPU compute path.
2. **`--parallel 1`** for single-user agent profile. Default auto-allocates 4 slots and fragments the prompt cache.
3. **`--cache-ram` to 32768 (or larger)**. The biggest single agent win — warm-turn TTFT drops from seconds to tens of ms because the cached prefix is reused.
4. **`--flash-attn on`** (explicit, not `auto`).
5. **`--cache-type-k q8_0 --cache-type-v q8_0`**. Halves KV memory at ~no quality cost. Required for KV q4_0 later.
6. **`--ctx-size`** sized to actual agent need. 65536 if your agent stays under 64k; 131072 for heavier loops; 262144 only if you genuinely use it (KV cost is linear).
7. **`--ubatch-size 1024–2048`** (sweep). Default 512 leaves prefill throughput on the table. Bigger ubatch helps long-prompt cold prefill the most.
8. **`--batch-size 4096`**. Logical batch ceiling; doesn't cost memory unless ubatch fills it.
9. **`--threads N` matched to perf cores only** (on big.LITTLE: `taskset -c <perf-core-ids>` and set `--threads N` = number of perf cores). On x86 SMT: physical-core count.
10. **`--spec-draft-n-max`** at 5–6 for code (higher acceptance than prose). Sweep 3/5/8 once stable.

Do **not** apply on unified-memory systems: `-cmoe`, `-ncmoe` (no benefit — same memory pool).

Output a config diff table: column 1 current flag value, column 2 proposed, column 3 one-line rationale per row. Don't bury the user in prose.

### Step 5 — User approval gate

Show the config diff, the predicted impact (e.g. "warm TTFT should drop ~100× because the 32GB prompt cache will catch repeated project context"), and **ask the user to accept**. The skill applies a destructive action — stopping their server and restarting — and the user may have in-flight requests or dependencies.

Phrase as a single yes/no question. Do not proceed without approval.

If declined, output the proposed config as a paste-ready command and stop.

### Step 6 — Benchmark BEFORE

Use `scripts/bench_agent.py` against the **currently running** server. The script runs five prompt sizes (256 / 2k / 8k / 32k / 64k) and reports both **cold** (`cache_prompt=false`) and **warm** (`cache_prompt=true`, same prompt re-sent) numbers. Warm numbers are the ones that matter for an agent.

```bash
python3 scripts/bench_agent.py --base http://127.0.0.1:<port> --out /tmp/bench-before.json
```

Save the result. Don't apply the config yet.

### Step 7 — Apply config

1. Capture the exact current `llama-server` command line (from `pgrep -fa`).
2. Stop the running server with `kill <pid>`, wait up to 20s for the port to free.
3. Launch the new config under `nohup`, redirecting stdout to a fresh log file. Use `taskset -c <perf-core-ids>` on big.LITTLE systems.
4. Poll `/health` until `{"status":"ok"}` or the process dies. If it dies, dump the last 50 lines of the log, restore the previous command line, and abort.

### Step 8 — Benchmark AFTER

Run `scripts/bench_agent.py` again with `--out /tmp/bench-after.json`. Same cases as before — apples-to-apples comparison only works if the prompt shapes match exactly, which the script guarantees.

### Step 9 — Final report

Read `references/report-format.md` for the report template. The report must contain:

- **Machine + model summary** (one paragraph).
- **Diff table** of flags changed (before → after, one-line rationale per row).
- **Benchmark table** with both cold and warm columns, plus a **delta column** for each.
- **Verdict line**: where the agent will feel this — usually "warm TTFT ↓ Nx" and "long-context decode ↑ N%".
- **Caveats**: which tunings depend on workload assumptions (e.g. "prompt cache only helps if the agent re-sends overlapping context across turns — confirm with logs after a real session").
- **Optional next experiments** with the single-flag commands to run them.

Output Step Completion Report after each major step using the format described below.

## Step Completion Report format

After each numbered step, emit:

```
◆ <step name> (step N of 9 — <one-line context>)
··································································
  <Check 1>:          √ pass — <one-line detail>
  <Check 2>:          × fail — <reason>
  <Check 3>:          √ pass
  ____________________________
  Result:             PASS | FAIL | PARTIAL
```

`PASS` lets the workflow continue. `FAIL` stops it — never apply a config or run a benchmark from a failed discovery state.

## Resources

- `scripts/bench_agent.py` — coding-agent shaped benchmark. Sends five prompt sizes, both cold and warm, parses the server's `timings` field, writes machine-readable JSON.
- `references/agent-tuning-knobs.md` — every flag this skill might recommend, why, and what it costs.
- `references/report-format.md` — the final-report template.

## Notes & gotchas

- **Never apply KV `q4_0` blindly** for a coding agent. It does lift long-ctx decode 15–30% but trades fidelity that matters when the model has to recall specific code references. Recommend it only as an *experiment* with a quality spot-check.
- **MTP draft acceptance is workload-dependent**. The number measured during a synthesis prompt (e.g. "summarize") is not the number you'll see on real code completions. Re-measure with a representative prompt before locking `--spec-draft-n-max`.
- **The benchmark prompt deliberately avoids early EOS** by asking for a long structured review. If you change the prompt template, verify `n_predict` is actually reached on at least the medium cases — otherwise decode tok/s is reliable but wall-clock comparisons are not.
- **Unified memory systems** (GB10, Grace-Hopper, Apple Silicon equivalents): MoE CPU offload (`-cmoe`, `-ncmoe`) is pointless — CPU and GPU share the same memory.
- **The skill is read-only on the model file**. It never re-quantizes, re-converts, or modifies the GGUF. Restart-only configuration changes.
