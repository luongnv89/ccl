# POC wizard

Date: 2026-04-10

## What this POC proves

A single interactive command (`bin/claude-codex-local`) takes a user from
"just installed" to "working single-command local coding session" in 9 steps,
without ever touching their official `~/.claude` or `~/.codex` config. The
daily driver is a one-word shell alias (`cc` or `cx`).

This POC closes the **Claude Code runtime gap** from the previous iteration:
Claude Code is now proven end-to-end against a local Ollama engine, not just
"config-real but runtime-unproven".

## The 9-step flow

| Step | Name                         | What happens |
|------|------------------------------|--------------|
| 2.1  | Discover environment         | Probe claude, codex, ollama, lmstudio, llama.cpp, llmfit, and free disk. Print a presence table. Fail fast if the minimum set is not met. |
| 2.2  | Install missing components   | If anything is missing, show install hints per category, wait for the user to install, then re-probe. Runs only when 2.1 detects gaps. |
| 2.3  | Pick preferences             | Interactive primary-harness and primary-engine picker. Skips prompts when only one option exists. Respects `--harness` / `--engine` overrides. |
| 2.4  | Pick a model (**user-first**)| Ask the user which model they want. Default path: accept a direct model name and map it into the selected engine's naming scheme. Opt-in `find-model` path: run llmfit, show a ranked list, let the user pick. Handles disk-aware download branches (exists / fits / too big / cancel). |
| 2.5  | Smoke test engine + model    | Run a minimal "Reply with exactly READY" prompt through the chosen engine. Fail fast if the engine rejects the model. Also measures and reports model speed in tokens/second so users can gauge throughput before committing. |
| 2.6  | Wire up harness              | Build a `WireResult` (argv + inline env). For Ollama this is just `ollama launch <harness> --model <tag>` — `ollama launch` sets the right env vars internally and execs the user's real `claude`/`codex` against their real `~/.claude` / `~/.codex`. For LM Studio / llama.cpp the env is set inline because `ollama launch` does not apply. **No isolated HOME**, no duplicated settings file. |
| 2.65 | Install helper script + aliases | Write `.claude-codex-local/bin/cc` (or `cx`) — a tiny bash wrapper that exports any inline env and execs the wire argv. Then append a per-harness fenced block (`# >>> claude-codex-local:claude >>>` … `# <<< claude-codex-local:claude <<<` for the Claude harness; `# >>> claude-codex-local:codex >>>` … `# <<< claude-codex-local:codex <<<` for Codex) to the user's `~/.zshrc` / `~/.bashrc` containing `alias cc=…`/`alias claude-local=…` (or the codex pair). Each harness's block is overwritten in place on re-run of its own harness, and the two blocks coexist so `cc` and `cx` can be installed simultaneously. Any legacy (pre-#16) unified block is migrated in place on first touch. |
| 2.7  | Verify launch command        | Actually run the wired argv with `-p "Reply with exactly READY"` (Ollama path uses `ollama launch <harness> -- -p …`; LM Studio / llama.cpp merges the inline env into `os.environ` and runs the argv directly) and assert `READY` in stdout. No `--bare --settings` — the verify uses the same code path the daily alias will use. |
| 2.8  | Generate `guide.md`          | Write a personalized per-machine guide: alias names, helper script path, shell rc file, troubleshooting notes, rollback instructions. |

Each step writes its progress to `.claude-codex-local/wizard-state.json`,
so `--resume` can pick up from the last completed step after a failure.

## Why user-first model pick

The previous iteration drove everything from `llmfit`. That worked, but it
bundled "recommend a model" into "run the wizard" — users who already knew
what they wanted had to fight the recommender. The new shape:

- **Default** — user types the model name they want. Fast path, zero magic.
- **Opt-in** — user picks "help me pick" in the wizard, or runs the
  standalone `claude-codex-local find-model` subcommand any time.

Both paths converge on the same downstream disk/download/smoke-test/wire-up
pipeline.

## Runtime bridge contracts

### Claude Code → Ollama

The wizard builds `argv = ["ollama", "launch", "claude", "--model", <tag>]`
with an empty env dict. `ollama launch` sets `ANTHROPIC_BASE_URL`,
`ANTHROPIC_API_KEY`, `ANTHROPIC_CUSTOM_MODEL_OPTION`, and
`CLAUDE_CODE_ATTRIBUTION_HEADER` internally and execs the user's real
`claude` binary against `~/.claude`. The wizard writes nothing to
`~/.claude`.

### Claude Code → LM Studio / llama.cpp

`ollama launch` does not apply, so the wizard sets the env vars inline
in the helper script:

| Env var                                      | LM Studio                  | llama.cpp                  |
|----------------------------------------------|----------------------------|----------------------------|
| `ANTHROPIC_BASE_URL`                         | `http://localhost:1234`    | `http://localhost:8001`    |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | `lmstudio`                 | `sk-local`                 |
| `ANTHROPIC_CUSTOM_MODEL_OPTION`              | `<tag>`                    | `<tag>`                    |
| `ANTHROPIC_CUSTOM_MODEL_OPTION_NAME`         | `Local (lmstudio) <tag>`   | `Local (llamacpp) <tag>`   |
| `CLAUDE_CODE_ATTRIBUTION_HEADER`             | `"0"`                      | `"0"`                      |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`   | `"1"`                      | `"1"`                      |

The helper script exports all of these and then execs
`claude --model <tag> "$@"`.

### Codex CLI → Ollama

```bash
ollama launch codex --model <tag> -- --oss --local-provider=ollama
```

`ollama launch codex` forwards `-m <tag>` to `codex` automatically, but
the `--oss --local-provider=ollama` flags must still be supplied after
`--` because Codex otherwise tries to route the request through the
user's ChatGPT account and rejects non-OpenAI model names.

### Codex CLI → LM Studio / llama.cpp

The helper script exports `OPENAI_BASE_URL` and `OPENAI_API_KEY`, then
execs `codex -m <tag> "$@"`.

## Known limitations

- **LM Studio + Claude Code + Qwen3** hits `400 thinking.type`. Root cause:
  Claude Code sends a `thinking` payload that Qwen3 reasoning models interpret
  as an unterminated `<think>` block. The wizard warns on Qwen3 model names at
  pick time and recommends Gemma 3 or Qwen 2.5 Coder instead. The wizard no
  longer auto-builds a `-cclocal` Ollama variant as a workaround (see
  `docs/poc-architecture.md`).
- **llama.cpp** detection works, but automatic server management is not.
  Users must start `llama-server` themselves.
- **Disk-size estimation** is still a stub — the disk-gated download branch
  runs, but for now it always falls through the "size unknown, warn-only" path.

## Proven paths

| Harness | Engine | Model               | Status |
|---------|--------|---------------------|--------|
| Claude  | Ollama | `gemma4:26b`        | ✅ verified end-to-end via `ollama launch claude` |
| Codex   | Ollama | `gemma4:26b`        | ✅ verified via `ollama launch codex -- --oss --local-provider=ollama` |
| Codex   | Ollama | `qwen2.5-coder:0.5b` | ✅ verified (from earlier POC) |
| Claude  | LM Studio | Qwen3 family    | ⚠️ blocked by `400 thinking.type`; wizard warns and recommends alternatives |
| Any     | llama.cpp | any              | ⚠️ inline-env code path exists, no live runtime proof |

## How to re-run

```bash
# Full clean run
rm -rf .claude-codex-local guide.md
bin/claude-codex-local setup --harness claude --engine ollama

# Non-interactive (CI-friendly)
bin/claude-codex-local setup --non-interactive --harness claude --engine ollama

# Resume after a failed step
bin/claude-codex-local setup --resume

# Standalone model recommendation (no setup)
bin/claude-codex-local find-model
```
