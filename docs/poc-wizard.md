# POC wizard

Date: 2026-04-10

## What this POC proves

A single interactive command (`bin/claude-codex-local`) takes a user from
"just installed" to "working single-command local coding session" in 8 steps,
without ever touching their official `~/.claude` or `~/.codex` config.

This POC closes the **Claude Code runtime gap** from the previous iteration:
Claude Code is now proven end-to-end against a local Ollama engine, not just
"config-real but runtime-unproven".

## The 8-step flow (PRD v1.2 §4.1)

| Step | Name                         | What happens |
|------|------------------------------|--------------|
| 2.1  | Discover environment         | Probe claude, codex, ollama, lmstudio, llama.cpp, llmfit, and free disk. Print a presence table. Fail fast if the minimum set is not met. |
| 2.2  | Install missing components   | If anything is missing, show install hints per category, wait for the user to install, then re-probe. Runs only when 2.1 detects gaps. |
| 2.3  | Pick preferences             | Interactive primary-harness and primary-engine picker. Skips prompts when only one option exists. Respects `--harness` / `--engine` overrides. |
| 2.4  | Pick a model (**user-first**)| Ask the user which model they want. Default path: accept a direct model name and map it into the selected engine's naming scheme. Opt-in `find-model` path: run llmfit, show a ranked list, let the user pick. Handles disk-aware download branches (exists / fits / too big / cancel). |
| 2.5  | Smoke test engine + model    | Run a minimal "Reply with exactly READY" prompt through the chosen engine. Fail fast if the engine rejects the model. |
| 2.6  | Wire up harness              | Write an isolated `settings.json` (Claude) or Ollama integration config (Codex) under the repo-local state dir. Sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` inside `settings.json` — these MUST live in the file, not as shell env vars. |
| 2.7  | Verify launch command        | Actually run `claude --bare --settings <iso> --model <m> -p "Reply with exactly READY"` and assert `READY` in stdout. This is the step that closes the Claude runtime gap. |
| 2.8  | Generate `guide.md`          | Write a personalized per-machine guide with the exact launch command, troubleshooting notes, and a rollback path. |

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

### Claude Code → any engine

| Env var                                | Value per engine                                             |
|----------------------------------------|--------------------------------------------------------------|
| `ANTHROPIC_BASE_URL`                   | `http://localhost:11434` (ollama) / `:1234` (lms) / `:8001` (llama.cpp) |
| `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` | `ollama` / `lmstudio` / `sk-local`                     |
| `CLAUDE_CODE_ATTRIBUTION_HEADER`       | `"0"` — **must** be in `settings.json`; shell env is ignored |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `"1"`                                                    |

The wizard uses `claude --bare --settings <path> --model <tag> -p ...` for
the verify step to sidestep keychain lookups and side-channel model calls
that can trip LM Studio's strict OpenAI-compatible endpoint.

### Codex CLI → Ollama

Reuses the existing proven path from the earlier POC:

```bash
codex --oss -m <model>
```

Integration config is still written via `ollama launch codex --config`.

## Known limitations

- **LM Studio + Claude Code verify** hits `400 thinking.type` on some models.
  Root cause: Claude Code sends a `thinking` payload that LM Studio's OpenAI
  endpoint rejects as an unknown discriminator. Workaround: use Ollama for the
  Claude path, or swap to a model with a chat template that strips `thinking`.
  A per-model Modelfile / llama.cpp `--chat-template-kwargs` fix is documented
  in `blog-posts/2026-04-03-run-claude-code-codex-local-gemma4` but is not yet
  automated by the wizard.
- **llama.cpp** detection works, but automatic server management is not.
  Users must start `llama-server` themselves.
- **Disk-size estimation** is still a stub — the disk-gated download branch
  runs, but for now it always falls through the "size unknown, warn-only" path.

## Proven paths (on this machine)

| Harness | Engine | Model        | Status |
|---------|--------|--------------|--------|
| Claude  | Ollama | `gemma4:26b` | ✅ verified end-to-end |
| Codex   | Ollama | `qwen2.5-coder:0.5b` | ✅ verified (from prior POC) |
| Claude  | LM Studio | `qwen/qwen3-coder-30b` | ⚠️ smoke-test passes, verify blocked by 400 thinking.type |
| Any     | llama.cpp | any            | ⚠️ detection only; no runtime proof |

## How to re-run

```bash
# Full clean run
rm -f .claude-codex-local/wizard-state.json guide.md
bin/claude-codex-local setup --harness claude --engine ollama

# Non-interactive (CI-friendly)
bin/claude-codex-local setup --non-interactive --harness claude --engine ollama

# Resume after a failed step
bin/claude-codex-local setup --resume

# Standalone model recommendation (no setup)
bin/claude-codex-local find-model
```
