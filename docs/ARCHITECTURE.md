# Architecture

This document describes the system design of `claude-codex-local`.

## Overview

`claude-codex-local` is a **local backend bridge** that sits between the Claude Code / Codex CLI harness (the AI coding tool the user already knows) and a locally-running LLM server. It does not replace or modify the harness — it teaches the harness to talk to a local model instead of the Anthropic / OpenAI cloud.

```
┌──────────────────────────────────────────────────────────┐
│  User terminal                                           │
│                                                          │
│   cc  (alias)  →  .claude-codex-local/bin/cc            │
│                          │                              │
│                          ▼                              │
│              ollama launch claude --model <tag>          │
│                    OR                                    │
│              OPENAI_BASE_URL=... claude                  │
│                          │                              │
│                          ▼                              │
│          Real ~/.claude config is used as-is            │
│          (skills, agents, MCP servers unchanged)        │
└─────────────────────────┬────────────────────────────────┘
                          │  OpenAI-compatible HTTP
                          ▼
            ┌─────────────────────────┐
            │  Local LLM engine       │
            │  Ollama / LM Studio /   │
            │  llama.cpp              │
            └─────────────────────────┘
```

## Three Layers

### 1. Machine profile + model recommendation (`claude_codex_local/core.py`)

- `profile` — dumps a JSON snapshot of installed harnesses, engines, `llmfit`, and free disk
- `recommend` — picks the best-fit installed coding model for the hardware
- `doctor` — pretty-prints the current wizard state and re-runs presence checks
- `adapters` — lists the registered `RuntimeAdapter` implementations (ollama, lmstudio, llamacpp)

These are reachable for debugging via `python -m claude_codex_local.core <cmd>`. There is no user-facing binary for them — they return JSON for scripting and introspection.

### 2. Interactive setup wizard + CLI (`claude_codex_local/wizard.py`)

The wizard is exposed as the `ccl` binary (installed by `[project.scripts]` in `pyproject.toml`).

A 9-step wizard that runs once (or with `--resume` after a failure):

| Step | Action |
|------|--------|
| 1 | Discover installed harnesses and engines |
| 2 | Report missing tools and installation hints |
| 3 | Ask which harness + engine to use |
| 4 | Ask which model (or auto-pick via `llmfit`) |
| 5 | Smoke-test the engine with the chosen model |
| 6 | Wire up the harness |
| 7 | Install helper script + shell aliases (`cc` / `cx`) |
| 8 | End-to-end verification |
| 9 | Generate personalized `guide.md` |

State is persisted to `.claude-codex-local/wizard-state.json` so a failed run can be resumed without starting over.

### 3. Helper scripts + shell aliases

The user-facing surface after setup:

- `.claude-codex-local/bin/cc` (or `cx`) — a short bash wrapper that invokes the configured launch command
- `~/.zshrc` / `~/.bashrc` — one fenced block per harness (`# >>> claude-codex-local:claude >>>` … `# <<< claude-codex-local:claude <<<` for the Claude harness, `# >>> claude-codex-local:codex >>>` … `# <<< claude-codex-local:codex <<<` for Codex); each block is idempotently replaced on re-run of its own harness, and the two blocks coexist so `cc` and `cx` can both be installed at once. A one-shot migration rewraps any legacy (pre-#16) unified block into the per-harness format.

## Engine Strategies

### Ollama (primary)

Uses `ollama launch claude --model <tag>`, an official Ollama subcommand that:

- Sets the right env vars internally
- Execs the user's real `claude` binary against the local Ollama daemon
- Preserves `~/.claude` exactly as-is — skills, agents, MCP servers all work

### LM Studio / llama.cpp (secondary)

Uses an inline-env approach: the helper script exports `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and related vars, then execs the harness. This works because both Claude Code and Codex CLI support OpenAI-compatible endpoints.

## Isolation Rule

**The wizard never writes to `~/.claude` or `~/.codex`.**

All state is isolated under `.claude-codex-local/` (or `$CLAUDE_CODEX_LOCAL_STATE_DIR`). The user's global config is always used read-only.

## Rollback

Remove the alias block from `~/.zshrc` / `~/.bashrc` and delete `.claude-codex-local/`. The original `claude` / `codex` commands are unaffected.

## Related docs

- [`poc-wizard.md`](poc-wizard.md) — detailed wizard step specification
- [`poc-architecture.md`](poc-architecture.md) — original POC architecture notes
- [`poc-bootstrap.md`](poc-bootstrap.md) — install / bootstrap flow
- [`poc-proof.md`](poc-proof.md) — design rationale and proof-of-concept validation
