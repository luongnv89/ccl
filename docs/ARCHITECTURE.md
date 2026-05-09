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
- `adapters` — lists the registered `RuntimeAdapter` implementations (ollama, lmstudio, llamacpp, vllm, 9router)

These are reachable for debugging via `python -m claude_codex_local.core <cmd>`. There is no user-facing binary for them — they return JSON for scripting and introspection.

### 2. Interactive setup wizard + CLI (`claude_codex_local/wizard.py`)

The wizard is exposed as the `ccl` binary (installed by `[project.scripts]` in `pyproject.toml`).

A 9-step wizard that runs once (or with `--resume` after a failure):

| Step | Action                                                                     |
| ---- | -------------------------------------------------------------------------- |
| 1    | Discover installed harnesses and engines (lazy llmfit — see `--run-llmfit`) |
| 2    | Report missing tools and installation hints                                |
| 3    | Ask which harness + engine to use                                          |
| 4    | Ask which model (or auto-pick via `llmfit`, installed on-demand if needed) |
| 5    | Smoke-test the engine with the chosen model                                |
| 6    | Wire up the harness                                                        |
| 7    | Install helper script + shell aliases (`cc` / `cx`)                        |
| 8    | End-to-end verification                                                    |
| 9    | Generate personalized `guide.md`                                           |

State is persisted to `.claude-codex-local/wizard-state.json` so a failed run can be resumed without starting over.

### 3. Helper scripts + shell aliases

The user-facing surface after setup:

- `.claude-codex-local/bin/cc` / `cx` / `cc9` / `cx9` — a short bash wrapper that invokes the configured launch command. The `cc9` / `cx9` helpers are installed when the user picks the 9router engine (issue #51); they coexist with the local-engine `cc` / `cx` so a single machine can run both backends.
- `~/.zshrc` / `~/.bashrc` — one fenced block per **install** (`# >>> claude-codex-local:claude >>>` for Claude+local-engine, `# >>> claude-codex-local:claude9 >>>` for Claude+9router, etc.). Fence tags are derived at the alias-emission site as `f"{harness}9"` for 9router and `harness` otherwise, so `state.primary_harness` stays semantic ("claude" / "codex") while the fence-tag stays presentational. Each block is idempotently replaced on re-run of its own install, and all four blocks coexist. A one-shot migration rewraps any legacy (pre-#16) unified block into the per-harness format.

### `WireResult.raw_env` — deferred-secret pattern

`WireResult` carries two env dicts:

- `env: dict[str, str]` — emitted into the helper script with `shlex.quote(value)`. Use for plain values like base URLs.
- `raw_env: dict[str, str]` — emitted **verbatim** into the helper script (no `shlex.quote`). Use ONLY for shell expressions originating in this codebase, never user input. The 9router wiring uses `raw_env={"ANTHROPIC_AUTH_TOKEN": '"$(cat /path/to/key)"'}` so the API key stays in a chmod-600 file and is read at exec time, never embedded in the script body or in `wizard-state.json`.

## Engine Strategies

### Ollama (primary)

Uses `ollama launch claude --model <tag>`, an official Ollama subcommand that:

- Sets the right env vars internally
- Execs the user's real `claude` binary against the local Ollama daemon
- Preserves `~/.claude` exactly as-is — skills, agents, MCP servers all work

### LM Studio / llama.cpp (secondary)

Uses an inline-env approach: the helper script exports `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and related vars, then execs the harness. This works because both Claude Code and Codex CLI support OpenAI-compatible endpoints.

### 9router (cloud-routing proxy, optional)

[9router](https://github.com/decolua/9router) is a local server that exposes an OpenAI-compatible API on `http://localhost:20128/v1` and forwards calls to paid cloud models (e.g. `kr/claude-sonnet-4.5`). The CCL adapter:

- **Detects via `GET /v1/models`** with a 5-second timeout — never `/chat/completions`, since each chat call burns paid quota.
- **Uses the deferred-secret pattern** (see `WireResult.raw_env` above): the API key is stored in `~/.claude-codex-local/9router-api-key` (chmod 0600) and the helper script reads it at exec time via `$(cat …)`.
- **Skips Step 7 chat-verification.** The standard verify step runs `claude --model <tag> -p "Reply with exactly READY"`. For 9router this would issue a real paid `/chat/completions` call, so the wizard short-circuits with a `/v1/models` reachability check and records `state.verify_result = {"ok": ..., "via": "9router-models-endpoint", "skipped_chat": True}`.
- **Does not download or score models.** Step 4 has a dedicated `_step_4_pick_model_9router` branch that prompts for an API key + model name (default `kr/claude-sonnet-4.5`) and skips llmfit / disk / download paths entirely.

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
