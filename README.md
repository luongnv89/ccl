# claude-codex-local

Local backend bridge for Claude Code and Codex.

Core idea:
- keep the existing Claude Code / Codex harness
- swap the backend to a best-fit local model/runtime
- keep workflow change as close to zero as possible

## POC status

This repo now has a real POC for the narrowest sensible path:

- runtimes: **LM Studio (MLX, preferred on Apple Silicon)** and **Ollama (fallback)**
- real proved harness: **Codex CLI**
- config-prepared harness: **Claude Code**
- model-fit helper: **llmfit**
- isolation rule: **repo-local state only; official configs untouched**

## Quickstart

### Prereqs

At least one harness (Claude Code or Codex), at least one engine (Ollama, LM
Studio, or llama.cpp), and `llmfit` on `PATH`. The wizard will tell you what's
missing and how to install it.

### First run (interactive wizard)

Install the Python dependencies (one-time):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the 8-step interactive setup:

```bash
./bin/claude-codex-local
```

The wizard will:

1. discover what you already have (harnesses, engines, llmfit, free disk)
2. tell you what's missing and how to install it
3. ask which harness + engine you want to use
4. ask which model you want (or help you pick via `llmfit`)
5. smoke-test the engine with that model
6. wire up an **isolated** harness config (your real `~/.claude` / `~/.codex` are untouched)
7. verify the launch command works end-to-end
8. write a personalized `guide.md` with your exact daily-use command
   (see [`guide.example.md`](./guide.example.md) for a sanitized example
   of what that generated output looks like — real values are filled in
   from your wizard run)

### Useful flags

```bash
./bin/claude-codex-local setup --harness claude --engine ollama   # skip the prefs picker
./bin/claude-codex-local setup --non-interactive                  # CI-friendly
./bin/claude-codex-local setup --resume                           # pick up after a failed step
./bin/claude-codex-local find-model                               # standalone llmfit recommendation
```

### Legacy POC helpers

The earlier POC scripts still work for one-off diagnostics:

```bash
./bin/poc-machine-profile   # dump the full machine profile as JSON
./bin/poc-doctor            # run the old doctor + Codex smoke test
./bin/poc-recommend         # llmfit-only recommendation (older shape)
./bin/codex-local           # legacy Codex wrapper (pre-wizard)
```

## Repo-local state

Everything local to the bridge is written under:

```text
.claude-codex-local/
```

You can override that with `CLAUDE_CODEX_LOCAL_STATE_DIR`.

## Included docs

- `idea.md`
- `validate.md`
- `prd.md`
- `tasks.md`
- `docs/poc-wizard.md` — **current** 8-step wizard architecture
- `docs/poc-bootstrap.md`
- `docs/poc-architecture.md`
- `docs/poc-proof.md`

## Positioning

This is **not** an offline replacement for Claude Code or Codex.
It is a local backend bridge that preserves the existing workflow while making local model usage practical.
