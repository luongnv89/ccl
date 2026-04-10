# claude-codex-local

Local backend bridge for Claude Code and Codex.

Core idea:
- keep the existing Claude Code / Codex harness, including all your real
  `~/.claude` and `~/.codex` config — skills, statusline, agents, plugins,
  MCP servers all keep working
- swap the backend to a best-fit local model/runtime
- install a short shell alias (`cc` for Claude, `cx` for Codex) so your
  daily command is one word

## POC status

This repo now has a real POC for the narrowest sensible path:

- runtimes: **Ollama (primary, via `ollama launch`)** and **LM Studio (MLX, secondary)**
- harnesses: **Claude Code** and **Codex CLI**
- model-fit helper: **llmfit**
- isolation rule: **the wizard never touches `~/.claude` or `~/.codex`**;
  it installs a helper script under `.claude-codex-local/bin/` and a
  single fenced alias block in `~/.zshrc` / `~/.bashrc`

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

Run the interactive setup:

```bash
./bin/claude-codex-local
```

The wizard will:

1. discover what you already have (harnesses, engines, llmfit, free disk)
2. tell you what's missing and how to install it
3. ask which harness + engine you want to use
4. ask which model you want (or help you pick via `llmfit`)
5. smoke-test the engine with that model
6. wire up the harness: for Ollama it captures
   `ollama launch claude|codex --model <tag>`; for LM Studio / llama.cpp
   it records the inline env needed to point the harness at the local
   server
7. write a helper script to `.claude-codex-local/bin/{cc,cx}` and install
   aliases (`cc` + `claude-local`, or `cx` + `codex-local`) into your
   shell rc file between fenced markers (idempotent — re-running replaces
   the block in place)
8. verify the launch command end-to-end
9. write a personalized `guide.md` with your exact daily-use command
   (see [`guide.example.md`](./guide.example.md) for a sanitized example
   of what that generated output looks like — real values are filled in
   from your wizard run)

After setup, open a new terminal (or `source ~/.zshrc`) and run `cc`
(or `cx`). That's it.

### Useful flags

```bash
./bin/claude-codex-local setup --harness claude --engine ollama   # skip the prefs picker
./bin/claude-codex-local setup --non-interactive                  # CI-friendly
./bin/claude-codex-local setup --resume                           # pick up after a failed step
./bin/claude-codex-local find-model                               # standalone llmfit recommendation
```

### Diagnostic helpers

```bash
./bin/poc-machine-profile   # dump the full machine profile as JSON
./bin/poc-doctor            # print the wizard state + recommendation
./bin/poc-recommend         # llmfit-only model recommendation
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
