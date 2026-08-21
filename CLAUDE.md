# CLAUDE.md

Python 3.10+ CLI (`ccl`, package `claude_codex_local`) that wires local LLM
engines — Ollama, LM Studio, llama.cpp, vLLM, OpenRouter, 9router — into
coding-agent harnesses.

## Environment setup

YOU MUST follow @docs/AGENT_SETUP.md before running anything: create `.venv`
and `pip install -e '.[dev]'`. A bare checkout cannot even collect tests
(`ModuleNotFoundError: questionary`). Env vars read by `_config.py` take effect
at import time only — export them before starting Python.

## Critical commands

Run inside the activated `.venv`:

```bash
python -m pytest -q --cov=claude_codex_local --cov-report=term-missing   # tests (CI-recorded)
ruff check .            # lint — must pass cleanly
ruff format --check .   # formatter check
```

The known-red baseline (pre-existing failures/skips on `main`) is recorded
verbatim in @docs/AGENT_SETUP.md. Do not mistake it for your own breakage and
do not "fix" it inside an unrelated change.

## Architecture map

- `claude_codex_local/core.py` and `claude_codex_local/wizard.py` are
  **compatibility facades** — thin re-export shims kept so existing imports
  keep working.
- **IMPORTANT:** behaviour changes belong in the implementation modules, never
  the facades: `_*.py` (engine adapters, config, doctor, lifecycle) and
  `wizard_*.py` (state, ui, discovery, steps, cli).
- Engine adapters also live in `claude_codex_local/engines/`.
- Tests mirror modules under `tests/` and monkeypatch through the facades;
  preserve call-time attribute resolution when refactoring.
- Full map: @docs/ARCHITECTURE.md.

## Hard rules

- NEVER commit secrets, `.env`, or anything under `dist/`.
- NEVER hand-edit generated artifacts (`guide.md`, `docs/changelog.html`,
  `docs/*.html`).
- Do not bump dependencies or alter the public API in refactor-only changes.
- Tests marked `local` auto-skip without real binaries — never delete skips to
  make a run green.
- Conventional Commits; branches named `feat/|fix/|docs/<short-description>`.

## Workflow preferences

- Smallest change that satisfies the task; no drive-by refactors.
- Run `ruff check .` after code changes; run the full suite before opening a PR.
- Docs live in `docs/` — update the existing note instead of creating a
  parallel one; keep commands in this file and @docs/AGENT_SETUP.md only.

## What NOT to include here

Do not duplicate into this file: install steps, env-var tables, baseline
numbers (@docs/AGENT_SETUP.md owns them) and subagent definitions
(@AGENTS.md owns them).

## Token Efficiency
- Never re-read files you just wrote or edited. You know the contents.
- Never re-run commands to "verify" unless the outcome was uncertain.
- Don't echo back large blocks of code or file contents unless asked.
- Batch related edits into single operations. Don't make 5 edits when 1 handles it.
- Skip confirmations like "I'll continue..." Just do it.
- If a task needs 1 tool call, don't use 3. Plan before acting.
- Do not summarize what you just did unless the result is ambiguous or you need additional input.
