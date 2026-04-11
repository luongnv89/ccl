# Development Guide

This document covers local development setup, tooling, and debugging.

## Prerequisites

- Python 3.10+
- git
- At least one engine installed for integration tests: Ollama, LM Studio, or llama.cpp
- (Optional) `llmfit` on `PATH`

## Setup

```bash
git clone https://github.com/luongnv89/claude-codex-local.git
cd claude-codex-local

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pre-commit install                 # set up git hooks
```

The editable install puts the `ccl` entry point on your `PATH` inside the virtualenv.

## Running the Wizard

```bash
ccl                                # interactive
ccl setup --non-interactive        # CI-friendly
ccl setup --resume                 # resume after failure
ccl find-model                     # standalone llmfit query
ccl --version                      # print version
ccl --help                         # top-level help
```

## Diagnostics

```bash
ccl doctor                                       # wizard state + presence checks
python -m claude_codex_local.core profile        # JSON hardware + software profile
python -m claude_codex_local.core recommend      # llmfit model recommendation only
python -m claude_codex_local.core adapters       # list all engine adapters
```

The `core` debug subcommands were previously exposed as a separate `ccl-bridge` binary. They're now only reachable via `python -m` since they are internal JSON dumpers, not a user-facing tool.

## Testing

```bash
pytest                                           # all tests
pytest -m "not local"                            # skip tests needing real binaries (CI default)
pytest --cov=. --cov-report=term-missing         # with coverage
```

Tests requiring real binaries (ollama, lm-studio, claude, codex, llmfit) are marked `@pytest.mark.local` and auto-skipped in CI.

End-to-end smoke test (requires a real engine):

```bash
bash scripts/e2e_smoke.sh
```

## Linting and Type Checking

```bash
ruff check .          # lint
ruff check . --fix    # auto-fix safe issues
mypy .                # type check
bandit -r .           # security scan
```

All of these also run automatically via pre-commit on `git commit`.

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com). Hooks include:

- `ruff` — lint and format
- `mypy` — type checking
- `bandit` — security scanning
- `detect-secrets` — credential leak detection

Run all hooks manually:

```bash
pre-commit run --all-files
```

## Key Files

| File | Purpose |
|------|---------|
| `claude_codex_local/wizard.py` | Interactive setup wizard + `ccl` CLI |
| `claude_codex_local/core.py` | Machine profile, engine adapters, llmfit bindings, doctor |
| `scripts/e2e_smoke.sh` | End-to-end smoke test for `ccl` + `core` debug CLI |
| `~/.claude-codex-local/` | Runtime state (override with `CLAUDE_CODEX_LOCAL_STATE_DIR`) |

## Wizard State

The wizard persists progress to `.claude-codex-local/wizard-state.json`. Delete this file to reset and start fresh:

```bash
rm -rf .claude-codex-local/
```

## Debugging

Inspect the machine profile JSON:

```bash
python -m claude_codex_local.core profile | python3 -m json.tool
```

## Adding a New Engine

1. Add detection logic in `claude_codex_local/core.py` (engine `*_detect()` / `*_info()` helpers and the `RuntimeAdapter` list)
2. Add wiring logic in `claude_codex_local/wizard.py` (`_wire_engine()`)
3. Add a new helper script template in `claude_codex_local/wizard.py` (`_render_helper_script()`)
4. Add tests in `tests/`
5. Update `docs/ARCHITECTURE.md`
