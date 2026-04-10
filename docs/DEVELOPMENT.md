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
pip install -r requirements-dev.txt

pre-commit install                 # set up git hooks
```

## Running the Wizard

```bash
./bin/claude-codex-local                          # interactive
./bin/claude-codex-local setup --non-interactive  # CI-friendly
./bin/claude-codex-local setup --resume           # resume after failure
./bin/claude-codex-local find-model               # standalone llmfit query
```

## Diagnostics

```bash
./bin/poc-machine-profile   # JSON hardware + software profile
./bin/poc-doctor            # wizard state + presence checks
./bin/poc-recommend         # llmfit model recommendation only
```

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
| `wizard.py` | Interactive setup wizard (core logic) |
| `poc_bridge.py` | Machine profile, model recommendation, doctor |
| `bin/claude-codex-local` | Main wizard entrypoint |
| `bin/poc-doctor` | Diagnostic: wizard state |
| `bin/poc-machine-profile` | Diagnostic: hardware profile |
| `bin/poc-recommend` | Diagnostic: model recommendation |
| `scripts/e2e_smoke.sh` | End-to-end smoke test |
| `.claude-codex-local/` | Runtime state (gitignored) |

## Wizard State

The wizard persists progress to `.claude-codex-local/wizard-state.json`. Delete this file to reset and start fresh:

```bash
rm -rf .claude-codex-local/
```

## Debugging

Run with verbose output:

```bash
PYTHONPATH=. python wizard.py --debug
```

Inspect the machine profile JSON:

```bash
./bin/poc-machine-profile | python3 -m json.tool
```

## Adding a New Engine

1. Add detection logic in `poc_bridge.py` (`_detect_engines()`)
2. Add wiring logic in `wizard.py` (`_wire_engine()`)
3. Add a new helper script template in `wizard.py` (`_render_helper_script()`)
4. Add tests in `tests/`
5. Update `docs/ARCHITECTURE.md`
