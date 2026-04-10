# Contributing to claude-codex-local

Thank you for your interest in contributing! This guide covers everything you need to get started.

## How to Contribute

- **Bug reports** — open an issue using the Bug Report template
- **Feature requests** — open an issue using the Feature Request template
- **Code contributions** — fork, branch, implement, test, PR
- **Documentation** — improvements to any `docs/` file or the README are always welcome

## Development Setup

**Prerequisites:** Python 3.10+, git

```bash
git clone https://github.com/luongnv89/claude-codex-local.git
cd claude-codex-local

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install
```

Run the wizard locally:

```bash
./bin/claude-codex-local
```

## Branching Strategy

- Branch from `main`
- Name feature branches: `feat/<short-description>`
- Name bug-fix branches: `fix/<short-description>`
- Name documentation branches: `docs/<short-description>`

```bash
git checkout -b feat/my-new-feature
```

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(scope): add something new
fix(scope): correct something broken
docs(scope): update documentation
refactor(scope): restructure without behaviour change
test(scope): add or update tests
chore(scope): maintenance tasks
```

Examples from this repo:

```
feat(install): add one-command remote installer
fix(wizard): add -- separator to Ollama Claude helper
docs(wizard): remind users to source shell rc before first run
```

## Pull Request Process

1. Ensure all tests pass: `pytest`
2. Ensure linting is clean: `ruff check . && mypy .`
3. Update `docs/CHANGELOG.md` under `[Unreleased]`
4. Open a PR against `main` using the PR template
5. A maintainer will review within a few days

## Coding Standards

- **Style:** enforced by `ruff` (line length 100, see `pyproject.toml`)
- **Types:** partial — `mypy` runs but `check_untyped_defs = false`
- **Security:** `bandit` and `detect-secrets` run in pre-commit hooks
- **Isolation rule:** the wizard must never write to `~/.claude` or `~/.codex` directly; all state goes under `.claude-codex-local/`

## Testing Requirements

```bash
pytest                      # run all tests
pytest -m "not local"       # skip tests that need real binaries (CI default)
pytest --cov=. --cov-report=term-missing   # with coverage
```

Tests that exercise real binaries (ollama, lm-studio, claude, codex, llmfit) are marked `@pytest.mark.local` and are skipped automatically in CI.

## Questions?

Open a [Discussion](https://github.com/luongnv89/claude-codex-local/discussions) or file an issue — we're happy to help.
