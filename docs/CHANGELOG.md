# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-11

### Added
- Smoke test now measures and reports model throughput in tokens/second for Ollama, LM Studio, and llama.cpp engines, with slow/acceptable/fast guidance and an interactive prompt to re-pick slow models (#18)
- Per-harness shell alias fences in `~/.zshrc` / `~/.bashrc` so `cc` (Claude Code) and `cx` (Codex) aliases coexist after setting up both harnesses, plus one-shot migration of legacy unified blocks (#19)
- Changelog section on the GitHub Pages landing page and content refresh for v0.3.0 features (#15)

## [0.3.0] - 2026-04-11

### Added
- llama.cpp backend adapter (`llamacpp` engine support) with `llama-server` integration (#10)
- Docker-based e2e test suite covering pip, uv, source, and extras install scenarios (#12)
- `pip install .[dev]` optional extras group (pytest, ruff, mypy, bandit, detect-secrets, pre-commit)
- GitHub Pages landing page with brand refresh and two-column hero layout

### Fixed
- Empty array expansion in `run_e2e_docker.sh` under `set -u` (source/extras scenarios)

## [0.2.0] - 2026-04-10

### Added
- One-command remote installer (`install.sh`) — no clone required
- `ollama launch` integration as primary engine path
- Shell alias installer with idempotent fenced block in `~/.zshrc` / `~/.bashrc`
- Personalized `guide.md` generation after wizard completes
- `--resume` flag to pick up after a failed wizard step
- `--non-interactive` flag for CI-friendly setup
- `find-model` subcommand for standalone `llmfit` recommendations
- Diagnostic helpers: `poc-doctor`, `poc-machine-profile`, `poc-recommend`
- `--` separator in Ollama Claude helper for correct arg forwarding
- Installable Python package structure for PyPI distribution

### Changed
- Wizard now uses `ollama launch` instead of isolated HOME and variant builder
- LM Studio support moved to secondary/fallback path

### Fixed
- Shell alias block replaced idempotently on re-run (no more duplicates)
- Users reminded to `source ~/.zshrc` before first `cc`/`cx` run

## [0.1.0] - 2026-04-01

### Added
- Initial proof-of-concept: interactive wizard (8 steps)
- Harness support: Claude Code, Codex CLI
- Engine support: Ollama, LM Studio, llama.cpp
- `llmfit` integration for hardware-aware model selection
- Machine profile and model recommendation diagnostics
- Pre-commit hooks: ruff, mypy, bandit, detect-secrets
- pytest test suite with `@pytest.mark.local` marker for integration tests
