# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
