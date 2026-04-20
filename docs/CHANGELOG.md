# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2026-04-20

### Fixed
- Wizard step IDs renumbered from the `2.x` scheme (`2.1`–`2.8`) to sequential integers (`1`–`8`), so progress indicators are consistent throughout the setup flow (#47)
- Documentation updated to reflect the new sequential step numbering (`1`–`11` across all wizard sections)
- E2e and unit tests updated to reference the new step IDs

## [0.8.1] - 2026-04-17

### Fixed
- Machine specifications table now shows real CPU, RAM, and GPU values — the wizard was reading `llmfit system --json` fields from the top level, but they are wrapped under a `system` key; Platform row now comes from `platform.system()` / `platform.machine()` since llmfit does not emit those keys (#46)
- llmfit ranking now uses **available** RAM instead of total — `llmfit fit --json` is invoked with `--ram <available_ram_gb>G` so the Speed/Balanced/Quality picks match what will actually fit on the host right now (#46)
- Embedding and reranker models are hidden from the installed-models picker for both Ollama and LM Studio — they cannot serve as chat coding models and were surfacing as confusing choices (e.g. `embeddinggemma:300m`, `nomic-embed-text:latest`) (#46)
- Step 4 (formerly 2.4) model picker is now grouped with visual separators — `Running server` / `Suggested by llmfit` / `Installed on this machine` / `Other` — so categories are visually distinct (#46)

## [0.8.0] - 2026-04-17

### Added
- vLLM backend adapter with unit and e2e test coverage — high-throughput inference engine now joins Ollama, LM Studio, and llama.cpp as a first-class engine option
- Wizard detects an already-running `llama-server` and offers its active model as a pick, so you can keep your warm process instead of re-pulling a GGUF
- Wizard pre-populates the model picker with models discovered on-host and recommendation profile picks, so the first press of Enter lands on a sensible default (#35, #36)
- Wizard welcome banner now shows the installed version and repository URL, so users know which build they are running and where to file issues (#37)
- Live progress for model downloads: `ollama pull`, `lms get`, and the Hugging Face CLI now stream their own progress bars (bytes, speed, ETA) straight to the terminal, and the wizard prints a post-download summary with the final size and elapsed time. Ctrl-C cleanly aborts an in-flight pull (#39)
- Fuzzy-search fallback for Hugging Face GGUF downloads: when a repo is not found, the wizard queries the Hub's search API, presents up to 3 closest matches as a numbered picker, and lets the user either pick one or re-enter a different name. When no similar models are found the wizard reports it and re-prompts for a new name (#38)

### Fixed
- Post-review polish for the fuzzy fallback and KI wizard flow (#45)
- vLLM adapter type annotations and lint warnings cleared under `mypy` and `ruff`
- Removed a stray agent worktree gitlink that broke CI on fresh clones

## [0.7.0] - 2026-04-12

### Added
- Machine specifications table (CPU cores/name, RAM total/available, GPU details) displayed during environment discovery step (#31)
- Comprehensive e2e test suite covering all `ccl` CLI commands: `setup`, `doctor`, `find-model`, and their flags — 26 tests total (#29, #32)

### Fixed
- `--resume` and `--non-interactive` flags are now available at the top-level `ccl` parser, so `ccl --resume` works without specifying the `setup` subcommand explicitly (#28, #30)

## [0.6.0] - 2026-04-11

### Added
- ASCII 3D welcome banner with project tagline displayed at wizard startup (#23, #25)

### Fixed
- HuggingFace CLI detection now checks both `hf` (modern) and `huggingface-cli` (legacy) binary names, uses the resolved binary in download commands, and injects the Python scripts directory into PATH immediately after pip install so the new binary is discoverable without reloading the shell (#21, #22)
- `llmfit` check made optional — environment discovery (Step 1, formerly 2.1) no longer gates on llmfit being installed; llmfit is now checked only on-demand when the user requests model selection help (#24, #26)

## [0.5.0] - 2026-04-11

### Changed
- **BREAKING:** Single canonical CLI binary. The package now installs one entry point, `ccl`, replacing the previous `claude-codex-local` and `ccl-bridge` commands. The command tree is unchanged (`ccl setup`, `ccl doctor`, `ccl find-model`) — only the binary name differs. Update any scripts, docs, or aliases that invoked the old names.
- **BREAKING:** Removed `ccl-bridge` entirely. Its debug subcommands (`profile`, `recommend`, `doctor`, `adapters`) were internal JSON dumpers, not user-facing tools. They are still reachable for debugging via `python -m claude_codex_local.core <cmd>`.
- **Internal rename:** `claude_codex_local/bridge.py` is now `claude_codex_local/core.py`. Anyone importing `claude_codex_local.bridge` directly must switch to `claude_codex_local.core`. The `core` module is the neutral home for the machine profile, engine adapters, and llmfit bindings — the old `bridge` name predated the package layout.
- **Removed legacy shims:** `bin/claude-codex-local` (bash wrapper) and the top-level `wizard.py` duplicate are deleted. Both predated the installable package and are no longer needed.
- `install.sh` now performs `pip install -e .` instead of installing raw `requirements.txt`, so the `ccl` entry point lands in the virtualenv automatically.
- `ccl --version` is now available at the top level. New global flags: `--no-color` (also honors the `NO_COLOR` env var), `--verbose`, `--quiet`.

### Migration

If you had the old binary on your shell:

```bash
# before
claude-codex-local setup --resume
ccl-bridge profile

# after
ccl setup --resume
python -m claude_codex_local.core profile
```

Reinstall the package to pick up the new entry point:

```bash
pip install --upgrade claude-codex-local      # PyPI
# or, from a clone:
pip install -e .
```

Your existing `~/.claude-codex-local/` state directory and the `cc` / `cx` shell aliases installed by a previous wizard run are unaffected.

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
