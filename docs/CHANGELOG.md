# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.10.0 — 2026-05-10

### Features

- **Non-interactive CLI**: Add `ccl run` subcommand with `-p/--prompt` flag for non-interactive sessions, enabling scripted and automated workflows (#70, #71)
- **vLLM Wizard Integration**: Wire vLLM into the setup wizard as a selectable backend option (#66)
- **9router Auto-install**: Wizard now offers to install 9router via npm when user selects it as their routing backend (#67)
- **Machine Profile Caching**: Cache machine specifications to avoid re-scanning hardware on subsequent setup runs, significantly improving wizard startup time (#58, #75)
- **llama.cpp Enhancements**: Upgrade to 128k context support, add reasoning model smoke test, implement `ccl serve` command, and enable automatic server restart on crashes (#60)

### Bug Fixes

- **Wizard Component Recheck**: Fix wizard to recheck selected setup components after user modifications, ensuring configuration consistency (#79, #81)
- **vLLM Detection**: vLLM detection now checks for CLI installation rather than just server reachability, preventing false positives (#78)
- **llama.cpp Model Matching**: Fix HuggingFace tag matching to use existing `_llamacpp_models_match` helper for consistent model resolution (#64)
- **Machine Profile Cache**: Write in-process machine profile cache to the correct symbol, fixing cache persistence issues (#77)

### Performance

- **Lazy llmfit Loading**: Optimize setup workflow with lazy llmfit initialization and cache-aware model picker, reducing unnecessary hardware scans (#79, #80)

### Tests

- **vLLM E2E Coverage**: Add comprehensive end-to-end test against a live vLLM server instance (#63)

### Documentation

- Refresh documentation and brand assets for improved clarity and visual consistency (#76)

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.9.0...v0.10.0

## v0.9.0 — 2026-05-05

### Features

- **9router Integration**: Add 9router as a cloud-routing backend provider (#51, #52)
  - Add Router9Adapter with smoke test support (@luongnv89)
  - Extend wizard with 9router setup flow and API key management (@luongnv89)
  - Support cc9/cx9 aliases for 9router alongside existing cc/cx aliases (@luongnv89)
  - Add fence-tag derivation and doctor checks for 9router (@luongnv89)
  - Implement key-file deferral for Claude and Codex 9router branches (@luongnv89)
  - Update wizard steps to handle 9router-specific configuration (@luongnv89)

### Bug Fixes

- Fix wizard to honor forced setup preferences (#51) (@luongnv89)
- Update DeepSeek model hub paths (@luongnv89)
- Fix step 2 install-hint loop to show 9router URL (#51) (@luongnv89)

### Documentation

- Document 9router engine and cc9/cx9 aliases in README and ARCHITECTURE.md (#51) (@luongnv89)
- Add 9router to primary_engine inline comments (#51) (@luongnv89)

### Tests

- Update e2e and vllm adapter registry assertions for 9router (#51) (@luongnv89)

### Refactoring

- Refactor wizard \_alias_block and \_write_helper_script to use 4-way dispatch (#51) (@luongnv89)
- Extend WireResult with raw_env field for deferred shell expressions (#51) (@luongnv89)

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.8.3...v0.9.0

## [0.8.3] - 2026-04-24

### Fixed

- Retired the qwen2.5-coder 0.5b verified path and removed related claims from the README, docs, model mapping, and static site (#49)
- Restored the bootstrap docs to point users to `ccl find-model` instead of a hardcoded tiny model download path (#49)

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
