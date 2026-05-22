# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.15.0] — 2026-05-22

### Added

- **Auto-fetch available models during remote engine model selection** (#134): when the wizard's step 4 model picker runs against a remote engine endpoint (Ollama, llama.cpp, or vLLM), it now calls the remote API to list available models instead of showing a local-only picker. This removes the "model not found" guesswork for remote setups — you see exactly what the remote server has installed.
- **Smart remote endpoint URL scheme detection** (#134): if the user enters a bare IP or hostname (e.g. `192.168.1.100:11434`) during the local-vs-remote wizard prompt, the step now auto-prepends `http://` and the engine's default port if missing, so typos like `gpu-box.local` or `192.168.1.100:8000` produce a valid URL instead of a confusing connection error.
- **Test coverage for the remote model-fetch and URL normalization** (#134): new unit tests cover the auto-fetch path (`probe_remote_models`), the URL-scheme normalizer (`_normalize_url`), the `VLLM_BASE_URL` env-key extraction for remote vLLM, and the error-handling boundaries (connection refused, 404, JSON parse failure).
- **Interactive local-vs-remote prompt during engine selection** (#122): the wizard now asks whether the chosen engine (`ollama`, `llamacpp`, or `vllm`) is local on this machine or a remote endpoint. Selecting remote prompts for the base URL (and, for `vllm`, an API key), stores the value in the engine's `*_BASE_URL` env var inside the helper script, and skips the local install/launch path entirely. Local selections behave exactly as before.
- **Test coverage for the interactive remote-engine wizard path** (#125): new unit and integration tests exercise the local-vs-remote prompt, env-keyfile materialization with `chmod 0600`, and the remote branching in healthcheck, info, and `start_server` for `llamacpp`.

### Fixed

- **`llamacpp` remote-mode branching** (#123): the llama.cpp helper script, healthcheck, `info`, and `start_server` no longer assume a local `llama-server` binary when `LLAMACPP_BASE_URL` points at a remote endpoint. Remote endpoints now skip binary discovery, model-file checks, and the spawn path; healthcheck targets the remote URL directly.

### Changed

- **`llamacpp-tuner` skill no-ops cleanly when llamacpp is remote** (#124): the tuner detects a remote `LLAMACPP_BASE_URL` and exits with a friendly message rather than attempting to introspect a non-existent local binary or rewrite a helper script it does not own.
- **README and wizard walkthrough lead with the interactive remote-engine flow** (#126): the quickstart and wizard documentation now show the local-vs-remote prompt and remote-endpoint setup as the primary path, with the all-local install positioned as one branch of that prompt.

### Bug Fixes

- **Rename Pi local shortcut `cp` → `ccp`** (#120): the wizard-installed `cp` alias shadowed the standard POSIX copy command, so running `cp source dest` in a shell with CCL installed launched a Pi session instead of copying files. The Pi local helper script and short alias are now `ccp` (long alias `pi-local` is unchanged because it never collided). On the next `ccl setup` run, an existing pre-#120 install is migrated automatically: the legacy `alias cp=` line is rewritten to `alias ccp=`, the orphaned `.claude-codex-local/bin/cp` binary is removed, and the wizard prints a warning explaining the change. `ccl status` still detects pre-migration installs and shows them on the Pi row so users know to re-run setup. Router-suffixed variants `cp9` (9router) and `cpo` (OpenRouter) are not affected because they do not collide with any system command.

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.14.0...v0.15.0

## v0.14.0 — 2026-05-20

### Features

- **MTP (Multi-Token Prediction) support for llama.cpp** (#102, #103): `llama-server` is now auto-launched with `--spec-type draft-mtp --spec-draft-n-max 5` whenever the chosen model is an MTP variant. Detection runs in two passes:
  1. **GGUF metadata probe** — read the file's header KV table for an architecture-specific `*.mtp.*` key, a key ending in `.nextn_predict_layers` (the cross-arch convention used by Qwen, GLM, and DeepSeek MTP releases), or `MTP` in `general.name` / `general.architecture`. Large scalar arrays (tokenizer vocabularies, token-type tables) are skipped with a single seek so the KV walk stays aligned past them.
  2. **Filename fallback** — match `*mtp*` (word-bounded, case-insensitive) on the basename when the probe is inconclusive.
  Two env vars override the decision: `LLAMACPP_MTP_ENABLED=0/1` forces off/on, and `LLAMACPP_SPEC_DRAFT_N_MAX=N` overrides the default 5 (valid range 1–16). The detector also recognizes flag combinations that llama.cpp does not yet pair with `--spec-type draft-mtp` (`--mmproj`, `-np`/`--parallel > 1`) and disables MTP with a warning when those flags are supplied. `llamacpp_start_server()` and `build_llamacpp_server_args()` now accept an `extra_argv` kwarg that flows into the conflict guard *and* lands at the tail of the spawned argv, so any caller layering in additional `llama-server` flags exercises the guard at runtime. An out-of-range or non-integer `LLAMACPP_SPEC_DRAFT_N_MAX` is surfaced as a `notes` entry on the MTP result (and as a wizard warning) instead of silently reverting to the default. Reference: <https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF>.
- **`llamacpp-tuner` skill** for coding-agent performance tuning: new Claude Code skill that helps optimize llama.cpp server configuration for coding-agent workloads. Includes benchmark agent and configuration profiles.

### Bug Fixes

- Resolve mypy and ruff failures from v0.13.1 CI.
- Apply ruff format and rename ambiguous `l` identifier in `bench_agent.py`.

### Chore

- Level up `llamacpp-tuner` skill to A grade.

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.13.1...v0.14.0

## v0.13.1 — 2026-05-19

### Features

- **`ccl status` command** (#98, #101): new top-level subcommand that prints the current ccl setup and shortcut availability at a glance. Lists all 9 shortcuts (3 harnesses × 3 engine types) with their aliases, selected model, engine name and live status, and an availability column (`available` / `unavailable` / `unconfigured`); follows with an overall setup summary (engines detected, engines running, default harness, default engine, selected model). Engine health is checked via each adapter's healthcheck.

### Bug Fixes

- **`ccl status` internal consistency**: each shortcut row now reflects its own helper script independently instead of projecting one harness's engine across the others, and the availability column is honest about whether the alias would actually run today.
  - Per-harness inference: `cc` / `cx` / `cp` read from their own script (or matching wizard state), so `cx` no longer inherits llamacpp from `cc`.
  - Local availability requires the helper script to exist **and** the engine to be installed; a llamacpp helper with no `llama-server` is now `unavailable` instead of a misleading `available`.
  - Router shortcuts (`cc9` / `cx9` / `cp9`, `cco` / `cxo` / `cpo`) require the helper script to exist before claiming `available`; bare API-key detection without a wired-up alias now reports `unconfigured`.
  - `Default harness/engine/model` only show wizard-state values; we no longer fabricate them from a single detected script, which contradicted `Engines detected` when the inferred engine wasn't installed.
  - Replace the buggy `fence_tag.replace("9","").replace("o","")` (which turned `codex` into `cdex`) with explicit lookup tables; fix `_infer_engine_from_script`'s return annotation to admit `None`.

### Tests

- Add 18 unit tests across `TestRunStatus`, `TestInferEngineFromScript`, and `TestDetectExistingShortcuts` pinning each cross-row / cross-summary inconsistency and the helper-inference regexes (llamacpp via `--model` / `ANTHROPIC_CUSTOM_MODEL_OPTION`, ollama via `pi --provider ccl-ollama`).

### Chore

- Ignore `.gstack/` workspace directory.

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.13.0...v0.13.1

## v0.13.0 — 2026-05-19

### Features

- **Cross-harness session bridge**: `ccl run` now auto-captures and auto-injects conversation context across Claude Code, Codex, and Pi (#62, #93)
  - Post-run capture (both interactive and one-shot): after the harness exits, CCL reads its native session file for `$PWD` (`~/.claude/projects/...`, `~/.codex/sessions/...`, `~/.pi/agent/sessions/...`) and imports cleaned messages into `~/.claude-codex-local/sessions/<harness>.jsonl`
  - Pre-run injection (one-shot `-p` only): the freshest *other* harness's transcript for `$PWD` is rendered as a `[prior context, agent=…]` block and prepended to the prompt
  - New `session_adapters.py` with per-harness JSONL readers; drops AGENTS.md/CLAUDE.md re-dumps, slash-command echoes, tool calls, reasoning traces, and other harness internals
  - Bridge is **cwd-scoped** (no context leaks across repos), capped at **7 days** staleness, with an injection banner that shows source age (`last activity 6m ago`)
  - macOS symlink-aware: `/var/folders/...` ↔ `/private/var/folders/...` resolved consistently when matching cwd against the harnesses' stored paths
  - Opt out per-call with `ccl run --no-context`, or globally with `CCL_SESSION_BRIDGE=0`
  - `ccl session` command group (`list` / `show` / `sync --from A --to B` / `truncate --keep N` / `clear`) for review and manual operations on the imported store
  - Best-effort redaction of common token shapes (OpenAI, AWS, GitHub PAT/OAuth, Slack, GitLab, Google API) on every import and sync
  - Idempotent: a content-hash dedup key skips already-imported messages, so re-runs are safe
  - Same-harness one-shot continuity (`cc -p` then `cc -p`) is *not* covered — use the harness's own `--resume` / `--continue` for that
  - State directory overridable via `CLAUDE_CODEX_LOCAL_STATE_DIR`; native-home base overridable via `CCL_NATIVE_HOME_OVERRIDE`
- **`ccl run --native-params` passthrough flag** (#97, #99): forward everything after `--native-params --` verbatim to the launched harness, as a generic escape hatch for options ccl does not wrap first-class (e.g. Claude Code's `--dangerously-skip-permissions`). Argv is sliced before argparse runs; first-occurrence boundary only, does not interpret option-consuming flags.
- **Wizard `llmfit` fallback on deferred scan** (#95, #100): when step 1 auto-detection defers the hardware scan (the `LLMFIT_SKIPPED` sentinel), opportunistically invoke `llmfit` and render its result in place of `(scan deferred)`. Result is persisted to the machine-profile cache so the next wizard run benefits from it; falls back silently to the deferred-scan message if `llmfit` is missing or errors.

### Documentation

- README: rewrite `## Sharing Context Between Agents` to describe the auto-bridge, scope guards, and the interactive-capture / one-shot-inject asymmetry
- README: use theme-aware logo for dark/light mode rendering on GitHub
- README: add PyPI download badges and experiment tag
- docs.html: update `ccl run` card with `--no-context` and the auto-bridge note; refresh the `ccl session` card to describe inspection rather than manual seeding

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.12.0...v0.13.0

## v0.12.0 — 2026-05-16

### Features

- **OpenRouter Integration**: Add OpenRouter as a hosted-SaaS cloud-routing backend alongside 9router (#83)
  - New `openrouter` engine with `OpenRouterAdapter` mirroring the 9router shape
  - Helper scripts `cco` / `cxo` / `cpo` (Claude / Codex / Pi via OpenRouter)
  - Default model `anthropic/claude-sonnet-4.6`; override via `CCL_OPENROUTER_MODEL`
  - Deferred-secret API key storage (chmod 0600) reused from the 9router pattern
  - Smoke test sends a minimal request to the selected OpenRouter model; verify still uses `/models` only to avoid an extra chat call
  - Doctor checks for OpenRouter key file mode, content, and model name validity

### Bug Fixes

- **OpenRouter smoke test**: Smoke test now targets the selected OpenRouter model (#85)

### Documentation

- Update landing page release history for recent versions

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.11.0...v0.12.0

## v0.11.0 — 2026-05-16

### Features

- **Pi Harness Support**: Add Pi as a supported harness alongside Claude Code and Codex CLI, enabling model-agnostic terminal coding workflows (#59, #82)
  - Wire Pi into the wizard setup flow with dedicated configuration
  - Add `cp` alias for Pi + local model sessions
  - Support Pi-specific `models.json` configuration
  - Update documentation and guide generation for Pi workflows

**Full Changelog**: https://github.com/luongnv89/ccl/compare/v0.10.0...v0.11.0

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
