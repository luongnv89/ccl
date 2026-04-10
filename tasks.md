# Development Tasks

> Generated from: `prd.md` v1.2
> Generated on: 2026-04-10
> Implementation language: **Python** (MVP). Candidate rewrite in Go post-MVP.

## Overview

### Development Phases
- **POC**: Prove that a familiar Claude/Codex-style workflow can talk to a local backend without changing the user’s mental model.
- **MVP**: Support Ollama, LM Studio, and llama.cpp with machine profiling, `llmfit` scoring, separate config, mode presets, switching, and doctor flow.
- **Full Release**: Add download guidance, benchmark sanity checks, stronger adapter coverage, and optional richer integrations.

### Key Dependencies
- Runtime adapter contract must be settled before multi-runtime support scales.
- Machine profiling and `llmfit` scoring must exist before model recommendation/preset logic is trustworthy.
- Config separation must be implemented before local/offical switching can be considered safe.
- Claude/Codex bridge behavior must be proven early to avoid building the wrong abstraction.

## Dependencies Map

### Visual Dependency Graph

```text
[Task 1.1] ──┬───────────────> [Task 1.4] ─────────────┬──────────────> [Task 2.2] ───────────────┐
             │                                          │                                           │
[Task 1.2] ──┼──> [Task 1.3] ───────────────────────────┘                                           │
             │                                                                                      │
[Task 2.1] ──┴──────────────────────────────────────────────> [Task 2.3] ──┬──> [Task 3.1] ────────┤
                                                                            │                       │
[Task 2.4] ─────────────────────────────────────────────────────────────────┼──> [Task 3.2] ────────┤
                                                                            │                       │
[Task 2.5] ─────────────────────────────────────────────────────────────────┘                       │
                                                                                                    │
[Task 3.3] ──────────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                    │
[Task 3.4] ──────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                  │
                                                                                  ├──> [Task 4.1]
                                                                                  ├──> [Task 4.2]
                                                                                  └──> [Task 4.3]
```

### Dependency Table

| Task ID | Task Title | Depends On | Blocks | Can Parallel With |
|---------|------------|------------|--------|-------------------|
| 1.1 | Define adapter and bridge architecture | None | 1.4, 2.2 | 1.2, 2.1 |
| 1.2 | Implement machine profiling core | None | 1.3, 2.3 | 1.1, 2.1 |
| 1.3 | Integrate `llmfit` scoring | 1.2 | 2.3, 2.4 | 1.4 |
| 1.4 | Prove one end-to-end local backend path | 1.1 | 2.2, 3.1 | 1.3 |
| 2.1 | Build config/state manager with safe separation | None | 2.5, 3.2 | 1.1, 1.2 |
| 2.2 | Add Ollama / LM Studio / llama.cpp adapters | 1.1, 1.4 | 3.1, 3.3 | 2.3 |
| 2.3 | Implement model recommendation engine + presets | 1.2, 1.3 | 3.2, 4.1 | 2.2 |
| 2.4 | Create setup flow | 1.3 | 3.2, 3.4 | 2.5 |
| 2.5 | Implement official/local switching | 2.1 | 3.2, 3.4 | 2.4 |
| 3.1 | Ship Claude/Codex bridge polish | 2.2, 1.4 | 4.2 | 3.2, 3.3 |
| 3.2 | Build doctor + recovery UX | 2.3, 2.4, 2.5 | 4.3 | 3.1, 3.3 |
| 3.3 | Add compatibility + integration tests | 2.2 | 4.2, 4.3 | 3.1, 3.2 |
| 3.4 | Add docs and onboarding commands | 2.4, 2.5 | 4.3 | 3.1, 3.2 |
| 4.1 | Add recommended model download guidance | 2.3 | — | 4.2, 4.3 |
| 4.2 | Add benchmark sanity-check mode | 3.1, 3.3 | — | 4.1, 4.3 |
| 4.3 | Hardening, packaging, and release prep | 3.2, 3.3, 3.4 | — | 4.1, 4.2 |

### Parallel Execution Groups

**Wave 1** (Start immediately):
- [ ] Task 1.1: Define adapter and bridge architecture
- [ ] Task 1.2: Implement machine profiling core
- [ ] Task 2.1: Build config/state manager with safe separation

**Wave 2** (After core foundations):
- [ ] Task 1.3: Integrate `llmfit` scoring *(requires: 1.2)*
- [ ] Task 1.4: Prove one end-to-end local backend path *(requires: 1.1)*

**Wave 3** (After proof works):
- [ ] Task 2.2: Add Ollama / LM Studio / llama.cpp adapters *(requires: 1.1, 1.4)*
- [ ] Task 2.3: Implement model recommendation engine + presets *(requires: 1.2, 1.3)*
- [ ] Task 2.4: Create setup flow *(requires: 1.3)*
- [ ] Task 2.5: Implement official/local switching *(requires: 2.1)*

**Wave 4** (MVP completion):
- [ ] Task 3.1: Ship Claude/Codex bridge polish *(requires: 2.2, 1.4)*
- [ ] Task 3.2: Build doctor + recovery UX *(requires: 2.3, 2.4, 2.5)*
- [ ] Task 3.3: Add compatibility + integration tests *(requires: 2.2)*
- [ ] Task 3.4: Add docs and onboarding commands *(requires: 2.4, 2.5)*

**Wave 5** (Post-MVP enhancement):
- [ ] Task 4.1: Add recommended model download guidance *(requires: 2.3)*
- [ ] Task 4.2: Add benchmark sanity-check mode *(requires: 3.1, 3.3)*
- [ ] Task 4.3: Hardening, packaging, and release prep *(requires: 3.2, 3.3, 3.4)*

### Critical Path

```text
Task 1.1 → Task 1.4 → Task 2.2 → Task 3.1 → Task 4.2
```

**Critical Path Tasks**: 1.1 → 1.4 → 2.2 → 3.1 → 4.2
**Estimated Length**: 5 major task groups

> ⚠️ If the bridge abstraction is wrong early, the whole project drifts. That is the main bottleneck.

---

## Sprint 1: Proof of Concept (POC)

### Task 1.1: Define adapter and bridge architecture

**Description**: Design the internal contract for runtime adapters and the backend-bridge layer so the product can keep the Claude/Codex harness mental model without hard-coding to one runtime. This is the architectural bet everything else rests on.

**Acceptance Criteria**:
- [ ] Document a shared runtime adapter interface (`detect`, `healthcheck`, `list_models`, `run_test`, `recommend_params`)
- [ ] Define how Claude-oriented and Codex-oriented bridge layers talk to runtime adapters
- [ ] Define config boundaries so official Claude/Codex setups are not overwritten by default
- [ ] Identify what is shared vs tool-specific in the bridge layer

**Dependencies**: None

**PRD Reference**: 1.4 Core Product Principle, 6.1 High-Level Architecture, 6.4 Runtime adapters

---

### Task 1.2: Implement machine profiling core

**Description**: Build the machine capability detector so later model recommendations are based on hardware reality instead of guesswork.

**Acceptance Criteria**:
- [ ] Detect OS, architecture, CPU, and RAM
- [ ] Detect GPU/VRAM when available, otherwise classify as CPU-only
- [ ] Capture acceleration hints (Metal/CUDA where applicable)
- [ ] Emit a stable machine-profile object consumable by scoring logic

**Dependencies**: None

**PRD Reference**: F2, 6.2 Core modules, 6.3 Configuration strategy

---

### Task 1.3: Integrate `llmfit` scoring

**Description**: Connect `llmfit` so the product can produce one coding-focused recommendation instead of exposing users to model-zoo chaos.

**Acceptance Criteria**:
- [ ] `llmfit` is invoked using the machine profile
- [ ] Balanced, Fast, and Quality recommendation outputs are supported
- [ ] A human-readable explanation is returned for the top recommendation
- [ ] Failures fall back to a clear explanation rather than silent breakage

**Dependencies**: Task 1.2

**PRD Reference**: F3, 3.1 Feature Matrix, 6.5 Modes

---

### Task 1.4: Prove one end-to-end local backend path

**Description**: Build a minimal working proof that one harness-oriented path can talk to one local backend end to end without forcing the user into a totally new workflow.

**Acceptance Criteria**:
- [ ] One supported local runtime can be reached through the bridge
- [ ] One harness-oriented local run path succeeds end to end
- [ ] Active runtime/model are shown clearly to the user
- [ ] Failure mode shows recovery guidance instead of raw internal errors

**Dependencies**: Task 1.1

**PRD Reference**: F5, 4.1 First-time setup flow, 4.2 Switch-to-local flow

---

## Sprint 2: MVP Foundation

### Task 2.1: Build config/state manager with safe separation

**Description**: Implement local config storage so the tool can remember runtime/model preferences without mutating official Claude/Codex state.

**Acceptance Criteria**:
- [ ] Local config is stored under a separate namespace
- [ ] Last-known-good profile can be saved and reloaded
- [ ] Official config remains untouched by default
- [ ] Config corruption is detectable and recoverable

**Dependencies**: None

**PRD Reference**: F7, 5.2 Reliability, 6.3 Configuration strategy

---

### Task 2.2: Add Ollama / LM Studio / llama.cpp adapters

**Description**: Implement the three runtime adapters needed for MVP and normalize them behind the shared contract.

**Acceptance Criteria**:
- [ ] Ollama adapter supports detect, healthcheck, list models, and test run
- [ ] LM Studio adapter supports detect, healthcheck, list models, and test run
- [ ] llama.cpp adapter supports detect, healthcheck, list models, and test run
- [ ] Adapter outputs are normalized for scoring and setup flows

**Dependencies**: Task 1.1, Task 1.4

**PRD Reference**: F1, F14, 3.1 Feature Matrix, 6.4 Runtime adapters

---

### Task 2.3: Implement model recommendation engine + presets

**Description**: Convert `llmfit` output and runtime/model inventory into the actual recommendation engine users experience.

**Acceptance Criteria**:
- [ ] Balanced is the default recommendation mode
- [ ] Fast and Quality presets change ranking and suggested params
- [ ] If no good local model is installed, one opinionated recommendation is produced
- [ ] Recommendation output includes reason/explanation and active runtime fit

**Dependencies**: Task 1.2, Task 1.3

**PRD Reference**: F3, F8, F9, 6.5 Modes

---

### Task 2.4: Create interactive first-run wizard (8-step flow)

**Description**: Build the interactive first-run experience that takes the user from "just installed" to "working single-command local coding session" without manual surgery. This is the product's main surface area.

**Sub-tasks** (map 1:1 to PRD §4.1 steps 2.1–2.8):

- [ ] **2.1 Discover environment** — detect Claude Code, Codex, Ollama, LM Studio, llama.cpp, `llmfit`, and free disk space. Verify at least one harness + one engine + `llmfit` are present.
- [ ] **2.2 Install missing components** — interactively prompt for which missing component to install and spawn a sub-process installer. Re-run discovery after each install.
- [ ] **2.3 Pick preferences** — if multiple harnesses or engines are present, prompt for the primary. Persist the primary to config and keep secondaries as fallbacks. Allow enabling both Claude and Codex paths.
- [ ] **2.4 Pick a model (user-first)** — ask the user which model they want to use. Default path: accept a direct model name and map it into the selected engine's scheme. Opt-in path: run `llmfit` against the machine profile, show a ranked list, and let the user pick. Expose `find-model` as a standalone subcommand too. Regardless of path, handle the download branches: already installed / fits and confirmed / too big and cleanup path / cancelled → re-ask or exit.
- [ ] **2.5 Smoke test engine + model** — run a minimal coding prompt through the selected engine + model and fail fast on errors.
- [ ] **2.6 Wire up harness** — write config so the selected harness starts against the selected engine + model via a single command.
- [ ] **2.7 Verify launch command** — actually run the single launch command end-to-end and confirm success.
- [ ] **2.8 Generate `guide.md`** — write a personalized `guide.md` with the exact launch command, harness, engine, model, and troubleshooting tips.

**Acceptance Criteria**:
- [ ] The wizard is idempotent: re-running after a partial failure resumes cleanly
- [ ] Every prompt is skippable via non-interactive flags for scripting
- [ ] Config is written atomically and never touches official Claude/Codex config
- [ ] Final state: either a working single-command launcher + `guide.md`, or a cancelled setup with a clear explanation

**Dependencies**: Task 1.3, Task 2.1, Task 2.2, Task 2.3

**PRD Reference**: F4, F15, F16, F17, F18, F19, F20, F21, §4.1 First-time setup flow

---

### Task 2.5: Implement official/local switching

**Description**: Make switching explicit and reversible so users can move between local and official backends without setup drama.

**Acceptance Criteria**:
- [ ] Local mode can be activated explicitly
- [ ] Official mode can be restored explicitly
- [ ] Switching does not require reinstalling the harness
- [ ] State after switching is visible and understandable to the user

**Dependencies**: Task 2.1

**PRD Reference**: F6, 4.2 Switch-to-local flow, 4.3 Switch-back-to-official flow

---

## Sprint 3: MVP Completion

### Task 3.1: Ship Claude/Codex bridge polish

**Description**: Harden the actual bridge layer so both `claude-local` and `codex-local` feel coherent and preserve the harness mental model.

**Acceptance Criteria**:
- [ ] Claude-oriented path works cleanly with the local backend bridge
- [ ] Codex-oriented path works cleanly with the local backend bridge
- [ ] Active runtime/model/mode are surfaced consistently
- [ ] Known tool-specific quirks are handled or documented

**Dependencies**: Task 2.2, Task 1.4

**PRD Reference**: F11, F12, 1.6 UX Constraint, 6.2 Core modules

---

### Task 3.2: Build doctor + recovery UX

**Description**: Add a strong diagnosis flow so users can recover from broken runtime/model/config situations without rage-quitting.

**Acceptance Criteria**:
- [ ] Doctor checks runtime availability, model availability, config validity, and backend health
- [ ] Doctor returns prioritized recovery actions
- [ ] Common failure states map to clear fixes
- [ ] Recovery instructions are usable by non-expert developers

**Dependencies**: Task 2.3, Task 2.4, Task 2.5

**PRD Reference**: F10, 4.4 Doctor flow, 7.3 Observability

---

### Task 3.3: Add compatibility + integration tests

**Description**: Add enough automated coverage to trust the MVP instead of hand-waving it.

**Acceptance Criteria**:
- [ ] Adapter-level tests cover supported runtimes with mocks or fixtures
- [ ] Integration tests cover setup, local run, and switch-back flow
- [ ] Regression coverage exists for config separation rules
- [ ] CI can fail on broken core flows

**Dependencies**: Task 2.2

**PRD Reference**: 5.2 Reliability, 5.4 Compatibility, 8.1 MVP

---

### Task 3.4: Add docs and onboarding commands

**Description**: Make the first-run and recovery experience understandable enough that people can actually adopt the product.

**Acceptance Criteria**:
- [ ] README explains what the product is and is not
- [ ] Setup / status / doctor / switching commands are documented
- [ ] Positioning avoids “offline Claude Code” parity claims
- [ ] Quickstart path gets a user to a local run fast

**Dependencies**: Task 2.4, Task 2.5

**PRD Reference**: 1.1 Product Vision, 1.4 Core Product Principle, 10.2 Positioning statement

---

## Sprint 4: Feature Enhancement

### Task 4.1: Add recommended model download guidance

**Description**: Improve setup by recommending one install path when no suitable local coding model exists.

**Acceptance Criteria**:
- [ ] Tool recommends one best-fit model download when necessary
- [ ] Recommendation varies by hardware profile and preset mode
- [ ] Download guidance does not overwhelm the user with too many options
- [ ] Failure to download/install leaves the setup resumable

**Dependencies**: Task 2.3

**PRD Reference**: F9, 8.2 Version 1.1

---

### Task 4.2: Add benchmark sanity-check mode

**Description**: Add a lightweight validation mode so users can test whether the recommended local model is actually good enough for coding on their machine.

**Acceptance Criteria**:
- [ ] User can run a lightweight coding-oriented sanity check
- [ ] Results can compare recommended model/preset combinations
- [ ] Benchmark mode is optional and not required for setup
- [ ] Output helps users tune Fast vs Balanced vs Quality decisions

**Dependencies**: Task 3.1, Task 3.3

**PRD Reference**: F13, 8.3 Version 2.0

---

### Task 4.3: Hardening, packaging, and release prep

**Description**: Prepare the product to be shared beyond internal use.

**Acceptance Criteria**:
- [ ] Packaging/install story is defined and reproducible
- [ ] Logging and debug modes are stable enough for external users
- [ ] Error handling is reviewed across setup/run/doctor/switch flows
- [ ] Release checklist and versioning plan exist

**Dependencies**: Task 3.2, Task 3.3, Task 3.4

**PRD Reference**: 5.2 Reliability, 7.3 Observability, 8 Release Planning

---

## Backlog: Future Iterations

### vLLM / additional runtime support
- Add server-grade runtime support after MVP if justified
- PRD reference: F14, 8.3 Version 2.0

### Team/shared machine presets
- Allow reusable hardware/runtime profiles across teams or machines
- PRD reference: 8.3 Version 2.0

### Richer editor / workflow integrations
- Expand beyond CLI once the backend-bridge core is stable
- PRD reference: 1.3 Business Objectives, 8.3 Version 2.0

### Transparent auto-fallback mode
- Consider optional “switch automatically when cloud fails” later, only after explicit/manual mode is solid
- PRD reference: 9.2 Assumptions

---

## Technical Decisions

### Implementation language: Python (MVP)

The MVP is built in **Python**, distributed via `pipx` or a thin install script. Rationale:

- Matches the existing `poc_bridge.py` and `llmfit` ecosystem in this repo.
- Best-in-class interactive-CLI libraries for the 8-step wizard (`rich`, `questionary`, `textual`).
- Mature hardware introspection via `psutil`.
- Fastest iteration path while the flow and adapters are still being proven.

Alternatives considered:

- **Bash** — rejected; too fragile for a structured interactive wizard.
- **Node.js** — viable, but adds a second ecosystem with no clear advantage over Python for this workload.
- **Go** — best candidate for a **v2 rewrite** once the flow stabilizes and single-binary distribution becomes more valuable than iteration speed.
- **Rust** — overkill for a setup wizard at MVP stage.

## Ambiguous Requirements

> The following items from the PRD still need clarification:

| Requirement | What Needs Clarification |
|-------------|-------------------------|
| `claude-local` / `codex-local` naming | Should these be standalone CLIs, symlinked wrappers, aliases, or subcommands? |
| Claude-oriented bridge strategy | What exact integration mechanism is acceptable without breaking official behavior? |
| Codex-oriented bridge strategy | How much native Codex backend configurability can be reused vs wrapped? |
| Download behavior | Should model install be integrated, semi-automated, or just recommended? |
| Telemetry | Is any opt-in telemetry acceptable, or should all analytics be local-only? |

## Technical Notes

- The real make-or-break is the bridge abstraction, not the pretty setup UX.
- Preserve official-tool trust by never silently mutating official configs.
- Start with one runtime + one harness proof before getting fancy across all adapters.
- Balanced should be the default preset unless evidence proves otherwise.
