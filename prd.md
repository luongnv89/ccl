# Product Requirements Document: claude-local / codex-local

| Field | Value |
|-------|-------|
| Product Name | claude-local / codex-local |
| Version | 1.2 |
| Last Updated | 2026-04-10 |
| Status | Draft |

---

## 1. Product Overview

### 1.1 Product Vision
Give Claude Code and Codex users a local backend mode that preserves their existing harness and workflow while swapping the underlying model backend to the best-fit local runtime/model available on their machine.

### 1.2 Target Users
- Claude Code power users who run into quota/rate/network/privacy constraints
- Codex users who want a local backend path without abandoning their existing CLI workflow
- Local-LLM tinkerers who want an easier coding-focused setup than manual runtime/model tuning
- Developers who want to keep official cloud tools as primary, but have a reliable local backup mode

### 1.3 Business Objectives
- Prove there is demand for a coding-focused local backend bridge, not just another generic LLM launcher
- Reduce time-to-first-local-session from “hours of setup” to “one setup command + one run command”
- Create a differentiated wedge around **workflow continuity** rather than raw model quality claims
- Build a strong CLI product that can later expand into richer local-agent or editor integrations

### 1.4 Core Product Principle
**Do not replace Claude Code or Codex. Keep the harness. Swap the backend.**

### 1.5 Naming
- **`claude-local`** for the Claude Code path
- **`codex-local`** for the Codex path

### 1.6 UX Constraint
There should be **no major change in how users use Claude Code or Codex**. The backend changes; the user workflow should stay as familiar as possible.

### 1.7 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Setup success rate | >80% | `setup` completion telemetry / logs |
| Time to first local session | <10 min | Start/end timing during setup flow |
| Local session success rate | >70% | Successful run completion without critical config errors |
| Runtime detection coverage | >90% for supported runtimes | Detection diagnostics |
| Model recommendation acceptance | >50% | User chooses top recommended model / config |
| Return-to-official reliability | >95% | Successful switch-back command runs |

---

## 2. User Personas

### Persona 1: Claude Code Power User
- **Demographics**: solo developer or indie hacker, uses terminal daily
- **Goals**: keep shipping even when Anthropic quota or connectivity becomes a problem
- **Pain Points**:
  - runs out of tokens at the worst time
  - local model setup is tedious and confusing
  - doesn’t want to lose existing Claude Code habits
- **User Journey**:
  - uses Claude Code normally
  - hits quota / wants local mode
  - runs setup once, then switches to local backend with one command
- **Quote**: “I don’t want a new tool. I want my current tool to keep working with a local brain.”

### Persona 2: Codex CLI User
- **Demographics**: terminal-native developer, comfortable with OpenAI tooling
- **Goals**: keep Codex-style workflow but use local models when needed
- **Pain Points**:
  - backend setup differs by runtime
  - model selection is unclear
  - local performance/quality tradeoffs are hard to reason about
- **User Journey**:
  - installs the tool
  - detects Ollama / LM Studio / llama.cpp
  - gets best-fit model recommendation via `llmfit`
  - runs local backend mode without touching official Codex config
- **Quote**: “If I already know Codex, I shouldn’t need to learn a second product to go local.”

### Persona 3: Privacy-Conscious Local LLM User
- **Demographics**: engineer or consultant handling sensitive code / working offline
- **Goals**: keep code local without giving up coding-agent ergonomics
- **Pain Points**:
  - generic local chat apps are not optimized for coding workflows
  - configuration across runtimes is fragmented
  - coding model choice is trial-and-error
- **User Journey**:
  - wants local-only mode from day one
  - uses setup + doctor commands to get an optimized coding profile
  - keeps cloud mode available but separate
- **Quote**: “I want local coding help that feels operational, not like a toy demo.”

---

## 3. Feature Requirements

### 3.1 Feature Matrix

| ID | Feature | Description | Priority | Acceptance Criteria | Dependencies |
|----|---------|-------------|----------|---------------------|--------------|
| F1 | Runtime detection | Detect Ollama, LM Studio, llama.cpp installations and health | Must-have | Detect installed/active runtimes with clear status | None |
| F2 | Machine profiling | Inspect CPU, RAM, GPU/VRAM, OS, available acceleration | Must-have | Build machine profile used by model scoring | F1 |
| F3 | `llmfit` scoring | Rank installed and candidate models for coding on this machine | Must-have | Return one best-fit recommendation per mode | F1, F2 |
| F4 | Setup command | One-command setup that configures local backend mode | Must-have | User can complete setup without manual file surgery | F1, F2, F3 |
| F5 | Local run command | Launch local backend mode for Claude/Codex-oriented usage | Must-have | User can start local session from CLI | F4 |
| F6 | Official/local switching | Explicitly switch between official and local backend mode | Must-have | Switch is reversible and does not break official setup | F4 |
| F7 | Separate config storage | Store tool state/config outside official Claude/Codex config | Must-have | No overwrite of official configs by default | F4 |
| F8 | Mode presets | Balanced / Fast / Quality selection | Should-have | Presets change recommendation + runtime params | F3 |
| F9 | Download recommendation | Suggest one model to install if no fit model exists | Should-have | User gets one opinionated install path | F3 |
| F10 | Doctor command | Diagnose missing runtime, model, or config issues | Should-have | Errors mapped to actionable fixes | F1, F2, F4 |
| F11 | Codex-oriented bridge mode | Codex-friendly local backend profile | Should-have | Codex users can run local path cleanly | F4 |
| F12 | Claude-oriented bridge mode | Claude Code-friendly local backend profile | Should-have | Claude users can run local path cleanly | F4 |
| F13 | Lightweight benchmark sanity check | Optional quick coding-oriented validation | Could-have | User can compare recommended models on a small eval | F3 |
| F14 | Additional runtime support | Add vLLM / other server runtimes later | Could-have | New adapter added without rewriting core flow | Core architecture |
| F15 | Interactive first-run wizard | Single guided flow from install → working local session | Must-have | 8-step wizard (discover → install missing → choose primary → pick model → smoke test → wire harness → verify → write guide.md) runs end-to-end | F1, F2, F3, F4 |
| F16 | Dependency installer | Sub-process installer for missing harness/engine/llmfit | Must-have | User can opt into installing any detected missing dependency without leaving the wizard | F15 |
| F17 | Disk space awareness | Detect free disk and gate model downloads against it | Must-have | Wizard refuses or warns when model size > free space and offers cleanup path | F2, F15 |
| F18 | Model download + mapping | Download the chosen model via selected engine, with cross-engine name mapping | Must-have | Model is fetchable through Ollama/LM Studio/llama.cpp with a normalized name | F1, F15 |
| F19 | User-specified model (default path) | User directly specifies the model they want to use | Must-have | Wizard accepts a direct model name and persists it | F15, F18 |
| F19b | Optional `find-model` helper | Opt-in llmfit-driven recommendation available both in-wizard and as a standalone subcommand | Must-have | Running `find-model` returns a ranked list of coding model candidates for this machine | F2, F3 |
| F20 | Generated `guide.md` | Personalized per-machine quickstart file | Must-have | `guide.md` written at end of setup with exact launch command, engine, model, harness | F15 |
| F21 | Single-command launcher | One command to start the chosen harness against the local model | Must-have | `claude-local` or `codex-local` launches harness + engine + model with zero extra flags | F15, F20 |

### 3.2 Feature Details

#### F1: Runtime detection

**Description**: Detect whether Ollama, LM Studio, and llama.cpp are installed and/or running, and normalize them into one internal adapter interface.

**User Stories**:
- As a user, I want the tool to find my local runtimes automatically so I don’t have to configure each one manually.

**Acceptance Criteria**:
- [ ] Tool detects whether Ollama is installed and reachable
- [ ] Tool detects whether LM Studio local server is available
- [ ] Tool detects whether llama.cpp server or compatible endpoint is available
- [ ] Tool prints a consistent status summary per runtime
- [ ] Detection failures explain what is missing and how to fix it

**Edge Cases**:
- runtime binary exists but service is not running
- multiple runtimes are available at once
- runtime exists but no compatible models are present

#### F2: Machine profiling

**Description**: Build a hardware profile to avoid recommending nonsense model setups.

**Acceptance Criteria**:
- [ ] Detect OS, architecture, CPU, RAM
- [ ] Detect GPU/VRAM or fallback to CPU-only mode
- [ ] Capture acceleration capability (Metal/CUDA where applicable)
- [ ] Output profile is usable by model scoring engine

#### F3: `llmfit` scoring

**Description**: Use `llmfit` to choose the best coding model for the user’s machine and preferred mode.

**Acceptance Criteria**:
- [ ] Returns one best-fit installed model for Balanced mode
- [ ] Returns alternative picks for Fast and Quality modes
- [ ] Recommends one download if no suitable installed model exists
- [ ] Explains why a model was chosen in short human-readable terms

#### F4: Setup command

**Description**: Configure local backend mode in one guided command.

**Acceptance Criteria**:
- [ ] `setup` runs detection + profiling + scoring
- [ ] Writes local tool config to a separate namespace
- [ ] Does not overwrite official Claude/Codex config by default
- [ ] Tests one local backend connection before completion

#### F5: Local run command

**Description**: Launch a local coding session while preserving the harness mental model.

**Acceptance Criteria**:
- [ ] User can run local mode with one command
- [ ] Tool shows which runtime/model/preset is active
- [ ] Failure states tell user how to recover or switch back

#### F6: Official/local switching

**Description**: Explicitly switch to local mode or back to official mode.

**Acceptance Criteria**:
- [ ] `use local` activates local backend profile
- [ ] `use official` restores official backend behavior
- [ ] Switching back does not require reinstalling anything

---

## 4. User Flows

### 4.1 First-time setup flow (interactive wizard)

**Description**: A user installs the tool and runs it for the first time. The tool launches an interactive 8-step wizard that takes them from a blank machine to a working single-command local coding session.

**Steps**:

1. **Install** — user installs the tool via package manager / script.
2. **First run** — user executes the tool; it detects that no config exists and enters the interactive wizard.
   - **2.1 Discover environment**
     - Detect installed harnesses: Claude Code, Codex
     - Detect installed engines: Ollama, LM Studio, llama.cpp
     - Detect `llmfit`
     - Detect free disk space
     - Verify user has at least: one harness (Claude Code or Codex) + one engine (Ollama / LM Studio / llama.cpp) + `llmfit`
   - **2.2 Install missing components**
     - If any required category is missing, prompt user to choose which one to install
     - Spawn a sub-process to install the chosen component
     - Re-run discovery after install
   - **2.3 Pick preferences**
     - If multiple harnesses are present, ask which one to use as primary (Claude or Codex, or both)
     - If multiple engines are present, ask which engine to prefer
     - Save primary selections to config; keep secondary ones as fallbacks
   - **2.4 Pick a model (user-first)**
     - The tool asks the user which model they want to use. Two paths:
       - **(default) Direct choice** — user types a model name (e.g. `qwen3-coder:30b` or `qwen/qwen3-coder-30b`). The tool maps the name into the selected engine's naming scheme if required.
       - **(opt-in) Find best model** — user chooses "help me pick" and the tool runs `llmfit` against the machine profile, presents a ranked list of candidates, and lets the user select one. `find-model` is also exposed as a standalone subcommand so users can run it any time, not just during setup.
     - Once a model is chosen, the download/disk branches apply identically to both paths:
       - if the model exists locally → skip to 2.5
       - if the model is missing and free disk space is sufficient → ask user to confirm download, then download
       - if the model is missing and free disk space is not sufficient → ask user whether to clean storage first, then download
       - if user cancels the download → re-ask for a different model, or exit cleanly
     - The final choice is persisted in config (including whether it came from direct input or `find-model`)
   - **2.5 Smoke test engine + model**
     - Run a minimal coding prompt against the selected engine + selected model
     - Fail fast with a clear error if the smoke test fails
   - **2.6 Wire up harness**
     - Write config so the selected harness can be launched against the selected engine + selected model via a single command
   - **2.7 Verify the launch command**
     - Actually run the single launch command end-to-end and confirm it succeeds
   - **2.8 Write `guide.md`**
     - Generate a personalized `guide.md` containing the exact launch command, harness, engine, model, and troubleshooting tips
3. **Daily use** — user runs the single configured command to start their harness against the local model.

```mermaid
flowchart TD
    A[Install tool] --> B[First run]
    B --> C[2.1 Discover harnesses, engines, llmfit, disk]
    C --> D{At least 1 harness + 1 engine + llmfit?}
    D -->|No| E[2.2 Ask user what to install + spawn installer]
    E --> C
    D -->|Yes| F[2.3 Pick primary harness + engine]
    F --> G[2.4 Ask user: which model?]
    G --> G1{User choice}
    G1 -->|Direct name| H
    G1 -->|Help me pick| G2[Run llmfit → show ranked list]
    G2 --> G3[User picks from list]
    G3 --> H{Model installed?}
    H -->|Yes| M[2.5 Smoke test engine + model]
    H -->|No| I{Fits in free disk?}
    I -->|Yes| J[Ask user to confirm download]
    I -->|No| K[Ask user to free space then download]
    J --> L{Download accepted?}
    K --> L
    L -->|Yes| M
    L -->|No| G
    L -->|Exit| P[Cancel setup with message]
    M --> Q[2.6 Wire up harness with single command]
    Q --> R[2.7 Verify launch command end to end]
    R --> S[2.8 Write personalized guide.md]
    S --> T[Setup complete — user runs daily command]
```

### 4.2 Switch-to-local flow

**Description**: A user already has the tool configured and wants to switch into local backend mode.

**Steps**:
1. User runs `use local` or `run --local`
2. Tool loads stored preferred runtime/model/profile
3. Tool confirms active backend and preset
4. Local session starts

### 4.3 Switch-back-to-official flow

**Description**: A user wants to return to official Claude/Codex models.

**Steps**:
1. User runs `use official`
2. Tool restores official backend settings/route
3. User continues with their normal cloud workflow

### 4.4 Doctor flow

**Description**: A user has a broken setup and wants guided diagnosis.

**Steps**:
1. User runs `doctor`
2. Tool checks runtime availability, model availability, config validity, and backend health
3. Tool prints concrete fixes in priority order

```mermaid
flowchart TD
    A[Run doctor] --> B[Check runtime status]
    B --> C[Check machine profile]
    C --> D[Check model availability]
    D --> E[Check config integrity]
    E --> F[Return prioritized fixes]
```

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Target | Notes |
|-------------|--------|-------|
| Setup completion | < 3 min excluding model downloads | Local checks should be quick |
| Runtime detection | < 10 sec | On a normal developer machine |
| Switching mode | < 5 sec | Should feel instant-ish |
| Doctor results | < 15 sec | Fast enough to be used often |
| Startup messaging | immediate | Always show active backend + model |

### 5.2 Reliability
- Setup should be resumable after partial failure
- Config should be recoverable if corrupted
- Switching back to official mode should be highly reliable
- No destructive modification of official tool config by default

### 5.3 Security
- No silent exfiltration of project data during local-mode setup
- Local config stored separately from official credentials
- Any optional telemetry must be opt-in
- Secrets/tokens handled minimally and never logged in plaintext

### 5.4 Compatibility

| Platform | Requirement |
|----------|-------------|
| macOS | Supported in MVP |
| Linux | Supported in MVP |
| Windows | Nice-to-have after MVP unless easy |
| Runtimes | Ollama, LM Studio, llama.cpp |
| Harnesses | Claude-oriented and Codex-oriented workflows |

### 5.5 UX constraints
- Product must never imply parity with frontier cloud models when local models are weaker
- Recommendations should be opinionated and minimal, not a giant model zoo menu
- Mode labels should be simple: Balanced / Fast / Quality

---

## 6. Technical Specifications

### 6.0 Implementation Language Decision

The tool is a CLI-first interactive wizard that must shell out to other processes (harnesses, engines, `llmfit`, package managers), probe hardware and disk, talk to local HTTP endpoints, run a TTY-driven flow, and write config files across macOS and Linux (Windows later).

| Language | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Bash** | Zero runtime deps, trivial shell-out | Weak at structured data, fragile interactive UX, poor hardware introspection, cross-platform quirks | ❌ Not suited for an 8-step interactive wizard |
| **Python** | Rich interactive CLI libs (`rich`, `questionary`, `textual`), mature hardware introspection (`psutil`), matches `llmfit` and existing `poc_bridge.py` in this repo, fastest iteration | Distribution requires venv/pipx, slower cold start | ✅ **Chosen for MVP** |
| **Node.js** | Excellent interactive CLI libs (ink, prompts, oclif), cross-platform | Adds a second ecosystem with no clear win over Python, weaker hardware introspection, not the language of `llmfit` | ⚠️ Reasonable alternative, not chosen |
| **Go** | Single static binary, fast, strong concurrency for parallel probes | Slower to iterate on an interactive wizard, no reuse of `llmfit` Python surface | 🔄 Best candidate for a **v2 rewrite** once the flow stabilizes and distribution-as-single-binary matters more than iteration speed |
| **Rust** | Single static binary, strong TUI (ratatui), safe | Highest iteration cost, overkill for a setup wizard | ❌ Overkill for MVP |

**Decision**: Build the MVP in **Python**, distributed via `pipx` / a thin install script, to preserve iteration speed and reuse the existing POC and `llmfit` integration. Re-evaluate a **Go** rewrite after the MVP to ship a single static binary with no runtime dependency.

### 6.1 High-Level Architecture

```mermaid
graph TB
    subgraph User CLI
        CLI[Main CLI]
    end
    subgraph Core Engine
        DETECT[Runtime Detection]
        PROFILE[Machine Profiler]
        SCORE[llmfit Scoring]
        CONFIG[Config Manager]
        SWITCH[Mode Switcher]
        DOCTOR[Doctor]
    end
    subgraph Runtime Adapters
        OLLAMA[Ollama Adapter]
        LMSTUDIO[LM Studio Adapter]
        LLAMACPP[llama.cpp Adapter]
    end
    subgraph Harness Layer
        CLAUDE[Claude-oriented backend bridge]
        CODEX[Codex-oriented backend bridge]
    end
    CLI --> DETECT
    CLI --> PROFILE
    CLI --> SCORE
    CLI --> CONFIG
    CLI --> SWITCH
    CLI --> DOCTOR
    DETECT --> OLLAMA
    DETECT --> LMSTUDIO
    DETECT --> LLAMACPP
    SCORE --> OLLAMA
    SCORE --> LMSTUDIO
    SCORE --> LLAMACPP
    SWITCH --> CLAUDE
    SWITCH --> CODEX
```

### 6.2 Core modules

| Module | Responsibility | Notes |
|--------|---------------|-------|
| Runtime detection | Discover runtimes and health | Adapter-based |
| Machine profiler | Capture hardware capabilities | Input to scoring |
| Model scoring | Call `llmfit` and rank candidates | Core differentiation |
| Config manager | Store local profiles separately | Must not clobber official setup |
| Mode switcher | Activate local or official path | Explicit and reversible |
| Doctor | Explain broken state and recovery | High leverage for DX |
| Harness bridges | Claude/Codex-oriented backend glue | Preserve user workflow |

### 6.3 Configuration strategy
- Store local tool config under a separate namespace, e.g. `~/.config/<tool>/`
- Keep track of:
  - preferred runtime
  - preferred model per mode
  - machine profile
  - last-known-good config
  - optional install recommendations
- Avoid mutating official Claude Code / Codex configs unless explicitly requested

### 6.4 Runtime adapters
Each supported runtime should implement a shared internal interface:
- `detect()`
- `healthcheck()`
- `list_models()`
- `run_test()`
- `recommend_params(mode)`

### 6.5 Modes
- **Balanced**: default; optimize for usable coding quality + acceptable speed
- **Fast**: prioritize responsiveness on weaker machines
- **Quality**: prioritize stronger model choice when hardware supports it

### 6.6 Proposed commands
- `claude-local` (Claude-oriented local backend path)
- `codex-local` (Codex-oriented local backend path)
- setup / status / doctor / use-local / use-official helpers may still exist behind the scenes

The visible product should minimize workflow change and keep the harness experience familiar.

---

## 7. Analytics & Monitoring

### 7.1 Key product metrics

| Category | Metric | Description | Target |
|----------|--------|-------------|--------|
| Activation | Setup success | Completed setup / started setup | >80% |
| Usability | Time to first local run | Setup start to first working local session | <10 min |
| Reliability | Doctor resolution rate | Issues resolved after doctor guidance | >60% |
| Adoption | Repeat local usage | Users coming back to local mode | >40% |
| Trust | Switch-back success | Official mode restored successfully | >95% |

### 7.2 Events to track (opt-in if telemetry exists)

| Event | Trigger | Properties |
|-------|---------|------------|
| setup_started | user runs setup | OS, runtime candidates |
| runtime_detected | runtime found | runtime type |
| model_scored | llmfit returns result | selected model, mode |
| local_run_started | local mode starts | runtime, model, preset |
| switch_to_official | user switches back | prior runtime/model |
| doctor_run | user runs doctor | failure category |

### 7.3 Observability
- Local structured logs for detection/setup/doctor flow
- Redact secrets/tokens from logs
- Provide user-visible verbose/debug modes for troubleshooting

---

## 8. Release Planning

### 8.1 MVP (v1.0)
**Goal**: ship a trustworthy local backend setup + switching flow with minimal user workflow change.

**Scope**:
- [ ] Detect Claude Code, Codex, Ollama, LM Studio, llama.cpp, `llmfit`, free disk
- [ ] Profile machine hardware
- [ ] Integrate `llmfit` and map model names into the selected engine
- [ ] Interactive 8-step first-run wizard (2.1 → 2.8)
- [ ] Sub-process installer for missing harness/engine/`llmfit`
- [ ] Disk-aware model download with user confirmation and cleanup prompt
- [ ] User-specified fallback model path
- [ ] Smoke test of engine + model
- [ ] Harness wire-up with single launch command
- [ ] Generated personalized `guide.md`
- [ ] Keep official configs untouched
- [ ] Explicit local/official switching
- [ ] Doctor command

**Out of scope for MVP**:
- full Windows polish
- vLLM support
- rich editor integrations
- heavy benchmarking suite

### 8.2 Version 1.1
- [ ] One-click recommended model download guidance
- [ ] Better error recovery / doctor UX
- [ ] Improved Claude/Codex bridge polish
- [ ] Smarter caching of last-good profile

### 8.3 Version 2.0
- [ ] vLLM or server-grade runtime support if justified
- [ ] lightweight coding benchmark / validation mode
- [ ] richer workflow integrations
- [ ] team/shared machine profile presets

---

## 9. Open Questions & Risks

### 9.1 Open Questions

| # | Question | Impact | Owner | Due |
|---|----------|--------|-------|-----|
| 1 | What exact CLI/product name should be used? | Medium | Product | Before MVP branding |
| 2 | What is the cleanest bridge strategy for Claude-oriented usage? | High | Engineering | Before implementation |
| 3 | How much Codex-specific behavior needs explicit handling vs generic backend switching? | High | Engineering | Before adapter work |
| 4 | Should model download be built-in or only recommended? | Medium | Product | Before setup finalization |
| 5 | What telemetry, if any, is acceptable for a privacy-sensitive audience? | Medium | Product | Before public release |

### 9.2 Assumptions

| # | Assumption | Risk if Wrong | Validation |
|---|------------|---------------|------------|
| 1 | Users care more about continuity than absolute parity | Product weaker if false | User interviews / dogfooding |
| 2 | `llmfit` can produce meaningfully better picks than static defaults | Differentiation weakens | Compare recommendation quality |
| 3 | Supported runtimes expose enough stable hooks for automation | Adapter complexity increases | Early prototype spikes |
| 4 | Users are okay with explicit switching instead of fully transparent magic | UX may feel clunky | Prototype testing |

### 9.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Local model quality disappoints | High | High | Honest positioning, recommended presets |
| Runtime differences create fragile adapter layer | Medium | High | Narrow MVP runtime set, adapter abstraction |
| Official harness integration is harder than expected | Medium | High | Bridge prototype early, keep fallback behavior explicit |
| Naming causes expectation trap | High | Medium | Avoid “offline Claude Code” framing |
| Users face config confusion | Medium | Medium | Strong setup/doctor UX |

---

## 10. Appendix

### 10.1 Competitive framing

| Product / Approach | What it does well | Gap vs this PRD |
|--------------------|-------------------|-----------------|
| Ollama / LM Studio / llama.cpp | Local runtime execution | No coding-focused fallback orchestration |
| Generic local wrappers | Launch local models | Weak workflow continuity |
| Model routers | Normalize backends | Often not coding/harness focused |
| Local coding tools | Provide coding UX | May replace workflow instead of preserving harness |

### 10.2 Positioning statement
**The product is not an offline Claude/Codex clone. It is a local backend bridge that keeps the existing harness while swapping the model engine.**

### 10.3 Naming statement
- `claude-local` = Claude Code with a local backend
- `codex-local` = Codex with a local backend
- The point is continuity, not replacement

### 10.4 Revision history

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.2 | 2026-04-10 | Olak | Added interactive 8-step first-run wizard (F15–F21), language decision (Python for MVP), disk-aware model download, personalized `guide.md`, single-command launcher |
| 1.1 | 2026-04-09 | Olak | Refined around `claude-local` / `codex-local` naming and no-workflow-change constraint |
| 1.0 | 2026-04-09 | Olak | Initial PRD draft |
