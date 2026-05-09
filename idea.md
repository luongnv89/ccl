# Idea: Local Claude/Codex Fallback

## Original Concept

A local backend layer for Claude Code and Codex that lets users keep using the same harness, while swapping the model backend to local runtimes when needed.

Core behavior:

- auto-detect installed local runtimes
- start with **Ollama**, **LM Studio**, and **llama.cpp** support
- use **llmfit** to detect the best-fit model for the current machine
- default to a balanced speed/quality mode, with user-selectable faster or higher-quality modes
- expose product names as **`claude-local`** and **`codex-local`**
- preserve the normal Claude Code / Codex usage style with **no workflow change for the user**
- do **not** overwrite or break the official Anthropic / OpenAI setups
- make it easy to switch back to official cloud models at any time

## Clarified Understanding

This should not be positioned as “offline Claude Code” because that implies parity with Anthropic’s official product and creates bad expectations.

The cleaner framing is:

**A local backend bridge for Claude Code and Codex users**.

Key product rule:

- users should not have to change how they normally use Claude Code or Codex
- the backend becomes local, but the harness mental model and usage style stay the same

Naming direction:

- **`claude-local`** for Claude Code users
- **`codex-local`** for Codex users

When the user runs out of Anthropic/OpenAI quota, or wants to work fully local/offline, the tool should:

1. detect available local inference runtimes
2. inspect the machine profile
3. use `llmfit` to identify the best-fit coding model
4. connect the existing Claude/Codex harness style to a local backend
5. keep official Claude Code/Codex untouched and easy to return to

## Target Audience

- Claude Code power users who hit usage limits
- Codex users who want a local fallback path
- developers who already have local model runtimes installed but do not want to manually tune everything
- privacy-conscious or offline-first developers who want local coding help without replacing their main cloud workflow

## Goals & Objectives

### Primary goal

Provide a **one-command fallback** from official cloud coding agents to a sane local coding setup.

### User promise

- "when official tokens run out, you still have a local coding copilot"
- "one command to go local, one command to come back"

### Success criteria

In 6-12 months, the product should:

- reliably detect local runtimes and usable coding models
- recommend or auto-select the best-fit local coding model for a machine
- support at least Claude-oriented and Codex-oriented fallback workflows
- become a trusted emergency/local mode for developers, not a novelty wrapper

## Technical Context

- Stack: likely CLI-first, with a runtime adapter layer and model-scoring layer
- Timeline: MVP should be possible in 2-4 weeks if tightly scoped
- Budget: bootstrapped / solo-friendly
- Constraints:
  - must not break official Claude Code config
  - should avoid pretending local models are equal to frontier cloud models
  - needs good machine/runtime detection or the UX falls apart

## Product Shape

### Best framing

- not “offline Claude Code installation”
- instead: **local backend for Claude Code / Codex**

### Naming

- **`claude-local`**
- **`codex-local`**

### Best UX shape

Preferred rule:

- keep usage as close as possible to normal Claude Code / Codex behavior
- avoid making users learn a second workflow unless absolutely necessary

Setup/status/doctor commands can still exist behind the scenes, but the user-facing experience should preserve the existing harness style.

### Modes

- balanced (default)
- fast
- quality

### Runtime detection

First support:

- Ollama
- LM Studio
- llama.cpp

Potential later support:

- vLLM

### Model selection

Use `llmfit` to:

- inspect hardware/resources
- inspect local runtimes
- rank already-installed models
- recommend one download if no suitable coding model exists

### Configuration rules

- official Claude Code config remains untouched
- local fallback config is stored separately
- switching between official and local should be explicit and reversible

## First-Run User Flow (v1.2)

The tool is built around a single interactive first-run experience that takes a user from "just installed" to "working local coding session" without manual surgery.

1. **Install** — user installs the tool via a single package command.
2. **First run** — user runs the tool for the first time and is dropped into a guided, interactive flow:
3. **Discover environment** — detect installed harnesses (Claude Code, Codex), engines (Ollama, LM Studio, llama.cpp), `llmfit`, and remaining disk space. Verify that the user has at least one harness, one engine, and `llmfit`.
4. **Install missing pieces** — if any required component is missing, interactively ask the user which one to install and spawn a sub-process to install it.
5. **Choose preferences** — if multiple harnesses or engines are present, ask the user to pick a primary. Save the primary choice to config but remember secondary choices as fallbacks. Users can opt to enable both Claude and Codex paths.
6. **Pick a model** — the user drives this step. The tool asks: _"Which model do you want to use?"_ and accepts one of:
   - a direct model name (mapped into the selected engine's naming scheme if needed), or
   - an opt-in `find-model` path that runs `llmfit` against the machine profile and ranks candidates, then lets the user pick one from the list.
   - Once a model is chosen, the standard download/disk branches apply:
     - if the model already exists locally → continue.
     - if the model is missing and fits in free space → ask the user to confirm download.
     - if the model is missing and larger than free space → ask the user to free space before downloading.
     - if the user cancels the download → re-ask for a different model, or exit cleanly.
   - The final choice is persisted in config. `find-model` is always available later as a standalone subcommand for users who want a recommendation on demand.
7. **Smoke test engine + model** — run a minimal test with the selected engine and selected model to verify it actually works.
8. **Wire up the harness** — configure the selected harness to start with the selected engine + selected model via a single command.
9. **Verify the final command** — run an end-to-end test of the exact command the user will use day-to-day.
10. **Write `guide.md`** — generate a short personalized guide telling the user exactly how to launch their chosen harness against the local model from now on.
11. **Daily use** — user runs their favorite harness against the local model via the single configured command.

Every step is idempotent: re-running the tool should re-use existing config, skip already-done steps, and only re-prompt on changes or failures.

## Discussion Notes

### Strong positioning insight

This product should win as a **local backend bridge**, not by claiming to replicate or replace Claude Code/Codex.

### Important product decision

Make it a **backend bridge + optimizer + switcher**, not a full agent harness from scratch.

### MVP shape

- runtime detection
- hardware detection
- `llmfit` integration for model choice
- one-command setup
- local backend mode with no major workflow change
- support Claude/Codex users directly
- explicit switch back to official cloud tooling

### Implementation language (round check)

The tool is CLI-first, needs to shell out to other processes (harnesses, engines, `llmfit`), inspect hardware, probe local HTTP endpoints, drive an interactive TTY, write config files, and run on macOS + Linux (Windows later).

| Language    | Pros                                                                                                                                                             | Cons                                                                                                                            | Fit                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Bash**    | Zero runtime deps, trivial to install, great at shelling out                                                                                                     | Weak at structured data, interactive UX, hardware detection, and cross-platform quirks; brittle at scale                        | Poor — fine for tiny helpers, not for an 8-step interactive flow               |
| **Python**  | Rich ecosystem (psutil, rich/textual, httpx), fast to build interactive flows, already the language of `llmfit` and `poc_bridge.py`, matches existing repo state | Distribution is messy (venv, pip, pyproject), slower cold start                                                                 | **Strong fit** for MVP — speed of iteration matters more than binary size here |
| **Node.js** | Great interactive CLI libs (ink, prompts, oclif), cross-platform, easy npm install                                                                               | Another runtime to require, weaker hardware-introspection story than Python, not the language of `llmfit`                       | Reasonable alternative, but adds a second ecosystem with no clear win          |
| **Go**      | Single static binary, fast, great for cross-platform CLIs, strong concurrency for parallel runtime probes                                                        | Slower to iterate on an interactive wizard, less mature TUI ecosystem, need to re-implement anything `llmfit` exposes in Python | Good for a v2 rewrite once the flow is proven                                  |
| **Rust**    | Single static binary, strong TUI (ratatui), safe                                                                                                                 | Highest iteration cost, overkill for a setup-wizard CLI, slowest to prototype                                                   | Not recommended for MVP                                                        |

**Recommendation**: build the MVP in **Python**. It matches the existing POC (`poc_bridge.py`), keeps parity with `llmfit`'s native language, and has the best libraries for a guided interactive setup (e.g. `rich`, `questionary`, `psutil`, `httpx`). Once the flow and adapters are stable, a **Go** rewrite is the natural path to a single-binary distribution without a Python runtime requirement.

### Risks already identified

- expectation trap if marketed as “offline Claude Code”
- local model quality gap vs official Claude/Codex
- runtime abstraction complexity across backends
- coding UX depends on tool-use quality, not just raw model benchmark strength
