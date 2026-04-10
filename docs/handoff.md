# Hand-off: claude-codex-local wizard POC

Date: 2026-04-10
Branch: `main` (uncommitted)

## TL;DR

The 8-step interactive wizard is **built and proven end-to-end** for the
Claude Code + Ollama path. The Claude Code runtime gap from the earlier POC
is closed: `claude --bare --settings <iso> --model gemma4:26b -p "Reply with
exactly READY"` returns `READY` with `~/.claude` untouched.

Read `docs/poc-wizard.md` first for the full architecture — this file is
just a session-boot hand-off.

## Where to start a fresh session

```bash
cd /Users/montimage/buildspace/luongnv89/claude-codex-local
source .venv/bin/activate      # questionary 2.1.1, rich 14.3.3 already installed
bin/claude-codex-local setup --non-interactive --harness claude --engine ollama
```

Expected result: all 8 steps pass, `guide.md` is regenerated at repo root,
`~/.claude/settings.json` is not modified.

## What exists now (uncommitted)

| File | Status | Purpose |
|------|--------|---------|
| `poc_bridge.py` | modified | +`disk_usage_for()`, +`llamacpp_detect()`, richer `machine_profile()` with `presence.has_minimum` |
| `wizard.py` | **new** (~650 LOC) | The 8-step wizard. `WizardState` dataclass, one function per step, checkpoints to `.claude-codex-local/wizard-state.json`, `--resume` support |
| `bin/claude-codex-local` | **new** | Bash shim preferring `.venv/bin/python` |
| `pyproject.toml` | **new** | deps: `questionary>=2.0`, `rich>=13.0` |
| `requirements.txt` | **new** | Same deps for non-pipx installs |
| `docs/poc-wizard.md` | **new** | Full architecture doc — read this first |
| `docs/handoff.md` | **new** | This file |
| `README.md` | modified | Quickstart points at the new wizard |
| `idea.md` / `prd.md` / `tasks.md` | modified | PRD bumped to v1.2 with user-first model pick flow |

All prior POC scripts (`bin/codex-local`, `bin/poc-doctor`, `bin/poc-recommend`,
`bin/poc-machine-profile`, `bin/claude-local`, `bin/claude-local-config`) are
still in place and still work.

## The 8 steps (quick reference)

- **2.1** discover env (harnesses/engines/llmfit/disk) — ✅ implemented
- **2.2** install missing (hints only, user runs installers) — ✅ implemented, conditional
- **2.3** pick primary harness + engine (respects `--harness` / `--engine`) — ✅ implemented
- **2.4** pick model **user-first** (direct name OR opt-in llmfit find-model) — ✅ implemented; non-interactive prefers already-installed models
- **2.5** smoke test engine+model — ✅ implemented
- **2.6** wire harness (writes isolated `settings.json` with `ATTRIBUTION_HEADER=0` + `DISABLE_NONESSENTIAL_TRAFFIC=1`) — ✅ implemented
- **2.7** verify (`claude --bare --settings <iso> -p "Reply with exactly READY"`) — ✅ implemented, **proven against Ollama**
- **2.8** generate personalized `guide.md` — ✅ implemented

## Proven matrix (this machine)

| Harness | Engine    | Model                  | Status |
|---------|-----------|------------------------|--------|
| Claude  | Ollama    | `gemma4:26b`           | ✅ end-to-end verify passes |
| Codex   | Ollama    | `qwen2.5-coder:0.5b`   | ✅ from prior POC |
| Claude  | LM Studio | `qwen/qwen3-coder-30b` | ⚠️ smoke test passes, verify blocked by `400 thinking.type` |
| Any     | llama.cpp | any                    | ⚠️ detection only, no runtime proof |

## Active known limitations

1. **LM Studio + Claude Code 400 thinking.type** — Claude Code sends a
   `thinking` payload that LM Studio rejects. Smoke test goes through
   `/v1/chat/completions` and works; `claude` itself fails. Blog post
   `blog-posts/2026-04-03-run-claude-code-codex-local-gemma4` documents the
   `--chat-template-kwargs '{"enable_thinking": false}'` fix for llama.cpp and
   a Modelfile `PARAMETER` fix for Ollama. **Not yet automated by the wizard.**
2. **Disk-size estimation is a stub** — `_estimate_model_size()` returns
   `None`, so the "fits in free space" branch always falls through the
   warn-only path. Wire `llmfit info <model>` size data here.
3. **llama.cpp lifecycle is manual** — detection works, but the wizard cannot
   start `llama-server` or manage GGUFs.
4. **No automated tests** — everything is verified by manual runs.

## Suggested next work (pick one)

1. **Automate the no-think fix** for Qwen3/Gemma4 so LM Studio + Claude Code
   actually verifies end-to-end. This unlocks the Apple-Silicon-preferred
   path. Likely touches `_wire_claude()` + a per-model Modelfile/template
   writer.
2. **Wire llmfit model size** into `_estimate_model_size()` so step 2.4's
   disk-gated download branch does real work.
3. **Add a `doctor` subcommand** to `wizard.py` that pretty-prints
   `wizard-state.json` + re-runs presence check.
4. **Commit what's here** — nothing is committed yet; a single "wizard POC"
   commit would be a clean checkpoint before further iteration. User has not
   asked for a commit — do not run `git commit` without explicit permission.

## Task list state (TaskList tool)

Tasks #1–#12 are all `completed`. Safe to clean up or keep as a history
trail. A fresh session can create a new list scoped to whichever of the
"next work" items is picked.

## Key design decisions to preserve

- **User-first model pick**: the tool asks "which model?" first. llmfit is
  an opt-in helper, not the default path. See PRD v1.2 §4.1 step 2.4.
- **Python MVP, Go candidate for v2**: locked in PRD §6.0 and
  `tasks.md` "Technical Decisions". Do not rewrite in another language
  without user approval.
- **Repo-local isolation**: all state under `.claude-codex-local/`, subprocess
  `HOME`/`XDG_*` overridden to that path. Official `~/.claude` and `~/.codex`
  are **never** touched. Any new wire-up must follow this rule.
- **`CLAUDE_CODE_ATTRIBUTION_HEADER=0` MUST live in `settings.json`** — shell
  env vars are ignored by Claude Code for that flag. This is the #1 local-route
  perf fix. Don't move it to the environment.
- **Verify uses `claude --bare --settings <iso> --dangerously-skip-permissions`**
  — `--bare` sidesteps keychain reads and side-channel model calls, which
  otherwise break local routing. Don't drop `--bare` from step 2.7.

## Quick commands cheat sheet

```bash
# Full clean run
rm -f .claude-codex-local/wizard-state.json guide.md
bin/claude-codex-local setup --harness claude --engine ollama

# Non-interactive (CI)
bin/claude-codex-local setup --non-interactive --harness claude --engine ollama

# Resume after a failed step
bin/claude-codex-local setup --resume

# Standalone llmfit recommendation (no setup)
bin/claude-codex-local find-model

# Legacy diagnostics still work
./bin/poc-machine-profile
./bin/poc-doctor
```

## Reference reading

- `docs/poc-wizard.md` — **current** architecture
- `docs/poc-proof.md` — prior POC (Codex + Ollama proof)
- `docs/poc-architecture.md` — prior architecture note
- `prd.md` §4.1 + §6.0 — the 8-step flow and the language decision
- `blog-posts/2026-04-03-run-claude-code-codex-local-gemma4/draft-v0.2.md` — the no-think fix reference
- `blog-posts/2026-03-10-claude-code-local-model-config/draft-v0.1.md` — Claude Code `settings.json` env vars
