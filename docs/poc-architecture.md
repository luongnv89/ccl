# POC architecture note

Last updated: 2026-04-10

## Narrow scope

This POC is deliberately tiny:

- primary runtime: **Ollama** (via `ollama launch`)
- secondary runtime: **LM Studio** (inline env fallback)
- harnesses: **Claude Code** and **Codex CLI**
- model-fit engine: **llmfit**
- config rule: **the wizard never touches `~/.claude` or `~/.codex`**;
  the user's real skills, statusline, agents, plugins, and MCP servers
  keep working exactly as they were

That is enough to prove the idea without pretending we solved every backend.

## The bridge shape

### Three layers

1. **Machine profile + model recommendation** (`poc_bridge.py`)
   - `profile`: dump a JSON snapshot of installed harnesses/engines/llmfit/disk
   - `recommend`: pick a best-fit installed coding model, biased toward
     models that actually load on this machine (not just paper-best)
   - `doctor`: pretty-print the wizard state and re-run presence checks

2. **Interactive wizard** (`wizard.py`)
   - 9 steps from discovery to ready-to-use daily alias
   - persists progress in `.claude-codex-local/wizard-state.json` so
     `--resume` picks up after a failure
   - writes a helper script and installs a shell alias block (see below)

3. **Helper scripts + shell aliases** (the user-facing surface)
   - `.claude-codex-local/bin/cc` (or `cx`): a short bash wrapper that
     either runs `ollama launch claude|codex --model <tag>` (Ollama path)
     or sets inline env vars and execs the real harness (LM Studio /
     llama.cpp path)
   - `~/.zshrc` / `~/.bashrc`: a fenced block (`# >>> claude-codex-local >>>`
     … `# <<< claude-codex-local <<<`) declaring the aliases `cc` +
     `claude-local` or `cx` + `codex-local`, all pointing at the helper
     script

### Why `ollama launch`

`ollama launch claude --model <tag>` is an official Ollama subcommand
that sets the right env vars internally and execs the user's real
`claude` binary against the local Ollama daemon. Same for
`ollama launch codex --model <tag> -- --oss --local-provider=ollama`.

Using it means:

- **no duplicated `~/.claude` directory** — the real one is used as-is,
  so all skills, statusline, agents, plugins, and MCP servers carry over
- **no baked-in model variant** — no `-cclocal` Ollama model, no
  `Modelfile`, no `ollama create`
- **no `ANTHROPIC_CUSTOM_MODEL_OPTION` to manage** — `ollama launch`
  handles the client-side model-name allowlist issue for us
- **no isolated `HOME` prefix** the user has to remember — `cc` just works

For LM Studio and llama.cpp we fall back to the inline-env approach
because `ollama launch` only knows about Ollama. The helper script holds
the long env block so the user's rc file stays short.

### Daily user surface

After setup, the user opens a new terminal and runs:

```bash
cc             # or claude-local — same thing
cc -p "hi"     # extra args flow through "$@"
```

To switch back to cloud mode, run `claude` / `codex` directly (no
prefix). Nothing in the user's global config needs to be touched.

### Rollback

```bash
# 1. Delete the fenced block from ~/.zshrc (between the marker lines)
# 2. Remove the state directory
rm -rf .claude-codex-local
```

## Codex bridge

Codex natively supports `--oss --local-provider=ollama -m <model>`. The
wizard's Codex wire path for Ollama just wraps this in
`ollama launch codex --model <tag> -- --oss --local-provider=ollama`, so
the `cx` alias is a one-liner.

For LM Studio / llama.cpp, Codex reads `OPENAI_BASE_URL` and
`OPENAI_API_KEY` from the environment — the helper script exports those
and execs `codex -m <tag>`.

## Claude Code bridge

Claude Code has a known client-side model-name allowlist that rejects
unrecognized model IDs before any request is sent. The official escape
hatch is `ANTHROPIC_CUSTOM_MODEL_OPTION`, which whitelists one custom
model ID.

For the Ollama path, `ollama launch claude --model <tag>` sets that
variable internally and points `ANTHROPIC_BASE_URL` at the local
daemon, so we do nothing extra.

For the LM Studio / llama.cpp path, the wizard writes those env vars
(`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_CUSTOM_MODEL_OPTION`,
`ANTHROPIC_CUSTOM_MODEL_OPTION_NAME`, `CLAUDE_CODE_ATTRIBUTION_HEADER=0`,
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`) into the helper script
before it execs `claude --model <tag>`.

## Qwen3 + Claude Code

Claude Code sends a `thinking` payload in its chat requests that Qwen3
reasoning models interpret as an unterminated `<think>` block, blowing
the context budget. The wizard detects Qwen3 model names at pick time
and prints a warning recommending Gemma 3 or Qwen 2.5 Coder instead.
In interactive mode the user can continue anyway; in non-interactive
mode the step fails.

Earlier POC versions worked around this by running `ollama create` on a
derived `-cclocal` model with `SYSTEM "/no_think"` baked in. That
design was dropped in favor of warning + recommending a compatible
model, because duplicating Ollama models silently was surprising and
hard to roll back.

## Superseded design (historical)

The first version of this POC wrote an isolated `HOME=<repo>/.claude-codex-local/home`
directory with duplicate `.claude/settings.json` / `.codex/config.toml`
files, and generated a `-cclocal` Ollama model variant. That design is
gone as of 2026-04-10. See the `refactor(wizard): use ollama launch +
shell aliases` commit for the full diff.

## What remains intentionally unproven

- llama.cpp path has inline-env support but has not been live-tested
  end-to-end in this repo
- LM Studio Qwen3-coder path is known to return 400 on the `thinking`
  payload; the wizard warns but does not currently auto-mitigate
