# POC proof note

Date: 2026-04-09

> **Historical note (2026-04-10):** This document captures the original
> POC proof against the isolated-HOME + `bin/codex-local` wrapper design.
> That design has since been superseded by `ollama launch` + shell
> aliases. The commands below (`./bin/codex-local`) no longer exist —
> use `cx` after running `bin/claude-codex-local`. See
> `docs/poc-architecture.md` and `docs/poc-wizard.md` for the current
> design.

## Environment used

- host: Linux x86_64 VM
- CPU: 4 vCPU
- RAM: ~1.9 GiB total
- GPU: none
- runtime: `ollama 0.15.2`
- Claude CLI: `2.1.81`
- Codex CLI: `0.91.0`
- llmfit: `0.9.3`

## Local model used

- installed for the POC: `qwen2.5-coder:0.5b`
- Ollama size shown locally: `397 MB`

Why this model:

- this machine is tiny
- it is the smallest practical coding model available in Ollama for a real local proof
- llmfit still warns the fit is tight, which is fair, but the model actually starts here

## Commands used

### 1. Verify the box

```bash
claude-codex-local doctor
```

### 2. Inspect the machine profile

```bash
ccl-bridge profile
```

### 3. Inspect the recommendation

```bash
ccl-bridge recommend
```

### 4. Run the local Codex path

Interactive:

```bash
./bin/codex-local
```

One-shot prompt:

```bash
./bin/codex-local exec --skip-git-repo-check 'reply with exactly READY'
```

## What was proven

A real local backend path works end to end with:

- harness: **Codex CLI**
- runtime: **Ollama**
- model: **qwen2.5-coder:0.5b**

The wrapper prints the active runtime/model before launching:

```text
claude-codex-local: runtime=ollama model=qwen2.5-coder:0.5b
```

And the non-interactive smoke test returned the expected `READY` response through Codex.

## What stayed isolated

No official Claude/Codex config was required for the POC.

All local bridge state is written under:

```text
.claude-codex-local/
```

That includes the generated Ollama integration config used by the wrappers.

## Gaps and rough edges

- Codex still logs a harmless `401 Unauthorized` model-refresh warning in local-only mode before succeeding.
- Claude got a safe config path, but this repo does **not** claim a validated Claude end-to-end smoke test yet.
- On this hardware, "quality mode" is basically a lie. Balanced/Fast collapse to the tiny coder model.
