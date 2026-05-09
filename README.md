<div align="center">

<img src="assets/logo/logo-full.svg" alt="CCL — Claude Codex Local" height="60" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/luongnv89/claude-codex-local/actions/workflows/ci.yml/badge.svg)](https://github.com/luongnv89/claude-codex-local/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

# Hit your limit? Need privacy? Just swap the model.

One alias. Claude Code or Codex on a local model. Skills, agents, MCP servers — all intact.

> **Quota hit mid-session?** `cc` keeps you going on a local model, no context lost.
> **Code that can't leave your machine?** Everything runs offline after model download.
> **Don't want to rewire your workflow?** Your `~/.claude`, skills, agents, and MCP servers carry over untouched.

[**Get Started →**](#quick-start) · [**Landing page →**](https://luongnv89.github.io/claude-codex-local/)

---

## Features

| Feature               | What you get                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Ollama first-class    | `ollama launch` — no duplicated config, no custom Modelfiles                                                                      |
| Config untouched      | All skills, statusline, agents, plugins, and MCP servers carry over                                                               |
| Smart model selection | `llmfit` analyses your hardware and picks the best quantization that fits (optional — wizard prompts to install only when needed) |
| Resume on failure     | Wizard persists progress — `--resume` picks up from the last completed step                                                       |
| Idempotent aliases    | Re-running the wizard replaces the existing alias block, never appends                                                            |
| Cloud fallback        | Run `claude` / `codex` directly (no prefix) to switch back instantly                                                              |

---

## Quick Start

### Install from PyPI (recommended)

```bash
pip install claude-codex-local
```

Or with uv:

```bash
uv tool install claude-codex-local
```

Then run the setup wizard:

```bash
ccl
```

### One-command install (no clone required)

```bash
bash <(curl -sSL https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
```

Or with wget:

```bash
bash <(wget -qO- https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
```

> Use `bash <(...)`, not `curl … | bash`. The wizard is interactive and needs a real TTY — piping steals stdin.

Override defaults with env vars:

```bash
CCL_REF=v0.9.0 CCL_INSTALL_DIR=~/tools/claude-codex-local \
  bash <(curl -sSL https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
```

### Install from a clone

```bash
git clone https://github.com/luongnv89/claude-codex-local.git
cd claude-codex-local
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```bash
ccl
```

### After setup

Reload your shell so the alias is available:

```bash
source ~/.zshrc   # or source ~/.bashrc
```

Then run:

```bash
cc        # Claude Code → local model
cx        # Codex CLI → local model
```

---

## Wizard Steps

```mermaid
graph TD
    A[1. Discover environment] --> B[2. Install missing components]
    B --> C[3. Pick harness + engine]
    C --> D[4. Pick model]
    D --> E[5. Smoke test engine]
    E --> F[6. Wire harness]
    F --> G[7. Install helper + aliases]
    G --> H[8. Verify launch end-to-end]
    H --> I[9. Generate guide.md]
```

See [`guide.example.md`](guide.example.md) for the personalized daily-use guide the wizard generates.

---

## Usage

```bash
ccl                                             # run the interactive first-run wizard
ccl setup --harness claude --engine ollama      # skip the prefs picker
ccl setup --non-interactive                     # CI-friendly install
ccl setup --resume                              # resume after a failure
ccl find-model                                  # standalone model recommendation
ccl doctor                                      # wizard state + presence check
ccl run                                         # launch the configured session interactively
ccl run -p "what is 2+2?"                       # one-shot: drive CCL from another agent
ccl --version                                   # print version and exit
```

`ccl run -p "<prompt>"` runs the harness in non-interactive mode (Claude
Code's `-p`, Codex's `exec`) so external agents and CI scripts can drive a
local model end-to-end without keystrokes. Without `-p`, behavior matches the
`cc` / `cx` alias and the session starts interactively.

Advanced / debug (no user binary — run as a Python module):

```bash
python -m claude_codex_local.core profile      # full hardware profile as JSON
python -m claude_codex_local.core recommend    # llmfit-only model recommendation
python -m claude_codex_local.core adapters     # list all engine adapters
```

---

## Prerequisites

- macOS or Linux with zsh or bash
- Python 3.10+
- At least one harness: [Claude Code](https://claude.ai/code) or [Codex CLI](https://github.com/openai/codex)
- At least one engine: [Ollama](https://ollama.com) (recommended), [LM Studio](https://lmstudio.ai), [vLLM](https://github.com/vllm-project/vllm), llama.cpp, or [9router](https://github.com/decolua/9router) (cloud-routing proxy)
- [`llmfit`](https://github.com/luongnv89/llmfit) on `PATH` (optional — for automatic model selection)

---

## Proven Paths

| Harness     | Engine    | Model                  | Status                                                                  |
| ----------- | --------- | ---------------------- | ----------------------------------------------------------------------- |
| Claude Code | Ollama    | `gemma4:26b`           | Verified end-to-end                                                     |
| Codex CLI   | Ollama    | `gemma4:26b`           | Verified                                                                |
| Claude Code | LM Studio | Qwen3 family           | Blocked — `400 thinking.type`; wizard warns and recommends alternatives |
| Any         | llama.cpp | any                    | Inline-env code path exists, no live proof yet                          |
| Any         | vLLM      | any                    | New in 0.8.0 — adapter shipped with tests                               |
| Claude Code | 9router   | `kr/claude-sonnet-4.5` | New in 0.9.0 — cloud-routed via `cc9` alias; existing `cc` is untouched |
| Codex CLI   | 9router   | `kr/claude-sonnet-4.5` | New in 0.9.0 — cloud-routed via `cx9` alias; existing `cx` is untouched |

---

## 9router quick-start

[9router](https://github.com/decolua/9router) is a local proxy that exposes an OpenAI-compatible API on `http://localhost:20128/v1` and routes calls to cloud models such as `kr/claude-sonnet-4.5`. Picking 9router as the engine adds a **new** `cc9` (Claude) or `cx9` (Codex) alias and leaves your existing `cc` / `cx` aliases untouched.

### Installing and running 9router

**Step 1: Install 9router**

```bash
# Using npm (recommended)
npm install -g 9router

# Or using yarn
yarn global add 9router

# Or using pnpm
pnpm add -g 9router
```

**Step 2: Get your API key**

1. Visit the [9router dashboard](https://9router.com/dashboard) and sign up or log in
2. Navigate to API Keys section
3. Create a new API key and copy it

**Step 3: Start the 9router service**

```bash
# Start 9router with your API key
9router start --api-key YOUR_API_KEY_HERE

# Or set it as an environment variable
export ROUTER9_API_KEY=YOUR_API_KEY_HERE
9router start

# The service will start on http://localhost:20128
```

**Step 4: Verify 9router is running**

```bash
# Check if the service is responding
curl http://localhost:20128/v1/models

# You should see a list of available models
```

**Step 5: Configure CCL to use 9router**

```bash
# Interactive setup (wizard will prompt for API key)
ccl setup --engine 9router

# Non-interactive (CI / scripted):
CCL_9ROUTER_API_KEY=<paste-here> CCL_9ROUTER_MODEL=kr/claude-sonnet-4.5 \
  ccl setup --engine 9router --harness claude --non-interactive
```

### How the wizard configures 9router

The wizard:

1. Asks for the 9router API key and writes it to `~/.claude-codex-local/9router-api-key` with `chmod 0600`. The helper script reads this file at exec time via `$(cat …)` — the key is never embedded in the script body or wizard state file.
2. Verifies reachability via `GET /v1/models` only. **It deliberately does not call `/chat/completions`** during smoke-test or verify, because 9router routes to paid cloud models. The verification record is `{"ok": true, "via": "9router-models-endpoint", "skipped_chat": true}`.
3. Installs `cc9` (or `cx9`) into your shell rc as a new fenced block (`# >>> claude-codex-local:claude9 >>>`), leaving any existing `cc` / `cx` block alone.

**Tip:** `cc9` and `cc` can coexist on the same machine — pick `cc9` when you want to burn cloud quota for a tough prompt, and `cc` (Ollama / LM Studio / llama.cpp) for everyday work.

### Claude Code → 9router env vars

| Env var                                    | 9router                                                       |
| ------------------------------------------ | ------------------------------------------------------------- |
| `ANTHROPIC_BASE_URL`                       | `http://localhost:20128/v1`                                   |
| `ANTHROPIC_AUTH_TOKEN`                     | `$(cat ~/.claude-codex-local/9router-api-key)` (read at exec) |
| `ANTHROPIC_API_KEY`                        | `$(cat ~/.claude-codex-local/9router-api-key)` (read at exec) |
| `ANTHROPIC_CUSTOM_MODEL_OPTION`            | `<tag>` (e.g. `kr/claude-sonnet-4.5`)                         |
| `ANTHROPIC_CUSTOM_MODEL_OPTION_NAME`       | `9router <tag>`                                               |
| `CLAUDE_CODE_ATTRIBUTION_HEADER`           | `"0"`                                                         |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `"1"`                                                         |

For Codex: `OPENAI_BASE_URL=http://localhost:20128/v1`, `OPENAI_API_KEY=$(cat …)`.

---

## Rollback

```bash
# Remove the fenced block from ~/.zshrc (between the marker lines)
rm -rf .claude-codex-local
```

Each fence block (`claude` / `codex` / `claude9` / `codex9`) is independent — you can remove just one without touching the others. Your `~/.claude` and `~/.codex` are unchanged.

---

<details>
<summary>Architecture details</summary>

### Three layers

1. **Machine profile + model recommendation** (`claude_codex_local/core.py`) — dumps a JSON snapshot of installed harnesses/engines/llmfit/disk, runs `llmfit` for ranked model recommendations, and provides a `doctor` command for pretty-printing wizard state.

2. **Interactive wizard** (`claude_codex_local/wizard.py`) — 9 steps from discovery to ready-to-use daily alias. Persists progress in `.claude-codex-local/wizard-state.json` so `--resume` picks up after a failure.

3. **Helper scripts + shell aliases** — `.claude-codex-local/bin/cc` (or `cx`) is a short bash wrapper. For Ollama it runs `ollama launch claude|codex --model <tag>`. For LM Studio / llama.cpp it sets inline env vars and execs the real harness. A fenced block in `~/.zshrc` / `~/.bashrc` declares the aliases.

### Why `ollama launch`

`ollama launch claude --model <tag>` is an official Ollama subcommand that sets the right env vars internally and execs the user's real `claude` binary against the local daemon — using `~/.claude` as-is.

This means:

- No duplicated `~/.claude` directory
- No custom Modelfile or `ollama create`
- No `ANTHROPIC_CUSTOM_MODEL_OPTION` to manage manually
- `cc` just works

### Claude Code → LM Studio / llama.cpp env vars

| Env var                                    | LM Studio                | llama.cpp                |
| ------------------------------------------ | ------------------------ | ------------------------ |
| `ANTHROPIC_BASE_URL`                       | `http://localhost:1234`  | `http://localhost:8001`  |
| `ANTHROPIC_API_KEY`                        | `lmstudio`               | `sk-local`               |
| `ANTHROPIC_CUSTOM_MODEL_OPTION`            | `<tag>`                  | `<tag>`                  |
| `ANTHROPIC_CUSTOM_MODEL_OPTION_NAME`       | `Local (lmstudio) <tag>` | `Local (llamacpp) <tag>` |
| `CLAUDE_CODE_ATTRIBUTION_HEADER`           | `"0"`                    | `"0"`                    |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `"1"`                    | `"1"`                    |

### Codex CLI → Ollama

```bash
ollama launch codex --model <tag> -- --oss --local-provider=ollama
```

The `--oss --local-provider=ollama` flags are required after `--` because Codex otherwise tries to route through the ChatGPT account and rejects non-OpenAI model names.

</details>

<details>
<summary>Project structure</summary>

```
.
├── claude_codex_local/
│   ├── __init__.py             # Package metadata + __version__
│   ├── wizard.py               # Interactive setup wizard + `ccl` CLI
│   └── core.py                 # Machine profile, engine adapters, llmfit bindings
├── scripts/
│   └── e2e_smoke.sh            # End-to-end smoke test
├── docs/
│   ├── poc-wizard.md           # 9-step wizard architecture
│   ├── poc-architecture.md     # System design overview
│   ├── poc-bootstrap.md        # Bootstrap / install flow
│   └── poc-proof.md            # Design rationale
├── tests/                      # pytest test suite
├── install.sh                  # One-command remote installer
└── pyproject.toml              # Project metadata and tool config
```

</details>

<details>
<summary>Tech stack</summary>

| Layer         | Tool                                                                                                |
| ------------- | --------------------------------------------------------------------------------------------------- |
| Language      | Python 3.10+                                                                                        |
| UI / prompts  | [questionary](https://github.com/tmbo/questionary), [rich](https://github.com/Textualize/rich)      |
| Linting       | [ruff](https://github.com/astral-sh/ruff)                                                           |
| Type checking | [mypy](https://mypy-lang.org)                                                                       |
| Testing       | [pytest](https://pytest.org) + pytest-cov                                                           |
| Security      | [bandit](https://github.com/PyCQA/bandit), [detect-secrets](https://github.com/Yelp/detect-secrets) |
| Pre-commit    | [pre-commit](https://pre-commit.com)                                                                |

</details>

<details>
<summary>Local state</summary>

Everything written by the bridge goes under `.claude-codex-local/`. Override with `CLAUDE_CODEX_LOCAL_STATE_DIR`.

</details>

<details>
<summary>Contributing</summary>

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

For security issues, see [SECURITY.md](SECURITY.md).

</details>

---

[MIT](LICENSE) — © 2026 Luong NGUYEN
