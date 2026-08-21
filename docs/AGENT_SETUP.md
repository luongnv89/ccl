# Agent Setup Notes: Install, Environment and Run

Reproduce the CI environment from a clean checkout. Nothing here can be safely
inferred from the tree alone — a bare checkout has no venv and no installed
package, so `python -m pytest --collect-only` fails with
`ModuleNotFoundError: questionary` until you follow the steps below.

## Prerequisites

- Python **3.10+** (CI matrix runs 3.10, 3.11 and 3.12)
- `git`

## Install from a clean checkout

```bash
git clone https://github.com/luongnv89/ccl.git
cd ccl

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e '.[dev]'
```

The `[dev]` extra installs the test/lint toolchain: `pytest`, `pytest-cov`,
`ruff`, `mypy`, `bandit`, `detect-secrets`, `pre-commit`.

## Recorded test command

This is the command CI runs (see `.github/workflows/ci.yml`):

```bash
python -m pytest -q --cov=claude_codex_local --cov-report=term-missing
```

## Recorded lint commands

```bash
ruff check .           # lint — expected to pass cleanly
ruff format --check .  # formatter check — expected to report 2 files (see below)
```

## Known-red baseline (do not mistake for your own breakage)

The current expected outcome of the commands above on `main` is:

> **8 failed, 12 skipped; `ruff format --check` reports 2 files**

In other words:

- The test suite currently ends with **8 failed, 12 skipped** — these failures
  are pre-existing on `main`, not caused by your change.
- `ruff check .` passes; `ruff format --check .` flags **2 files**
  (`tests/test_e2e_llamacpp_real.py`, `tests/test_wizard_unit.py`) as needing
  reformatting.
- Tests marked `local` / requiring real binaries (ollama, lm-studio, claude,
  codex, pi, llmfit) auto-skip when the tool is missing. Skip counts vary with
  what is installed locally; CI records the numbers above.

## Environment variables read at import time

`claude_codex_local/_config.py` reads the following environment variables **once,
at import time**. Setting them after importing the package has no effect — export
them before starting the Python process (or use a fresh process).

| Variable | Default | Used for |
|------------------------------------|--------------------------------------|----------------------------------------|
| `HOME` | system home directory | base path for the default state dir |
| `CLAUDE_CODEX_LOCAL_STATE_DIR` | `$HOME/.claude-codex-local` | runtime state directory |
| `LMS_SERVER_PORT` | `1234` | LM Studio server port |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_API_KEY` | *(empty)* | Ollama API key |
| `LLAMACPP_SERVER_PORT` | `8001` | llama-server port |
| `LLAMACPP_SERVER_HOST` | `127.0.0.1` | llama-server host |
| `LLAMACPP_BASE_URL` | `http://localhost:<LLAMACPP_SERVER_PORT>` | llama.cpp base URL |
| `LLAMACPP_API_KEY` | *(empty)* | llama.cpp API key |
| `LLAMACPP_CTX_SIZE` | `131072` | llama-server context size |
| `LLAMACPP_N_GPU_LAYERS` | *(unset)* | llama-server GPU layers |
| `LLAMACPP_THREADS` | *(unset)* | llama-server threads |
| `LLAMACPP_MTP_ENABLED` | *(unset)* | llama-server MTP toggle |
| `LLAMACPP_SPEC_DRAFT_N_MAX` | *(unset; internal default `5`)* | speculative-draft token cap |
| `CCL_9ROUTER_BASE_URL` | `http://localhost:20128/v1` | 9router base URL |
| `CCL_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM base URL |
| `VLLM_API_KEY` | *(empty)* | vLLM API key |

None of these are required to install or collect tests — defaults are used when
they are unset.

## Verify your environment

After following the install steps above, this must complete **without
collection errors** (it may still report the known failures on a full run):

```bash
python -m pytest --collect-only -q
```

If it reports `ModuleNotFoundError` or collection errors, the editable install
did not complete — re-run `pip install -e '.[dev]'` inside the activated venv.
