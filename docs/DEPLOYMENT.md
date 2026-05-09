# Deployment

This document covers how to release and distribute `claude-codex-local`.

## Release Process

1. Update the version in `pyproject.toml`
2. Update `docs/CHANGELOG.md` — move `[Unreleased]` items under the new version heading
3. Commit: `git commit -m "chore(release): bump version to X.Y.Z"`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push origin main --tags`

The `install.sh` remote installer fetches a specific tag via `CCL_REF` (defaults to `main`). After tagging, users can pin to a specific release:

```bash
CCL_REF=v0.9.0 bash <(curl -sSL https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
```

## Installer (`install.sh`)

The installer is designed to work without cloning the repo:

1. Downloads a tarball of `CCL_REF` from GitHub
2. Extracts to `CCL_INSTALL_DIR` (default: `~/.claude-codex-local-src`)
3. Creates a virtualenv and runs `pip install -e .` inside it — this puts the `ccl` binary at `<venv>/bin/ccl`
4. Launches `.venv/bin/ccl` automatically

Environment variables:

| Variable          | Default                     | Description                                 |
| ----------------- | --------------------------- | ------------------------------------------- |
| `CCL_REF`         | `main`                      | Git ref (branch, tag, or commit) to install |
| `CCL_INSTALL_DIR` | `~/.claude-codex-local-src` | Install directory                           |

## CI

GitHub Actions CI runs on every push and PR (`.github/workflows/ci.yml`):

- Unit tests (`pytest -m "not local"`)
- Linting (`ruff check .`)
- Type checking (`mypy .`)
- Security scan (`bandit`)

Tests marked `@pytest.mark.local` are skipped in CI since they require real binaries.

## PyPI

The package is published to PyPI as `claude-codex-local`. To publish a new release:

```bash
pip install build twine
python -m build
twine upload dist/*
```

The `pyproject.toml` is already configured for `setuptools` build.
