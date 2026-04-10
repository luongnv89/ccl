# Local coding guide (example)

> **Note:** This is an example of the `guide.md` file that
> `claude-codex-local` generates on your machine after a successful
> wizard run. Real values (absolute paths, chosen model, engine,
> harness) will be filled in from your actual run. The real
> `guide.md` is gitignored — this `guide.example.md` is committed
> only as reference documentation.

## What was set up

- **Harness**: `claude`
- **Engine**: `ollama`
- **Model**: `<model-name>:<size>`
- **Isolated HOME**: `<REPO_ROOT>/.claude-codex-local/home`

## Daily use

Run this single command to start your local coding session:

```bash
claude --model <model-name>:<size>
```

The wizard wrote isolated config under `<REPO_ROOT>/.claude-codex-local/home`
so your official `~/.claude` and `~/.codex` directories are untouched. You
can switch back to cloud mode at any time by running `claude` or `codex`
directly (without the `claude-codex-local` wrapper).

## Troubleshooting

- **Slow second turn in Claude Code?** Check that
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` is set inside
  `<REPO_ROOT>/.claude-codex-local/home/.claude/settings.json`. It will not
  work as a shell env var.
- **Engine not responding?** Re-run the smoke test:
  ```bash
  ./bin/poc-doctor
  ```
- **Model missing?** Re-run the wizard — it will detect the gap and offer
  to re-download: `python3 -m wizard`
- **Switch to a different model?** Run the standalone `find-model` helper:
  ```bash
  python3 -m wizard find-model
  ```

## Return to official mode

The wizard never mutates your global `~/.claude` or `~/.codex` config.
Just run `claude` or `codex` directly (outside the wrapper) and you are
back on cloud.

## Rollback

To wipe all local bridge state:

```bash
rm -rf <REPO_ROOT>/.claude-codex-local
rm -f <REPO_ROOT>/guide.md
```
