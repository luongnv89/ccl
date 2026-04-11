# Local coding guide (example)

> **Note:** This is an example of the `guide.md` file that `ccl`
> generates on your machine after a successful wizard run. Real
> values (absolute paths, chosen model, engine, harness) will be
> filled in from your actual run. The real `guide.md` is gitignored —
> this `guide.example.md` is committed only as reference documentation.

## What was set up

- **Harness**: `<harness>`
- **Engine**: `<engine>`
- **Model**: `<model>:<size>`
- **Aliases**: `cc`, `claude-local` (installed in `~/.zshrc`)
- **Helper script**: `<REPO_ROOT>/.claude-codex-local/bin/cc`

## Daily use

> **First time after setup?** Reload your shell so the new alias is on
> your `PATH` — run `source ~/.zshrc` or open a new terminal. You only
> need to do this once per shell session.

Then run:

```bash
cc
```

That's it. The alias execs `<REPO_ROOT>/.claude-codex-local/bin/cc`, which
either runs `ollama launch <harness>` (Ollama path) or sets the right env
vars and execs `<harness>` directly (LM Studio / llama.cpp path).

Your real `~/.claude` and `~/.codex` are used as-is, so all your skills,
statusline, agents, plugins, and MCP servers keep working.

You can still pass extra args: `cc -p "what does foo.py do?"`.

## Troubleshooting

- **`cc: command not found`?** Open a new terminal or run
  `source ~/.zshrc`.
- **Engine not responding?** Re-run the wizard smoke test:
  ```bash
  ccl doctor
  ```
- **Want to switch models?** Re-run the wizard:
  ```bash
  ccl setup --resume
  ```

## Return to official mode

Your global `~/.claude` and `~/.codex` are unchanged. Run `claude` or
`codex` directly (without `cc`) to use the cloud backend.

## Rollback

Each harness (claude / codex) has its own fenced block, so you can remove
just one harness without touching any other you may have set up.

To wipe only the claude harness:

1. Delete the fenced block from `~/.zshrc` (between the
   `# >>> claude-codex-local:claude >>>` and
   `# <<< claude-codex-local:claude <<<` markers).
2. `rm -f <REPO_ROOT>/.claude-codex-local/bin/cc`
3. `rm -f <REPO_ROOT>/guide.md`

To wipe the local bridge entirely:

1. Delete every `# >>> claude-codex-local:<harness> >>>` block from
   `~/.zshrc`.
2. `rm -rf <REPO_ROOT>/.claude-codex-local`
3. `rm -f <REPO_ROOT>/guide.md`
