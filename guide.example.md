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
- **Aliases**: `cc`, `claude-local` (or `cp`, `pi-local` for Pi)
  (installed in `~/.zshrc`)
- **Helper script**: `<REPO_ROOT>/.claude-codex-local/bin/cc`

## Daily use

> **First time after setup?** Reload your shell so the new alias is on
> your `PATH` — run `source ~/.zshrc` or open a new terminal. You only
> need to do this once per shell session.

Then run:

```bash
cc
```

That's it. The alias execs `<REPO_ROOT>/.claude-codex-local/bin/cc`. Claude
and Codex helpers either run `ollama launch <harness>` (Ollama path) or set
the right env vars (LM Studio / llama.cpp / vLLM). Pi helpers set
`PI_CODING_AGENT_DIR` to Pi's normal config directory and launch
`pi --provider ccl-<engine> --model <model>`.

Your real `~/.claude`, `~/.codex`, and Pi config are used as-is for those
harnesses. For Pi, CCL adds/updates only its `ccl-*` provider in the normal
`models.json`, so installed Pi extensions, packages, skills, prompts, themes,
settings, and auth stay available from `cp`.

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

Your global `~/.claude` and `~/.codex` are unchanged. Pi keeps using its
normal config directory; CCL only adds a `ccl-*` provider to `models.json`.
Run `claude`, `codex`, or `pi` directly (without `cc`/`cx`/`cp`) to use the
official backend/model selection.

## Rollback

Each harness (claude / codex / pi, plus 9router variants) has its own fenced
block, so you can remove just one harness without touching any other you may
have set up.

To wipe only the claude harness:

1. Delete the fenced block from `~/.zshrc` (between the
   `# >>> claude-codex-local:claude >>>` and
   `# <<< claude-codex-local:claude <<<` markers).
2. `rm -f <REPO_ROOT>/.claude-codex-local/bin/cc`
3. `rm -f <REPO_ROOT>/guide.md`
4. For Pi installs, optionally remove this harness's `ccl-*` provider from
   Pi's normal `models.json`.

To wipe the local bridge entirely:

1. Delete every `# >>> claude-codex-local:<harness> >>>` block from
   `~/.zshrc`.
2. `rm -rf <REPO_ROOT>/.claude-codex-local`
3. `rm -f <REPO_ROOT>/guide.md`
4. For Pi installs, optionally remove all `ccl-*` providers from Pi's normal
   `models.json`.
