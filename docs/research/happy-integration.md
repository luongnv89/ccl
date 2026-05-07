# Analysis: Happy Integration with ccl-configured Model

## Happy's Invocation Mechanism

Happy is an Electron app that uses `@anthropic-ai/claude-agent-sdk` directly via `worker_threads`. Key findings:

- **No CLI invocation**: Happy never calls the `claude` or `codex` CLI binary. It uses the SDK's `query()` function programmatically.
- **Model selection**: Via `AgentStartOptions.model` passed from the UI layer through IPC to the worker. Fully configurable at runtime.
- **Environment variables**: Only sets `ANTHROPIC_API_KEY` when explicitly provided in options.
- **PATH lookups**: None whatsoever.
- **TTY handling**: Not applicable — headless `postMessage` communication.
- **Source files**:
  - `packages/codium/sources/boot/main/agent-worker/worker.ts` — agent worker using SDK
  - `packages/codium/sources/boot/main/agent-worker/host.ts` — IPC host
  - `packages/codium/sources/plugins/anthropic/index.ts` — Anthropic plugin with model picker
  - `packages/codium/sources/plugins/codex/index.ts` — Codex plugin (auth-only, not yet implemented)

## ccl's Alias/Wrapper Mechanism

ccl's wizard generates a complete alias system:

- **Helper scripts**: Written to `~/.claude-codex-local/bin/cc` (bash scripts with `set -e`, env export, `exec claude --model <tag> "$@"`)
- **Shell rc integration**: Fenced alias blocks appended to `~/.zshrc` or `~/.bashrc` (e.g., `alias cc=~/.claude-codex-local/bin/cc`)
- **State storage**: `~/.claude-codex-local/wizard-state.json` contains full config (engine, model, wire_result, alias_names)
- **Secret handling**: API keys stored as `chmod 600` files, referenced via `raw_env` shell expressions like `$(cat /path/to/key)`
- **Key files**: `core.py` (adapters, STATE_DIR), `wizard.py` (alias generation, helper scripts, shell rc installation)

## Feasibility Assessment: **ALIAS WORKAROUND IS NOT FEASIBLE**

The core working hypothesis — wrapping the `claude` binary to intercept Happy's calls — is **fundamentally impossible** because:

1. Happy **never invokes the `claude` CLI**. It imports `@anthropic-ai/claude-agent-sdk` as a Node.js dependency and calls `query()` directly via `worker_threads`.
2. Happy has no PATH resolution, no `child_process` spawning, no binary invocation. It's a pure Electron app with the SDK as a runtime dependency.
3. The model is selected via an `AgentStartOptions.model` string passed through IPC — not through CLI args or environment variables that a binary shim could intercept.

## Alternative Approaches (for future consideration)

While the alias wrapper approach is not feasible, there are alternative directions worth exploring in a follow-up spike:

### Option A: Happy Plugin System Hook
Happy has a plugin system (`packages/codium/sources/plugins/`). A ccl plugin could:
- Read `~/.claude-codex-local/wizard-state.json` at runtime
- Override the Anthropic plugin's model list and API key with ccl-configured values
- Intercept model selection in Happy's UI

**Feasibility**: High — requires modifying Happy's plugin code or injecting a plugin.

### Option B: Environment Variable Preload
If Happy reads `ANTHROPIC_API_KEY` or `ANTHROPIC_BASE_URL` from `process.env` when not explicitly set in options, a shell wrapper around the `happy` command (not `claude`) could set these vars before launching the Electron app.

**Feasibility**: Unclear — requires verifying Happy's SDK defaults when `apiKey` is not provided in `AgentStartOptions`.

### Option C: Happy Configuration File
Happy may read a local config file (e.g., `.happy/config.json` or similar) that specifies default model/API key. If so, ccl could generate/modify this file.

**Feasibility**: Unknown — requires further investigation of Happy's config system.

## Recommendation

**Close this issue as won't-fix** for the alias wrapper approach. The working hypothesis is invalid because Happy uses the SDK directly, not the CLI binary.

**Propose a follow-up spike** for Option A (Happy plugin integration) if the user wants to pursue this. This would be a higher-effort but more promising direction that aligns with Happy's actual architecture.

## Acceptance Criteria Verification

| Criterion | Status | Notes |
|-----------|--------|-------|
| Happy's invocation mechanism documented | Done | SDK-only, no CLI, worker_threads, `query()` |
| ccl's alias/wrapper mechanism documented | Done | Helper scripts + shell rc fence blocks |
| Feasibility assessment produced | Done | Alias wrapper NOT feasible; alternative approaches proposed |
| Proposed approach with constraints/risks | Done | 3 alternatives documented |
| Infeasibility reason documented | Done | Happy never calls `claude` binary |
