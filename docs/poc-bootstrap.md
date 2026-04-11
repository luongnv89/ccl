# POC bootstrap

Date: 2026-04-09

## What this POC assumes is already installed

These are treated as existing machine-level tools:

- `ollama`
- `claude`
- `codex`

The repo does **not** install those for you.

## What this POC adds locally

This POC expects `llmfit` on `PATH`.

Tested version during implementation:

- `llmfit 0.9.3`

## Minimal llmfit install used for this POC

Manual install into `~/.local/bin`:

```bash
REPO='AlexsJones/llmfit'
BINARY='llmfit'
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
  Linux) OS='unknown-linux-musl' ;;
  Darwin) OS='apple-darwin' ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac
case "$ARCH" in
  x86_64|amd64) ARCH='x86_64' ;;
  aarch64|arm64) ARCH='aarch64' ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac
PLATFORM="${ARCH}-${OS}"
TAG=$(curl -fsSI "https://github.com/${REPO}/releases/latest" | grep -i '^location:' | head -1 | sed 's|.*/tag/||' | tr -d '\r\n')
ASSET="${BINARY}-${TAG}-${PLATFORM}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
curl -fsSL "$URL" -o "$TMPDIR/$ASSET"
if curl -fsSL --max-time 10 "${URL}.sha256" -o "$TMPDIR/${ASSET}.sha256"; then
  (cd "$TMPDIR" && sha256sum -c "${ASSET}.sha256")
fi
tar -xzf "$TMPDIR/$ASSET" -C "$TMPDIR"
install -d "$HOME/.local/bin"
install -m 0755 "$TMPDIR/$BINARY" "$HOME/.local/bin/$BINARY"
export PATH="$HOME/.local/bin:$PATH"
llmfit --version
```

## Bootstrap flow

1. ensure `ollama`, `claude`, `codex` exist
2. install `llmfit`
3. pull the tiny local coder model if needed
4. run the repo doctor

```bash
ollama pull qwen2.5-coder:0.5b
ccl doctor
```

## What the doctor does

- verifies tool presence
- records a machine profile
- checks local Ollama models
- configures repo-local Ollama integration state for both Codex and Claude
- runs a real Codex -> Ollama smoke test when `--run-codex-smoke` is supplied
