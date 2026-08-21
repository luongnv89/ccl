#!/usr/bin/env bash
# ============================================================================
# claude-codex-local installer
#
# Downloads the repo, sets up a virtualenv, and runs the interactive wizard.
# No git clone required.
#
# Usage (interactive — recommended):
#   bash <(curl -sSL https://raw.githubusercontent.com/luongnv89/ccl/main/install.sh)
#
# Or with wget:
#   bash <(wget -qO- https://raw.githubusercontent.com/luongnv89/ccl/main/install.sh)
#
# IMPORTANT: use the `bash <(...)` form, not `curl ... | bash`. The wizard is
# interactive and needs a real TTY on stdin — piping steals stdin.
#
# Environment overrides:
#   CCL_REPO         owner/repo              (default: luongnv89/ccl)
#   CCL_REF          branch/tag/sha          (default: latest release tag)
#   CCL_INSTALL_DIR  install target          (default: $HOME/.claude-codex-local-src)
#   CCL_PYTHON       python interpreter      (default: python3)
#   CCL_NO_RUN       if set, skip running the wizard after install
# ============================================================================
set -euo pipefail

CCL_REPO="${CCL_REPO:-luongnv89/ccl}"
CCL_INSTALL_DIR="${CCL_INSTALL_DIR:-$HOME/.claude-codex-local-src}"
CCL_PYTHON="${CCL_PYTHON:-python3}"

# Last known-good release tag, used when the GitHub API cannot be reached to
# resolve the latest release (F-SEC-013: never default to the mutable main).
CCL_FALLBACK_REF="v0.17.0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ OK ]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERR ]${NC}  %s\n" "$*" >&2; }
die()   { err "$@"; exit 1; }

need() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

detect_os() {
    case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
        linux*)  echo "linux" ;;
        darwin*) echo "macos" ;;
        mingw*|msys*|cygwin*) echo "windows" ;;
        *) die "Unsupported OS: $(uname -s)" ;;
    esac
}

check_python() {
    if ! command -v "$CCL_PYTHON" >/dev/null 2>&1; then
        die "Python not found. Install Python 3.10+ and re-run, or set CCL_PYTHON=/path/to/python3."
    fi
    local ver
    ver="$("$CCL_PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    local major minor
    major="${ver%.*}"
    minor="${ver#*.}"
    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
        die "Python >= 3.10 required, found $ver at $(command -v "$CCL_PYTHON")."
    fi
    ok "Python $ver at $(command -v "$CCL_PYTHON")"
}

check_venv_module() {
    if ! "$CCL_PYTHON" -c 'import venv' >/dev/null 2>&1; then
        die "Python 'venv' module missing. On Debian/Ubuntu: sudo apt install python3-venv"
    fi
}

pick_downloader() {
    if command -v curl >/dev/null 2>&1; then
        echo "curl"
    elif command -v wget >/dev/null 2>&1; then
        echo "wget"
    else
        die "Neither curl nor wget found. Install one and retry."
    fi
}

download_tarball() {
    local url="$1" out="$2" dl="$3"
    info "Downloading $url"
    if [ "$dl" = "curl" ]; then
        curl -fsSL "$url" -o "$out" || die "Download failed: $url"
    else
        wget -qO "$out" "$url" || die "Download failed: $url"
    fi
}

# Quiet, non-fatal fetch: returns non-zero when the URL is unavailable.
fetch_to() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --max-time 30 "$1" -o "$2" 2>/dev/null
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$2" -T 30 "$1" 2>/dev/null
    else
        return 1
    fi
}

# Resolve the latest published release tag via the GitHub API.
latest_release_tag() {
    local api="https://api.github.com/repos/${CCL_REPO}/releases/latest" tag=""
    tag="$(fetch_to "$api" /dev/stdout | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)" || true
    printf '%s' "$tag"
}

# Pick CCL_REF: explicit override wins, else latest release tag, else the
# pinned fallback. Never defaults to a mutable branch.
resolve_ref() {
    if [ -n "${CCL_REF:-}" ]; then
        info "Using requested ref: $CCL_REF"
        return 0
    fi
    CCL_REF="$(latest_release_tag)"
    if [ -z "$CCL_REF" ]; then
        warn "Could not resolve latest release — falling back to pinned $CCL_FALLBACK_REF"
        CCL_REF="$CCL_FALLBACK_REF"
    fi
}

# Checksums recorded at release time for tags whose release does not (yet)
# publish a SHA256SUMS asset. Kept POSIX-safe (no associative arrays) so the
# installer also runs under macOS's stock bash 3.2.
pinned_sha256() {
    case "$1" in
        v0.17.0) echo "10a8a782e4edf84cd08a73f2735e402dcfeeabeb2a82b1783d5f9549058668c5" ;;
        *) return 1 ;;
    esac
}

tarball_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "No sha256 tool found (need sha256sum or shasum) — refusing unverified install."
    fi
}

# Verify the downloaded tarball against a checksum published with the release
# (SHA256SUMS asset), falling back to the pin table above. Fails closed.
verify_checksum() {
    local tarball="$1" expected actual sums_url sums_tmp
    actual="$(tarball_sha256 "$tarball")"
    sums_url="https://github.com/${CCL_REPO}/releases/download/${CCL_REF}/SHA256SUMS"

    sums_tmp="$(mktemp)"
    expected=""
    if fetch_to "$sums_url" "$sums_tmp"; then
        expected="$(awk 'NR==1{print $1}' "$sums_tmp")"
        rm -f "$sums_tmp"
        [ -n "$expected" ] || die "Checksum file for ${CCL_REF} is empty or malformed."
        info "Verifying tarball against published SHA256SUMS for ${CCL_REF}"
    elif pinned_sha256 "$CCL_REF" > /dev/null 2>&1; then
        expected="$(pinned_sha256 "$CCL_REF")"
        info "No SHA256SUMS asset for ${CCL_REF} — using checksum pinned in this installer."
    else
        rm -f "$sums_tmp"
        die "No published checksum for ${CCL_REF}; refusing unverified install. Set CCL_REF to a tagged release that publishes SHA256SUMS."
    fi

    if [ "$actual" != "$expected" ]; then
        die "Checksum mismatch for ${CCL_REF}: expected $expected, got $actual"
    fi
    ok "Tarball checksum verified ($actual)"
}

extract_repo() {
    local tarball="$1" target="$2"
    need tar
    mkdir -p "$target"
    # GitHub tarballs extract to <repo>-<ref>/ — strip that top-level dir.
    tar -xzf "$tarball" -C "$target" --strip-components=1 \
        || die "Failed to extract $tarball"
}

install_repo() {
    local dl tarball tmpdir url
    dl="$(pick_downloader)"
    url="https://codeload.github.com/${CCL_REPO}/tar.gz/${CCL_REF}"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    tarball="$tmpdir/repo.tar.gz"

    if [ -d "$CCL_INSTALL_DIR" ] && [ -n "$(ls -A "$CCL_INSTALL_DIR" 2>/dev/null || true)" ]; then
        warn "$CCL_INSTALL_DIR already exists and is not empty — it will be refreshed."
        rm -rf "$CCL_INSTALL_DIR"
    fi
    mkdir -p "$CCL_INSTALL_DIR"

    download_tarball "$url" "$tarball" "$dl"
    verify_checksum "$tarball"
    extract_repo "$tarball" "$CCL_INSTALL_DIR"
    ok "Repo extracted to $CCL_INSTALL_DIR"
}

setup_venv() {
    local venv="$CCL_INSTALL_DIR/.venv"
    info "Creating virtualenv at $venv"
    "$CCL_PYTHON" -m venv "$venv" || die "Failed to create virtualenv"

    local pip="$venv/bin/pip"
    [ -x "$pip" ] || die "pip not found in virtualenv"

    info "Upgrading pip"
    "$pip" install --quiet --upgrade pip || warn "pip upgrade failed, continuing"

    info "Installing claude-codex-local (editable)"
    "$pip" install --quiet -e "$CCL_INSTALL_DIR" \
        || die "Failed to install claude-codex-local"
    ok "Package installed — 'ccl' available at $venv/bin/ccl"
}

run_wizard() {
    local entry="$CCL_INSTALL_DIR/.venv/bin/ccl"
    [ -x "$entry" ] || die "ccl entry point missing: $entry"

    if [ -n "${CCL_NO_RUN:-}" ]; then
        info "CCL_NO_RUN set — skipping wizard."
        info "To run it later: $entry"
        return 0
    fi

    if [ ! -t 0 ]; then
        warn "stdin is not a TTY — the interactive wizard needs a terminal."
        warn "Run the installer with: bash <(curl -sSL <url>)   (not: curl | bash)"
        info "Install is complete. To run the wizard manually:"
        info "    $entry"
        return 0
    fi

    info "Launching interactive wizard (ccl)…"
    printf '\n'
    exec "$entry"
}

main() {
    resolve_ref
    info "claude-codex-local installer"
    info "repo=$CCL_REPO ref=$CCL_REF dir=$CCL_INSTALL_DIR"
    info "============================================"

    detect_os >/dev/null
    check_python
    check_venv_module
    need tar
    need mktemp

    install_repo
    setup_venv

    ok "Install complete."
    info "============================================"
    run_wizard
}

main "$@"
