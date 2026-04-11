#!/usr/bin/env bash
# ============================================================================
# claude-codex-local installer
#
# Downloads the repo, sets up a virtualenv, and runs the interactive wizard.
# No git clone required.
#
# Usage (interactive — recommended):
#   bash <(curl -sSL https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
#
# Or with wget:
#   bash <(wget -qO- https://raw.githubusercontent.com/luongnv89/claude-codex-local/main/install.sh)
#
# IMPORTANT: use the `bash <(...)` form, not `curl ... | bash`. The wizard is
# interactive and needs a real TTY on stdin — piping steals stdin.
#
# Environment overrides:
#   CCL_REPO         owner/repo              (default: luongnv89/claude-codex-local)
#   CCL_REF          branch/tag/sha          (default: main)
#   CCL_INSTALL_DIR  install target          (default: $HOME/.claude-codex-local-src)
#   CCL_PYTHON       python interpreter      (default: python3)
#   CCL_NO_RUN       if set, skip running the wizard after install
# ============================================================================
set -euo pipefail

CCL_REPO="${CCL_REPO:-luongnv89/claude-codex-local}"
CCL_REF="${CCL_REF:-main}"
CCL_INSTALL_DIR="${CCL_INSTALL_DIR:-$HOME/.claude-codex-local-src}"
CCL_PYTHON="${CCL_PYTHON:-python3}"

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
    ok "Package installed — `ccl` available at $venv/bin/ccl"
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
