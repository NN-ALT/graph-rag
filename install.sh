#!/usr/bin/env bash
# Graph RAG — macOS / Linux bootstrap installer
#
# Usage:
#   chmod +x install.sh && ./install.sh
#
# What this does:
#   1. Checks that Python 3.11+ is available
#   2. Installs Python via Homebrew (macOS) or apt (Linux) if missing
#   3. Hands off to install.py for the rest of the setup

set -euo pipefail

PYTHON=""

# ── Helpers ────────────────────────────────────────────────────────────────────

info()  { echo "  >>  $*"; }
ok()    { echo "  OK  $*"; }
warn()  { echo "  !!  $*"; }
die()   { echo ""; warn "$*"; exit 1; }

python_version_ok() {
    # Returns 0 if the given Python binary is 3.11 or newer
    local py="$1"
    "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null
}

find_python() {
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" &>/dev/null && python_version_ok "$candidate"; then
            PYTHON=$(command -v "$candidate")
            return 0
        fi
    done
    return 1
}

# ── Detect OS ─────────────────────────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macOS" ;;
    Linux)  PLATFORM="Linux" ;;
    *)      die "Unsupported platform: $OS. Use install.py directly on Windows." ;;
esac

echo ""
echo "============================================================"
echo "  Graph RAG — Bootstrap Installer ($PLATFORM)"
echo "============================================================"
echo ""

# ── Check / install Python ────────────────────────────────────────────────────

info "Looking for Python 3.11+..."

if find_python; then
    ok "Found: $PYTHON ($("$PYTHON" --version))"
else
    warn "Python 3.11+ not found."

    if [ "$PLATFORM" = "macOS" ]; then
        if ! command -v brew &>/dev/null; then
            die "Homebrew is required to install Python. Install it first:\n  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
        info "Installing Python 3 via Homebrew..."
        brew install python@3.13
        # Homebrew installs as python3.13 — find it
        if ! find_python; then
            die "Python install succeeded but binary not found. Restart your terminal and re-run."
        fi
        ok "Python installed: $PYTHON"

    elif [ "$PLATFORM" = "Linux" ]; then
        if ! command -v apt-get &>/dev/null; then
            die "Automatic Python install requires apt-get. Please install Python 3.11+ manually."
        fi
        info "Installing Python 3 via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv
        if ! find_python; then
            die "Python install succeeded but binary not found. Try: sudo apt-get install python3.11"
        fi
        ok "Python installed: $PYTHON"
    fi
fi

# ── Check pip ─────────────────────────────────────────────────────────────────

info "Checking pip..."
if ! "$PYTHON" -m pip --version &>/dev/null; then
    info "pip not found — installing..."
    if [ "$PLATFORM" = "macOS" ]; then
        brew install python@3.13 || true   # pip usually bundled
    else
        sudo apt-get install -y python3-pip
    fi
fi
ok "pip ready."

# ── Hand off to install.py ────────────────────────────────────────────────────

echo ""
info "Launching install.py with $PYTHON..."
echo ""

exec "$PYTHON" "$(dirname "$0")/install.py" "$@"
