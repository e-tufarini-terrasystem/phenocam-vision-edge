#!/bin/sh
# Configure one extracted Raspberry Pi runtime package.
# The script validates architecture and Python, creates the package-local
# virtual environment, installs runtime requirements, and creates runtime
# directories without privilege escalation.

set -u

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 1
requirements="$root/requirements/runtime.txt"
environment="$root/.venv"

if [ "$(uname -m 2>/dev/null)" != "aarch64" ]; then
    echo "error: Raspberry Pi aarch64 is required" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1 ||
   ! python3 -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 13))' >/dev/null 2>&1; then
    echo "error: Python 3.13 is required" >&2
    exit 1
fi

if ! python3 -c 'import venv' >/dev/null 2>&1; then
    echo "error: python3-venv is required" >&2
    exit 1
fi

if [ ! -f "$requirements" ] || [ -L "$requirements" ]; then
    echo "error: runtime requirements do not exist" >&2
    exit 1
fi

if [ -e "$environment" ] || [ -L "$environment" ]; then
    echo "error: virtual environment already exists" >&2
    exit 1
fi

if ! python3 -m venv "$environment" >/dev/null 2>&1; then
    echo "error: virtual environment could not be created" >&2
    exit 1
fi

if ! "$environment/bin/python" -m pip install --no-cache-dir -r "$requirements" >/dev/null 2>&1; then
    echo "error: runtime dependencies could not be installed" >&2
    exit 1
fi

if ! mkdir -p "$root/input" 2>/dev/null; then
    echo "error: input directory could not be created" >&2
    exit 1
fi

if ! mkdir -p "$root/output" 2>/dev/null; then
    echo "error: output directory could not be created" >&2
    exit 1
fi

echo "Installation complete"
