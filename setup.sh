#!/usr/bin/env bash
# One-click guided setup: configures the web server (and optionally the HPC
# cluster), creates .venv, installs requirements.txt, runs preflight checks,
# and starts the server. All flags pass through to scripts/setup_instance.py;
# run scripts/setup_instance.py --help for the full list.
set -euo pipefail
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  exec python3 scripts/setup_instance.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python scripts/setup_instance.py "$@"
else
  echo "setup.sh: Python 3.11+ was not found." >&2
  echo "Install Python from https://www.python.org/downloads/ (or activate an" >&2
  echo "environment that provides 'python3'/'python'), then rerun ./setup.sh" >&2
  exit 1
fi
