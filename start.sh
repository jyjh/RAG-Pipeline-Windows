#!/usr/bin/env bash
# Start the RAG instance with no setup prompts: provisions each HPC cluster
# only if its deployed source is stale (or the SIF is missing), then launches
# the web server. Missing Python dependencies are installed automatically.
#
# Run a full interactive setup (SSH keys, config, prompts) with ./setup.sh
# instead. To force a redeploy of the remote clusters, run:
#   ./start.sh --provision-hpc
set -euo pipefail
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "start.sh: Python 3.11+ was not found. Install Python, then rerun ./start.sh" >&2
  echo "(or run ./setup.sh for the full guided setup)." >&2
  exit 1
fi
exec "$PY" scripts/setup_instance.py \
    --non-interactive \
    --start \
    --provision-if-needed \
    "$@"
