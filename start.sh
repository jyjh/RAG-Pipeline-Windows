#!/usr/bin/env bash
# Start the RAG instance with no setup prompts: provisions each HPC cluster
# only if its deployed source is stale (or the SIF is missing), then launches
# the auto-reconnecting tunnel (HPC mode) and the web server.
#
# Run a full interactive setup (SSH keys, config, prompts) with ./setup.sh
# instead. To force a redeploy of the remote clusters, run:
#   ./start.sh --provision-hpc
set -euo pipefail
cd "$(dirname "$0")"
exec python3 scripts/setup_instance.py \
    --non-interactive \
    --start \
    --provision-if-needed \
    "$@"
