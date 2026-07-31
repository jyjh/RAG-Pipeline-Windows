#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
exec python3 scripts/setup_instance.py "$@"
