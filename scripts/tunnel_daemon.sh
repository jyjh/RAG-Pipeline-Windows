#!/bin/bash
# ==============================================================================
# SSH Tunnel Daemon (Linux / Bash)
# Maintains an auto-reconnecting SSH port-forwarding tunnel to NUS HPC for Ollama.
#
# Two modes, selected by what you pass:
#
# 1. Single-hop (default): forward a local port straight to an SSH host that
#    runs Ollama itself. Use this when Ollama is reachable on a login/dev node.
#
# 2. Two-hop (GPU compute node): pass --jump-host <login> --host-file <path>.
#    The daemon sshes the jump host to read the live compute-node name from
#    --host-file (written by scripts/nus_hpc_serve.pbs), then forwards to it
#    via ProxyJump. Re-reads the file on every reconnect so a new serving job
#    on a different node is picked up automatically. Use this when Ollama runs
#    inside a PBS job on a private-network GPU node.
# ==============================================================================

set -e

# Default configuration from environment variables or fallback defaults
HPC_USER="${HPC_USER:-${USER:-$(whoami 2>/dev/null || echo "user")}}"
HPC_HOST="${HPC_HOST:-nus_hpc_gpu}"
LOCAL_PORT="${LOCAL_PORT:-11434}"
REMOTE_PORT="${REMOTE_PORT:-11434}"
SSH_KEY="${SSH_KEY:-}"
RECONNECT_DELAY="${RECONNECT_DELAY:-5}"
JUMP_HOST="${JUMP_HOST:-}"
HOST_FILE="${HOST_FILE:-}"

show_help() {
    cat << 'EOF'
Usage: tunnel_daemon.sh [OPTIONS] [USER] [HOST] [LOCAL_PORT] [REMOTE_PORT] [KEY]

Cross-platform SSH tunnel daemon with auto-reconnect logic for Ollama port forwarding.

Options:
  -u, --user USER         HPC SSH username (default: $HPC_USER or current user)
  -h, --host HOST         HPC target host/hostname (default: $HPC_HOST or 'nus_hpc_gpu')
  -l, --local-port PORT   Local listening port (default: $LOCAL_PORT or 11434)
  -r, --remote-port PORT  Remote target port (default: $REMOTE_PORT or 11434)
  -i, --key PATH          Path to SSH private key file (optional)
  -j, --jump-host HOST    Login/jump host for 2-hop mode (ProxyJump). Enables
                          two-hop forwarding to a GPU compute node.
  -f, --host-file PATH    Discovery file (on the jump host) written by the
                          serving PBS job, containing the live compute hostname.
                          Recomputed on each reconnect. Requires --jump-host.
  --help                  Show this help message and exit

Positional Arguments:
  1: USER                 HPC SSH username
  2: HOST                 HPC target host
  3: LOCAL_PORT           Local listening port
  4: REMOTE_PORT          Remote target port
  5: KEY                  Path to SSH private key

Environment Variables:
  HPC_USER, HPC_HOST, LOCAL_PORT, REMOTE_PORT, SSH_KEY, RECONNECT_DELAY,
  JUMP_HOST, HOST_FILE

Examples:
  # Single-hop to a login host running Ollama:
  ./tunnel_daemon.sh -h nus_hpc

  # Two-hop to a GPU compute node (Ollama runs in scripts/nus_hpc_serve.pbs):
  ./tunnel_daemon.sh -j nus_hpc -f ~/.rag_ollama_serving_host
EOF
}

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--user)
            HPC_USER="$2"
            shift 2
            ;;
        -h|--host)
            HPC_HOST="$2"
            shift 2
            ;;
        -l|--local-port)
            LOCAL_PORT="$2"
            shift 2
            ;;
        -r|--remote-port)
            REMOTE_PORT="$2"
            shift 2
            ;;
        -i|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -j|--jump-host)
            JUMP_HOST="$2"
            shift 2
            ;;
        -f|--host-file)
            HOST_FILE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    [ -n "${POSITIONAL_ARGS[0]}" ] && HPC_USER="${POSITIONAL_ARGS[0]}"
    [ -n "${POSITIONAL_ARGS[1]}" ] && HPC_HOST="${POSITIONAL_ARGS[1]}"
    [ -n "${POSITIONAL_ARGS[2]}" ] && LOCAL_PORT="${POSITIONAL_ARGS[2]}"
    [ -n "${POSITIONAL_ARGS[3]}" ] && REMOTE_PORT="${POSITIONAL_ARGS[3]}"
    [ -n "${POSITIONAL_ARGS[4]}" ] && SSH_KEY="${POSITIONAL_ARGS[4]}"
fi

SSH_PID=""

cleanup() {
    echo "[$(date)] Stopping SSH tunnel daemon..."
    trap - SIGINT SIGTERM EXIT
    if [ -n "$SSH_PID" ] && kill -0 "$SSH_PID" 2>/dev/null; then
        kill "$SSH_PID" 2>/dev/null || true
        wait "$SSH_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Resolve the current tunnel target. In 2-hop mode this sshes the jump host to
# read the discovery file (so a freshly-rescheduled serving job on a different
# node is picked up on reconnect). In single-hop mode it just echoes $HPC_HOST.
resolve_target() {
    if [ -n "${JUMP_HOST}" ]; then
        if [ -z "${HOST_FILE}" ]; then
            echo "tunnel_daemon: --host-file is required when --jump-host is set" >&2
            return 1
        fi
        local discovered
        discovered=$(ssh -o ExitOnForwardFailure=no "${JUMP_HOST}" "cat '${HOST_FILE}' 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
        if [ -z "${discovered}" ]; then
            echo "tunnel_daemon: discovery file '${HOST_FILE}' on '${JUMP_HOST}' is empty or missing" >&2
            echo "              (is scripts/nus_hpc_serve.pbs running? It writes this file at startup.)" >&2
            return 1
        fi
        echo "${discovered}"
    else
        echo "${HPC_HOST}"
    fi
}

echo "[$(date)] Starting SSH tunnel daemon..."
echo "[$(date)] User: ${HPC_USER}"
if [ -n "${JUMP_HOST}" ]; then
    echo "[$(date)] Mode: 2-hop (jump='${JUMP_HOST}', host-file='${HOST_FILE}')"
else
    echo "[$(date)] Mode: single-hop (target='${HPC_HOST}')"
fi
echo "[$(date)] Port Forwarding: ${LOCAL_PORT} -> localhost:${REMOTE_PORT}"

while true; do
    TARGET_HOST="$(resolve_target)" || {
        echo "[$(date)] Could not resolve target; retrying in ${RECONNECT_DELAY}s..."
        sleep "${RECONNECT_DELAY}"
        continue
    }

    SSH_CMD=(ssh -N -T -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3)

    if [ -n "${JUMP_HOST}" ]; then
        SSH_CMD+=(-J "${JUMP_HOST}")
    fi
    if [ -n "${SSH_KEY}" ]; then
        SSH_CMD+=(-i "$SSH_KEY")
    fi
    SSH_CMD+=("${HPC_USER}@${TARGET_HOST}")

    echo "[$(date)] Establishing SSH tunnel to ${HPC_USER}@${TARGET_HOST}..."
    "${SSH_CMD[@]}" &
    SSH_PID=$!

    wait "$SSH_PID" 2>/dev/null || true
    SSH_PID=""

    echo "[$(date)] Tunnel disconnected. Reconnecting in ${RECONNECT_DELAY}s..."
    sleep "$RECONNECT_DELAY"
done
