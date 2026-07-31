#!/usr/bin/env python3
"""Admin CLI for issuing and managing per-user API keys.

API keys gate mutating ``/api/*`` requests (uploads, reindex, delete, edit).
They are issued by an admin out-of-band and never through the web UI, so this
script is the only path that ever sees a key's plaintext secret -- and it
prints that secret exactly once, at creation time. The store keeps only the
``sha256`` hash; identify keys later by their ``prefix`` (``rag_…<last4>``).

The store lives at ``data/.api_keys.json`` (the same sidecar pattern as the
PDF registry). Cross-process safety is provided by ``portalocker``, so it is
safe to run this while the server is up.

Examples::

    # Issue a standard key (shown once, then never again):
    python scripts/manage_api_keys.py create --label "alice laptop"

    # Issue an admin key that never expires:
    python scripts/manage_api_keys.py create --label "ops" --role admin

    # Issue a key valid until end of 2026, with its own 120 req/min cap:
    python scripts/manage_api_keys.py create --label "ci" \\
        --expires 2026-12-31 --rate-limit 120

    # List every key with role/status/usage (secrets are never shown):
    python scripts/manage_api_keys.py list

    # Disable (revoke access without deleting the record) then later re-enable:
    python scripts/manage_api_keys.py disable rag_…a1b2
    python scripts/manage_api_keys.py enable  rag_…a1b2

    # Rotate (issue a replacement secret for the same record; old secret dies):
    python scripts/manage_api_keys.py rotate rag_…a1b2

    # Delete a record outright / reset its usage counter:
    python scripts/manage_api_keys.py delete rag_…a1b2
    python scripts/manage_api_keys.py reset-usage rag_…a1b2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put the repo root on sys.path so ``from src...`` works regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api_key_auth import (  # noqa: E402  (import after sys.path bootstrap)
    DEFAULT_KEY_PREFIX,
    ApiKeyAuthenticator,
    KeyStore,
    ROLE_ADMIN,
    ROLE_USER,
    STATUS_ACTIVE,
    STATUS_DISABLED,
)


def _store_path(data_dir: str | None) -> Path:
    from src.api_key_auth import STORE_FILENAME

    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    return Path(root) / STORE_FILENAME


def _build_authenticator(data_dir: str | None, rate_limit: int, persist_interval: int) -> ApiKeyAuthenticator:
    from src.api_key_auth import RATE_WINDOW_SECONDS

    store = KeyStore(_store_path(data_dir))
    return ApiKeyAuthenticator(
        store,
        default_rate_limit=rate_limit,
        persist_interval=persist_interval,
        window_seconds=RATE_WINDOW_SECONDS,
    )


def _resolve_prefix(auth: ApiKeyAuthenticator, prefix: str) -> str:
    """Resolve a prefix to the key's hash id, or exit with an error."""
    record = auth.store.find_by_prefix(prefix)
    if record is None:
        _fail(f"No API key found with prefix {prefix!r}.")
    return record["key_id"]


def _fail(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    full_key, record = auth.create_key(
        label=args.label,
        role=args.role,
        expires_at=args.expires,
        rate_limit_per_minute=args.rate_limit,
        prefix=DEFAULT_KEY_PREFIX,
    )
    print("API key created. Store this now -- it cannot be shown again.\n")
    print(f"  Secret : {full_key}")
    print(f"  Prefix : {record['prefix']}")
    print(f"  Label  : {record['label']}")
    print(f"  Role   : {record['role']}")
    print(f"  Expires: {record['expires_at'] or '(never)'}")
    if record.get("rate_limit_per_minute"):
        print(f"  Rate   : {record['rate_limit_per_minute']} req/min")
    print(
        "\nSend it on mutating requests via the header  X-API-Token: <secret>"
        "\nor the query param  ?token=<secret>."
    )
    return 0


def _format_row(record: dict) -> dict:
    usage = record.get("usage", {}) or {}
    return {
        "prefix": record.get("prefix", ""),
        "label": record.get("label", ""),
        "role": record.get("role", "user"),
        "status": record.get("status", "active"),
        "expires_at": record.get("expires_at") or "(never)",
        "requests": usage.get("requests", 0),
        "last_used_at": usage.get("last_used_at") or "(never)",
    }


def cmd_list(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    records = auth.store.list_keys()
    if not records:
        print("No API keys configured.")
        print("\nCreate one with:  python scripts/manage_api_keys.py create --label \"...\"")
        return 0
    rows = [_format_row(r) for r in records]
    # Order: admins first, then by label, for easy scanning.
    rows.sort(key=lambda r: (r["role"] != "admin", r["label"].lower(), r["prefix"]))
    headers = ["Prefix", "Label", "Role", "Status", "Expires", "Reqs", "Last used"]
    width = {
        "prefix": max(len(headers[0]), max(len(r["prefix"]) for r in rows)),
        "label": max(len(headers[1]), max(len(r["label"]) for r in rows) + 1, 8),
        "role": max(len(headers[2]), max(len(r["role"]) for r in rows)),
        "status": max(len(headers[3]), max(len(r["status"]) for r in rows)),
        "expires_at": max(len(headers[4]), max(len(r["expires_at"]) for r in rows)),
        "requests": max(len(headers[5]), max(len(str(r["requests"])) for r in rows)),
    }
    fmt = (
        "{prefix:<{pw}}  {label:<{lw}}  {role:<{rw}}  {status:<{sw}}  "
        "{expires_at:<{ew}}  {requests:>{kw}}  {last}"
    )
    print(
        fmt.format(
            prefix=headers[0], label=headers[1], role=headers[2], status=headers[3],
            expires_at=headers[4], requests=headers[5], last=headers[6],
            pw=width["prefix"], lw=width["label"], rw=width["role"],
            sw=width["status"], ew=width["expires_at"], kw=width["requests"],
        )
    )
    print("-" * (sum(width.values()) + 30))
    for r in rows:
        print(
            fmt.format(
                prefix=r["prefix"], label=r["label"], role=r["role"], status=r["status"],
                expires_at=r["expires_at"], requests=r["requests"], last=r["last_used_at"],
                pw=width["prefix"], lw=width["label"], rw=width["role"],
                sw=width["status"], ew=width["expires_at"], kw=width["requests"],
            )
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    record = auth.store.find_by_prefix(args.prefix)
    if record is None:
        _fail(f"No API key found with prefix {args.prefix!r}.")
    row = _format_row(record)
    print(f"Prefix  : {row['prefix']}")
    print(f"Label   : {row['label']}")
    print(f"Role    : {row['role']}")
    print(f"Status  : {row['status']}")
    print(f"Created : {record.get('created_at', '(unknown)')}")
    print(f"Expires : {row['expires_at']}")
    if record.get("rate_limit_per_minute"):
        print(f"Rate    : {record['rate_limit_per_minute']} req/min")
    print(f"Requests: {row['requests']}")
    print(f"Last use: {row['last_used_at']}")
    return 0


def _set_status(args: argparse.Namespace, status: str) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    key_hash = _resolve_prefix(auth, args.prefix)
    auth.set_status(key_hash, status)
    verb = "Disabled" if status == STATUS_DISABLED else "Enabled"
    print(f"{verb} API key {args.prefix}.")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    return _set_status(args, STATUS_DISABLED)


def cmd_enable(args: argparse.Namespace) -> int:
    return _set_status(args, STATUS_ACTIVE)


def cmd_delete(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    key_hash = _resolve_prefix(auth, args.prefix)
    if not args.yes:
        confirm = input(f"Permanently delete API key {args.prefix}? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1
    auth.delete_key(key_hash)
    print(f"Deleted API key {args.prefix}.")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    """Issue a fresh secret for an existing record and drop the old one.

    The label/role/expiry/rate-limit are preserved; only the secret (and its
    hash/prefix) change. The old secret is removed and the new one added in a
    SINGLE locked write, so a crash mid-rotate cannot leave both active.
    """
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    record = auth.store.find_by_prefix(args.prefix)
    if record is None:
        _fail(f"No API key found with prefix {args.prefix!r}.")
    result = auth.rotate_key(record["key_id"])
    if result is None:
        _fail(f"No API key found with prefix {args.prefix!r}.")
    full_key, new_record = result
    print("API key rotated. The old secret no longer works. Store the new one now.\n")
    print(f"  Secret : {full_key}")
    print(f"  Prefix : {new_record['prefix']}  (was {args.prefix})")
    return 0


def cmd_reset_usage(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    key_hash = _resolve_prefix(auth, args.prefix)
    auth.reset_usage(key_hash)
    print(f"Reset usage counter for API key {args.prefix}.")
    return 0


def cmd_set_role(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    key_hash = _resolve_prefix(auth, args.prefix)
    auth.set_role(key_hash, args.role)
    print(f"Set role of {args.prefix} to {args.role}.")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_api_keys",
        description="Issue and manage per-user API keys for the RAG pipeline web app.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data directory containing .api_keys.json (default: ./data).",
    )
    parser.add_argument(
        "--default-rate-limit",
        type=int,
        default=60,
        help="Default requests/minute shown in help text (does not reconfigure the server).",
    )
    parser.add_argument(
        "--persist-interval",
        type=int,
        default=50,
        help="Usage-flush throttle (server-side knob; irrelevant for this CLI).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Issue a new API key (secret shown once).")
    p_create.add_argument("--label", required=True, help="Human-readable label, e.g. 'alice laptop'.")
    p_create.add_argument("--role", choices=[ROLE_USER, ROLE_ADMIN], default=ROLE_USER, help="Key role.")
    p_create.add_argument(
        "--expires", default=None, help="Expiry as YYYY-MM-DD or ISO timestamp (blank = never)."
    )
    p_create.add_argument(
        "--rate-limit", type=int, default=None, help="Per-key requests/minute override."
    )
    p_create.set_defaults(func=cmd_create)

    p_list = sub.add_parser("list", help="List all keys (secrets are never shown).")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show details for one key (by prefix).")
    p_show.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_show.set_defaults(func=cmd_show)

    for name, func, help_text in [
        ("disable", cmd_disable, "Revoke access for a key without deleting it."),
        ("enable", cmd_enable, "Re-enable a previously disabled key."),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
        p.set_defaults(func=func)

    p_del = sub.add_parser("delete", help="Permanently delete a key record.")
    p_del.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_del.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    p_del.set_defaults(func=cmd_delete)

    p_rot = sub.add_parser("rotate", help="Issue a fresh secret for an existing key.")
    p_rot.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_rot.set_defaults(func=cmd_rotate)

    p_reset = sub.add_parser("reset-usage", help="Reset the usage counter for a key.")
    p_reset.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_reset.set_defaults(func=cmd_reset_usage)

    p_role = sub.add_parser("set-role", help="Change a key's role (user/admin).")
    p_role.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_role.add_argument("--role", choices=[ROLE_USER, ROLE_ADMIN], required=True)
    p_role.set_defaults(func=cmd_set_role)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # Surface a clean message instead of a traceback.
        _fail(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
