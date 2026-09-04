#!/usr/bin/env python3
"""Admin CLI for issuing and managing per-user API keys and permission sets.

API keys authenticate remote clients (the local machine is auto-authenticated
as admin). Each key is assigned a permission set that restricts category
access/visibility and carries ``can_write``/``admin`` flags. Keys are issued by
an admin out-of-band; this script is one of the paths that ever sees a key's
plaintext secret -- and it prints that secret exactly once, at creation time.
The store keeps only the ``sha256`` hash; identify keys later by their
``prefix`` (``rag_…<last4>``).

The store lives at ``data/.api_keys.json`` (the same sidecar pattern as the
PDF registry). Cross-process safety is provided by ``portalocker``, so it is
safe to run this while the server is up.

Examples::

    # Issue a standard key (shown once, then never again):
    python scripts/manage_api_keys.py create --label "alice laptop"

    # Issue a key scoped to a custom permission set:
    python scripts/manage_api_keys.py create --label "team-a" \\
        --permission-set team-a-read-only

    # Issue an admin key that never expires:
    python scripts/manage_api_keys.py create --label "ops" --role admin

    # Create a read-only set that can only see two categories:
    python scripts/manage_api_keys.py create-set team-a-read-only \\
        --label "Team A (read only)" --categories general,team-a --read-only

    # Every category, write-capable, no admin powers:
    python scripts/manage_api_keys.py create-set contributors --all-categories

    # List every key with set/status/usage (secrets are never shown):
    python scripts/manage_api_keys.py list
    python scripts/manage_api_keys.py list-sets

    # Adjust a set (applies to all of its keys immediately):
    python scripts/manage_api_keys.py update-set team-a-read-only \\
        --categories general --writable

    # Delete a set (refused while keys still reference it):
    python scripts/manage_api_keys.py delete-set team-a-read-only

    # Rotate (issue a replacement secret for the same record; old secret dies):
    python scripts/manage_api_keys.py rotate rag_…a1b2
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
    if args.permission_set:
        full_key, record = auth.create_key(
            label=args.label,
            expires_at=args.expires,
            rate_limit_per_minute=args.rate_limit,
            prefix=DEFAULT_KEY_PREFIX,
            permission_set=args.permission_set,
        )
    else:
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
    print(f"  Set    : {record.get('permission_set', '(legacy role)')}")
    print(f"  Role   : {record['role']}")
    print(f"  Expires: {record['expires_at'] or '(never)'}")
    if record.get("rate_limit_per_minute"):
        print(f"  Rate   : {record['rate_limit_per_minute']} req/min")
    print(
        "\nSend it on gated requests via the header  X-API-Token: <secret>"
        "\nor the query param  ?token=<secret>."
    )
    return 0


def _format_row(record: dict) -> dict:
    usage = record.get("usage", {}) or {}
    return {
        "prefix": record.get("prefix", ""),
        "label": record.get("label", ""),
        "permset": record.get("permission_set", ""),
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
    headers = ["Prefix", "Label", "Set", "Role", "Status", "Expires", "Reqs", "Last used"]
    width = {
        "prefix": max(len(headers[0]), max(len(r["prefix"]) for r in rows)),
        "label": max(len(headers[1]), max(len(r["label"]) for r in rows) + 1, 8),
        "permset": max(len(headers[2]), max(len(r["permset"]) for r in rows), 5),
        "role": max(len(headers[3]), max(len(r["role"]) for r in rows)),
        "status": max(len(headers[4]), max(len(r["status"]) for r in rows)),
        "expires_at": max(len(headers[5]), max(len(r["expires_at"]) for r in rows)),
        "requests": max(len(headers[6]), max(len(str(r["requests"])) for r in rows)),
    }
    fmt = (
        "{prefix:<{pw}}  {label:<{lw}}  {permset:<{gw}}  {role:<{rw}}  {status:<{sw}}  "
        "{expires_at:<{ew}}  {requests:>{kw}}  {last}"
    )
    print(
        fmt.format(
            prefix=headers[0], label=headers[1], permset=headers[2], role=headers[3],
            status=headers[4], expires_at=headers[5], requests=headers[6], last=headers[7],
            pw=width["prefix"], lw=width["label"], gw=width["permset"], rw=width["role"],
            sw=width["status"], ew=width["expires_at"], kw=width["requests"],
        )
    )
    print("-" * (sum(width.values()) + 40))
    for r in rows:
        print(
            fmt.format(
                prefix=r["prefix"], label=r["label"], permset=r["permset"], role=r["role"],
                status=r["status"], expires_at=r["expires_at"], requests=r["requests"],
                last=r["last_used_at"],
                pw=width["prefix"], lw=width["label"], gw=width["permset"], rw=width["role"],
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
    print(f"Set     : {row['permset'] or '(legacy role)'}")
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


def cmd_set_permission(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    key_hash = _resolve_prefix(auth, args.prefix)
    try:
        updated = auth.set_permission_set(key_hash, args.permission_set)
    except ValueError as exc:
        _fail(f"Error: {exc}")
        return 1
    if updated is None:
        _fail(f"No API key found with prefix {args.prefix!r}.")
        return 1
    print(
        f"Assigned {args.prefix} to permission set {args.permission_set} "
        f"(role now {updated.get('role')})."
    )
    return 0


def cmd_create_set(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    categories = "*" if args.all_categories else args.categories
    try:
        record = auth.create_permission_set(
            name=args.name,
            label=args.label,
            categories=categories,
            can_write=not args.read_only,
            admin=args.admin,
        )
    except ValueError as exc:
        _fail(f"Error: {exc}")
        return 1
    categories_text = "all" if record["categories"] == ["*"] else ", ".join(record["categories"])
    print(f"Permission set created: {record['name']}")
    print(f"  Label     : {record['label']}")
    print(f"  Categories: {categories_text}")
    print(f"  Write     : {'yes' if record['can_write'] else 'no (read-only)'}")
    print(f"  Admin     : {'yes' if record['admin'] else 'no'}")
    print(
        "\nAssign a key with:  "
        f"python scripts/manage_api_keys.py create --label \"...\" --permission-set {record['name']}"
    )
    return 0


def cmd_list_sets(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    sets = auth.list_permission_sets()
    if not sets:
        print("No permission sets configured.")
        return 0
    counts = auth.permission_set_key_counts()
    headers = ["Name", "Label", "Categories", "Write", "Admin", "Keys"]
    rows = [
        {
            "name": str(s.get("name", "")),
            "label": str(s.get("label", "")),
            "categories": "all" if s.get("categories") == ["*"] else ", ".join(s.get("categories") or []),
            "write": "yes" if s.get("can_write") else "no",
            "admin": "yes" if s.get("admin") else "no",
            "keys": str(counts.get(str(s.get("name")), 0)),
            "builtin": bool(s.get("builtin")),
        }
        for s in sets
    ]
    width = {
        "name": max(len(headers[0]), max(len(r["name"]) for r in rows)),
        "label": max(len(headers[1]), max(len(r["label"]) for r in rows)),
        "categories": max(len(headers[2]), max(len(r["categories"]) for r in rows), 10),
    }
    fmt = "{name:<{nw}}  {label:<{lw}}  {categories:<{cw}}  {write:<5}  {admin:<5}  {keys}"
    print(fmt.format(name=headers[0], label=headers[1], categories=headers[2],
                     write=headers[3], admin=headers[4], keys=headers[5],
                     nw=width["name"], lw=width["label"], cw=width["categories"]))
    print("-" * (sum(width.values()) + 30))
    for r in rows:
        suffix = "  (built-in)" if r["builtin"] else ""
        print(fmt.format(name=r["name"], label=r["label"], categories=r["categories"],
                         write=r["write"], admin=r["admin"], keys=r["keys"],
                         nw=width["name"], lw=width["label"], cw=width["categories"]) + suffix)
    return 0


def cmd_update_set(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    mutate: dict = {}
    if args.label is not None:
        mutate["label"] = args.label
    if args.all_categories:
        mutate["categories"] = "*"
    elif args.categories is not None:
        mutate["categories"] = args.categories
    if args.admin is not None:
        mutate["admin"] = args.admin
    if args.writable is not None:
        mutate["can_write"] = args.writable
    if not mutate:
        _fail("Nothing to update: pass --label, --categories, --all-categories, --admin/--no-admin, "
              "--writable/--read-only.")
        return 1
    try:
        record = auth.update_permission_set(args.name, mutate)
    except ValueError as exc:
        _fail(f"Error: {exc}")
        return 1
    if record is None:
        _fail(f"No permission set named {args.name!r}.")
        return 1
    categories_text = "all" if record["categories"] == ["*"] else ", ".join(record["categories"])
    print(f"Updated permission set {record['name']}: categories={categories_text}, "
          f"write={record['can_write']}, admin={record['admin']}.")
    return 0


def cmd_delete_set(args: argparse.Namespace) -> int:
    auth = _build_authenticator(args.data_dir, args.default_rate_limit, args.persist_interval)
    try:
        deleted = auth.delete_permission_set(args.name)
    except ValueError as exc:
        _fail(f"Error: {exc}")
        return 1
    if not deleted:
        _fail(f"No permission set named {args.name!r}.")
        return 1
    print(f"Deleted permission set {args.name}.")
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
    p_create.add_argument("--role", choices=[ROLE_USER, ROLE_ADMIN], default=ROLE_USER, help="Legacy two-value role (ignored when --permission-set is given).")
    p_create.add_argument(
        "--permission-set",
        default=None,
        help="Assign the key to a named permission set (see create-set / list-sets).",
    )
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

    p_role = sub.add_parser("set-role", help="Change a key's legacy role (user/admin) via the builtin sets.")
    p_role.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_role.add_argument("--role", choices=[ROLE_USER, ROLE_ADMIN], required=True)
    p_role.set_defaults(func=cmd_set_role)

    p_perm = sub.add_parser("set-permission", help="Assign a key to a permission set.")
    p_perm.add_argument("prefix", help="Key prefix, e.g. rag_…a1b2.")
    p_perm.add_argument("--permission-set", required=True, help="Permission set name (see list-sets).")
    p_perm.set_defaults(func=cmd_set_permission)

    p_cset = sub.add_parser("create-set", help="Create a permission set.")
    p_cset.add_argument("name", help="Short slug name, e.g. team-a-read-only.")
    p_cset.add_argument("--label", default="", help="Display label (defaults to the name).")
    p_cset.add_argument("--categories", default=None, help="Comma-separated category keys, e.g. 'general,team-a'.")
    p_cset.add_argument("--all-categories", action="store_true", help="Allow every category, including ones created later (default).")
    p_cset.add_argument("--read-only", action="store_true", help="Forbid uploads/edits for keys in this set.")
    p_cset.add_argument("--admin", action="store_true", help="Grant admin powers (key/set management, maintenance).")
    p_cset.set_defaults(func=cmd_create_set)

    p_lset = sub.add_parser("list-sets", help="List permission sets with key counts.")
    p_lset.set_defaults(func=cmd_list_sets)

    p_uset = sub.add_parser("update-set", help="Update a permission set's label/categories/flags.")
    p_uset.add_argument("name", help="Permission set name.")
    p_uset.add_argument("--label", default=None, help="New display label.")
    p_uset.add_argument("--categories", default=None, help="Comma-separated category keys.")
    p_uset.add_argument("--all-categories", action="store_true", help="Allow every category.")
    p_uset.add_argument("--admin", dest="admin", action="store_true", default=None, help="Grant admin powers.")
    p_uset.add_argument("--no-admin", dest="admin", action="store_false", help="Revoke admin powers.")
    p_uset.add_argument("--writable", dest="writable", action="store_true", default=None, help="Allow uploads/edits.")
    p_uset.add_argument("--read-only", dest="writable", action="store_false", help="Forbid uploads/edits.")
    p_uset.set_defaults(func=cmd_update_set)

    p_dset = sub.add_parser("delete-set", help="Delete a custom permission set (refused while keys reference it).")
    p_dset.add_argument("name", help="Permission set name.")
    p_dset.set_defaults(func=cmd_delete_set)

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
