"""Make a client API key and the hashed entry the gateway stores for it.

    python scripts/make_api_key.py web                            # prints the key once, and its entry
    python scripts/make_api_key.py web --append config/api-keys    # also appends the entry to a keys file

The key (``gw_`` + 32 random bytes, URL-safe) goes to the client; the gateway only needs ``web:sha256:<hex>``, in
$GATEWAY_API_KEYS or the file named by $GATEWAY_API_KEYS_FILE. Rotation: add the new entry next to the old one,
restart the replicas one at a time, move the client to the new key, then delete the old entry and restart again.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import secrets
import sys
from pathlib import Path


def make_key(app_id: str) -> tuple[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", app_id):
        raise ValueError("app id must be 1-64 letters, digits, '.', '_' or '-'")
    key = "gw_" + secrets.token_urlsafe(32)
    return key, f"{app_id}:sha256:{hashlib.sha256(key.encode()).hexdigest()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("app_id")
    parser.add_argument("--append", help="keys file to append the hashed entry to")
    args = parser.parse_args(argv)
    key, entry = make_key(args.app_id)
    if args.append:
        path = Path(args.append)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry + "\n")
        print(f"appended the entry for {args.app_id!r} to {path}")
    else:
        print(f"entry for the gateway: {entry}")
    print(f"key for the client (shown once, not stored): {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
