"""Manage releases from the command line (works on the database directly).

    python -m gateway.manage status
    python -m gateway.manage activate v2-gemma --note "promote after canary"
    python -m gateway.manage rollback
    python -m gateway.manage canary v2-gemma 0.1
    python -m gateway.manage canary-off
    python -m gateway.manage history
"""

from __future__ import annotations

import argparse
import sys

from gateway.config import load_config, resolve_path
from gateway.manage.releases import ReleaseError, ReleaseManager
from gateway.monitoring.store import Store


def describe(manager: ReleaseManager) -> str:
    state = manager.state()
    lines = [f"active:  {state.active}  (from {state.source}{', ' + state.ts if state.ts else ''})"]
    release = manager.config.releases[state.active]
    lines.append(f"         cheap={release.cheap} strong={release.strong} prompt={release.prompt}")
    if release.batch:
        lines.append(f"         batch: cheap={release.batch.cheap} strong={release.batch.strong}")
    if state.canary:
        canary = manager.config.releases[state.canary]
        lines.append(f"canary:  {state.canary} on {state.canary_share:.0%} of traffic")
        lines.append(f"         cheap={canary.cheap} strong={canary.strong} prompt={canary.prompt}")
    else:
        lines.append("canary:  none")
    lines.append("releases: " + ", ".join(manager.config.releases))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gateway.manage", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default: $GATEWAY_CONFIG or config/gateway.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show the active release and canary")
    activate = sub.add_parser("activate", help="make a release active")
    activate.add_argument("release")
    activate.add_argument("--note", default="")
    rollback = sub.add_parser("rollback", help="re-activate the previous release and stop the canary")
    rollback.add_argument("--note", default="")
    canary = sub.add_parser("canary", help="send a share of traffic to a release")
    canary.add_argument("release")
    canary.add_argument("share", type=float, help="0-1, e.g. 0.1 for 10%%")
    canary.add_argument("--note", default="")
    canary_off = sub.add_parser("canary-off", help="stop the canary")
    canary_off.add_argument("--note", default="")
    history = sub.add_parser("history", help="deployment log, newest first")
    history.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    store = Store(resolve_path(config.storage.db_path))
    manager = ReleaseManager(config, store)
    try:
        if args.command == "activate":
            manager.activate(args.release, args.note)
        elif args.command == "rollback":
            manager.rollback(args.note)
        elif args.command == "canary":
            manager.set_canary(args.release, args.share, args.note)
        elif args.command == "canary-off":
            manager.clear_canary(args.note)
        elif args.command == "history":
            rows = manager.history(args.limit)
            if not rows:
                print("no deployment changes yet (state comes from the config)")
            for row in rows:
                canary = f" canary={row['canary_release']}@{row['canary_share']:.0%}" if row["canary_release"] else ""
                note = f"  # {row['note']}" if row["note"] else ""
                print(f"{row['ts']}  {row['action']:<10} active={row['active']}{canary}{note}")
            return 0
        print(describe(manager))
        return 0
    except ReleaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
