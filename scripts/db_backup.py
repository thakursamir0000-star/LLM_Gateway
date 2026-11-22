"""Back up and restore the gateway database, and prove a backup restores (the drill).

    python scripts/db_backup.py backup                      # into var/backups/ (git-ignored)
    python scripts/db_backup.py restore var/backups/gateway-....dump --target gateway_restored --yes
    python scripts/db_backup.py drill                       # backup, restore into a scratch database, compare

PostgreSQL (the Compose ``postgres`` service, or ``--local`` for pg_dump/pg_restore on this machine with
$GATEWAY_DATABASE_URL): custom-format ``pg_dump`` archives. SQLite (development, $GATEWAY_DB): the online backup
API, so the gateway can keep running. Restoring over the live database needs ``--target`` set to its own name and
``--overwrite-live`` as well as ``--yes``; by default a restore goes into a new database next to it.

Backups hold request metadata and, with prompt logging off, no prompt text; they are still sensitive (app ids,
costs, feedback comments): keep them encrypted at rest and off the repository (var/ is git-ignored).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import database_target, load_config  # noqa: E402

TABLES = ("requests", "calls", "feedback", "deployments", "alerts", "budget_reservations")
DRILL_DB = "gateway_restore_drill"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Postgres:
    """pg_dump / pg_restore / psql, inside the Compose postgres container or on this machine."""

    def __init__(self, url: str, local: bool, compose: list[str]):
        from sqlalchemy.engine import make_url

        parsed = make_url(url)
        self.user, self.database = parsed.username or "postgres", parsed.database or "postgres"
        self.local, self.compose = local, compose
        self.env = {**os.environ, "PGPASSWORD": parsed.password or ""} if local else dict(os.environ)
        self.host_args = (["-h", parsed.host or "localhost", "-p", str(parsed.port or 5432)] if local else [])

    def run(self, tool: str, *args: str, stdin: bytes | None = None, check: bool = True) -> bytes:
        command = [tool, *self.host_args, "-U", self.user, *args]
        if not self.local:
            command = [*self.compose, "exec", "-T", "postgres", *command]
        result = subprocess.run(command, input=stdin, capture_output=True, env=self.env)
        if check and result.returncode != 0:
            raise SystemExit(f"{tool} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
        return result.stdout

    def sql(self, statement: str, database: str = "postgres") -> str:
        return self.run("psql", "-d", database, "-v", "ON_ERROR_STOP=1", "-tAc", statement).decode().strip()

    def backup(self, out: Path) -> Path:
        path = out / f"gateway-{stamp()}.dump"
        path.write_bytes(self.run("pg_dump", "-d", self.database, "-Fc", "--no-owner"))
        return path

    def restore(self, archive: Path, target: str, clean: bool) -> None:
        if not clean:
            self.sql(f'CREATE DATABASE "{target}"')
        args = ["-d", target, "--no-owner", "--exit-on-error"] + (["--clean", "--if-exists"] if clean else [])
        self.run("pg_restore", *args, stdin=archive.read_bytes())

    def counts(self, database: str) -> dict[str, int]:
        return {t: int(self.sql(f"SELECT count(*) FROM {t}", database)) for t in TABLES}

    def drop(self, database: str) -> None:
        self.sql(f'DROP DATABASE IF EXISTS "{database}"')


def sqlite_backup(db: Path, out: Path) -> Path:
    path = out / f"gateway-{stamp()}.db"
    with sqlite3.connect(db) as source, sqlite3.connect(path) as target:
        source.backup(target)
    return path


def sqlite_counts(db: Path) -> dict[str, int]:
    with sqlite3.connect(db) as connection:
        return {t: connection.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "var" / "backups"))
    parser.add_argument("--local", action="store_true", help="use pg_dump/pg_restore on this machine, not Compose")
    parser.add_argument("--compose", default="docker compose", help="Compose command (with -p/-f as needed)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backup")
    restore = sub.add_parser("restore")
    restore.add_argument("archive")
    restore.add_argument("--target", required=True, help="database (PostgreSQL) or file (SQLite) to restore into")
    restore.add_argument("--overwrite-live", action="store_true")
    restore.add_argument("--yes", action="store_true")
    sub.add_parser("drill")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = database_target(load_config())
    if isinstance(target, Path):  # SQLite
        if args.command == "backup":
            print(f"backup: {sqlite_backup(target, out)}")
        elif args.command == "restore":
            dest = Path(args.target)
            if dest.resolve() == target.resolve() and not args.overwrite_live:
                raise SystemExit("refusing to overwrite the live database without --overwrite-live")
            if not args.yes:
                raise SystemExit("restore replaces the target: pass --yes")
            with sqlite3.connect(args.archive) as source, sqlite3.connect(dest) as dest_db:
                source.backup(dest_db)
            print(f"restored {args.archive} into {dest}")
        else:
            path = sqlite_backup(target, out)
            before, after = sqlite_counts(target), sqlite_counts(path)
            print(f"live:     {before}\nrestored: {after}")
            if before != after:
                raise SystemExit("drill FAILED: row counts differ")
            print(f"drill passed: {path}")
        return 0

    pg = Postgres(str(target), args.local, args.compose.split())
    if args.command == "backup":
        print(f"backup: {pg.backup(out)}")
    elif args.command == "restore":
        live = args.target == pg.database
        if live and not args.overwrite_live:
            raise SystemExit("refusing to overwrite the live database without --overwrite-live")
        if not args.yes:
            raise SystemExit("restore writes to the target database: pass --yes")
        pg.restore(Path(args.archive), args.target, clean=live)
        print(f"restored {args.archive} into database {args.target}")
    else:
        archive = pg.backup(out)
        pg.drop(DRILL_DB)
        try:
            pg.restore(archive, DRILL_DB, clean=False)
            before, after = pg.counts(pg.database), pg.counts(DRILL_DB)
            version = pg.sql("SELECT version_num FROM alembic_version", DRILL_DB)
        finally:
            pg.drop(DRILL_DB)
        print(f"backup:   {archive} ({archive.stat().st_size} bytes)")
        print(f"live:     {before}\nrestored: {after}\nalembic revision in the restore: {version}")
        if before != after:
            raise SystemExit("drill FAILED: row counts differ (was the gateway writing during the drill?)")
        print("drill passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
