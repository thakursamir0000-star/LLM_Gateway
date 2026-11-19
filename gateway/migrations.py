"""One-shot Alembic migration and schema-revision commands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from gateway.config import ROOT, resolve_path

CORE_COLUMNS = {
    "requests": {"id", "ts", "day", "app_id", "status", "cost_usd"},
    "calls": {"request_id", "seq", "ts", "status"},
    "feedback": {"id", "request_id", "ts", "app_id", "score"},
    "deployments": {"id", "ts", "action", "active"},
}


class MigrationError(RuntimeError):
    pass


def sqlite_url(path: str | Path) -> str:
    return f"sqlite:///{resolve_path(path).as_posix()}"


def configured_database_url() -> str:
    return os.environ.get("GATEWAY_DATABASE_URL") or sqlite_url(os.environ.get("GATEWAY_DB", "var/gateway.db"))


def alembic_config(database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def head_revision(database_url: str) -> str:
    return ScriptDirectory.from_config(alembic_config(database_url)).get_current_head()


def current_revision(database_url: str) -> str | None:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def validate_existing_schema(database_url: str) -> None:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            tables = set(inspector.get_table_names())
            missing_tables = set(CORE_COLUMNS) - tables
            if missing_tables:
                raise MigrationError(f"existing database is missing tables: {sorted(missing_tables)}")
            for table, required in CORE_COLUMNS.items():
                actual = {column["name"] for column in inspector.get_columns(table)}
                missing = required - actual
                if missing:
                    raise MigrationError(f"existing table {table!r} is missing columns: {sorted(missing)}")
    finally:
        engine.dispose()


def upgrade_database(database_url: str, *, adopt_existing: bool = False) -> str:
    config = alembic_config(database_url)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names()) - {"alembic_version"}
            revision = MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
    if tables and revision is None:
        if not adopt_existing:
            raise MigrationError("pre-Alembic database requires --adopt-existing")
        validate_existing_schema(database_url)
        command.stamp(config, "0001_core_schema")
    command.upgrade(config, "head")
    return current_revision(database_url) or ""


def verify_database_revision(database_url: str) -> None:
    current = current_revision(database_url)
    expected = head_revision(database_url)
    if current != expected:
        raise MigrationError(f"database revision is {current or 'unversioned'}; expected {expected}")


def downgrade_database(database_url: str, revision: str, *, confirm_destructive: bool = False) -> str | None:
    if not confirm_destructive:
        raise MigrationError("destructive downgrade requires --confirm-destructive")
    old = os.environ.get("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE")
    os.environ["GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE"] = "1"
    try:
        command.downgrade(alembic_config(database_url), revision)
    finally:
        if old is None:
            os.environ.pop("GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE", None)
        else:
            os.environ["GATEWAY_CONFIRM_DESTRUCTIVE_DOWNGRADE"] = old
    return current_revision(database_url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=configured_database_url())
    subparsers = parser.add_subparsers(dest="command", required=True)
    upgrade = subparsers.add_parser("upgrade")
    upgrade.add_argument("revision", nargs="?", default="head")
    upgrade.add_argument("--adopt-existing", action="store_true")
    downgrade = subparsers.add_parser("downgrade")
    downgrade.add_argument("revision")
    downgrade.add_argument("--confirm-destructive", action="store_true")
    subparsers.add_parser("verify")
    args = parser.parse_args()
    if args.command == "upgrade":
        revision = upgrade_database(args.database_url, adopt_existing=args.adopt_existing)
        print(f"database upgraded to {revision}")
    elif args.command == "downgrade":
        revision = downgrade_database(
            args.database_url, args.revision, confirm_destructive=args.confirm_destructive
        )
        print(f"database downgraded to {revision or 'base'}")
    else:
        verify_database_revision(args.database_url)
        print(f"database is at {head_revision(args.database_url)}")


if __name__ == "__main__":
    main()
