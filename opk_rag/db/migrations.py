from __future__ import annotations

from pathlib import Path
from urllib.parse import ParseResult, urlparse, urlunparse

from .config import DatabaseConfigError

CORE_MIGRATION_PATH = Path("supabase/migrations/202607170001_create_core_rag_schema.sql")
MIGRATIONS_DIR = Path("supabase/migrations")


class DatabaseMigrationError(RuntimeError):
    """Raised when test database preparation or migration execution fails."""


def _require_psycopg():
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise DatabaseMigrationError(
            "psycopg is required for database migration helpers. Install project dependencies first."
        ) from exc
    return psycopg


def _replace_database(parsed: ParseResult, database_name: str) -> str:
    return urlunparse(parsed._replace(path=f"/{database_name}"))


def validate_test_database_url(database_url: str) -> None:
    parsed = urlparse(database_url)
    database_name = parsed.path.lstrip("/")
    if "test" not in database_name.lower():
        raise DatabaseConfigError(
            "DATABASE_URL must target a dedicated test database whose name contains 'test'."
        )


def maintenance_database_url(database_url: str) -> str:
    parsed = urlparse(database_url)
    return _replace_database(parsed, "postgres")


def recreate_database(database_url: str) -> None:
    validate_test_database_url(database_url)
    psycopg = _require_psycopg()

    parsed = urlparse(database_url)
    database_name = parsed.path.lstrip("/")

    with psycopg.connect(maintenance_database_url(database_url), autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                select pg_terminate_backend(pid)
                from pg_stat_activity
                where datname = %s and pid <> pg_backend_pid()
                """,
                (database_name,),
            )
            identifier = psycopg.sql.Identifier(database_name)
            cursor.execute(psycopg.sql.SQL("drop database if exists {}").format(identifier))
            cursor.execute(psycopg.sql.SQL("create database {}").format(identifier))


def apply_sql_migration(
    database_url: str,
    migration_path: Path = CORE_MIGRATION_PATH,
) -> None:
    psycopg = _require_psycopg()
    sql = migration_path.read_text()

    with psycopg.connect(database_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)


def migration_paths(migrations_dir: Path = MIGRATIONS_DIR) -> tuple[Path, ...]:
    return tuple(sorted(migrations_dir.glob("*.sql")))


def apply_all_migrations(
    database_url: str,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> None:
    for migration_path in migration_paths(migrations_dir):
        apply_sql_migration(database_url, migration_path)
