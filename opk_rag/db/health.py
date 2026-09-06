from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import PostgresConfig


class DatabaseHealthCheckError(RuntimeError):
    """Raised when the database health check cannot complete successfully."""


EXPECTED_TABLES = (
    "knowledge_bases",
    "documents",
    "chunks",
    "index_configurations",
    "index_runs",
    "index_failures",
    "chunk_lexical_statistics",
    "chunk_lexical_terms",
    "knowledge_base_lexical_statistics",
    "knowledge_base_lexical_terms",
    "conversation_sessions",
    "conversation_turns",
    "conversation_turn_citations",
)


@dataclass(frozen=True)
class HealthReport:
    database_reachable: bool
    pgvector_enabled: bool
    tables_present: tuple[str, ...]


def _require_psycopg():
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise DatabaseHealthCheckError(
            "psycopg is required for live database health checks. Install project dependencies first."
        ) from exc
    return psycopg


def check_database_health(
    config: PostgresConfig,
    connection_factory: Callable[[str], object] | None = None,
) -> HealthReport:
    factory = connection_factory
    if factory is None:
        psycopg = _require_psycopg()
        factory = psycopg.connect

    try:
        with factory(config.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select exists (select 1 from pg_extension where extname = 'vector')")
                pgvector_row = cursor.fetchone()
                pgvector_enabled = bool(pgvector_row and pgvector_row[0])

                cursor.execute(
                    """
                    select table_name
                    from information_schema.tables
                    where table_schema = 'public'
                      and table_type = 'BASE TABLE'
                      and table_name = any(%s)
                    order by array_position(%s, table_name)
                    """,
                    (list(EXPECTED_TABLES), list(EXPECTED_TABLES)),
                )
                tables_present = tuple(row[0] for row in cursor.fetchall())
    except Exception as exc:
        raise DatabaseHealthCheckError(
            f"Unable to check database health for {config.redacted_database_url}."
        ) from exc

    missing_tables = tuple(table for table in EXPECTED_TABLES if table not in tables_present)
    if missing_tables:
        raise DatabaseHealthCheckError(
            "Missing required database tables: " + ", ".join(missing_tables)
        )
    if not pgvector_enabled:
        raise DatabaseHealthCheckError("Required PostgreSQL extension 'vector' is not installed.")

    return HealthReport(
        database_reachable=True,
        pgvector_enabled=pgvector_enabled,
        tables_present=tables_present,
    )
