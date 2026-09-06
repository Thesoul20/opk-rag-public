from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from .config import PostgresConfig, load_postgres_config


def connect_postgres(database_url: str):
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError("psycopg is required for PostgreSQL access.") from exc

    return psycopg.connect(database_url, prepare_threshold=None)


def connect_from_config(config: PostgresConfig):
    return connect_postgres(config.database_url)


def connect_from_env(env: Mapping[str, str]):
    return connect_from_config(load_postgres_config(env))


@contextmanager
def transaction(connection) -> Iterator[None]:
    with connection.transaction():
        yield
