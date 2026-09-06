from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class DatabaseConfigError(ValueError):
    """Raised when required database configuration is missing or invalid."""


def _mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _require_non_empty(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise DatabaseConfigError(f"Missing required environment variable: {key}")
    return value


def _optional_non_empty(env: Mapping[str, str], key: str) -> str | None:
    return env.get(key, "").strip() or None


def _validate_http_url(value: str, env_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DatabaseConfigError(f"{env_name} must be a valid http(s) URL.")
    return value


def _validate_database_url(value: str, env_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path:
        raise DatabaseConfigError(f"{env_name} must be a valid PostgreSQL connection URL.")
    return value


@dataclass(frozen=True)
class PostgresConfig:
    database_url: str

    @property
    def redacted_database_url(self) -> str:
        parsed = urlparse(self.database_url)
        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{_mask_secret(parsed.password)}"
            userinfo += "@"
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{userinfo}{host}{port}{parsed.path}"


DatabaseConfig = PostgresConfig


@dataclass(frozen=True)
class SupabaseApiConfig:
    supabase_url: str
    publishable_key: str | None = None
    service_role_key: str | None = None

    def require_publishable_key(self) -> str:
        if self.publishable_key is None:
            raise DatabaseConfigError("Missing required environment variable: SUPABASE_PUBLISHABLE_KEY")
        return self.publishable_key

    def require_service_role_key(self) -> str:
        if self.service_role_key is None:
            raise DatabaseConfigError("Missing required environment variable: SUPABASE_SERVICE_ROLE_KEY")
        return self.service_role_key

    @property
    def redacted_publishable_key(self) -> str | None:
        if self.publishable_key is None:
            return None
        return _mask_secret(self.publishable_key)

    @property
    def redacted_service_role_key(self) -> str | None:
        if self.service_role_key is None:
            return None
        return _mask_secret(self.service_role_key)


def load_postgres_config(env: Mapping[str, str]) -> PostgresConfig:
    database_url = _validate_database_url(_require_non_empty(env, "DATABASE_URL"), "DATABASE_URL")
    return PostgresConfig(database_url=database_url)


def load_database_config(env: Mapping[str, str]) -> DatabaseConfig:
    return load_postgres_config(env)


def load_supabase_api_config(env: Mapping[str, str]) -> SupabaseApiConfig:
    supabase_url = _validate_http_url(_require_non_empty(env, "SUPABASE_URL"), "SUPABASE_URL")
    publishable_key = _optional_non_empty(env, "SUPABASE_PUBLISHABLE_KEY") or _optional_non_empty(
        env, "SUPABASE_ANON_KEY"
    )
    service_role_key = _optional_non_empty(env, "SUPABASE_SERVICE_ROLE_KEY")

    if publishable_key is None and service_role_key is None:
        raise DatabaseConfigError(
            "Missing required environment variable: one of SUPABASE_PUBLISHABLE_KEY, "
            "SUPABASE_ANON_KEY, or SUPABASE_SERVICE_ROLE_KEY"
        )

    return SupabaseApiConfig(
        supabase_url=supabase_url,
        publishable_key=publishable_key,
        service_role_key=service_role_key,
    )
