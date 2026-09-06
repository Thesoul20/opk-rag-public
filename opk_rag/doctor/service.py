from __future__ import annotations

import json
import os
import socket
import ssl
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from opk_rag.answer.config import AnswerConfigError, AnswerGenerationConfig, load_answer_generation_config
from opk_rag.answer.provider import AnswerProviderError, OpenAICompatibleLocalChatProvider, ProviderErrorDetail, RemoteLLMNotAllowedError
from opk_rag.db.connection import connect_postgres
from opk_rag.db.config import DatabaseConfigError, PostgresConfig, SupabaseApiConfig, load_database_config, load_supabase_api_config
from opk_rag.db.health import DatabaseHealthCheckError, check_database_health
from opk_rag.embedding.config import EmbeddingConfig, EmbeddingConfigError, load_embedding_config
from opk_rag.runtime.config_sources import ConfigFieldSource, resolve_config_field_source
from opk_rag.runtime.dotenv import EnvironmentLoadResult, load_project_env
from opk_rag.vault.paths import VaultPathError, normalize_vault_root

from .models import DoctorCheck, DoctorOverallStatus, DoctorReport

DOCTOR_SCHEMA_VERSION = "opk-rag.doctor.v1"
DEFAULT_TIMEOUT_SECONDS = 3.0
_REFERENCE_DEEPSEEK_MODEL = "deepseek-v4-flash"
_REFERENCE_RESPONSE_FORMAT = "json_object"
_REFERENCE_THINKING_MODE = "disabled"
_REFERENCE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_REFERENCE_EMBEDDING_DIMENSION = 1024


@dataclass(frozen=True, slots=True)
class _DoctorContext:
    env: Mapping[str, str]
    load_result: EnvironmentLoadResult
    database_config: PostgresConfig | None
    supabase_config: SupabaseApiConfig | None
    answer_config: AnswerGenerationConfig | None
    embedding_config: EmbeddingConfig | None


_CONFIG_SOURCE_SPECS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("OPK_RAG_VAULT_PATH", (), False),
    ("DATABASE_URL", (), False),
    ("SUPABASE_URL", (), False),
    ("SUPABASE_PUBLISHABLE_KEY", (), False),
    ("SUPABASE_ANON_KEY", (), False),
    ("SUPABASE_SERVICE_ROLE_KEY", (), False),
    ("OPK_RAG_LLM_BASE_URL", (), True),
    ("OPK_RAG_LLM_API_KEY", (), False),
    ("OPK_RAG_LLM_MODEL", ("OPK_RAG_LLM_MODEL_ID",), True),
    ("OPK_RAG_LLM_MODEL_ID", (), False),
    ("OPK_RAG_LLM_RESPONSE_FORMAT", (), True),
    ("OPK_RAG_LLM_THINKING_MODE", (), False),
    ("OPK_RAG_LLM_ALLOW_REMOTE", (), True),
    ("OPK_RAG_EMBEDDING_MODEL", ("OPK_RAG_EMBEDDING_MODEL_NAME",), True),
    ("OPK_RAG_EMBEDDING_MODEL_NAME", (), False),
    ("OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY", (), True),
)


def run_doctor(
    *,
    env: Mapping[str, str] | None = None,
    load_result: EnvironmentLoadResult | None = None,
    remote_supabase: bool = False,
    remote_deepseek: bool = False,
) -> DoctorReport:
    runtime_env = os.environ if env is None else env
    if load_result is None:
        try:
            load_result = load_project_env()
        except Exception:
            load_result = EnvironmentLoadResult(Path(".env"), found=False, loaded=False)

    checks: list[DoctorCheck] = []
    context = _DoctorContext(
        env=runtime_env,
        load_result=load_result,
        database_config=None,
        supabase_config=None,
        answer_config=None,
        embedding_config=None,
    )

    checks.append(_check_dotenv(context))

    database_config = _safe_load_database_config(runtime_env, checks)
    supabase_config = _safe_load_supabase_config(runtime_env, checks)
    answer_config = _safe_load_answer_config(runtime_env, checks)
    embedding_config = _safe_load_embedding_config(runtime_env, checks)

    context = _DoctorContext(
        env=runtime_env,
        load_result=load_result,
        database_config=database_config,
        supabase_config=supabase_config,
        answer_config=answer_config,
        embedding_config=embedding_config,
    )

    checks.extend(
        (
            _check_config_conflicts(context),
            _check_vault(context),
            _check_url_formats(context),
            _check_reference_runtime_contract(context),
        )
    )

    if remote_supabase:
        checks.extend(_check_remote_supabase(context))
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.remote",
                component="supabase.remote",
                status="skipped",
                summary="remote Supabase checks not requested",
            )
        )

    if remote_deepseek:
        checks.extend(_check_remote_deepseek(context))
    else:
        checks.append(
            DoctorCheck(
                check_id="deepseek.remote",
                component="deepseek.remote",
                status="skipped",
                summary="remote DeepSeek checks not requested",
            )
        )

    overall_status = _overall_status(checks)
    summary = _render_summary(checks)
    config_sources = _collect_config_sources(context)
    return DoctorReport(
        schema_version=DOCTOR_SCHEMA_VERSION,
        overall_status=overall_status,
        remote_requested=remote_supabase or remote_deepseek,
        checks=tuple(checks),
        summary=summary,
        config_sources=config_sources,
    )


def render_doctor_text(report: DoctorReport) -> str:
    lines = ["OPK-RAG Doctor", ""]
    for check in report.checks:
        label = {
            "pass": "PASS",
            "warning": "WARN",
            "fail": "FAIL",
            "skipped": "SKIP",
        }[check.status]
        state = f" ({check.component_state})" if check.component_state else ""
        lines.append(f"[{label}] {check.component}{state}: {check.summary}")
        for detail in check.details:
            lines.append(f"  - {detail}")
    if report.config_sources:
        lines.append("")
        lines.append("Configuration sources:")
        for source in report.config_sources:
            lines.append(f"  - {source.name}: source={source.source}, configured={'true' if source.configured else 'false'}")
    lines.append("")
    lines.append(f"Overall: {report.overall_status}")
    return "\n".join(lines)


def _check_dotenv(context: _DoctorContext) -> DoctorCheck:
    result = context.load_result
    if result.found and result.loaded:
        return DoctorCheck(
            check_id="dotenv",
            component="dotenv",
            status="pass",
            summary="repository .env loaded",
            details=(f"repository_root={_redact_path(result.env_path.parent)}",),
        )
    if result.found:
        return DoctorCheck(
            check_id="dotenv",
            component="dotenv",
            status="warning",
            summary="repository .env found but no new keys were loaded",
            details=("shell environment already supplied the needed values",),
        )
    return DoctorCheck(
        check_id="dotenv",
        component="dotenv",
        status="warning",
        summary="repository .env not found",
        details=("shell environment will be used when available",),
    )


def _collect_config_sources(context: _DoctorContext) -> tuple[ConfigFieldSource, ...]:
    env = context.env
    load_result = context.load_result
    sources = [
        resolve_config_field_source(
            env,
            load_result,
            name,
            aliases=aliases,
            default_present=default_present,
        )
        for name, aliases, default_present in _CONFIG_SOURCE_SPECS
    ]
    return tuple(sources)


def _safe_load_database_config(env: Mapping[str, str], checks: list[DoctorCheck]) -> PostgresConfig | None:
    try:
        config = load_database_config(env)
    except DatabaseConfigError as exc:
        checks.append(
            DoctorCheck(
                check_id="config.database",
                component="config.database",
                status="fail",
                summary="database configuration is invalid or missing",
                details=(_sanitize_message(str(exc)),),
            )
        )
        return None

    checks.append(
        DoctorCheck(
            check_id="config.database",
            component="config.database",
            status="pass",
            summary="database configuration parsed",
            details=("DATABASE_URL present and parseable",),
        )
    )
    return config


def _safe_load_supabase_config(env: Mapping[str, str], checks: list[DoctorCheck]) -> SupabaseApiConfig | None:
    try:
        config = load_supabase_api_config(env)
    except DatabaseConfigError as exc:
        checks.append(
            DoctorCheck(
                check_id="config.supabase",
                component="config.supabase",
                status="fail",
                summary="Supabase configuration is invalid or incomplete",
                details=(_sanitize_message(str(exc)),),
            )
        )
        return None

    key_presence = []
    if config.publishable_key is not None:
        key_presence.append("publishable_key")
    if config.service_role_key is not None:
        key_presence.append("service_role_key")
    checks.append(
        DoctorCheck(
            check_id="config.supabase",
            component="config.supabase",
            status="pass",
            summary="Supabase configuration parsed",
            details=(
                "SUPABASE_URL present and parseable",
                f"key_variants={'/'.join(key_presence) if key_presence else 'none'}",
            ),
        )
    )
    return config


def _safe_load_answer_config(env: Mapping[str, str], checks: list[DoctorCheck]) -> AnswerGenerationConfig | None:
    try:
        config = load_answer_generation_config(env)
    except (AnswerConfigError, ValueError) as exc:
        checks.append(
            DoctorCheck(
                check_id="config.deepseek",
                component="config.deepseek",
                status="fail",
                summary="DeepSeek / answer generation configuration is invalid",
                details=(_sanitize_message(str(exc)),),
            )
        )
        return None

    details = [
        "LLM base URL present and parseable",
        f"response_format={config.response_format_type}",
        f"thinking_mode={config.thinking_mode or 'unset'}",
        f"allow_remote={'true' if config.allow_remote else 'false'}",
    ]
    if not _has_env_value(env, "OPK_RAG_LLM_API_KEY"):
        checks.append(
            DoctorCheck(
                check_id="config.deepseek",
                component="config.deepseek",
                status="fail",
                summary="DeepSeek API key is missing",
                details=("OPK_RAG_LLM_API_KEY is required for the reference runtime",),
            )
        )
        return config

    status = "pass"
    if _is_reference_deepseek_mismatch(config):
        status = "warning"
        details.append("configured values deviate from the frozen reference runtime")

    checks.append(
        DoctorCheck(
            check_id="config.deepseek",
            component="config.deepseek",
            status=status,
            summary="DeepSeek configuration parsed",
            details=tuple(details),
        )
    )
    return config


def _safe_load_embedding_config(env: Mapping[str, str], checks: list[DoctorCheck]) -> EmbeddingConfig | None:
    try:
        config = load_embedding_config(env)
    except EmbeddingConfigError as exc:
        checks.append(
            DoctorCheck(
                check_id="config.embedding",
                component="config.embedding",
                status="fail",
                summary="embedding configuration is invalid",
                details=(_sanitize_message(str(exc)),),
            )
        )
        return None

    status = "pass"
    details = [
        f"model_name={_REFERENCE_EMBEDDING_MODEL}",
        f"dimension={_REFERENCE_EMBEDDING_DIMENSION}",
    ]
    if config.model_name != _REFERENCE_EMBEDDING_MODEL:
        status = "warning"
        details.append("embedding model differs from the frozen reference runtime")
    if config.dimension != _REFERENCE_EMBEDDING_DIMENSION:
        status = "fail"
        details.append("embedding dimension must remain fixed at 1024")
    if not config.local_files_only:
        status = "warning" if status == "pass" else status
        details.append("local_files_only is disabled")
    checks.append(
        DoctorCheck(
            check_id="config.embedding",
            component="config.embedding",
            status=status,
            summary="embedding configuration parsed",
            details=tuple(details),
        )
    )
    return config


def _check_config_conflicts(context: _DoctorContext) -> DoctorCheck:
    env = context.env
    conflicts: list[str] = []
    if _has_env_value(env, "OPK_RAG_ANSWERABILITY_MIN_EVIDENCE") and _has_env_value(env, "OPK_RAG_ANSWER_MIN_EVIDENCE_ITEMS"):
        if env["OPK_RAG_ANSWERABILITY_MIN_EVIDENCE"].strip() != env["OPK_RAG_ANSWER_MIN_EVIDENCE_ITEMS"].strip():
            conflicts.append("answerability.min_evidence vs answer.min_evidence_items")
    if _has_env_value(env, "SUPABASE_PUBLISHABLE_KEY") and _has_env_value(env, "SUPABASE_ANON_KEY"):
        if env["SUPABASE_PUBLISHABLE_KEY"].strip() != env["SUPABASE_ANON_KEY"].strip():
            conflicts.append("supabase publishable_key vs anon_key")
    if _has_env_value(env, "OPK_RAG_LLM_ALLOW_REMOTE") and _has_env_value(env, "OPK_RAG_LLM_BASE_URL"):
        if _looks_remote_base_url(env["OPK_RAG_LLM_BASE_URL"]) and env["OPK_RAG_LLM_ALLOW_REMOTE"].strip().lower() in {"0", "false", "no", "off"}:
            conflicts.append("deepseek.remote disabled while remote base URL is configured")

    if conflicts:
        return DoctorCheck(
            check_id="config.conflicts",
            component="config.conflicts",
            status="fail",
            summary="conflicting configuration values detected",
            details=tuple(conflicts),
        )
    return DoctorCheck(
        check_id="config.conflicts",
        component="config.conflicts",
        status="pass",
        summary="no conflicting alias values detected",
    )


def _check_vault(context: _DoctorContext) -> DoctorCheck:
    env = context.env
    vault_path = env.get("OPK_RAG_VAULT_PATH", "").strip()
    if not vault_path:
        return DoctorCheck(
            check_id="vault",
            component="vault",
            status="fail",
            summary="vault path is missing",
            details=("OPK_RAG_VAULT_PATH is required for the reference runtime",),
        )

    try:
        root = normalize_vault_root(vault_path)
    except VaultPathError as exc:
        return DoctorCheck(
            check_id="vault",
            component="vault",
            status="fail",
            summary="vault path is invalid or unreadable",
            details=(_sanitize_message(str(exc)),),
        )

    markdown_files = _count_markdown_files(root.path)
    status: str = "pass" if markdown_files > 0 else "warning"
    summary = "vault path is configured and readable" if markdown_files > 0 else "vault is readable but contains no Markdown files"
    details = (
        "vault path configured",
        f"markdown_files={markdown_files}",
    )
    return DoctorCheck(
        check_id="vault",
        component="vault",
        status=status,
        summary=summary,
        details=details,
    )


def _check_url_formats(context: _DoctorContext) -> DoctorCheck:
    env = context.env
    issues: list[str] = []
    for name, allowed in (
        ("DATABASE_URL", {"postgres", "postgresql"}),
        ("SUPABASE_URL", {"http", "https"}),
        ("OPK_RAG_LLM_BASE_URL", {"http", "https"}),
    ):
        value = env.get(name, "").strip()
        if not value:
            issues.append(f"{name} missing")
            continue
        parsed = _parse_url(value)
        if parsed is None:
            issues.append(f"{name} is not a valid URL")
            continue
        if parsed.scheme not in allowed:
            issues.append(f"{name} uses an unexpected scheme")
            continue
        if not parsed.hostname:
            issues.append(f"{name} is missing a hostname")
            continue
        try:
            _ = parsed.port
        except ValueError:
            issues.append(f"{name} contains an invalid port")
            continue

    if issues:
        return DoctorCheck(
            check_id="config.urls",
            component="config.urls",
            status="fail",
            summary="one or more URLs are invalid",
            details=tuple(issues),
        )
    return DoctorCheck(
        check_id="config.urls",
        component="config.urls",
        status="pass",
        summary="database, Supabase, and DeepSeek URLs are parseable",
    )


def _check_reference_runtime_contract(context: _DoctorContext) -> DoctorCheck:
    env = context.env
    details: list[str] = []
    status: str = "pass"

    answer_config = context.answer_config
    if answer_config is None:
        return DoctorCheck(
            check_id="runtime.contract",
            component="runtime.contract",
            status="fail",
            summary="reference runtime contract cannot be evaluated",
            details=("answer configuration is unavailable",),
        )

    if answer_config.response_format_type != _REFERENCE_RESPONSE_FORMAT:
        status = "warning"
        details.append("response format differs from the frozen reference runtime")
    if answer_config.thinking_mode not in {None, _REFERENCE_THINKING_MODE}:
        status = "warning"
        details.append("thinking mode differs from the frozen reference runtime")
    if not answer_config.allow_remote:
        status = "warning"
        details.append("remote LLM access is disabled")
    if not _looks_remote_base_url(answer_config.base_url):
        status = "warning"
        details.append("DeepSeek base URL is not a remote endpoint")

    embedding_config = context.embedding_config
    if embedding_config is not None:
        if getattr(embedding_config, "model_name", None) != _REFERENCE_EMBEDDING_MODEL:
            status = "warning" if status == "pass" else status
            details.append("embedding model differs from the frozen reference runtime")
        if getattr(embedding_config, "dimension", None) != _REFERENCE_EMBEDDING_DIMENSION:
            status = "fail"
            details.append("embedding dimension is not 1024")

    if not details:
        details.append("reference runtime settings match the frozen baseline")
    return DoctorCheck(
        check_id="runtime.contract",
        component="runtime.contract",
        status=status,
        summary="reference runtime contract evaluated",
        details=tuple(details),
    )


def _check_remote_supabase(context: _DoctorContext) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    supabase_config = context.supabase_config
    database_config = context.database_config
    api_checks: list[DoctorCheck]
    postgres_checks: list[DoctorCheck]
    api_available = supabase_config is not None
    postgres_available = database_config is not None

    if api_available:
        api_checks = _check_supabase_api_branch(supabase_config)
    else:
        api_checks = [
            DoctorCheck(
                check_id="supabase.api",
                component="supabase.api",
                status="warning",
                component_state="optional_unavailable",
                summary="Supabase Data API configuration is incomplete",
                details=("SUPABASE_URL or API key is missing",),
            ),
            _skipped("supabase.api.dns", "Supabase Data API DNS check skipped because configuration is incomplete"),
            _skipped("supabase.api.network", "Supabase Data API network check skipped because configuration is incomplete"),
            _skipped("supabase.api.auth", "Supabase Data API auth check skipped because configuration is incomplete"),
        ]

    if postgres_available:
        postgres_checks = _check_supabase_postgres_branch(database_config)
    else:
        postgres_checks = [
            DoctorCheck(
                check_id="supabase.postgres",
                component="supabase.postgres",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL configuration is incomplete",
                details=("DATABASE_URL is missing or invalid",),
            ),
            _skipped("supabase.postgres.dns", "PostgreSQL DNS check skipped because configuration is incomplete"),
            _skipped("supabase.postgres.network", "PostgreSQL network check skipped because configuration is incomplete"),
            _skipped("supabase.postgres.auth", "PostgreSQL auth check skipped because configuration is incomplete"),
            _skipped("supabase.database", "database check skipped because PostgreSQL configuration is incomplete"),
            _skipped("supabase.isolation", "isolation check skipped because PostgreSQL configuration is incomplete"),
        ]

    checks.extend(api_checks)
    checks.extend(postgres_checks)
    checks.append(_summarize_supabase_remote(api_checks, postgres_checks))
    return checks


def _check_supabase_api_branch(config: SupabaseApiConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    host = _parse_url(config.supabase_url).hostname or ""
    dns_ok, dns_detail = _check_hostname_resolution(host, timeout=DEFAULT_TIMEOUT_SECONDS, label="Supabase API")
    if dns_ok:
        checks.append(
            DoctorCheck(
                check_id="supabase.api.dns",
                component="supabase.api.dns",
                status="pass",
                summary="Supabase Data API DNS resolution succeeded",
                details=("hostname resolved",),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.api.dns",
                component="supabase.api.dns",
                status="warning",
                component_state="optional_unavailable",
                summary="Supabase Data API DNS resolution failed",
                details=(dns_detail or "dns_resolution_failure",),
            )
        )
        checks.append(_skipped("supabase.api.network", "Supabase Data API network check skipped after DNS failure"))
        checks.append(_skipped("supabase.api.auth", "Supabase Data API auth check skipped after DNS failure"))
        return checks

    network_ok, network_detail = _check_tcp_connectivity(host, 443, timeout=DEFAULT_TIMEOUT_SECONDS, use_tls=True)
    if network_ok:
        checks.append(
            DoctorCheck(
                check_id="supabase.api.network",
                component="supabase.api.network",
                status="pass",
                summary="Supabase Data API TLS handshake succeeded",
                details=("TCP 443 and TLS handshake succeeded",),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.api.network",
                component="supabase.api.network",
                status="warning",
                component_state="optional_unavailable",
                summary="Supabase Data API network timed out or was unreachable",
                details=(network_detail or "network_connection_failure",),
            )
        )
        checks.append(_skipped("supabase.api.auth", "Supabase Data API auth check skipped after network failure"))
        checks.append(
            DoctorCheck(
                check_id="supabase.api",
                component="supabase.api",
                status="warning",
                component_state="optional_unavailable",
                summary="Supabase Data API network path is unavailable",
                details=("Data API is optional for the current Phase 2 reference runtime",),
            )
        )
        return checks

    checks.append(_check_supabase_api_auth(config))
    api_status = _branch_status(checks, prefix="supabase.api")
    checks.append(
        DoctorCheck(
            check_id="supabase.api",
            component="supabase.api",
            status="pass" if api_status == "pass" else "warning",
            component_state=None if api_status == "pass" else "optional_unavailable",
            summary="Supabase Data API read-only probe completed",
            details=("optional branch", "read-only metadata request was used"),
        )
    )
    return checks


def _check_supabase_api_auth(config: SupabaseApiConfig) -> DoctorCheck:
    key = config.service_role_key or config.publishable_key
    if key is None:
        return DoctorCheck(
            check_id="supabase.api.auth",
            component="supabase.api.auth",
            status="warning",
            component_state="optional_unavailable",
            summary="Supabase Data API key is missing",
            details=("a publishable or service-role key is required",),
        )

    endpoint = config.supabase_url.rstrip("/") + "/rest/v1/knowledge_bases?select=id&limit=0"
    request = Request(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            response.read(0)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            return DoctorCheck(
                check_id="supabase.api.auth",
                component="supabase.api.auth",
                status="warning",
                component_state="optional_unavailable",
                summary="Supabase Data API authentication failed",
                details=(_classify_http_status(exc.code), _sanitize_message(_read_http_error_body(exc))),
            )
        return DoctorCheck(
            check_id="supabase.api.auth",
            component="supabase.api.auth",
            status="warning",
            component_state="optional_unavailable",
            summary="Supabase Data API request failed",
            details=(_classify_http_status(exc.code), _sanitize_message(_read_http_error_body(exc))),
        )
    except URLError as exc:
        return DoctorCheck(
            check_id="supabase.api.auth",
            component="supabase.api.auth",
            status="warning",
            component_state="optional_unavailable",
            summary="Supabase Data API request timed out or was unreachable",
            details=(_classify_url_error(exc.reason),),
        )
    except TimeoutError:
        return DoctorCheck(
            check_id="supabase.api.auth",
            component="supabase.api.auth",
            status="warning",
            component_state="optional_unavailable",
            summary="Supabase Data API request timed out",
            details=("timeout",),
        )

    return DoctorCheck(
        check_id="supabase.api.auth",
        component="supabase.api.auth",
        status="pass",
        summary="Supabase Data API responded to a read-only metadata request",
        details=(
            "GET /rest/v1/knowledge_bases?select=id&limit=0",
            "apikey and Authorization headers were sent",
        ),
    )


def _check_supabase_postgres_branch(database_config: PostgresConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    host, port = _parse_postgres_target(database_config.database_url)
    dns_ok, dns_detail = _check_hostname_resolution(host, timeout=DEFAULT_TIMEOUT_SECONDS, label="PostgreSQL")
    if dns_ok:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.dns",
                component="supabase.postgres.dns",
                status="pass",
                summary="PostgreSQL DNS resolution succeeded",
                details=("hostname resolved",),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.dns",
                component="supabase.postgres.dns",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL DNS resolution failed",
                details=(dns_detail or "dns_resolution_failure",),
            )
        )
        checks.append(_skipped("supabase.postgres.network", "PostgreSQL network check skipped after DNS failure"))
        checks.append(_skipped("supabase.postgres.auth", "PostgreSQL auth check skipped after DNS failure"))
        checks.append(_skipped("supabase.database", "database check skipped after DNS failure"))
        checks.append(_skipped("supabase.isolation", "isolation check skipped after DNS failure"))
        return checks

    network_ok, network_detail = _check_tcp_connectivity(host, port, timeout=DEFAULT_TIMEOUT_SECONDS, use_tls=False)
    if network_ok:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.network",
                component="supabase.postgres.network",
                status="pass",
                summary="PostgreSQL socket connection succeeded",
                details=("TCP connectivity succeeded",),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.network",
                component="supabase.postgres.network",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL network connection failed",
                details=(network_detail or "network_connection_failure",),
            )
        )
        checks.append(_skipped("supabase.postgres.auth", "PostgreSQL auth check skipped after network failure"))
        checks.append(_skipped("supabase.database", "database check skipped after PostgreSQL network failure"))
        checks.append(_skipped("supabase.isolation", "isolation check skipped after PostgreSQL network failure"))
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres",
                component="supabase.postgres",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL network path failed",
                details=("core PostgreSQL checks are required for the reference runtime",),
            )
        )
        return checks

    auth_ok, auth_detail = _check_supabase_postgres_auth(database_config)
    if auth_ok:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.auth",
                component="supabase.postgres.auth",
                status="pass",
                summary="PostgreSQL authentication succeeded",
                details=("read-only connection probe succeeded",),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres.auth",
                component="supabase.postgres.auth",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL authentication failed",
                details=(auth_detail or "authentication_failure",),
            )
        )
        checks.append(_skipped("supabase.database", "database check skipped after PostgreSQL auth failure"))
        checks.append(_skipped("supabase.isolation", "isolation check skipped after PostgreSQL auth failure"))
        checks.append(
            DoctorCheck(
                check_id="supabase.postgres",
                component="supabase.postgres",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL authentication failed",
                details=("core PostgreSQL checks are required for the reference runtime",),
            )
        )
        return checks

    try:
        db_report = check_database_health(database_config)
    except DatabaseHealthCheckError as exc:
        checks.append(
            DoctorCheck(
                check_id="supabase.database",
                component="supabase.database",
                status="fail",
                component_state="core_unavailable",
                summary="PostgreSQL health contract failed",
                details=(_sanitize_message(str(exc)),),
            )
        )
        checks.append(_skipped("supabase.isolation", "isolation check skipped after database failure"))
        return checks
    else:
        checks.append(
            DoctorCheck(
                check_id="supabase.database",
                component="supabase.database",
                status="pass",
                summary="PostgreSQL health contract passed",
                details=(
                    f"pgvector_enabled={'true' if db_report.pgvector_enabled else 'false'}",
                    f"tables_checked={len(db_report.tables_present)}",
                ),
            )
        )

    isolation_status = _check_supabase_isolation(database_config)
    checks.append(isolation_status)
    postgres_status = _branch_status(checks, prefix="supabase.postgres")
    checks.append(
        DoctorCheck(
            check_id="supabase.postgres",
            component="supabase.postgres",
            status=postgres_status,
            component_state=None if postgres_status == "pass" else ("core_unavailable" if postgres_status == "fail" else "core_degraded"),
            summary="PostgreSQL core path completed",
            details=("core PostgreSQL checks passed",),
        )
    )
    return checks


def _check_supabase_postgres_auth(database_config: PostgresConfig) -> tuple[bool, str | None]:
    try:
        with connect_postgres(database_config.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1")
                cursor.fetchone()
    except Exception as exc:
        return False, _classify_postgres_error(exc)
    return True, None


def _summarize_supabase_remote(api_checks: list[DoctorCheck], postgres_checks: list[DoctorCheck]) -> DoctorCheck:
    api_status = _branch_status(api_checks, prefix="supabase.api")
    postgres_status = _branch_status(postgres_checks, prefix="supabase.postgres")

    if postgres_status == "fail":
        return DoctorCheck(
            check_id="supabase.remote",
            component="supabase.remote",
            status="fail",
            summary="Supabase PostgreSQL path failed",
            details=("core PostgreSQL checks failed", "Data API is optional for the current Phase 2 reference runtime"),
        )
    if postgres_status == "warning":
        return DoctorCheck(
            check_id="supabase.remote",
            component="supabase.remote",
            status="warning",
            summary="Supabase PostgreSQL core path is degraded",
            details=("core PostgreSQL checks passed with warnings", "Data API remains optional"),
        )
    if api_status == "warning":
        return DoctorCheck(
            check_id="supabase.remote",
            component="supabase.remote",
            status="warning",
            summary="Supabase PostgreSQL passed and Data API is optional-unavailable",
            details=("PostgreSQL core checks passed", "Data API failure does not block the reference runtime"),
        )
    if api_status == "fail":
        return DoctorCheck(
            check_id="supabase.remote",
            component="supabase.remote",
            status="warning",
            summary="Supabase PostgreSQL passed and Data API is unavailable",
            details=("PostgreSQL core checks passed", "Data API is optional for the current Phase 2 reference runtime"),
        )
    return DoctorCheck(
        check_id="supabase.remote",
        component="supabase.remote",
        status="pass",
        summary="Supabase PostgreSQL and optional Data API checks passed",
        details=("PostgreSQL core checks passed", "Data API optional check passed"),
    )


def _branch_status(checks: list[DoctorCheck], *, prefix: str) -> str:
    statuses = [check.status for check in checks if check.component.startswith(prefix)]
    if not statuses:
        return "pass"
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"


def _parse_postgres_target(database_url: str) -> tuple[str, int]:
    parsed = _parse_url(database_url)
    if parsed is None or parsed.hostname is None:
        return "", 5432
    return parsed.hostname, parsed.port or 5432


def _check_supabase_isolation(database_config: PostgresConfig) -> DoctorCheck:
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        return DoctorCheck(
            check_id="supabase.isolation",
            component="supabase.isolation",
            status="skipped",
            summary="psycopg is unavailable",
            details=(_sanitize_message(str(exc)),),
        )

    try:
        with psycopg.connect(database_config.database_url, prepare_threshold=None) as connection:
            with connection.cursor() as cursor:
                cursor.execute("select relrowsecurity from pg_class where oid = 'public.knowledge_bases'::regclass")
                row_security = bool(cursor.fetchone()[0])
                cursor.execute(
                    """
                    select exists (
                        select 1
                        from pg_constraint
                        where conrelid = 'public.knowledge_bases'::regclass
                          and conname = 'knowledge_bases_root_path_key'
                    )
                    """
                )
                has_root_path_constraint = bool(cursor.fetchone()[0])
                cursor.execute(
                    """
                    select exists (
                        select 1
                        from pg_proc
                        where proname = 'match_chunks'
                    )
                    """
                )
                has_match_chunks = bool(cursor.fetchone()[0])
    except Exception as exc:
        return DoctorCheck(
            check_id="supabase.isolation",
            component="supabase.isolation",
            status="fail",
            summary="knowledge base isolation metadata could not be verified",
            details=(_sanitize_message(str(exc)),),
        )

    if not row_security or not has_root_path_constraint or not has_match_chunks:
        missing: list[str] = []
        if not row_security:
            missing.append("row-level security disabled")
        if not has_root_path_constraint:
            missing.append("knowledge_bases.root_path uniqueness missing")
        if not has_match_chunks:
            missing.append("match_chunks function missing")
        return DoctorCheck(
            check_id="supabase.isolation",
            component="supabase.isolation",
            status="warning",
            summary="knowledge base isolation metadata is incomplete",
            details=tuple(missing),
        )

    return DoctorCheck(
        check_id="supabase.isolation",
        component="supabase.isolation",
        status="pass",
        summary="knowledge base isolation metadata is present",
        details=("row_level_security=enabled", "root_path_constraint=present", "match_chunks=function-present"),
    )

def _check_remote_deepseek(context: _DoctorContext) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    answer_config = context.answer_config
    if answer_config is None:
        checks.append(
            DoctorCheck(
                check_id="deepseek.remote",
                component="deepseek.remote",
                status="fail",
                summary="DeepSeek configuration is incomplete",
                details=("required configuration is missing",),
            )
        )
        checks.append(_skipped("deepseek.dns", "DNS check skipped because configuration is incomplete"))
        checks.append(_skipped("deepseek.network", "network check skipped because configuration is incomplete"))
        checks.append(_skipped("deepseek.provider", "provider probe skipped because configuration is incomplete"))
        return checks

    base_url = answer_config.base_url
    parsed = _parse_url(base_url)
    if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname:
        checks.append(
            DoctorCheck(
                check_id="deepseek.config",
                component="deepseek.config",
                status="fail",
                summary="DeepSeek base URL is invalid",
                details=("invalid_base_url",),
            )
        )
        checks.append(_skipped("deepseek.dns", "DNS check skipped because the base URL is invalid"))
        checks.append(_skipped("deepseek.network", "network check skipped because the base URL is invalid"))
        checks.append(_skipped("deepseek.provider", "provider probe skipped because the base URL is invalid"))
        return checks

    config_issue = _deepseek_config_issue(context.env, answer_config)
    if config_issue is not None:
        checks.append(config_issue)
        if config_issue.status == "fail":
            checks.append(_skipped("deepseek.dns", "DNS check skipped after configuration failure"))
            checks.append(_skipped("deepseek.network", "network check skipped after configuration failure"))
            checks.append(_skipped("deepseek.provider", "provider probe skipped after configuration failure"))
            return checks
    else:
        checks.append(
            DoctorCheck(
                check_id="deepseek.config",
                component="deepseek.config",
                status="pass",
                summary="DeepSeek configuration is complete",
                details=(
                    "base_url present and parseable",
                    "api_key present",
                    f"model_id={_REFERENCE_DEEPSEEK_MODEL if answer_config.model_id == _REFERENCE_DEEPSEEK_MODEL else 'configured'}",
                ),
            )
        )

    host = parsed.hostname or ""
    timeout = DEFAULT_TIMEOUT_SECONDS
    try:
        _resolve_hostname(host, timeout=timeout)
    except TimeoutError:
        checks.append(
            DoctorCheck(
                check_id="deepseek.dns",
                component="deepseek.dns",
                status="fail",
                summary="DeepSeek DNS resolution timed out",
                details=("dns_resolution_failure",),
            )
        )
        checks.append(_skipped("deepseek.network", "network check skipped after DNS failure"))
        checks.append(_skipped("deepseek.provider", "provider probe skipped after DNS failure"))
        return checks
    except (socket.gaierror, OSError):
        checks.append(
            DoctorCheck(
                check_id="deepseek.dns",
                component="deepseek.dns",
                status="fail",
                summary="DeepSeek DNS resolution failed",
                details=("dns_resolution_failure",),
            )
        )
        checks.append(_skipped("deepseek.network", "network check skipped after DNS failure"))
        checks.append(_skipped("deepseek.provider", "provider probe skipped after DNS failure"))
        return checks

    try:
        _probe_tcp(host, parsed.port or (443 if parsed.scheme == "https" else 80), timeout=timeout, use_tls=parsed.scheme == "https")
    except ConnectionRefusedError:
        checks.append(
            DoctorCheck(
                check_id="deepseek.network",
                component="deepseek.network",
                status="fail",
                summary="DeepSeek connection was refused",
                details=("connection_refused",),
            )
        )
        checks.append(_skipped("deepseek.provider", "provider probe skipped after network failure"))
        return checks
    except TimeoutError:
        checks.append(
            DoctorCheck(
                check_id="deepseek.network",
                component="deepseek.network",
                status="fail",
                summary="DeepSeek connection timed out",
                details=("provider_timeout",),
            )
        )
        checks.append(_skipped("deepseek.provider", "provider probe skipped after timeout"))
        return checks
    except ssl.SSLError:
        checks.append(
            DoctorCheck(
                check_id="deepseek.network",
                component="deepseek.network",
                status="fail",
                summary="DeepSeek TLS negotiation failed",
                details=("tls_failure",),
            )
        )
        checks.append(_skipped("deepseek.provider", "provider probe skipped after TLS failure"))
        return checks
    except OSError as exc:
        checks.append(
            DoctorCheck(
                check_id="deepseek.network",
                component="deepseek.network",
                status="fail",
                summary="DeepSeek network check failed",
                details=(_classify_socket_message(exc),),
            )
        )
        checks.append(_skipped("deepseek.provider", "provider probe skipped after network failure"))
        return checks
    else:
        checks.append(
            DoctorCheck(
                check_id="deepseek.network",
                component="deepseek.network",
                status="pass",
                summary="DeepSeek network and TLS checks succeeded",
                details=("endpoint reachable",),
            )
        )

    provider_check = _probe_deepseek_provider(context.env, answer_config)
    checks.append(provider_check)
    return checks


def _deepseek_config_issue(env: Mapping[str, str], config: AnswerGenerationConfig) -> DoctorCheck | None:
    details: list[str] = []
    status = "pass"

    if not _has_env_value(env, "OPK_RAG_LLM_API_KEY"):
        return DoctorCheck(
            check_id="deepseek.config",
            component="deepseek.config",
            status="fail",
            summary="DeepSeek API key is missing",
            details=("OPK_RAG_LLM_API_KEY is required",),
        )

    if config.model_id != _REFERENCE_DEEPSEEK_MODEL:
        status = "warning"
        details.append("model differs from the frozen reference runtime")
    if config.response_format_type != _REFERENCE_RESPONSE_FORMAT:
        status = "warning"
        details.append("response format differs from the frozen reference runtime")
    if config.thinking_mode not in {None, _REFERENCE_THINKING_MODE}:
        status = "warning"
        details.append("thinking mode differs from the frozen reference runtime")
    if not config.allow_remote:
        status = "warning"
        details.append("remote LLM access is disabled")

    if not details:
        details.append("DeepSeek configuration matches the reference runtime baseline")
    return DoctorCheck(
        check_id="deepseek.config",
        component="deepseek.config",
        status=status,
        summary="DeepSeek configuration parsed",
        details=tuple(details),
    )


def _probe_deepseek_provider(env: Mapping[str, str], config: AnswerGenerationConfig) -> DoctorCheck:
    if not _has_env_value(env, "OPK_RAG_LLM_MODEL") and not _has_env_value(env, "OPK_RAG_LLM_MODEL_ID"):
        config = replace(config, model_id=_REFERENCE_DEEPSEEK_MODEL)
    try:
        provider = OpenAICompatibleLocalChatProvider(config, api_key=env.get("OPK_RAG_LLM_API_KEY", "").strip() or None)
    except RemoteLLMNotAllowedError as exc:
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek remote endpoint is blocked by configuration",
            details=(_sanitize_message(str(exc)),),
        )

    payload = _build_deepseek_probe_payload(provider)
    endpoint = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    started = monotonic()
    try:
        raw_text, attempts = provider._post_with_retries(endpoint, json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers)
    except AnswerProviderError as exc:
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider probe failed",
            details=_classify_provider_error(exc.detail, exc),
        )

    duration_ms = int((monotonic() - started) * 1000)
    try:
        envelope = json.loads(raw_text)
    except json.JSONDecodeError:
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider returned non-JSON output",
            details=("invalid_provider_response",),
            duration_ms=duration_ms,
        )

    if not isinstance(envelope, dict):
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider returned an invalid response envelope",
            details=("invalid_provider_response",),
            duration_ms=duration_ms,
        )
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider response is missing choices",
            details=("invalid_provider_response",),
            duration_ms=duration_ms,
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider returned empty content",
            details=("invalid_provider_response",),
            duration_ms=duration_ms,
        )

    parsed = _parse_json_object(content)
    if parsed is None or not isinstance(parsed, dict):
        return DoctorCheck(
            check_id="deepseek.provider",
            component="deepseek.provider",
            status="fail",
            summary="DeepSeek provider content is not a JSON object",
            details=("invalid_provider_response",),
            duration_ms=duration_ms,
        )

    return DoctorCheck(
        check_id="deepseek.provider",
        component="deepseek.provider",
        status="pass",
        summary="DeepSeek provider completed a minimal read-only request",
        details=("minimal JSON response verified",),
        duration_ms=duration_ms,
    )


def _build_deepseek_probe_payload(provider: OpenAICompatibleLocalChatProvider) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": provider.config.model_id,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": "Return a JSON object with a single boolean field named ok.",
            }
        ],
        "response_format": _probe_response_format(provider.config),
        "thinking": {"type": "disabled"},
    }
    return payload


def _probe_response_format(config: AnswerGenerationConfig) -> dict[str, object]:
    if config.response_format_type == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "opk_rag_doctor_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
    return {"type": "json_object"}


def _classify_provider_error(detail: ProviderErrorDetail | None, exc: AnswerProviderError) -> tuple[str, ...]:
    if detail is None:
        return ("provider_error", _sanitize_message(str(exc)))
    reason = detail.reason_code
    summary = (detail.error_summary or "").lower()
    if "supported api model names" in summary and "you passed" in summary:
        status = "invalid_model"
    elif "thinking" in summary:
        status = "thinking_mode_failure"
    else:
        mapping = {
            "authentication_failure": "authentication_failure",
            "authorization_failure": "authorization_failure",
            "model_not_found": "model_not_found",
            "invalid_model": "invalid_model",
            "unsupported_response_format": "unsupported_response_format",
            "thinking_mode_unsupported": "thinking_mode_failure",
            "thinking_mode_failure": "thinking_mode_failure",
            "provider_timeout": "provider_timeout",
            "connection_timeout": "provider_timeout",
            "network_failure": "network_failure",
            "dns_or_host_error": "network_failure",
            "connection_refused": "network_failure",
            "tls_error": "network_failure",
            "proxy_error": "network_failure",
            "provider_error": "provider_error",
            "unknown_provider_error": "provider_error",
        }
        status = mapping.get(reason, "provider_error")
    details = [status]
    if detail.error_summary:
        details.append(_sanitize_message(detail.error_summary))
    elif str(exc):
        details.append(_sanitize_message(str(exc)))
    return tuple(details)


def _render_summary(checks: list[DoctorCheck]) -> str:
    counts = {"pass": 0, "warning": 0, "fail": 0, "skipped": 0}
    for check in checks:
        counts[check.status] += 1
    return (
        f"{counts['pass']} pass, {counts['warning']} warning, "
        f"{counts['fail']} fail, {counts['skipped']} skipped"
    )


def _overall_status(checks: list[DoctorCheck]) -> DoctorOverallStatus:
    if any(check.status == "fail" for check in checks):
        return "unhealthy"
    if any(check.status == "warning" for check in checks):
        return "degraded"
    return "healthy"


def _skipped(check_id: str, summary: str) -> DoctorCheck:
    return DoctorCheck(check_id=check_id, component=check_id, status="skipped", summary=summary)


def _check_hostname_resolution(host: str, *, timeout: float, label: str) -> tuple[bool, str | None]:
    try:
        _resolve_hostname(host, timeout=timeout)
    except TimeoutError:
        return False, f"{label} DNS resolution timed out"
    except (socket.gaierror, OSError):
        return False, f"{label} DNS resolution failed"
    return True, None


def _check_tcp_connectivity(host: str, port: int, *, timeout: float, use_tls: bool) -> tuple[bool, str | None]:
    try:
        _probe_tcp(host, port, timeout=timeout, use_tls=use_tls)
    except ConnectionRefusedError:
        return False, "connection_refused"
    except TimeoutError:
        return False, "timeout"
    except ssl.SSLError:
        return False, "tls_failure"
    except OSError as exc:
        return False, _classify_socket_message(exc)
    return True, None


def _has_env_value(env: Mapping[str, str], key: str) -> bool:
    return bool(env.get(key, "").strip())


def _looks_remote_base_url(value: str) -> bool:
    parsed = _parse_url(value)
    if parsed is None or parsed.hostname is None:
        return False
    host = parsed.hostname.lower()
    return host not in {"127.0.0.1", "localhost", "::1"} and not host.startswith("127.")


def _parse_url(value: str):
    try:
        parsed = urlparse(value)
    except Exception:
        return None
    if not parsed.scheme:
        return None
    return parsed


def _resolve_hostname(host: str, *, timeout: float) -> list[tuple]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError("DNS resolution timed out.") from exc


def _probe_tcp(host: str, port: int, *, timeout: float, use_tls: bool) -> None:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        if use_tls:
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=host):
                return


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _classify_http_status(status: int) -> str:
    if status == 401:
        return "authentication_failure"
    if status == 403:
        return "authorization_failure"
    if status in {404, 405, 406, 415, 422}:
        return "api_contract_failure"
    return f"http_{status}"


def _classify_url_error(reason: object) -> str:
    if isinstance(reason, socket.gaierror):
        return "dns_resolution_failure"
    if isinstance(reason, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(reason, TimeoutError):
        return "timeout"
    if isinstance(reason, ssl.SSLError):
        return "tls_failure"
    lowered = str(reason).lower()
    if "timed out" in lowered:
        return "timeout"
    if "name or service not known" in lowered or "nodename nor servname" in lowered:
        return "dns_resolution_failure"
    if "certificate" in lowered or "ssl" in lowered:
        return "tls_failure"
    return "network_connection_failure"


def _classify_socket_message(exc: OSError) -> str:
    if isinstance(exc, ConnectionRefusedError):
        return "connection_refused"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ssl.SSLError):
        return "tls_failure"
    message = str(exc).lower()
    if "timed out" in message:
        return "timeout"
    if "refused" in message:
        return "connection_refused"
    if "ssl" in message or "certificate" in message:
        return "tls_failure"
    if "name or service not known" in message or "nodename nor servname" in message:
        return "dns_resolution_failure"
    return "network_connection_failure"


def _classify_postgres_error(exc: Exception) -> str:
    message = _sanitize_message(str(exc)).lower()
    if "password authentication failed" in message or "authentication failed" in message:
        return "authentication_failure"
    if "permission denied" in message or "insufficient privilege" in message:
        return "authorization_failure"
    if "timeout" in message:
        return "timeout"
    if "could not connect" in message or "connection refused" in message:
        return "connection_refused"
    if "ssl" in message or "certificate" in message:
        return "tls_failure"
    return "network_connection_failure"


def _sanitize_message(value: str) -> str:
    text = value.replace("\r", " ").replace("\n", " ").strip()
    text = " ".join(text.split())
    for marker in ("Bearer ", "api_key=", "access_token="):
        if marker in text:
            head, tail = text.split(marker, 1)
            replacement = tail.split(" ", 1)[-1] if " " in tail else ""
            text = head + marker + "<redacted>" + (f" {replacement}" if replacement else "")
    return text[:500]


def _redact_path(path: Path) -> str:
    return f"<redacted>/{path.name}" if path.name else "<redacted>"


def _count_markdown_files(root: Path) -> int:
    count = 0
    for path in root.rglob("*.md"):
        if path.is_file():
            count += 1
    return count


def _parse_json_object(value: str) -> dict | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_reference_deepseek_mismatch(config: AnswerGenerationConfig) -> bool:
    return (
        config.model_id != _REFERENCE_DEEPSEEK_MODEL
        or config.response_format_type != _REFERENCE_RESPONSE_FORMAT
        or config.thinking_mode not in {None, _REFERENCE_THINKING_MODE}
        or not config.allow_remote
        or not _looks_remote_base_url(config.base_url)
    )
