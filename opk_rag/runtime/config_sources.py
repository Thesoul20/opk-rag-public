from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .dotenv import EnvironmentLoadResult


ConfigSource = Literal["shell", "dotenv", "default", "compatibility_alias", "missing"]


@dataclass(frozen=True, slots=True)
class ConfigFieldSource:
    name: str
    source: ConfigSource
    configured: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "configured": self.configured,
        }


def resolve_config_field_source(
    env: Mapping[str, str],
    load_result: EnvironmentLoadResult | None,
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    default_present: bool = False,
) -> ConfigFieldSource:
    if name in env:
        return ConfigFieldSource(name=name, source=_env_origin(load_result, name), configured=_is_configured(env[name]))

    for alias in aliases:
        if alias in env:
            return ConfigFieldSource(name=name, source="compatibility_alias", configured=_is_configured(env[alias]))

    if default_present:
        return ConfigFieldSource(name=name, source="default", configured=True)
    return ConfigFieldSource(name=name, source="missing", configured=False)


def _env_origin(load_result: EnvironmentLoadResult | None, key: str) -> ConfigSource:
    if load_result is None:
        return "shell"
    if key in load_result.loaded_keys:
        return "dotenv"
    if key in load_result.skipped_existing_keys:
        return "shell"
    return "shell"


def _is_configured(value: str) -> bool:
    return bool(value.strip())
