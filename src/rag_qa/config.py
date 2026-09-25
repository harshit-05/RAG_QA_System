"""Configuration loading: YAML in, a validated and frozen :class:`RagConfig` out.

The only reader of the config file (ARCHITECTURE.md §1.2). Everything that can be wrong
with a config — a mistyped key, a reference to nothing, an absolute path, an empty
file — surfaces here as one :class:`ConfigError` naming the file, the location and the
fix, before any component is built (FR-8). The rules themselves live in
:mod:`rag_qa.schema`; this module reads the file and renders what they report.

Paths in the config file are relative to *the config file's own directory*, never to
the process working directory. That is what makes a fresh clone runnable from anywhere
and closes ISS-02 / NFR-11 — the old config hardcoded absolute paths into another
machine's home directory. See :class:`rag_qa.schema.Paths` for the full rule.
"""

import os
from pathlib import Path

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from rag_qa.schema import PathContext, RagConfig
from rag_qa.settings import ENV_CONFIG, PATH_ENV_VARS, EnvSettings

DEFAULT_CONFIG_FILENAME = "config.yaml"


class ConfigError(Exception):
    """The config file is missing, unreadable or invalid.

    The message is written for the person fixing the file: it names the file, then one
    ``location: problem`` line per error.
    """


def resolve_config_path(
    config_path: str | os.PathLike[str] | None = None, env: EnvSettings | None = None
) -> Path:
    """Decide which config file to load.

    Precedence: explicit argument, then ``$RAG_CONFIG``, then ``config.yaml`` in the
    current directory.
    """
    env = env or EnvSettings()
    chosen = config_path or env.config or DEFAULT_CONFIG_FILENAME
    return Path(chosen).expanduser()


def load_config(config_path: str | os.PathLike[str] | None = None) -> RagConfig:
    """Load, validate and freeze the config. Raises :class:`ConfigError` on any problem."""
    env = EnvSettings()
    path = resolve_config_path(config_path, env).resolve()
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. "
            f"Pass a path, or set ${ENV_CONFIG}, or run from the repo root."
        )

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Config file {path} is not valid YAML:\n{e}") from e

    if raw is None:
        raise ConfigError(
            f"Config file {path} is empty. It needs 'components' and 'pipeline' "
            f"sections (and optionally 'paths'); the repo's config.yaml shows the shape."
        )
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file {path} must be a mapping with 'components' and 'pipeline' "
            f"sections, got a {type(raw).__name__}."
        )

    context = PathContext(
        base_dir=path.parent, overrides=env.path_overrides(), env_vars=PATH_ENV_VARS
    )
    try:
        return RagConfig.model_validate(raw, context=context)
    except ValidationError as e:
        # `from None`: the rendered message says everything the Pydantic dump would,
        # and a chained dump is the "1 validation error for RagConfig" wall FR-8 bans.
        raise ConfigError(_render(path, e)) from None


def _render(path: Path, error: ValidationError) -> str:
    lines = [f"Invalid config: {path}"]
    for detail in error.errors():
        where = ".".join(str(part) for part in detail["loc"])
        for message in _message(detail).splitlines():
            lines.append(f"  {where}: {message}" if where else f"  {message}")
    return "\n".join(lines)


def _message(detail: ErrorDetails) -> str:
    if detail["type"] == "missing":
        return "required key is missing"
    if detail["type"] == "extra_forbidden":  # backstop; the schema's own check fires first
        return "unknown key"
    return detail["msg"].removeprefix("Value error, ")
