"""Environment overrides, read through pydantic-settings.

The only reader of the ``RAG_*`` environment variables (ARCHITECTURE.md §1.2), and of the
two ``OTEL_*`` ones our code reads itself; the OpenTelemetry SDK reads its others. An
empty variable counts as unset, so ``RAG_DATA_PATH= rag-ingest`` means "use the config
file's value", as it always has.

Six groups: :class:`EnvSettings`, the overrides every command reads;
:class:`StoreSettings`, the ``RAG_QDRANT_*`` ones the vector store reads (DEC-19);
:class:`FallbackSettings`, the hosted fallback's key (DEC-21); :class:`LogSettings`, the log
format every command reads (DEC-22); :class:`TraceSettings`, whether and as what a command
sends its traces (DEC-22, S3-7b); and :class:`ServerSettings`, the ``RAG_API_*`` settings
only ``rag-serve`` reads (DEC-18). ``config.yaml`` stays about the pipeline, and a secret
never lives in it: a component key named like one is refused at load
(:meth:`rag_qa.schema.Components.check_api_keys`).
"""

import ipaddress
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_qa.schema import collection_problem

#: Selects which config file to load.
ENV_CONFIG = "RAG_CONFIG"
#: Overrides ``paths.data`` (the document corpus).
ENV_DATA_PATH = "RAG_DATA_PATH"
#: Overrides ``paths.state``: the writer lock, which is not part of the index (DEC-19).
ENV_STATE_PATH = "RAG_STATE_PATH"
#: v0.3's index folder. Refused while set, never ignored (DEC-19): a scratch ingest that set
#: it to stay off the real index (caveat 9) would otherwise write the real index.
ENV_VECTOR_STORE_PATH = "RAG_VECTOR_STORE_PATH"

#: Each ``paths`` key, and the variable that overrides it.
PATH_ENV_VARS = {
    "data": ENV_DATA_PATH,
    "state": ENV_STATE_PATH,
}

#: The prefix of the vector store's settings: ``RAG_QDRANT_URL``, ``RAG_QDRANT_COLLECTION``,
#: ``RAG_QDRANT_API_KEY``.
ENV_QDRANT_PREFIX = "RAG_QDRANT_"
ENV_QDRANT_COLLECTION = f"{ENV_QDRANT_PREFIX}COLLECTION"


class EnvSettings(BaseSettings):
    """The ``RAG_*`` overrides as they are *now*.

    Instantiated on every :func:`rag_qa.config.load_config` call, never at import, so
    a long-lived process and a test's ``monkeypatch`` both see the current environment.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_ignore_empty=True, frozen=True)

    config: str | None = None
    data_path: str | None = None
    state_path: str | None = None
    vector_store_path: str | None = None  # read only to be refused

    def path_overrides(self) -> dict[str, str | None]:
        """Each ``paths`` key mapped to its override, ``None`` when unset."""
        return {"data": self.data_path, "state": self.state_path}


class StoreSettings(BaseSettings):
    """The vector store's overrides and its key (DEC-19), from the environment only.

    ``url`` and ``collection`` override the store component's own. A scratch index is
    another ``collection``: the name of the alias its generations are published under. The
    API key has no config counterpart at all, so it can never be committed (§9). Like
    :class:`EnvSettings`, read on every use, never at import.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_QDRANT_PREFIX, env_ignore_empty=True, frozen=True
    )

    url: str | None = None
    collection: str | None = None
    api_key: SecretStr | None = None

    @field_validator("collection")
    @classmethod
    def check_collection(cls, collection: str | None) -> str | None:
        problem = None if collection is None else collection_problem(collection)
        if problem is not None:
            raise ValueError(problem)
        return collection


#: The hosted fallback's key (DEC-21, S3-6): one variable whatever the provider.
ENV_FALLBACK_PREFIX = "RAG_FALLBACK_"
ENV_FALLBACK_API_KEY = f"{ENV_FALLBACK_PREFIX}API_KEY"


class FallbackSettings(BaseSettings):
    """The hosted fallback's API key, from the environment only (DEC-21, amended in S3-6).

    One provider-neutral variable, which our code passes to the fallback's model itself.
    Left to its own default, ``ChatOpenAI`` would read ``OPENAI_API_KEY``, or a LangSmith
    gateway key, and send it to whatever ``base_url`` the config names. Like
    :class:`EnvSettings`, read on every use, never at import. The shell may load it from a
    gitignored ``.env`` (``set -a; . ./.env; set +a``); rag_qa never reads that file.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_FALLBACK_PREFIX, env_ignore_empty=True, frozen=True
    )

    api_key: SecretStr | None = None


#: The log format's variable (DEC-22, S3-7): ``RAG_LOG_FORMAT``.
ENV_LOG_PREFIX = "RAG_LOG_"


class LogSettings(BaseSettings):
    """How every command formats its log lines (DEC-22, S3-7), from the environment only.

    ``format`` is ``json`` or ``console``; unset, it is JSON when stderr is not a terminal
    and console lines when it is (:func:`rag_qa.observability.logs.configure_logging`). Like
    :class:`EnvSettings`, read on every use, never at import.
    """

    model_config = SettingsConfigDict(env_prefix=ENV_LOG_PREFIX, env_ignore_empty=True, frozen=True)

    format: Literal["json", "console"] | None = None


#: The OpenTelemetry variables our code reads (DEC-22, S3-7b): ``OTEL_EXPORTER_OTLP_ENDPOINT``
#: and ``OTEL_SERVICE_NAME``.
ENV_OTEL_PREFIX = "OTEL_"


class TraceSettings(BaseSettings):
    """Whether, and as what, a command sends its traces (DEC-22, S3-7b), from the environment
    only.

    ``exporter_otlp_endpoint`` is the collector's base address, an http(s) URL; the exporter
    adds ``/v1/traces``. Unset, nothing is traced (:func:`rag_qa.observability.tracing.configure`).
    ``service_name`` names the service; unset, it is the command's own. The SDK reads the other
    ``OTEL_*`` variables itself, the timeout (in seconds, in Python) and the headers among them.
    Like :class:`EnvSettings`, read on every use, never at import.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_OTEL_PREFIX, env_ignore_empty=True, frozen=True
    )

    exporter_otlp_endpoint: AnyHttpUrl | None = None
    service_name: str | None = None


#: The prefix of the HTTP server's settings: ``RAG_API_TOKEN``, ``RAG_API_MAX_CONCURRENT``,
#: ``RAG_API_HOST``, ``RAG_API_PORT``, ``RAG_API_METRICS_PORT``, ``RAG_API_TRUSTED_PROXIES``.
ENV_API_PREFIX = "RAG_API_"


def proxy_addresses(value: str) -> tuple[str, ...]:
    """``RAG_API_TRUSTED_PROXIES`` as addresses: IP addresses separated by commas, each one
    proxy's own, normalised. A range, ``*`` or a name raises ``ValueError`` (DEC-23, design
    review 2): in the stack a published port's connections arrive from the network's gateway,
    inside the network's range, and would then name any address they liked."""
    found = []
    for item in value.split(","):
        try:
            found.append(str(ipaddress.ip_address(item.strip())))
        except ValueError:
            raise ValueError(
                "must be IP addresses separated by commas, each one proxy's own: a range "
                "(with /), a wildcard or a host name is refused"
            ) from None
    return tuple(found)


class ServerSettings(BaseSettings):
    """The HTTP server's settings (DEC-18), from the environment only.

    ``token`` unset means no authentication, which ``rag-serve`` allows on a loopback host
    only. ``max_concurrent`` is how many answers stream at once; a CPU host generates one at
    a time. ``host`` and ``port`` are the defaults ``rag-serve --host/--port`` override.
    ``metrics_port`` is where Prometheus scrapes the metrics, on the API's own host: a
    listener of its own, off unless set (DEC-22, S3-7). ``trusted_proxies`` are the addresses
    whose ``X-Forwarded-For`` names the client (:func:`proxy_addresses`); unset, nobody's is
    believed, loopback's included (DEC-23, S3-8).
    """

    model_config = SettingsConfigDict(env_prefix=ENV_API_PREFIX, env_ignore_empty=True, frozen=True)

    token: SecretStr | None = None
    max_concurrent: int = Field(default=1, ge=1)
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8000, ge=1, le=65535)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)
    #: A string, not a list: pydantic-settings would read a list from the environment as JSON.
    trusted_proxies: str | None = None

    @field_validator("trusted_proxies")
    @classmethod
    def check_trusted_proxies(cls, value: str | None) -> str | None:
        return None if value is None else ",".join(proxy_addresses(value))

    @property
    def proxies(self) -> list[str]:
        """The trusted proxies' addresses, none when unset: what uvicorn is given."""
        return [] if self.trusted_proxies is None else list(proxy_addresses(self.trusted_proxies))

    @field_validator("token")
    @classmethod
    def check_token(cls, token: SecretStr | None) -> SecretStr | None:
        """Visible ASCII with no spaces: what a client can send in a header, compared as
        bytes. A token pasted with a stray space or a non-ASCII letter would otherwise
        never match, and look like a broken server."""
        if token is not None:
            value = token.get_secret_value()
            if not all("!" <= char <= "~" for char in value):
                raise ValueError(
                    "must be visible ASCII characters with no spaces (for example the output "
                    "of `openssl rand -hex 32`)"
                )
        return token


def is_loopback(host: str) -> bool:
    """Whether ``host``, a name or an address with no port, is this machine itself.

    ``localhost`` and every loopback address (``127.0.0.0/8``, ``::1``). Any other name is
    not, even one that resolves to loopback today: without a token, the server must never
    listen on, or answer to, a name someone else's DNS controls (DEC-18).
    """
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def describe_settings_error(error: ValidationError, prefix: str = ENV_API_PREFIX) -> str:
    """One ``<PREFIX><NAME>: problem`` line per error, never echoing a value: a token's or
    a key's own value must not reach a terminal or a log."""
    lines = []
    for detail in error.errors():
        name = prefix + "_".join(str(part) for part in detail["loc"]).upper()
        lines.append(f"{name}: {detail['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)
