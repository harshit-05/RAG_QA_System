"""``rag-serve``: the HTTP API under uvicorn (FR-6, DEC-18).

By default it listens on 127.0.0.1:8000 and needs no token. ``RAG_API_TOKEN`` turns on
bearer-token authentication, and only then may ``--host`` be anything but loopback. Over
plain HTTP the token travels in cleartext until Phase 3 adds TLS, so ``rag-serve`` warns
when it listens beyond this machine.

Before uvicorn starts, it checks that the language model's server answers. A failure
inside uvicorn's start-up would end in a traceback, not a reason (Starlette hands uvicorn
the formatted traceback of a failed lifespan).

uvicorn reports ASGI spec 2.3, under which a client disconnect cancels its answer at once,
during the model's prompt evaluation too. That is why this command runs uvicorn and
nothing else (DEC-18).

``rag-serve`` exit codes: **0** the server ran and was stopped; **2** it could not start
(invalid settings, a non-loopback host without a token, a configuration error, the
model's server not answering, the address in use, or a usage error).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

import uvicorn
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from rag_qa.api.app import create_app
from rag_qa.components import aclose_llm, build_llm, llm_reachable
from rag_qa.config import ConfigError, load_config
from rag_qa.schema import RagConfig
from rag_qa.settings import (
    ENV_CONFIG,
    ServerSettings,
    describe_settings_error,
    is_loopback,
)

EXIT_OK = 0
EXIT_CANNOT_START = 2

#: Seconds a stopping server waits for answers in flight before cancelling them. An answer
#: takes minutes on CPU, and a cancel is what stops its generation (DEC-14).
GRACEFUL_SHUTDOWN_S = 5

logger = logging.getLogger("rag_qa.api.server")

OLLAMA_HINT = "Is Ollama running? Try: systemctl status ollama"


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{value} is not a port (1-65535)")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-serve",
        description="Serve questions and ingestion over HTTP: answers stream as "
        "Server-Sent Events. Set RAG_API_TOKEN to require a bearer token.",
        epilog="Exit status: 0 the server ran and was stopped; 2 could not start (invalid "
        "settings, a non-loopback host without RAG_API_TOKEN, a configuration error, the "
        "language model's server not answering, the address in use, or a usage error).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    parser.add_argument(
        "--host",
        help="address to listen on (default: $RAG_API_HOST, then 127.0.0.1); anything but "
        "loopback needs RAG_API_TOKEN",
    )
    parser.add_argument(
        "--port", type=_port, help="port to listen on (default: $RAG_API_PORT, then 8000)"
    )
    return parser


def configure_logging() -> None:
    """Show the package's own log lines (progress, readiness, ingest jobs) beside uvicorn's.

    Idempotent, so calling ``main`` twice in one process never prints a line twice.
    """
    package = logging.getLogger("rag_qa")
    if any(getattr(handler, "rag_serve", False) for handler in package.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.rag_serve = True  # type: ignore[attr-defined]
    package.addHandler(handler)
    package.setLevel(logging.INFO)


def _reason(error: Exception) -> str:
    """What a failed model build says, without pydantic's wrapping around it."""
    if isinstance(error, ValidationError):
        return "; ".join(
            str(detail["msg"]).removeprefix("Value error, ") for detail in error.errors()
        )
    return f"{type(error).__name__}: {error}"


async def _reachable_then_close(llm: BaseChatModel) -> bool | None:
    try:
        return await llm_reachable(llm)
    finally:
        await aclose_llm(llm)


def preflight(config: RagConfig) -> str | None:
    """Why the language model's server cannot be used, or ``None`` when it answers.

    It builds the model, which with ``validate_model_on_init`` asks Ollama for its models,
    then asks once more with a time limit (the readiness check's), then closes the
    model's clients. The server builds its own afterwards.
    """
    ref = config.pipeline.query.llm
    target = config.component(ref).spec()["_target_"]
    hint = f"\n{OLLAMA_HINT}" if str(target).startswith("langchain_ollama.") else ""
    try:
        llm = build_llm(config)
    # Deliberately broad: whatever stops the model being built stops the server, and
    # it deserves a reason rather than a traceback.
    except Exception as e:  # noqa: BLE001
        return f"cannot build the language model {ref}: {_reason(e)}{hint}"
    if asyncio.run(_reachable_then_close(llm)) is False:
        return f"the language model {ref} did not answer within 2 s{hint}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-serve``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work
    try:
        settings = ServerSettings()
    except ValidationError as e:
        print(f"Error: invalid server settings:\n{describe_settings_error(e)}", file=sys.stderr)
        return EXIT_CANNOT_START
    host = args.host or settings.host
    port = args.port if args.port is not None else settings.port
    if settings.token is None and not is_loopback(host):
        print(
            f"Error: refusing to listen on {host} without a token. Set RAG_API_TOKEN to "
            f"serve beyond this machine, or listen on 127.0.0.1.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    configure_logging()
    if not is_loopback(host):
        logger.warning(
            "Listening on %s: the bearer token travels in cleartext over HTTP until Phase 3 "
            "adds TLS. Use this only on a network you trust.",
            host,
        )
    problem = preflight(config)
    if problem is not None:
        print(f"Error: {problem}", file=sys.stderr)
        return EXIT_CANNOT_START

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(config, settings),
            host=host,
            port=port,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
        )
    )
    try:
        server.run()
    except SystemExit:  # uvicorn exits when it cannot bind the address, after saying why
        return EXIT_CANNOT_START
    return EXIT_OK if server.started else EXIT_CANNOT_START  # a failed lifespan: logged


if __name__ == "__main__":
    sys.exit(main())
