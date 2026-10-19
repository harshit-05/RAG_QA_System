"""``rag-serve``: the HTTP API under uvicorn (FR-6, DEC-18).

By default it listens on 127.0.0.1:8000 and needs no token. ``RAG_API_TOKEN`` turns on
bearer-token authentication, and only then may ``--host`` be anything but loopback. Over
plain HTTP the token travels in cleartext until Phase 3 adds TLS, so ``rag-serve`` warns
when it listens beyond this machine.

Before uvicorn starts, it checks Qdrant (DEC-19), then that the language model's server
answers (:func:`rag_qa.components.preflight`, which ``rag-query`` runs too). A failure
inside uvicorn's start-up would end in a traceback, not a reason (Starlette hands uvicorn
the formatted traceback of a failed lifespan).

uvicorn reports ASGI spec 2.3, under which a client disconnect cancels its answer at once,
during the model's prompt evaluation too. That is why this command runs uvicorn and
nothing else (DEC-18).

``rag-serve`` exit codes: **0** the server ran and was stopped with Ctrl-C; **2** it could
not start (invalid settings, a non-loopback host without a token, a configuration error,
Qdrant unusable, the model's server not answering, the address in use, or a usage error).
A SIGTERM shuts
it down the same way, and uvicorn then ends the process by that signal, as service
managers expect (status 143 in a shell).
"""

import argparse
import logging
import sys
from collections.abc import Sequence

import uvicorn
from pydantic import ValidationError

from rag_qa.api.app import create_app
from rag_qa.components import build_store, preflight
from rag_qa.config import ConfigError, load_config
from rag_qa.settings import (
    ENV_CONFIG,
    ServerSettings,
    describe_settings_error,
    is_loopback,
)
from rag_qa.vectorstore import StoreUnavailableError

EXIT_OK = 0
EXIT_CANNOT_START = 2

#: Seconds a stopping server waits for answers in flight before cancelling them. An answer
#: takes minutes on CPU, and a cancel is what stops its generation (DEC-14).
GRACEFUL_SHUTDOWN_S = 5

logger = logging.getLogger("rag_qa.api.server")


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
        epilog="Exit status: 0 the server ran and was stopped with Ctrl-C; 2 could not "
        "start (invalid settings, a non-loopback host without RAG_API_TOKEN, a "
        "configuration error, the language model's server not answering, the address in "
        "use, or a usage error).",
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
        store = build_store(config)
        store.check()  # Qdrant first (DEC-19): before the model's server, and uvicorn
    except (ConfigError, StoreUnavailableError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START
    store.close()  # the server builds its own, which lives as long as it does

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
    except KeyboardInterrupt:
        # Ctrl-C: uvicorn shut down gracefully, then re-raised the SIGINT it captured, and
        # asyncio's runner turned that into KeyboardInterrupt (S2-9's end-to-end run). The
        # stop was asked for, so the status below reports it, without a traceback.
        pass
    return EXIT_OK if server.started else EXIT_CANNOT_START  # a failed lifespan: logged


if __name__ == "__main__":
    sys.exit(main())
