"""``rag-serve``: the HTTP API under uvicorn (FR-6, DEC-18).

By default it listens on 127.0.0.1:8000 and needs no token. ``RAG_API_TOKEN`` turns on
bearer-token authentication, and only then may ``--host`` be anything but loopback. Over
plain HTTP the token travels in cleartext until Phase 3 adds TLS, so ``rag-serve`` warns
when it listens beyond this machine.

Before uvicorn starts, it checks Qdrant (DEC-19), then that the language model's server
answers (:func:`rag_qa.components.preflight`, which ``rag-query`` runs too). A failure
inside uvicorn's start-up would end in a traceback, not a reason (Starlette hands uvicorn
the formatted traceback of a failed lifespan). With a hosted fallback configured, Ollama
down is a warning in the log, and the fallback answers until it is back (DEC-21, S3-6).

uvicorn reports ASGI spec 2.3, under which a client disconnect cancels its answer at once,
during the model's prompt evaluation too. That is why this command runs uvicorn and
nothing else (DEC-18).

**Logs, metrics and traces** (NFR-10, DEC-22, S3-7, S3-7b). Its log lines, uvicorn's
included, go through :func:`rag_qa.observability.logs.configure_logging`: JSON when stderr is
not a terminal, or with ``RAG_LOG_FORMAT=json``. With ``RAG_API_METRICS_PORT`` set, Prometheus
scrapes the metrics there, on the API's own host, from a listener of its own that starts
before uvicorn and stops after it. With ``OTEL_EXPORTER_OTLP_ENDPOINT`` set, every request
but the health probes is traced, and its spans go to that collector
(:mod:`rag_qa.observability.tracing`).

``rag-serve`` exit codes: **0** the server ran and was stopped with Ctrl-C; **2** it could
not start (invalid settings, a trace endpoint that is not an http(s) URL, a non-loopback host
without a token, a configuration error, Qdrant unusable, the model's server not answering
with no hosted fallback configured, the address in use, the metrics port in use or the API's
own, or a usage error).
A SIGTERM shuts
it down the same way, and uvicorn then ends the process by that signal, as service
managers expect (status 143 in a shell).
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from rag_qa.api.app import create_app
from rag_qa.components import build_store, preflight
from rag_qa.config import ConfigError, load_config
from rag_qa.observability import metrics, tracing
from rag_qa.observability.logs import LogSettingsError, configure_logging
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
        "configuration error, the language model's server not answering with no hosted "
        "fallback configured, the address in use, the metrics port in use or the API's "
        "own, or a usage error). Set RAG_API_METRICS_PORT for Prometheus's metrics, and "
        "OTEL_EXPORTER_OTLP_ENDPOINT to send traces to a collector.",
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


def uvicorn_config(
    app: FastAPI, settings: ServerSettings, host: str, port: int, **options: Any
) -> uvicorn.Config:
    """How ``rag-serve`` runs ``app`` under uvicorn; ``options`` are for the tests, which run
    the same configuration on a socket of their own.

    **``forwarded_allow_ips`` is always passed** (DEC-23, design review 2): the trusted proxies'
    addresses (``RAG_API_TRUSTED_PROXIES``), or none. Left out, uvicorn would believe the
    ``X-Forwarded-For`` of any loopback client (``FORWARDED_ALLOW_IPS``, by default
    ``127.0.0.1,::1``), and the rate limits would count whatever address it named.
    """
    chosen: dict[str, Any] = {
        "host": host,
        "port": port,
        "forwarded_allow_ips": settings.proxies,
        "timeout_graceful_shutdown": GRACEFUL_SHUTDOWN_S,
        # uvicorn's lines go through our formatter (S3-7): no config of its own, and so it sets
        # its loggers' level itself.
        "log_config": None,
        "log_level": "info",
    }
    return uvicorn.Config(app, **(chosen | options))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-serve``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work
    try:
        configure_logging(detailed=True)
    except LogSettingsError as e:
        print(f"Error: invalid log settings: {e}", file=sys.stderr)
        return EXIT_CANNOT_START
    try:
        tracing.configure("rag-serve")
    except tracing.TraceSettingsError as e:
        print(f"Error: invalid trace settings: {e}", file=sys.stderr)
        return EXIT_CANNOT_START
    try:
        return _serve(args)
    finally:
        tracing.shutdown()  # what the lifespan's flush left, on every ending but a SIGTERM


def _serve(args: argparse.Namespace) -> int:
    """Everything after the logs and the traces are set up: see :func:`main`."""
    try:
        settings = ServerSettings()
    except ValidationError as e:
        print(f"Error: invalid server settings:\n{describe_settings_error(e)}", file=sys.stderr)
        return EXIT_CANNOT_START
    host = args.host or settings.host
    port = args.port if args.port is not None else settings.port
    if settings.metrics_port == port:
        print(
            f"Error: RAG_API_METRICS_PORT is {port}, the API's own port: give the metrics "
            f"another one.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_START
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

    if not is_loopback(host):
        logger.warning(
            "Listening on %s: the bearer token travels in cleartext over HTTP until Phase 3 "
            "adds TLS. Use this only on a network you trust.",
            host,
        )
    check = preflight(config)
    if check.problem is not None:
        print(f"Error: {check.problem}", file=sys.stderr)
        return EXIT_CANNOT_START
    if check.warning is not None:  # Ollama down, and the hosted fallback answers (DEC-21)
        logger.warning("%s", check.warning)
    fallback = config.pipeline.query.fallback_llm
    if fallback is not None and check.warning is None:
        logger.info(
            "When the language model cannot answer, the hosted fallback %s does: the question "
            "and its retrieved chunks are then sent to %s.",
            fallback,
            config.component(fallback).spec()["base_url"],
        )

    server = uvicorn.Server(uvicorn_config(create_app(config, settings), settings, host, port))
    stop_metrics: Callable[[], None] | None = None
    if settings.metrics_port is not None:
        try:
            stop_metrics = metrics.serve(host, settings.metrics_port)
        except OSError as e:
            print(
                f"Error: cannot serve the metrics on port {settings.metrics_port} "
                f"({e.strerror or e}): set RAG_API_METRICS_PORT to a free port.",
                file=sys.stderr,
            )
            return EXIT_CANNOT_START
        logger.info("Serving the metrics on %s, port %d, at /metrics", host, settings.metrics_port)
    try:
        server.run()
    except SystemExit:  # uvicorn exits when it cannot bind the address, after saying why
        return EXIT_CANNOT_START
    except KeyboardInterrupt:
        # Ctrl-C: uvicorn shut down gracefully, then re-raised the SIGINT it captured, and
        # asyncio's runner turned that into KeyboardInterrupt (S2-9's end-to-end run). The
        # stop was asked for, so the status below reports it, without a traceback.
        pass
    finally:
        if stop_metrics is not None:
            stop_metrics()
    return EXIT_OK if server.started else EXIT_CANNOT_START  # a failed lifespan: logged


if __name__ == "__main__":
    sys.exit(main())
