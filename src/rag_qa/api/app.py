"""``create_app``: the FastAPI application and its lifespan (DEC-18).

The lifespan builds what the API serves (:class:`rag_qa.api.state.Serving`) once, in a
thread, before the server accepts a connection. It closes the model's HTTP clients at
shutdown, on the loop that used them. The server starts without an index, or with one the
config cannot use: it reports not-ready until an ingest gives it one.

Two refusals apply to every route, through the app's own dependencies: requests from a
web page (``Origin``) and, with no token set, a ``Host`` that is not loopback. The
interactive docs are re-added as ordinary routes, so they sit behind the token too. The
first dependency gives each request's log lines an ID of their own (S3-7).

While it serves, the metrics read each breaker's state and the documents served at every
scrape (:func:`rag_qa.observability.metrics.watch`, S3-7).

**Traces** (S3-7b). While a command has tracing on, every request but the health probes gets
a server span from the contrib instrumentation (:func:`rag_qa.observability.tracing.instrument`),
and the lifespan sends what is left at shutdown. **FastAPI's own OpenTelemetry is always off**,
by explicit flags. FastAPI 0.143 has its own, which switches on whenever a global provider is
set, all of it on by default. It ends a dropped client's server span as an error, where a
cancel is the client's choice, and it wraps every request in a context attach of its own. Its
``auto_configure`` would add exporters of its own under ``FASTAPI_OTEL_AUTO_CONFIGURE=true``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.routing import APIRoute

from rag_qa import __version__
from rag_qa.api import routes
from rag_qa.api.state import Serving
from rag_qa.observability import metrics, tracing
from rag_qa.schema import RagConfig
from rag_qa.settings import ServerSettings

logger = logging.getLogger(__name__)


def create_app(
    config: RagConfig, settings: ServerSettings, *, serving: Serving | None = None
) -> FastAPI:
    """The API for ``config``. ``serving`` is for tests, which pass their own; otherwise
    the lifespan builds it from ``config`` (seconds: the models load)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        served = serving
        if served is None:
            served = await asyncio.to_thread(
                Serving.build, config, max_concurrent=settings.max_concurrent
            )
        app.state.serving = served
        live: Serving = served  # one name, never reassigned, for the lambdas
        metrics.watch(lambda: breaker_states(live), lambda: documents_served(live))
        try:
            yield
        finally:
            metrics.unwatch()
            running = served.jobs.running
            if running is not None:
                logger.warning(
                    "Ingest job %s is still running and is abandoned. The live index is "
                    "unchanged, and the next ingest cleans up what it left.",
                    running.job_id,
                )
            try:
                await served.aclose()
            finally:
                # Here, not only at the end of rag-serve's main: a SIGTERM ends the process
                # once uvicorn has shut down, before any finally of main runs. In a thread,
                # since a dead collector may hold it for the exporter's timeout.
                await asyncio.to_thread(tracing.flush)

    app = FastAPI(
        title="RAG_QA_System",
        version=__version__,
        summary="Answers questions about a document corpus, streamed over SSE.",
        lifespan=lifespan,
        docs_url=None,  # re-added in routes.docs, behind the token
        redoc_url=None,
        openapi_url=None,
        # FastAPI's own OpenTelemetry, off whatever the environment holds: the module docstring.
        telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
        dependencies=[
            Depends(routes.bind_request),  # first, so every line of the request carries it
            Depends(routes.log_spec_version),
            Depends(routes.refuse_browsers),
            Depends(routes.check_host),
        ],
    )
    app.state.settings = settings
    app.include_router(routes.health)
    app.include_router(routes.v1)
    app.include_router(routes.docs)
    # Before the first request builds the middleware stack. The probes are asked every few
    # seconds and would bury the answers' traces.
    probes = [route.path for route in routes.health.routes if isinstance(route, APIRoute)]
    tracing.instrument(app, untraced=probes)
    return app


def breaker_states(serving: Serving) -> Mapping[str, str]:
    """Each breaker's state, by the role ``GET /v1/health`` names it by."""
    return {role: breaker.state for role, breaker in serving.breakers().items()}


def documents_served(serving: Serving) -> int:
    """The documents in the index generation served, 0 while none is."""
    manifest = serving.served.manifest
    return 0 if manifest is None else len(manifest.documents)
