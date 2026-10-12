"""``create_app``: the FastAPI application and its lifespan (DEC-18).

The lifespan builds what the API serves (:class:`rag_qa.api.state.Serving`) once, in a
thread, before the server accepts a connection. It closes the model's HTTP clients at
shutdown, on the loop that used them. The server starts without an index, or with one the
config cannot use: it reports not-ready until an ingest gives it one.

Two refusals apply to every route, through the app's own dependencies: requests from a
web page (``Origin``) and, with no token set, a ``Host`` that is not loopback. The
interactive docs are re-added as ordinary routes, so they sit behind the token too.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from rag_qa import __version__
from rag_qa.api import routes
from rag_qa.api.state import Serving
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
        try:
            yield
        finally:
            running = served.jobs.running
            if running is not None:
                logger.warning(
                    "Ingest job %s is still running and is abandoned. The live index is "
                    "unchanged, and the next ingest cleans up what it left.",
                    running.job_id,
                )
            await served.aclose()

    app = FastAPI(
        title="RAG_QA_System",
        version=__version__,
        summary="Answers questions about a document corpus, streamed over SSE.",
        lifespan=lifespan,
        docs_url=None,  # re-added in routes.docs, behind the token
        redoc_url=None,
        openapi_url=None,
        dependencies=[
            Depends(routes.log_spec_version),
            Depends(routes.refuse_browsers),
            Depends(routes.check_host),
        ],
    )
    app.state.settings = settings
    app.include_router(routes.health)
    app.include_router(routes.v1)
    app.include_router(routes.docs)
    return app
