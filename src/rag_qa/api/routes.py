"""The HTTP endpoints (ARCHITECTURE.md §2.4), and the refusals that run before them.

**Every refusal happens before the first byte** (DEC-18). A streaming route's body runs
only once the 200 headers are on their way, so a refusal from inside it would be too
late. Each check is therefore a dependency, and they run in this order:

1. ``Origin`` present: 403, on every route (:func:`refuse_browsers`).
2. With no token set, a ``Host`` that is not loopback: 403 (:func:`check_host`).
3. With a token set, a missing or wrong bearer token: 401 (:func:`require_token`).
   Every route needs it except the two health probes.
4. ``POST /v1/query`` only: no free slot, 503 with ``Retry-After`` (:func:`take_slot`).
5. ``POST /v1/query`` only: no index the config can use, 503 (:func:`served_pipeline`).
6. ``POST /v1/query`` only: a body that breaks the request model, then a source the served
   index does not hold, 422 (:func:`known_sources`).

**The body comes in two steps** (S2-7's first review). FastAPI parses it before it runs any
dependency, so a body that is not valid JSON gets its 422 ahead of the 403 and the 401
above; nothing runs, but the body has been read before the token is checked. A body that
parses but breaks the request model is validated after the dependencies, so that 422 is
reported last. FastAPI also refuses a POST whose body is not ``application/json`` (strict
content type), which is what a web page cannot send cross-site without a CORS preflight.

**The answer stream** (``POST /v1/query``) is Server-Sent Events: ``sources``, then
``token``…, then ``done``, or ``error`` if the answer fails after the stream began.
:func:`rag_qa.answering.stream_answer` runs inside the route's own generator, and the
route spawns no task. A client that disconnects cancels it there: Starlette notices the
disconnect (uvicorn reports ASGI spec 2.3), and the request's teardown cancels FastAPI's
own producer task, whose cancel lands in the model's read and closes the stream to
Ollama. The slot's teardown closes the stream as well, in case the cancel landed between
events instead (:meth:`rag_qa.api.state.Slot.aclose`).
"""

import asyncio
import logging
import secrets
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import asdict
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from rag_qa.answering import AnswerEvent, Sources, Token, stream_answer
from rag_qa.api.jobs import IngestJob, JobRunningError
from rag_qa.api.models import (
    DocumentInfo,
    Documents,
    Health,
    IndexHealth,
    IngestAccepted,
    IngestJobOut,
    IngestReportOut,
    IngestRequest,
    LlmHealth,
    QueryRequest,
)
from rag_qa.api.state import NO_INDEX_YET, REINGEST, Serving, Slot
from rag_qa.chain import IncompatibleIndexError, NoIndexError, QueryPipeline
from rag_qa.components import llm_reachable
from rag_qa.settings import ServerSettings, is_loopback

logger = logging.getLogger(__name__)

#: How long a busy server asks a client to wait before asking again, in seconds. An
#: answer takes minutes on CPU; this is how often it is reasonable to check.
RETRY_AFTER_S = 30

#: Seconds Ollama has to answer the readiness check (ARCHITECTURE.md §2.4).
LLM_CHECK_TIMEOUT_S = 2.0


# --- dependencies: the refusals, in order -------------------------------------------------


def get_serving(request: Request) -> Serving:
    return cast(Serving, request.app.state.serving)


def get_settings(request: Request) -> ServerSettings:
    return cast(ServerSettings, request.app.state.settings)


def log_spec_version(request: Request) -> None:
    """Log, once, which ASGI spec version the server reports.

    The disconnect handling depends on it. Below 2.4, Starlette listens for a disconnect
    while a response streams, during a silent prefill too. From 2.4 it notices only when a
    send fails, which for an answer stream means at the next keep-alive ping (15 s).
    uvicorn reports 2.3; this says so if a later server does not (DEC-18).
    """
    state = request.app.state
    if getattr(state, "spec_version_logged", False):
        return
    state.spec_version_logged = True
    spec = str(request.scope.get("asgi", {}).get("spec_version", "2.0"))
    if tuple(int(part) for part in spec.split(".")) >= (2, 4):
        logger.warning(
            "The server reports ASGI spec %s: a client that disconnects while the model reads "
            "the prompt is noticed only at the next keep-alive ping, so generation can run "
            "on for up to 15 s after it left. rag-serve's uvicorn reports 2.3.",
            spec,
        )
    else:
        logger.info("The server reports ASGI spec %s: a disconnect cancels its answer.", spec)


def refuse_browsers(request: Request) -> None:
    """403 for any request that carries ``Origin``: browsers add it, and nothing else does.

    A cross-site "simple" POST from a web page reaches a server on 127.0.0.1 with a
    loopback ``Host``, so the Host check cannot stop it; ``Origin`` does. curl, the CLI and
    scripts send none, and the API serves no browser UI (DEC-18).
    """
    if "origin" in request.headers:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "requests from a web page (with an Origin header) are refused",
        )


def host_of(header: str) -> str:
    """The host in a ``Host`` header, without its port: ``[::1]:8000`` gives ``::1``."""
    if header.startswith("["):
        return header[1:].partition("]")[0]
    return header.rpartition(":")[0] if header.count(":") == 1 else header


def check_host(
    request: Request, settings: Annotated[ServerSettings, Depends(get_settings)]
) -> None:
    """Without a token, 403 for a ``Host`` that is not loopback: DNS rebinding.

    A web page can point its own domain at 127.0.0.1 and then call the server as
    same-origin; the request still names that domain in ``Host``. With a token set, the
    server may be reached by any name, and a rebound page has no token to send.
    """
    if settings.token is None and not is_loopback(host_of(request.headers.get("host", ""))):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "without RAG_API_TOKEN this server answers only to a loopback Host header",
        )


def require_token(
    request: Request, settings: Annotated[ServerSettings, Depends(get_settings)]
) -> None:
    """With a token set, 401 unless the request carries ``Authorization: Bearer <token>``.

    Exactly that: the scheme in any case, one space, then the token, nothing stripped. The
    comparison is constant-time and on bytes: ``secrets.compare_digest`` raises
    ``TypeError`` for a non-ASCII ``str``, which would turn a hostile header into a 500.
    Every failure gets the same answer, so none says which part was wrong.
    """
    if settings.token is None:
        return
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    expected = settings.token.get_secret_value().encode("ascii")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        credentials.encode("utf-8"), expected
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "a valid bearer token is required (Authorization: Bearer <token>)",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def take_slot(serving: Annotated[Serving, Depends(get_serving)]) -> AsyncIterator[Slot]:
    """A slot for one answer, or 503 with ``Retry-After`` rather than a wait in a queue.

    The teardown (after the response, ``scope="request"``) closes whatever answer stream
    the slot still holds, then releases the slot. It covers a client that leaves before the
    stream's own ``finally`` could run, and a stream paused between events.
    """
    slot = serving.slots.take()
    if slot is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "busy: the server is answering as many questions as it can; ask again later",
            headers={"Retry-After": str(RETRY_AFTER_S)},
        )
    try:
        yield slot
    finally:
        await slot.aclose()


async def served_pipeline(serving: Annotated[Serving, Depends(get_serving)]) -> QueryPipeline:
    """The pipeline to answer with, after picking up any new index generation; 503 when
    there is no index the config can use."""
    await serving.refresh()
    if serving.served.problem is not None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, serving.served.problem)
    return serving.pipeline


def known_sources(
    body: QueryRequest,
    serving: Annotated[Serving, Depends(get_serving)],
    _: Annotated[QueryPipeline, Depends(served_pipeline)],
) -> QueryRequest:
    """The request, once every source it names is in the served index; 422 naming the
    unknown ones otherwise (S3-5, FR-3).

    The body is declared here rather than on the route, after the index's own check, so the
    refusals keep their order: the index's 503, then a body that breaks the model (422),
    then unknown sources (422). The names come from the client, so repeating them leaks
    nothing; the model bounds their count and length.
    """
    manifest = serving.served.manifest
    held = manifest.documents if manifest is not None else {}
    unknown = [source for source in dict.fromkeys(body.sources) if source not in held]
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"unknown sources, not in the index: {', '.join(map(repr, unknown))}. A source is a "
            f"path relative to the corpus, as GET /v1/documents lists them",
        )
    return body


# --- routes -------------------------------------------------------------------------------

#: The health probes: never behind the token, so a supervisor can always ask.
health = APIRouter(prefix="/v1/health", tags=["health"])
#: Everything else under /v1, behind the token when one is set.
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])
#: The interactive docs and the schema. Behind the token too: they describe the surface.
docs = APIRouter(dependencies=[Depends(require_token)], include_in_schema=False)


def error_message(error: Exception) -> str:
    """A fixed message for an answer that failed after its stream began.

    Fixed, because the exceptions' own text carries paths and URLs: the full exception
    goes to the server log only (DEC-18). The model's errors are told apart by class name,
    so this module imports neither httpx nor ollama; the text is for people, never logic.
    """
    if isinstance(error, IncompatibleIndexError):
        return REINGEST
    if isinstance(error, NoIndexError):
        return NO_INDEX_YET
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {"ConnectError", "ConnectTimeout", "ConnectionError"}:
        return "the language model is unreachable"
    if names & {"TimeoutException", "TimeoutError"}:
        return "the language model timed out"
    if "ResponseError" in names:
        return "the language model returned an error"
    return "the answer failed; the server log has the detail"


def answer_event(event: AnswerEvent) -> ServerSentEvent:
    """One answer event as SSE. The data is a plain dict: FastAPI writes it as JSON with
    every non-ASCII character escaped, so no character inside it can split a line."""
    if isinstance(event, Sources):
        return ServerSentEvent(
            event="sources", data={"sources": [asdict(ref) for ref in event.sources]}
        )
    if isinstance(event, Token):
        return ServerSentEvent(event="token", data={"text": event.text})
    return ServerSentEvent(
        event="done", data={"ttft_ms": event.ttft_ms, "total_ms": event.total_ms}
    )


@v1.post(
    "/query",
    response_class=EventSourceResponse,
    responses={
        401: {"description": "a token is set, and the request did not carry it"},
        422: {"description": "a malformed body, or a source the served index does not hold"},
        503: {"description": "busy (with Retry-After), or no index the config can use"},
    },
)
async def query(
    slot: Annotated[Slot, Depends(take_slot, scope="request")],
    pipeline: Annotated[QueryPipeline, Depends(served_pipeline)],
    body: Annotated[QueryRequest, Depends(known_sources)],
) -> AsyncIterator[ServerSentEvent]:
    """Answer a question as a stream: ``sources``, ``token``…, then ``done`` or ``error``.
    With ``sources``, only those corpus documents are searched."""
    answer = cast(
        AsyncGenerator[AnswerEvent, None],
        stream_answer(pipeline, body.question, sources=tuple(dict.fromkeys(body.sources))),
    )
    slot.hold(answer)
    try:
        async for event in answer:
            yield answer_event(event)
    except asyncio.CancelledError:
        logger.info("Answer cancelled: its client disconnected")
        raise
    # Deliberately broad: after the 200, an error event is the only way left to say the
    # answer failed. A stream that ends with neither done nor error was cut off.
    except Exception as e:
        logger.exception("An answer failed after its stream began")
        yield ServerSentEvent(
            event="error", data={"type": type(e).__name__, "message": error_message(e)}
        )
    finally:
        slot.release()


def job_out(job: IngestJob) -> IngestJobOut:
    report = None if job.report is None else IngestReportOut(**asdict(job.report))
    return IngestJobOut(
        job_id=job.job_id,
        status=job.status,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        report=report,
        log=list(job.log),
    )


@v1.post("/ingest", status_code=status.HTTP_202_ACCEPTED, responses={409: {}})
async def start_ingest(
    body: IngestRequest,
    response: Response,
    serving: Annotated[Serving, Depends(get_serving)],
) -> IngestAccepted:
    """Start an ingest job, as ``rag-ingest`` (``--rebuild`` with ``rebuild: true``)."""
    try:
        job = serving.jobs.start(rebuild=body.rebuild)
    except JobRunningError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"ingest job {e.job.job_id} is running; one runs at a time",
        ) from None
    response.headers["Location"] = f"/v1/ingest/{job.job_id}"
    return IngestAccepted(job_id=job.job_id, status=job.status)


@v1.get("/ingest/{job_id}", responses={404: {}})
async def ingest_status(
    job_id: str, serving: Annotated[Serving, Depends(get_serving)]
) -> IngestJobOut:
    """An ingest job: its status, ``rag-ingest``'s exit code, the report and the log."""
    job = serving.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such ingest job (the last 20 are kept)")
    return job_out(job)


@v1.get("/documents", responses={503: {}})
async def documents(serving: Annotated[Serving, Depends(get_serving)]) -> Documents:
    """The documents in the index being served, from its manifest."""
    await serving.refresh()
    manifest = serving.served.manifest
    if manifest is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, serving.served.problem or NO_INDEX_YET
        )
    return Documents(
        documents=[
            DocumentInfo(
                source=name,
                sha256=record.sha256,
                chunks=record.chunks,
                loader=record.loader,
                ingested_at=record.ingested_at,
            )
            for name, record in sorted(manifest.documents.items())
        ]
    )


@health.get("", responses={503: {"model": Health}})
async def readiness(
    response: Response, serving: Annotated[Serving, Depends(get_serving)]
) -> Health:
    """Ready: an index the config can use is served, and the model's server answers
    within 2 s. 503 otherwise, with what is missing."""
    await serving.refresh()
    served = serving.served
    if served.ready and served.generation is not None and served.manifest is not None:
        index = IndexHealth(
            ready=True,
            generation=served.generation,
            documents=len(served.manifest.documents),
            detail="serving",
        )
    else:
        index = IndexHealth(
            ready=served.ready,
            generation=None,
            documents=None,
            detail=served.problem or "serving",
        )
    reachable = await llm_reachable(serving.pipeline.llm, LLM_CHECK_TIMEOUT_S)
    llm = LlmHealth(
        ready=reachable is not False,
        detail={
            True: "the model's server answered",
            False: f"the model's server did not answer within {LLM_CHECK_TIMEOUT_S:g} s",
            None: "no server to ask: the model is not served by Ollama",
        }[reachable],
    )
    ready = index.ready and llm.ready
    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return Health(status="ready" if ready else "not ready", index=index, llm=llm)


@health.get("/live")
async def liveness() -> dict[str, str]:
    """200 while the process runs."""
    return {"status": "alive"}


@docs.get("/openapi.json")
async def openapi(request: Request) -> JSONResponse:
    return JSONResponse(request.app.openapi())


@docs.get("/docs")
async def swagger(request: Request) -> HTMLResponse:
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{request.app.title} - docs")


@docs.get("/redoc")
async def redoc(request: Request) -> HTMLResponse:
    return get_redoc_html(openapi_url="/openapi.json", title=f"{request.app.title} - ReDoc")
