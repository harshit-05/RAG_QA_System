"""The HTTP endpoints (ARCHITECTURE.md §2.4), and the refusals that run before them.

**Every refusal happens before the first byte** (DEC-18). A streaming route's body runs
only once the 200 headers are on their way, so a refusal from inside it would be too
late. Each check is therefore a dependency, and they run in this order:

1. ``Origin`` present: 403, on every route (:func:`refuse_browsers`).
2. With no token set, a ``Host`` that is not loopback: 403 (:func:`check_host`).
3. With a token set, a missing or wrong bearer token: 401 (:func:`require_token`).
   Every route needs it except the two health probes.
4. ``POST /v1/query`` only: a dependency the answer needs is down, its breaker open, 503
   with ``Retry-After`` (:func:`refuse_while_down`, DEC-21). It takes no slot.
5. ``POST /v1/query`` only: no free slot, 503 with ``Retry-After`` (:func:`take_slot`).
6. ``POST /v1/query`` only: no index the config can use, 503 (:func:`served_pipeline`).
7. ``POST /v1/query`` only: a body that breaks the request model, then a source the served
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

**Logs and metrics** (NFR-10, DEC-22, S3-7). Every request's log lines carry its
``request_id`` (:func:`bind_request`). The refusals count in ``rag_refusals_total``; an answer
times its ``sources`` event, its first token and its end, counts its tokens, and logs one line
when complete; how it ended counts in ``rag_answers_total``. The times run from the request's
first dependency, not from the route's body, which FastAPI starts only once every dependency
has resolved: the alias read before each answer counts, and Qdrant's retries of it (S3-7's
second review). All of it synchronous: the cancel paths await nothing new.
"""

import asyncio
import logging
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import asdict
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from rag_qa import resilience
from rag_qa.answering import AnswerEvent, Done, Sources, Token, answered_line, stream_answer
from rag_qa.api.jobs import IngestJob, JobRunningError
from rag_qa.api.models import (
    BreakerHealth,
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
from rag_qa.api.state import (
    NO_INDEX_YET,
    REINGEST,
    STORE_DOWN,
    Serving,
    Slot,
    breaker_refusal,
    wait_s,
)
from rag_qa.chain import IncompatibleIndexError, NoIndexError, QueryPipeline
from rag_qa.components import FALLBACK_SERVICE, llm_reachable, raised_by_fallback
from rag_qa.observability import logs, metrics
from rag_qa.observability.genai import model_identity
from rag_qa.resilience import OPEN, BreakerOpenError, CircuitBreaker
from rag_qa.settings import ServerSettings, is_loopback
from rag_qa.vectorstore import StoreError

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


async def bind_request(request: Request) -> None:
    """Give the request's log lines a ``request_id`` of their own (S3-7), and note when it
    arrived, which an answer's times start from (``request.state.received_at``). The first of
    the app's dependencies, and ``async``: FastAPI runs a plain function in a worker thread,
    whose context the request never sees, while this runs in the request's own task, which the
    SSE producer task copies."""
    request.state.received_at = time.perf_counter()
    logs.bind_request()


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


def refuse_while_down(serving: Annotated[Serving, Depends(get_serving)]) -> None:
    """503 with ``Retry-After`` while a dependency the answer needs is known to be down: its
    breaker is open (DEC-21, :meth:`rag_qa.api.state.Serving.unavailable`). Before the slot,
    so a refusal holds none, and before the first byte, as every refusal (DEC-18)."""
    down = serving.unavailable()
    if down is not None:
        detail, retry_after = down
        metrics.REFUSALS.labels(code="503", reason=metrics.DEPENDENCY_DOWN).inc()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail,
            headers={"Retry-After": str(retry_after)},
        )


async def take_slot(serving: Annotated[Serving, Depends(get_serving)]) -> AsyncIterator[Slot]:
    """A slot for one answer, or 503 with ``Retry-After`` rather than a wait in a queue.

    The teardown (after the response, ``scope="request"``) closes whatever answer stream
    the slot still holds, then releases the slot. It covers a client that leaves before the
    stream's own ``finally`` could run, and a stream paused between events.
    """
    slot = serving.slots.take()
    if slot is None:
        metrics.REFUSALS.labels(code="503", reason=metrics.BUSY).inc()
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
        metrics.REFUSALS.labels(code="503", reason=metrics.NO_INDEX).inc()
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
    goes to the server log only (DEC-18). The models' errors are told apart by
    :func:`rag_qa.resilience.classify`, which reads class names, so this module imports
    neither httpx, ollama nor openai; the text is for people, never logic. A service is
    named, never its address: the store, the model, or the hosted fallback (S3-6).
    """
    if isinstance(error, IncompatibleIndexError):
        return REINGEST
    if isinstance(error, NoIndexError):
        return NO_INDEX_YET
    if isinstance(error, BreakerOpenError):
        return f"{error.service} is not answering; ask again in a while"
    if isinstance(error, StoreError):
        return "the vector store could not be used; the server log has the detail"
    who = FALLBACK_SERVICE if raised_by_fallback(error) else "the language model"
    verdict = resilience.classify(error)
    if verdict == resilience.CONNECTION:
        return f"{who} is unreachable"
    if verdict == resilience.TIMEOUT:
        return f"{who} timed out"
    if verdict in (resilience.UNAVAILABLE, resilience.ANSWERED):
        return f"{who} returned an error"
    return "the answer failed; the server log has the detail"


def answer_event(event: AnswerEvent) -> ServerSentEvent:
    """One answer event as SSE. The data is a plain dict: FastAPI writes it as JSON with
    every non-ASCII character escaped, so no character inside it can split a line. ``done``
    names the model that answered (S3-6): its config reference; and its token counts,
    ``null`` when the provider reported none (S3-7)."""
    if isinstance(event, Sources):
        return ServerSentEvent(
            event="sources", data={"sources": [asdict(ref) for ref in event.sources]}
        )
    if isinstance(event, Token):
        return ServerSentEvent(event="token", data={"text": event.text})
    return ServerSentEvent(
        event="done",
        data={
            "ttft_ms": event.ttft_ms,
            "total_ms": event.total_ms,
            "model": event.model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
        },
    )


def answered(event: Done, pipeline: QueryPipeline, seconds: float) -> None:
    """Record a complete answer: its time, its tokens under the model that gave them, and
    one log line; and say so when the hosted fallback gave it, since then the question and
    its chunks left this host (DEC-21)."""
    metrics.ANSWER_SECONDS.observe(seconds)
    fallback = event.model != pipeline.llm_ref and pipeline.fallback_llm is not None
    model = pipeline.fallback_llm if fallback else pipeline.llm
    metrics.observe_tokens(model_identity(model), event.input_tokens, event.output_tokens)
    if fallback:
        metrics.FALLBACK_ANSWERS.inc()
        logger.warning(
            "Answered by the hosted fallback %s: the question and its retrieved chunks left "
            "this host",
            event.model,
        )
    logger.info(answered_line(event), extra=asdict(event))


@v1.post(
    "/query",
    response_class=EventSourceResponse,
    responses={
        401: {"description": "a token is set, and the request did not carry it"},
        422: {"description": "a malformed body, or a source the served index does not hold"},
        503: {
            "description": "a dependency down or busy (both with Retry-After), or no index "
            "the config can use"
        },
    },
)
async def query(
    request: Request,
    _: Annotated[None, Depends(refuse_while_down)],
    slot: Annotated[Slot, Depends(take_slot, scope="request")],
    pipeline: Annotated[QueryPipeline, Depends(served_pipeline)],
    body: Annotated[QueryRequest, Depends(known_sources)],
) -> AsyncIterator[ServerSentEvent]:
    """Answer a question as a stream: ``sources``, ``token``…, then ``done`` or ``error``.
    With ``sources``, only those corpus documents are searched. ``done`` names the model
    that answered, which is the hosted fallback's when the language model could not, and its
    token counts."""
    answer = cast(
        AsyncGenerator[AnswerEvent, None],
        stream_answer(pipeline, body.question, sources=tuple(dict.fromkeys(body.sources))),
    )
    slot.hold(answer)
    # From the request's arrival: this body starts only after the dependencies, the alias read
    # in served_pipeline among them, and the client waited for those too.
    received: float = request.state.received_at
    first_token = True
    # Cancelled unless it ends otherwise; the slot counts it at its one release, which every
    # ending reaches, a drop between events included (Slot's docstring).
    slot.outcome = metrics.CANCELLED
    try:
        async for event in answer:
            if isinstance(event, Sources):
                metrics.SOURCES_SECONDS.observe(time.perf_counter() - received)
            elif isinstance(event, Token):
                if first_token:
                    first_token = False
                    metrics.TTFT_SECONDS.observe(time.perf_counter() - received)
            else:
                slot.outcome = metrics.ANSWERED
                answered(event, pipeline, time.perf_counter() - received)
            yield answer_event(event)
    except asyncio.CancelledError:
        logger.info("Answer cancelled: its client disconnected")
        raise
    # Deliberately broad: after the 200, an error event is the only way left to say the
    # answer failed. A stream that ends with neither done nor error was cut off.
    except Exception as e:
        slot.outcome = metrics.FAILED
        refusal = breaker_refusal(e)
        if refusal is not None:
            # An expected refusal, not a fault: one line, no traceback (S3-6's backlog line).
            wait = wait_s(refusal.retry_after_s)
            logger.warning(
                "An answer was refused after its stream began: %s is not answering, and is "
                "asked again in %d s",
                refusal.service,
                wait,
                extra={"service": refusal.service, "retry_after_s": wait},
            )
        else:
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


def breaker_health(breaker: CircuitBreaker) -> BreakerHealth:
    """One breaker as ``GET /v1/health`` reports it."""
    state = breaker.state
    wait = wait_s(breaker.retry_after_s()) if state == OPEN else None
    return BreakerHealth(state=state, retry_after_s=wait)


@health.get("", responses={503: {"model": Health}})
async def readiness(
    response: Response, serving: Annotated[Serving, Depends(get_serving)]
) -> Health:
    """Ready: an index the config can use is served and Qdrant's breaker is not open, and
    the model's server answers within 2 s, or the hosted fallback can answer in its place
    (its breaker not open). 503 otherwise, with what is missing. ``breakers`` reports each
    dependency's breaker, by role (DEC-21)."""
    await serving.refresh()
    served = serving.served
    pipeline = serving.pipeline
    if serving.store.breaker.state == OPEN:
        index = IndexHealth(ready=False, generation=None, documents=None, detail=STORE_DOWN)
    elif served.ready and served.generation is not None and served.manifest is not None:
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
    reachable = await llm_reachable(pipeline.llm, LLM_CHECK_TIMEOUT_S)
    covered = (
        reachable is False
        and pipeline.fallback_llm is not None
        and pipeline.fallback_breaker.state != OPEN
    )
    silent = f"the model's server did not answer within {LLM_CHECK_TIMEOUT_S:g} s"
    llm = LlmHealth(
        ready=reachable is not False or covered,
        detail=(
            f"{silent}; {FALLBACK_SERVICE} answers in its place"
            if covered
            else {
                True: "the model's server answered",
                False: silent,
                None: "no server to ask: the model is not served by Ollama",
            }[reachable]
        ),
    )
    ready = index.ready and llm.ready
    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    breakers = {role: breaker_health(b) for role, b in serving.breakers().items()}
    return Health(status="ready" if ready else "not ready", index=index, llm=llm, breakers=breakers)


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
