"""The HTTP endpoints (ARCHITECTURE.md §2.4), and the refusals that run before them.

**Every refusal happens before the first byte** (DEC-18). A streaming route's body runs
only once the 200 headers are on their way, so a refusal from inside it would be too
late. Each check is therefore a dependency, and they run in this order:

1. ``Origin`` present: 403, on every route (:func:`refuse_browsers`).
2. With no credential configured (no keys file, no token), a ``Host`` that is not loopback:
   403 (:func:`check_host`).
3. Who asks (:func:`authenticate`, DEC-23), on every route but the two health probes: with a
   credential configured, no valid key, 401, or 429 once its address has failed too often;
   a keys file that cannot be used, 503 with ``Retry-After``; then the key's own rate
   limit, 429 with ``Retry-After``. With none, its address's rate limit.
4. The route's scope (:func:`scope`): ``query`` for questions and the document list,
   ``ingest`` for the ingest jobs; 403 without it. The docs and the schema need none.
5. ``POST /v1/query`` only: a dependency the answer needs is down, its breaker open, 503
   with ``Retry-After`` (:func:`refuse_while_down`, DEC-21). It takes no slot.
6. ``POST /v1/query`` only: no free slot, 503 with ``Retry-After`` (:func:`take_slot`).
7. ``POST /v1/query`` only: no index the config can use, 503 (:func:`served_pipeline`).
8. ``POST /v1/query`` only: a body that breaks the request model, then a source the served
   index does not hold, 422 (:func:`known_sources`).

**The body comes in three steps** (S2-7's first review, S3-8). Its size first: a body over
64 KiB gets 413 before FastAPI reads it, ahead of everything above
(:class:`rag_qa.api.limits.BodyLimit`, a middleware). FastAPI then parses it before it runs any
dependency, so a body that is not valid JSON gets its 422 ahead of the 403s, the 401 and the
429s above, and draws on no bucket;
nothing runs, and the body read before the token is checked is at most 64 KiB. A body that
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
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from typing import Annotated, NoReturn, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.types import Scope

from rag_qa import resilience
from rag_qa.answering import AnswerEvent, Done, Sources, Token, answered_line, stream_answer
from rag_qa.api import auth
from rag_qa.api.auth import ApiKey, KeyRing
from rag_qa.api.jobs import KEEP, IngestJob, JobRunningError
from rag_qa.api.limits import AUTH_FAILURES_PER_MINUTE, Limits, client_key
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
    KeysHealth,
    LlmHealth,
    QueryRequest,
)
from rag_qa.api.state import (
    LLM_CHECK_TIMEOUT_S,
    NO_INDEX_YET,
    REINGEST,
    STORE_DOWN,
    Serving,
    Slot,
    breaker_refusal,
    wait_s,
)
from rag_qa.chain import IncompatibleIndexError, NoIndexError, QueryPipeline
from rag_qa.components import FALLBACK_SERVICE, raised_by_fallback
from rag_qa.observability import logs, metrics, tracing
from rag_qa.observability.genai import model_identity
from rag_qa.resilience import OPEN, BreakerOpenError, CircuitBreaker
from rag_qa.settings import ServerSettings, is_loopback
from rag_qa.vectorstore import StoreError

logger = logging.getLogger(__name__)

#: How long a busy server asks a client to wait before asking again, in seconds. An
#: answer takes minutes on CPU; this is how often it is reasonable to check.
RETRY_AFTER_S = 30


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
    """Without a credential, 403 for a ``Host`` that is not loopback: DNS rebinding.

    A web page can point its own domain at 127.0.0.1 and then call the server as
    same-origin; the request still names that domain in ``Host``. With a keys file or a token,
    the server may be reached by any name, a proxy's public one included, and a rebound page
    has no key to send (DEC-23, design review 2).
    """
    if not settings.has_credential and not is_loopback(host_of(request.headers.get("host", ""))):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "without RAG_API_KEYS_FILE or RAG_API_TOKEN this server answers only to a "
            "loopback Host header",
        )


def asks_a_question(scope: Scope) -> bool:
    """Whether a request is ``POST /v1/query``: what ``rag_refusals_total`` counts (S3-7)."""
    return scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/v1/query"


def refuse(
    request: Request,
    code: int,
    reason: str,
    detail: str,
    *,
    headers: dict[str, str] | None = None,
) -> NoReturn:
    """Refuse ``request`` with ``code``, counted under ``reason`` when it is a question."""
    if asks_a_question(request.scope):
        metrics.refused(reason)
    raise HTTPException(code, detail, headers=headers)


#: Who asks when no credential is configured: anyone on this machine, with every scope.
ANONYMOUS = ApiKey(name="", scopes=frozenset(auth.SCOPES), sha256=b"", created="")

UNAUTHENTICATED = "a valid API key is required (Authorization: Bearer <token>)"


async def authenticate(request: Request) -> None:
    """Who is asking, rate-limited (DEC-23, S3-8): ``request.state.caller``, or a refusal.

    - **No credential configured** (no keys file and no token): every caller has every scope,
      and is limited by its address.
    - **Otherwise** a key is required, as ``Authorization: Bearer <token>``: the scheme in any
      case, one space, then the token, nothing stripped. :meth:`rag_qa.api.auth.KeyRing.match`
      compares the token's hash with every key's, on bytes. A request without a valid key
      draws on its address's failure bucket: 401, or 429 once that is empty. Every 401 says
      the same thing, so none says which part was wrong.
    - **A keys file that cannot be used** refuses every bearer token but ``RAG_API_TOKEN`` with
      503, as the 401 would, after drawing on the same failure bucket: otherwise wrong guesses
      at the token would meet unthrottled 503s, and the right one a 200.
    - **A valid key** draws on its own bucket (429), and its name, never its token, is bound
      into the request's log lines (``api_key``) and set on its server span
      (``rag.api_key.name``).

    ``async``, so it runs in the request's own task, where the log binding and the span are.
    """
    keys: KeyRing = request.app.state.keys
    limits: Limits = request.app.state.limits
    address = client_key(request.client.host if request.client is not None else None)
    if not keys.configured:
        wait = limits.addresses.take(address)
        if wait is not None:
            limited(request, metrics.RATE_LIMITED, wait, "this address")
        request.state.caller = ANONYMOUS
        return
    scheme, _, credentials = request.headers.get("authorization", "").partition(" ")
    bearer = scheme.lower() == "bearer"
    key = keys.match(credentials) if bearer else None
    if key is None:
        wait = limits.failures.take(address)
        if wait is not None:
            limited(request, metrics.AUTH_LIMITED, wait, "failed authentications from this address")
        if bearer and keys.problem is not None:
            refuse(
                request,
                status.HTTP_503_SERVICE_UNAVAILABLE,
                metrics.KEYS_UNREADABLE,
                KEYS_UNUSABLE,
                headers={"Retry-After": str(wait_s(auth.RECHECK_S))},
            )
        refuse(
            request,
            status.HTTP_401_UNAUTHORIZED,
            metrics.UNAUTHENTICATED,
            UNAUTHENTICATED,
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Named before its bucket is drawn on, so a key's own 429 says whose it is (S3-8's second
    # review).
    logs.bind(api_key=key.name)
    tracing.annotate({tracing.API_KEY_NAME: key.name})
    wait = limits.keys.take(key.name)
    if wait is not None:
        limited(request, metrics.RATE_LIMITED, wait, "this key")
    request.state.caller = key


#: Why a key is refused while the keys file cannot be used: fixed text, never its path.
KEYS_UNUSABLE = (
    "the server's keys file cannot be used just now, so API keys are refused; ask again after "
    "Retry-After"
)


def limited(request: Request, reason: str, wait: float, whose: str) -> NoReturn:
    """429 with ``Retry-After``: ``whose`` bucket is empty for ``wait`` more seconds."""
    settings = get_settings(request)
    per_minute = AUTH_FAILURES_PER_MINUTE if reason == metrics.AUTH_LIMITED else settings.rate_limit
    refuse(
        request,
        status.HTTP_429_TOO_MANY_REQUESTS,
        reason,
        f"too many requests: {whose} may make {per_minute} a minute; ask again after Retry-After",
        headers={"Retry-After": str(wait_s(wait))},
    )


def scope(name: str) -> Callable[[Request], Awaitable[None]]:
    """A dependency: 403 unless the caller's key holds the scope ``name`` (DEC-23)."""

    async def require(request: Request) -> None:
        caller: ApiKey = request.state.caller
        if name not in caller.scopes:
            refuse(
                request,
                status.HTTP_403_FORBIDDEN,
                metrics.NO_SCOPE,
                f"this API key lacks the {name} scope",
            )

    return require


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

#: The health probes: never behind a key, nor rate-limited, so a supervisor can always ask.
health = APIRouter(prefix="/v1/health", tags=["health"])
#: Everything else under /v1, behind a key when a credential is configured, each route with
#: the scope it needs.
v1 = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])
#: The interactive docs and the schema: any valid key, since they describe the surface.
docs = APIRouter(dependencies=[Depends(authenticate)], include_in_schema=False)
#: Each scope's dependency (DEC-23).
QUERY_SCOPE = Depends(scope(auth.QUERY))
INGEST_SCOPE = Depends(scope(auth.INGEST))


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


#: What every route behind a key may answer before its own work (DEC-23, S3-8).
REFUSED: dict[int | str, dict[str, object]] = {
    401: {"description": "a credential is configured, and the request carried no valid key"},
    403: {"description": "the key lacks this route's scope"},
    413: {"description": "a body over 64 KiB"},
    429: {"description": "a rate limit, with Retry-After"},
    503: {"description": "the keys file cannot be used just now, with Retry-After"},
}


@v1.post(
    "/query",
    response_class=EventSourceResponse,
    dependencies=[QUERY_SCOPE],
    responses={
        **REFUSED,
        422: {"description": "a malformed body, or a source the served index does not hold"},
        503: {
            "description": "a dependency down or busy (both with Retry-After), no index the "
            "config can use, or the keys file unusable (with Retry-After)"
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
    # Cancelled, and waiting for its sources, unless it gets further; the slot counts it at
    # its one release, which every ending reaches, a drop between events included (Slot's
    # docstring).
    slot.begin(received)
    try:
        async for event in answer:
            if isinstance(event, Sources):
                metrics.SOURCES_SECONDS.observe(time.perf_counter() - received)
                slot.waiting_for = metrics.WAITING_FOR_FIRST_TOKEN
            elif isinstance(event, Token):
                if slot.waiting_for is not None:  # the first token
                    slot.waiting_for = None
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
    report = None if job.report is None else IngestReportOut.model_validate(job.report)
    return IngestJobOut(
        job_id=job.job_id,
        status=job.status,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        report=report,
        log=list(job.log),
        abandoned=job.abandoned,
    )


@v1.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[INGEST_SCOPE],
    responses={**REFUSED, 409: {}},
)
async def start_ingest(
    body: IngestRequest,
    request: Request,
    response: Response,
    serving: Annotated[Serving, Depends(get_serving)],
) -> IngestAccepted:
    """Start an ingest job, as ``rag-ingest`` (``--rebuild`` with ``rebuild: true``)."""
    caller: ApiKey = request.state.caller
    try:
        job = await serving.jobs.start(rebuild=body.rebuild, api_key=caller.name or None)
    except JobRunningError as e:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"ingest job {e.job.job_id} is running; one runs at a time",
        ) from None
    response.headers["Location"] = f"/v1/ingest/{job.job_id}"
    return IngestAccepted(job_id=job.job_id, status=job.status)


@v1.get("/ingest/{job_id}", dependencies=[INGEST_SCOPE], responses={**REFUSED, 404: {}})
async def ingest_status(
    job_id: str, serving: Annotated[Serving, Depends(get_serving)]
) -> IngestJobOut:
    """An ingest job, whichever replica ran it: its status, ``rag-ingest``'s exit code, the
    report and the log; ``abandoned`` when its process died before it ended (DEC-23)."""
    job = await serving.jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no such ingest job (the last {KEEP} are kept)"
        )
    return job_out(job)


@v1.get(
    "/documents",
    dependencies=[QUERY_SCOPE],
    responses={
        **REFUSED,
        503: {"description": "no index yet, or the keys file unusable (with Retry-After)"},
    },
)
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
    request: Request, response: Response, serving: Annotated[Serving, Depends(get_serving)]
) -> Health:
    """Ready: an index the config can use is served and Qdrant's breaker is not open, the
    model's server answers within 2 s, or the hosted fallback can answer in its place (its
    breaker not open), and the keys file, when one is set, can be used. 503 otherwise, with
    what is missing. ``breakers`` reports each dependency's breaker, by role (DEC-21).

    Never rate-limited, and cheap (DEC-23): the alias and the model's server are each asked at
    most once every 5 s, whoever polls (:class:`rag_qa.api.state.Probe`); the breakers and
    the keys file are read as they are now."""
    keys = keys_health(request.app.state.keys)
    await serving.index_probe.get()
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
    reachable = await serving.llm_probe.get()
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
    ready = index.ready and llm.ready and (keys is None or keys.ready)
    response.status_code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    breakers = {role: breaker_health(b) for role, b in serving.breakers().items()}
    return Health(
        status="ready" if ready else "not ready",
        index=index,
        llm=llm,
        keys=keys,
        breakers=breakers,
    )


def keys_health(keys: KeyRing) -> KeysHealth | None:
    """The keys file as ``GET /v1/health`` reports it, ``None`` while none is set. It is
    looked at again first, at most every 5 s. Fixed text, never the path (DEC-18), nor a key's
    name, its place or how many keys it holds: health answers anyone (S3-8's second review)."""
    if keys.path is None:
        return None
    keys.refresh()
    if keys.problem is not None:
        return KeysHealth(
            ready=False,
            detail=f"{keys.problem}: every API key is refused with 503 until it is fixed",
        )
    return KeysHealth(ready=True, detail="the keys file is read")


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
