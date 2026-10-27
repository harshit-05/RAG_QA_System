"""Traces: spans for every answer and every ingest run, sent to a collector (NFR-10, DEC-22,
S3-7b).

**Off unless a collector is named.** :func:`configure`, which ``rag-serve``, ``rag-query`` and
``rag-ingest`` call once their logging is set up, turns tracing on only when
``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (:class:`rag_qa.settings.TraceSettings`). A tracer
provider of our own then batches the spans and sends them over OTLP/HTTP to that address plus
``/v1/traces``; the exporter reads the SDK's other variables itself, the timeout among them
(``OTEL_EXPORTER_OTLP_TIMEOUT``, in seconds in Python, not the spec's milliseconds). A dead
collector costs at most that timeout at exit, with the exporter's warnings in the log. Off,
every helper below does nothing at all: no span, no context touched, and our code imports
neither the exporter nor the instrumentation. (langsmith, which langchain-core imports, takes
the SDK's ``TracerProvider`` as a type whenever the SDK is installed, and configures nothing.)
``rag-eval`` never calls it: its record is its files, and an endpoint left in the
shell must not trace a tier-2 run.

**Scrubbed on the way out** (:mod:`rag_qa.observability.scrub`). The collector is not this
host, so every span leaves with its exception events cut to the class name and no status
description: the contrib server span records an unhandled exception's message and stacktrace
itself, and has no switch to stop. The full traceback is in the log, on this host.

**Never the global provider.** The shared code, retrieval and the answer stream, gets a no-op
tracer unless a command turned tracing on here, whatever the environment holds. And FastAPI
0.143 has OpenTelemetry of its own, which switches on whenever a global provider is set (the
API turns it off besides, :func:`rag_qa.api.app.create_app`).

**Three shapes of span** (DEC-22):

- :func:`span`, for synchronous code: current while it runs, so its children find it, and
  ended in a ``finally``. In a coroutine it may enclose ``await``\\ s, never a ``yield``: a
  context attached in one task and detached in another fails ("Failed to detach context").
- :func:`start`, for :func:`rag_qa.answering.stream_answer`, an async generator: an explicit
  parent and never current, so no ``yield`` carries it from task to task; its caller ends it
  in a ``finally``.
- :func:`current` and :func:`attached`, to carry a parent into a thread of our own (an ingest
  job). An executor thread gets a copy of the whole context instead
  (:func:`rag_qa.api.state.serialized`).

**What a span records.** A failure sets status ERROR and ``error.type`` to the exception's
class name, never its message, which may carry a path or a document's text (:func:`failed`). A
cancel adds the event and the attribute ``rag.cancelled``, and leaves the status unset: the
client's choice, not an error (:func:`cancelled`). No question, prompt, chunk or answer text
goes into any span, and no exception's message or stacktrace leaves the host in one, whichever
instrumentation recorded it.

Loads no model stack (``tests/test_architecture.py``): the OpenTelemetry API, and the SDK only
once tracing is on.
"""

import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from opentelemetry import context as otel_context
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import (
    INVALID_SPAN,
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    get_current_span,
)
from opentelemetry.util.types import Attributes, AttributeValue
from pydantic import ValidationError

from rag_qa import __version__
from rag_qa.settings import ENV_OTEL_PREFIX, TraceSettings, describe_settings_error

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

#: The instrumentation scope of our own spans.
SCOPE: Final = "rag_qa"
#: What every helper returns while tracing is off: a span that records nothing.
NO_SPAN: Final = INVALID_SPAN

# --- the spans, and the attributes of our own -------------------------------------------------

#: ``rag-query``'s span per question, the root of its trace (the API's root is its server span).
QUERY: Final = "rag.query"
#: One retrieval: the query embedding, the search and the rerank beneath it.
RETRIEVE: Final = "rag.retrieve"
EMBED_QUERY: Final = "rag.embed_query"
VECTOR_SEARCH: Final = "rag.vector_search"
RERANK: Final = "rag.rerank"
#: One ingest run, and its steps: walking the corpus, loading and splitting what changed,
#: embedding it, publishing the new generation. The steps time what ``IngestReport`` times.
INGEST: Final = "rag.ingest"
INGEST_SCAN: Final = "rag.ingest.scan"
INGEST_PREPARE: Final = "rag.ingest.prepare"
INGEST_EMBED: Final = "rag.ingest.embed"
INGEST_PUBLISH: Final = "rag.ingest.publish"

#: How many corpus documents a question was limited to (FR-3); 0 for the whole corpus.
SOURCES: Final = "rag.sources"
#: How many chunks a retrieval returned, or an ingest step loaded, split or embedded.
CHUNKS: Final = "rag.chunks"
#: A retriever's mode: ``dense`` or ``hybrid`` (DEC-20).
MODE: Final = "rag.mode"
#: The rerank's input, output and cut (DEC-16): the retriever's candidates, the chunks kept,
#: and the score below which none is kept (set only when configured).
RERANK_CANDIDATES: Final = "rag.rerank.candidates"
RERANK_KEPT: Final = "rag.rerank.kept"
RERANK_MIN_SCORE: Final = "rag.rerank.min_score"
#: An ingest run's counts, as its report has them.
INGEST_REBUILD: Final = "rag.ingest.rebuild"  # asked for (--rebuild)
INGEST_FULL: Final = "rag.ingest.full_rebuild"  # done, asked for or not
INGEST_FILES: Final = "rag.ingest.files"
INGEST_PAGES: Final = "rag.ingest.pages"
INGEST_POINTS: Final = "rag.ingest.points"
INGEST_ADDED: Final = "rag.ingest.added"
INGEST_CHANGED: Final = "rag.ingest.changed"
INGEST_REMOVED: Final = "rag.ingest.removed"
INGEST_KEPT: Final = "rag.ingest.kept"
INGEST_FAILED: Final = "rag.ingest.failed"
#: ``db.system.name`` for Qdrant. No well-known value applies, and then the database
#: conventions let an instrumentation use a custom one.
QDRANT: Final = "qdrant"
#: A cancel's event and attribute: the client left, or the user pressed Ctrl-C.
CANCELLED: Final = "rag.cancelled"
#: The name of the API key a request authenticated with (S3-8, DEC-23): never its token.
API_KEY_NAME: Final = "rag.api_key.name"


class TraceSettingsError(ValueError):
    """``OTEL_EXPORTER_OTLP_ENDPOINT`` is set to something that is not an http(s) URL: the
    command exits 2 with this message, which names the variable."""


_provider: "TracerProvider | None" = None
_tracer: Tracer | None = None
#: Whether :func:`configure` built the provider, which :func:`shutdown` then closes; one a
#: test installed is the test's to close.
_owned = False
#: The apps :func:`instrument` wired, so :func:`uninstall` can undo it (the tests do).
_instrumented: list["FastAPI"] = []


def configure(service: str) -> bool:
    """Turn tracing on for this process when ``OTEL_EXPORTER_OTLP_ENDPOINT`` names a
    collector, and say whether it did. ``service`` is the command, which ``service.name``
    takes unless ``OTEL_SERVICE_NAME`` names another. Raises :class:`TraceSettingsError` for an
    endpoint that is not an http(s) URL.
    """
    try:
        settings = TraceSettings()
    except ValidationError as e:
        raise TraceSettingsError(describe_settings_error(e, ENV_OTEL_PREFIX)) from None
    if settings.exporter_otlp_endpoint is None:
        return False
    # Imported only now: with tracing off, no command loads the SDK or the exporter.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME, SERVICE_VERSION

    from rag_qa.observability.scrub import ScrubbingExporter

    name = settings.service_name or service
    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: name, SERVICE_VERSION: __version__})
    )
    # The exporter reads its endpoint, its timeout and its headers from the SDK's variables.
    # Scrubbed, since the collector is not this host: no span leaves it with exception text.
    provider.add_span_processor(BatchSpanProcessor(ScrubbingExporter(OTLPSpanExporter())))
    install(provider)
    global _owned
    _owned = True
    logger.info(
        "Sending traces to %s over OTLP/HTTP, as service %s.",
        _where(str(settings.exporter_otlp_endpoint)),
        name,
    )
    return True


def _where(url: str) -> str:
    """``url``'s scheme, host and port: never a path, a query or credentials."""
    split = urlsplit(url)
    host = split.hostname or ""
    if ":" in host:  # an IPv6 address
        host = f"[{host}]"
    return f"{split.scheme}://{host}" + (f":{split.port}" if split.port else "")


def install(provider: "TracerProvider") -> None:
    """Trace with ``provider`` from now on: :func:`configure`'s, or a test's."""
    global _provider, _tracer
    _provider = provider
    _tracer = provider.get_tracer(SCOPE, __version__)


def uninstall() -> None:
    """Stop tracing, without shutting the provider down: undo :func:`install` and every
    :func:`instrument`. For :func:`shutdown`, and for the tests."""
    global _provider, _tracer, _owned
    if _instrumented:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        while _instrumented:
            FastAPIInstrumentor.uninstrument_app(_instrumented.pop())
    _provider = None
    _tracer = None
    _owned = False


def enabled() -> bool:
    """Whether a command turned tracing on in this process."""
    return _tracer is not None


def flush() -> None:
    """Send the spans that have ended and wait for it: the API's lifespan calls it, since a
    SIGTERM ends ``rag-serve`` before any ``finally`` of ``main`` runs."""
    if _provider is not None:
        _provider.force_flush()


def shutdown() -> None:
    """Send what is left and stop tracing: the end of every command that called
    :func:`configure`. Nothing at all while tracing is off, or for a provider a test
    installed, which the test closes."""
    provider = _provider
    if not _owned or provider is None:
        return
    uninstall()
    provider.shutdown()


@contextmanager
def span(
    name: str, attributes: Attributes = None, *, kind: SpanKind = SpanKind.INTERNAL
) -> Iterator[Span]:
    """A span around synchronous work, current while it runs and ended when it ends: see the
    module docstring for where it may and may not be used. Yields :data:`NO_SPAN` while
    tracing is off."""
    if _tracer is None:
        yield NO_SPAN
        return
    with _tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes,
        record_exception=False,  # an exception's message may carry text: failed() says why
        set_status_on_exception=False,
    ) as started:
        try:
            yield started
        except Exception as e:
            failed(started, e)
            raise
        except BaseException:
            cancelled(started)
            raise


def start(
    name: str,
    parent: "Context | None",
    attributes: Attributes = None,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Span:
    """A span under ``parent`` that is never made current, for an async generator: its caller
    ends it in a ``finally``. :data:`NO_SPAN` while tracing is off."""
    if _tracer is None:
        return NO_SPAN
    return _tracer.start_span(
        name,
        context=parent,
        kind=kind,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    )


def current() -> "Context | None":
    """The context a span started now would take as its parent, to hand to another thread or
    to :func:`start`; ``None`` while tracing is off."""
    return None if _tracer is None else otel_context.get_current()


@contextmanager
def attached(parent: "Context | None") -> Iterator[None]:
    """Make ``parent`` (:func:`current`'s, from another thread) this thread's context while the
    block runs. Attached and detached in the one thread; nothing at all for ``None``."""
    if parent is None:
        yield
        return
    token = otel_context.attach(parent)
    try:
        yield
    finally:
        otel_context.detach(token)


def annotate(attributes: Mapping[str, AttributeValue]) -> None:
    """Set ``attributes`` on the span current now: in a request, its server span. Nothing while
    tracing is off."""
    if _tracer is not None:
        get_current_span().set_attributes(attributes)


def failed(span: Span, error: BaseException) -> None:
    """Mark ``span`` failed by ``error``: status ERROR and ``error.type``, the class name only.
    Never the message, nor an exception event: either may carry a path, or a document's text."""
    span.set_status(Status(StatusCode.ERROR))
    span.set_attribute(ERROR_TYPE, type(error).__name__)


def cancelled(span: Span) -> None:
    """Mark ``span`` cancelled: the event and the attribute ``rag.cancelled``. The status stays
    unset, since a client that leaves is not an error."""
    span.add_event(CANCELLED)
    span.set_attribute(CANCELLED, True)


def instrument(app: "FastAPI", *, untraced: Sequence[str]) -> None:
    """Give ``app`` its HTTP server spans, while tracing is on: the contrib instrumentation
    (``opentelemetry-instrumentation-fastapi``), with our provider and no OTel metrics, since
    the metrics are Prometheus's (S3-7). Its ``receive`` and ``send`` spans are left out, so an
    SSE event is never a span and Starlette reads the raw ``receive`` (DEC-22, checked in
    0.66b1); the paths in ``untraced`` (the health probes) get no span at all."""
    if _provider is None:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.metrics import NoOpMeterProvider

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=_provider,
        meter_provider=NoOpMeterProvider(),
        exclude_spans=["receive", "send"],
        excluded_urls=",".join(f"{re.escape(path)}$" for path in untraced),
    )
    _instrumented.append(app)
