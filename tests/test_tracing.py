"""Traces (NFR-10, DEC-22, S3-7b), checked hermetically.

- **Off unless a collector is named.** No endpoint, no provider. With one, the spans reach a
  local fake collector at ``/v1/traces`` as OTLP/HTTP protobuf, under the command's service
  name. A dead collector costs at most the exporter's timeout at exit, and an endpoint that is
  not a URL stops each command with exit 2.
- **The helpers' rules.** A failure is its class name, never its message; a cancel is marked,
  its status unset; off, nothing is recorded and no context is touched; a parent reaches a
  thread, and ``serialized``'s executor thread runs in the caller's context.
- **The API's spans.** One server span per request, from the contrib instrumentation, none for
  the health probes, and none from FastAPI's own telemetry, even with a global provider set;
  an answer's retrieval and chat spans beneath its server span; no question, chunk, answer or
  host path in any span.
- **What leaves the host.** An unhandled exception's server span reaches the exporter as its
  class name alone: contrib records the message and the stacktrace itself, and the scrubbing
  exporter drops them (first review).
- **Ingest.** A run is a span with its four steps and its counts, and an API job's run is in
  the trace of the request that started it.

The spans of an answer's attempts are ``tests/test_answering.py``'s; ``rag-query``'s are
``tests/test_cli.py``'s; that ``rag-eval generate`` exports nothing is
``tests/test_eval_generation.py``'s. The cancel tests run with tracing on too, in
``test_answering.py`` and ``test_api_uvicorn.py``.
"""

import asyncio
import logging
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import (
    FAKE_ANSWER,
    RERANK_CANDIDATES,
    RERANK_TOP_N,
    FakeCollector,
    MakeConfig,
    SpanRecorder,
    StreamingFakeChatModel,
    Traced,
    fake_serving,
    live_generation,
    sse_events,
    use_fakes,
    use_fakes_and_reranker,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from opentelemetry import context as otel_context
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.attributes.db_attributes import DB_COLLECTION_NAME, DB_SYSTEM_NAME
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.service_attributes import SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import SpanKind, Status, StatusCode
from qdrant_client import QdrantClient

from rag_qa import __version__, cli, ingest
from rag_qa.api import server
from rag_qa.api.app import create_app
from rag_qa.api.state import serialized
from rag_qa.chain import RetrievalQuery, _rerank_attributes
from rag_qa.config import load_config
from rag_qa.observability import genai, tracing
from rag_qa.observability.scrub import ScrubbingExporter, scrubbed
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ServerSettings

ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
#: The contrib instrumentation's scope, and FastAPI 0.143's own.
CONTRIB_SCOPE = "opentelemetry.instrumentation.fastapi"
FASTAPI_SCOPE = "fastapi"


def dead_url() -> str:
    """A local port nothing listens on, so a connection there is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def exported(collector: FakeCollector) -> list[tuple[dict[str, str], str, list[str]]]:
    """What the collector was sent, request by request: the resource's attributes, the scope
    and the spans' names."""
    found = []
    for path, content_type, body in collector.requests:
        assert (path, content_type) == ("/v1/traces", "application/x-protobuf")
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            resource = {a.key: a.value.string_value for a in resource_spans.resource.attributes}
            for scope_spans in resource_spans.scope_spans:
                names = [s.name for s in scope_spans.spans]
                found.append((resource, scope_spans.scope.name, names))
    return found


# --- configure: off, on, the collector, the settings -------------------------------------------


def test_without_an_endpoint_nothing_is_traced() -> None:
    assert tracing.configure("rag-test") is False
    assert not tracing.enabled()
    assert tracing.current() is None


def test_an_endpoint_sends_the_spans_to_its_v1_traces(
    fake_collector: FakeCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENDPOINT, fake_collector.url)
    assert tracing.configure("rag-test")
    with tracing.span("rag.test"):
        pass
    tracing.shutdown()
    assert not tracing.enabled()
    ((resource, scope, names),) = exported(fake_collector)
    assert (resource[SERVICE_NAME], resource[SERVICE_VERSION]) == ("rag-test", __version__)
    assert (scope, names) == (tracing.SCOPE, ["rag.test"])


def test_otel_service_name_names_the_service(
    fake_collector: FakeCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENDPOINT, fake_collector.url)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "rag-staging")
    assert tracing.configure("rag-test")
    with tracing.span("rag.test"):
        pass
    tracing.shutdown()
    ((resource, _, _),) = exported(fake_collector)
    assert resource[SERVICE_NAME] == "rag-staging"


def test_a_dead_collector_costs_at_most_its_timeout_at_exit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ENDPOINT, dead_url())
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "2")  # seconds: Python's SDK reads seconds
    caplog.set_level(logging.WARNING, logger="opentelemetry")
    assert tracing.configure("rag-test")
    with tracing.span("rag.test"):
        pass
    started = time.monotonic()
    tracing.shutdown()
    assert time.monotonic() - started < 2 + 1.5
    assert "Failed to export spans batch" in caplog.text  # the warning the log shows


def test_the_start_line_names_the_collector_but_no_path_or_credentials(
    fake_collector: FakeCollector,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    where = fake_collector.url.removeprefix("http://")
    monkeypatch.setenv(ENDPOINT, f"http://user:hunter2@{where}/base")
    caplog.set_level(logging.INFO, logger="rag_qa")
    assert tracing.configure("rag-test")
    tracing.shutdown()
    assert f"Sending traces to http://{where} over OTLP/HTTP, as service rag-test." in caplog.text
    assert "hunter2" not in caplog.text
    assert "/base" not in caplog.text


@pytest.mark.parametrize(
    ("url", "where"),
    [
        ("http://jaeger:4318", "http://jaeger:4318"),
        ("https://user:secret@collector.example.com/otlp?x=1", "https://collector.example.com"),
        ("http://[::1]:4318/", "http://[::1]:4318"),
    ],
)
def test_the_start_line_names_scheme_host_and_port_only(url: str, where: str) -> None:
    assert tracing._where(url) == where


@pytest.mark.parametrize(
    "main", [server.main, cli.main, ingest.main], ids=["rag-serve", "rag-query", "rag-ingest"]
)
def test_an_endpoint_that_is_not_a_url_stops_the_command_with_exit_2(
    main: Callable[[list[str]], int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENDPOINT, "localhost:4318")  # the scheme forgotten
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "Error: invalid trace settings: OTEL_EXPORTER_OTLP_ENDPOINT:" in err
    assert not tracing.enabled()


# --- the helpers -------------------------------------------------------------------------------


def test_a_failed_span_names_the_class_and_never_the_message(traced: Traced) -> None:
    with pytest.raises(ValueError), tracing.span("rag.test"):
        raise ValueError("a path, or a document's text")
    span = traced.one("rag.test")
    assert span.status.status_code is StatusCode.ERROR
    assert not span.status.description
    assert (span.attributes or {})[ERROR_TYPE] == "ValueError"
    assert list(span.events) == []  # no exception event
    assert "a document's text" not in traced.text()


@pytest.mark.parametrize("stop", [asyncio.CancelledError, KeyboardInterrupt, GeneratorExit])
def test_a_cancel_is_marked_and_leaves_the_status_unset(
    traced: Traced, stop: type[BaseException]
) -> None:
    with pytest.raises(stop), tracing.span("rag.test"):
        raise stop()
    span = traced.one("rag.test")
    assert [event.name for event in span.events] == [tracing.CANCELLED]
    assert (span.attributes or {})[tracing.CANCELLED] is True
    assert span.status.status_code is StatusCode.UNSET


def test_off_the_helpers_record_nothing_and_touch_no_context() -> None:
    before = otel_context.get_current()
    with tracing.span("rag.test") as span:
        assert span is tracing.NO_SPAN
        assert otel_context.get_current() is before
    with tracing.attached(None):
        assert otel_context.get_current() is before
    assert tracing.start("rag.test", None) is tracing.NO_SPAN
    assert tracing.current() is None


def test_attached_makes_another_thread_s_context_this_one_s(traced: Traced) -> None:
    with tracing.span("parent"):
        parent = tracing.current()

    def child() -> None:
        with tracing.attached(parent), tracing.span("child"):
            pass

    thread = threading.Thread(target=child)
    thread.start()
    thread.join()
    assert [s.name for s in traced.children(traced.one("parent"))] == ["child"]


def test_serialized_runs_the_retrieval_in_a_copy_of_the_caller_s_context(traced: Traced) -> None:
    def retrieve(asked: RetrievalQuery) -> list[Document]:
        with tracing.span("inner"):
            return []

    retrieval = serialized(RunnableLambda(retrieve), threading.Lock())
    with tracing.span("outer"):
        asyncio.run(retrieval.ainvoke(RetrievalQuery("q")))
    assert [s.name for s in traced.children(traced.one("outer"))] == ["inner"]


def test_negative_control_run_in_executor_alone_carries_no_context(traced: Traced) -> None:
    # Why serialized copies the context: without the copy, the thread's span is a root.
    def inner() -> None:
        with tracing.span("inner"):
            pass

    async def ask() -> None:
        await asyncio.get_running_loop().run_in_executor(None, inner)

    with tracing.span("outer"):
        asyncio.run(ask())
    assert traced.one("inner").parent is None


def test_the_rerank_span_says_its_cut_only_when_the_reranker_has_one() -> None:
    assert _rerank_attributes(SimpleNamespace(min_score=-4), 20) == {  # type: ignore[arg-type]
        tracing.RERANK_CANDIDATES: 20,
        tracing.RERANK_MIN_SCORE: -4.0,
    }
    assert _rerank_attributes(SimpleNamespace(min_score=None), 20) == {  # type: ignore[arg-type]
        tracing.RERANK_CANDIDATES: 20
    }


# --- the API's spans ------------------------------------------------------------------------------


def serve(config: RagConfig, **kw: Any) -> TestClient:
    return TestClient(create_app(config, ServerSettings(), **kw), base_url="http://127.0.0.1")


@pytest.fixture
def reranked(
    make_config: MakeConfig,
    sample_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_cross_encoder: Any,
) -> RagConfig:
    """The real build over the fakes, with the reranker on, and the sample corpus ingested."""
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    config = load_config(make_config(use_fakes_and_reranker))
    ingest.ingest(config)
    return config


def test_an_answer_is_its_server_span_with_the_retrieval_and_the_chat_beneath(
    reranked: RagConfig, traced: Traced, qdrant: QdrantClient, sample_corpus: Path
) -> None:
    question = "How many staff are at the station?"
    with serve(reranked) as client:
        before = len(traced.spans())  # the ingest's spans, made before the server started
        response = client.post("/v1/query", json={"question": question})
    assert response.status_code == 200
    assert [name for name, _ in sse_events(response.text)][-1] == "done"
    spans = traced.spans()[before:]
    (server_span,) = [s for s in spans if s.kind is SpanKind.SERVER]
    assert server_span.name == "POST /v1/query"
    assert server_span.parent is None
    retrieve, chat = traced.children(server_span)
    assert retrieve.name == tracing.RETRIEVE
    assert dict(retrieve.attributes or {}) == {tracing.SOURCES: 0, tracing.CHUNKS: RERANK_TOP_N}
    embed, search, rerank = traced.children(retrieve)
    assert (embed.name, search.name, rerank.name) == (
        tracing.EMBED_QUERY,
        tracing.VECTOR_SEARCH,
        tracing.RERANK,
    )
    assert search.kind is SpanKind.CLIENT
    assert dict(search.attributes or {}) == {
        DB_SYSTEM_NAME: tracing.QDRANT,
        DB_COLLECTION_NAME: live_generation(qdrant),
        tracing.MODE: "dense",
    }
    assert dict(rerank.attributes or {}) == {
        tracing.RERANK_CANDIDATES: RERANK_CANDIDATES,
        tracing.RERANK_KEPT: RERANK_TOP_N,
    }
    assert chat.name == "chat FakeListChatModel"
    assert (chat.attributes or {})[genai.MODEL_REF] == "components.llms.fake"
    # No question, answer, chunk or host path in any span.
    text = traced.text()
    chunks = [
        point.payload["page_content"]
        for point in qdrant.scroll(live_generation(qdrant), with_payload=True, limit=100)[0]
        if point.payload
    ]
    for secret in (question, FAKE_ANSWER, str(sample_corpus), *chunks):
        assert secret not in text, f"{secret!r} is in a span"


def test_one_request_makes_one_server_span_even_with_a_global_provider_set(
    fake_config: RagConfig, traced: Traced, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FastAPI 0.143's own telemetry switches on whenever a global provider is set (an
    # opentelemetry-instrument wrapper sets one): create_app's flags keep it off, so contrib's
    # server span is the only one, and a FastAPI release that moved its defaults fails here.
    every_provider = TracerProvider()
    every_provider.add_span_processor(traced.recorder)
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: every_provider)
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, serving=serving) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
    (server_span,) = [s for s in traced.spans() if s.kind is SpanKind.SERVER]
    assert server_span.instrumentation_scope is not None
    assert server_span.instrumentation_scope.name == CONTRIB_SCOPE
    assert FASTAPI_SCOPE not in {
        s.instrumentation_scope.name for s in traced.spans() if s.instrumentation_scope
    }


def test_negative_control_a_fastapi_app_without_the_flags_traces_itself(
    traced: Traced, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control for the test above: with the same global provider, a FastAPI app built
    # without the flags makes server spans of its own.
    every_provider = TracerProvider()
    every_provider.add_span_processor(traced.recorder)
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: every_provider)
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {}

    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
    assert FASTAPI_SCOPE in {
        s.instrumentation_scope.name for s in traced.spans() if s.instrumentation_scope
    }


def test_with_tracing_off_a_global_provider_gets_no_span_from_the_api(
    fake_config: RagConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tracing off is no instrumentation at all, whatever else configured a provider.
    recorder = SpanRecorder()
    every_provider = TracerProvider()
    every_provider.add_span_processor(recorder)
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: every_provider)
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, serving=serving) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
    assert recorder.ended() == []
    every_provider.shutdown()


def test_the_health_probes_make_no_span(fake_config: RagConfig, traced: Traced) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, serving=serving) as client:
        assert client.get("/v1/health/live").status_code == 200
        client.get("/v1/health")  # ready or not, a probe
        assert traced.spans() == []
        assert client.get("/openapi.json").status_code == 200  # any other route is traced
    assert traced.names() == ["GET /openapi.json"]


def test_an_untraced_app_has_no_otel_middleware(fake_config: RagConfig) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    app = create_app(fake_config, ServerSettings(), serving=serving)
    stack, seen = app.build_middleware_stack(), []
    while stack is not None:
        seen.append(type(stack).__module__)
        stack = getattr(stack, "app", None)
    assert not [module for module in seen if module.startswith("opentelemetry")]


# --- ingest ------------------------------------------------------------------------------------


def test_an_ingest_run_is_a_span_with_its_four_steps_and_its_counts(
    make_config: MakeConfig,
    sample_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    traced: Traced,
    qdrant: QdrantClient,
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    config = load_config(make_config(use_fakes))
    report = ingest.ingest(config)
    run = traced.one(tracing.INGEST)
    assert run.parent is None
    scan, prepare, embed, publish = traced.children(run)
    assert [s.name for s in (scan, prepare, embed, publish)] == [
        tracing.INGEST_SCAN,
        tracing.INGEST_PREPARE,
        tracing.INGEST_EMBED,
        tracing.INGEST_PUBLISH,
    ]
    attributes = dict(run.attributes or {})
    assert attributes == {
        tracing.INGEST_REBUILD: False,
        tracing.INGEST_FULL: True,  # a first run builds the index in full
        tracing.INGEST_PAGES: report.documents,
        tracing.CHUNKS: report.chunks,
        tracing.INGEST_POINTS: report.points,
        tracing.INGEST_ADDED: report.added,
        tracing.INGEST_CHANGED: 0,
        tracing.INGEST_REMOVED: 0,
        tracing.INGEST_KEPT: 0,
        tracing.INGEST_FAILED: len(report.failed),
        DB_COLLECTION_NAME: live_generation(qdrant),
    }
    assert (scan.attributes or {})[tracing.INGEST_FILES] == report.found
    assert (embed.attributes or {})[tracing.CHUNKS] == report.chunks
    assert (publish.attributes or {})[DB_COLLECTION_NAME] == report.published
    assert str(sample_corpus) not in traced.text()
    # An update that finds nothing changed: the walk, and no other step.
    ingest.ingest(config)
    _, again = traced.spans(tracing.INGEST)
    assert [s.name for s in traced.children(again)] == [tracing.INGEST_SCAN]


def test_an_ingest_job_s_run_is_in_the_trace_of_its_request(
    make_config: MakeConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch, traced: Traced
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    config = load_config(make_config(use_fakes))
    with serve(config) as client:
        started = client.post("/v1/ingest", json={"rebuild": False})
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        deadline = time.monotonic() + 30
        while client.get(f"/v1/ingest/{job_id}").json()["status"] not in ("succeeded", "failed"):
            assert time.monotonic() < deadline, "the ingest job did not finish"
            time.sleep(0.02)
    request = traced.one("POST /v1/ingest")
    run = traced.one(tracing.INGEST)
    assert run.parent is not None and request.context is not None
    assert run.parent.span_id == request.context.span_id
    assert run.context is not None and run.context.trace_id == request.context.trace_id


# --- what leaves the host: no exception text in any span (first review) ------------------------

SECRET = "WHO-IS-THE-SECRET-EMPLOYEE-QUESTION"


def scrubbing_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """A provider whose spans reach an in-memory exporter through the same wrapper
    :func:`tracing.configure` puts in front of the OTLP one."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ScrubbingExporter(memory)))
    return provider, memory


def everything(span: ReadableSpan) -> str:
    """All a span would say to a collector: its JSON form holds name, attributes, events and
    status (description included)."""
    return span.to_json()


def raising_app(config: RagConfig, message: str) -> TestClient:
    app = create_app(
        config, ServerSettings(), serving=fake_serving(config, StreamingFakeChatModel())
    )

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError(message)

    return TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)


def test_negative_control_the_server_span_records_an_unhandled_exception_s_message(
    fake_config: RagConfig,
) -> None:
    # Why the wrapper exists: contrib's server span records the exception itself, message and
    # stacktrace (and its FastAPI wrapper writes "Type: message" into the status, which the 500
    # response's own status then replaces). Nothing here scrubs it.
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracing.install(provider)
    with raising_app(fake_config, f"failed on {SECRET}") as client:
        assert client.get("/boom").status_code == 500
    (server_span,) = memory.get_finished_spans()
    (exception,) = [e for e in server_span.events if e.name == "exception"]
    assert SECRET in str(dict(exception.attributes or {}))
    assert "Traceback" in str(dict(exception.attributes or {}))


def test_an_unhandled_exception_leaves_the_server_span_as_its_class_name_alone(
    fake_config: RagConfig,
) -> None:
    provider, memory = scrubbing_provider()
    tracing.install(provider)
    with raising_app(fake_config, f"failed on {SECRET}, in /home/someone/corpus/secret.pdf") as c:
        assert c.get("/boom").status_code == 500
    (server_span,) = memory.get_finished_spans()
    assert server_span.kind is SpanKind.SERVER
    assert server_span.status.status_code is StatusCode.ERROR  # still an error...
    assert server_span.status.description is None  # ...without its text
    (exception,) = [e for e in server_span.events if e.name == "exception"]
    assert dict(exception.attributes or {}) == {"exception.type": "RuntimeError"}
    text = everything(server_span)
    assert SECRET not in text and "/home/someone" not in text and "Traceback" not in text


def test_the_collector_is_sent_the_class_name_alone_through_configure(
    fake_config: RagConfig, fake_collector: FakeCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real path end to end: configure() builds the provider, the batch processor sends
    # protobuf to the collector. Not a byte of the message, nor of the stacktrace, is in it.
    monkeypatch.setenv(ENDPOINT, fake_collector.url)
    assert tracing.configure("rag-test")
    with raising_app(fake_config, f"failed on {SECRET}") as client:
        assert client.get("/boom").status_code == 500
    tracing.shutdown()
    bodies = b"".join(body for _, _, body in fake_collector.requests)
    assert b"RuntimeError" in bodies  # the type is there
    assert SECRET.encode() not in bodies and b"Traceback" not in bodies
    assert b"site-packages" not in bodies  # no host path from a stacktrace


def span_with(events: list[Event], status: Status | None = None, **attributes: Any) -> ReadableSpan:
    # Room for two events and one attribute: the oldest event and the second attribute are
    # dropped, so the span has drop counts to keep.
    provider = TracerProvider(
        span_limits=SpanLimits(max_events=2, max_attributes=1, max_event_attributes=128)
    )
    span = provider.get_tracer("t").start_span("op", attributes=attributes or None)
    for event in events:
        span.add_event(event.name, event.attributes, event.timestamp)
    span.set_status(status or Status(StatusCode.UNSET))
    span.end()
    assert isinstance(span, ReadableSpan)
    return span


def test_a_span_without_exception_text_reaches_the_exporter_untouched() -> None:
    span = span_with([Event("rag.first_token", {})], Status(StatusCode.ERROR))
    assert scrubbed(span) is span


def test_only_exception_events_and_the_status_description_are_rewritten() -> None:
    detail = {
        "exception.type": "ValueError",
        "exception.message": "private",
        "exception.stacktrace": "Traceback ... private",
        "exception.escaped": "False",
    }
    span = span_with(
        [
            Event("dropped", {}, 1),
            Event("rag.first_token", {"k": "v"}, 5),
            Event("exception", detail, 7),
        ],
        Status(StatusCode.ERROR, "ValueError: private"),
        **{"rag.chunks": 3, "rag.dropped": 4},
    )
    assert (span.dropped_events, span.dropped_attributes) == (1, 1)  # the premise
    clean = scrubbed(span)
    assert clean is not span
    first, exception = clean.events
    assert (first.name, dict(first.attributes or {}), first.timestamp) == (
        "rag.first_token",
        {"k": "v"},
        5,
    )
    assert (exception.name, dict(exception.attributes or {}), exception.timestamp) == (
        "exception",
        {"exception.type": "ValueError"},
        7,
    )
    assert clean.status.status_code is StatusCode.ERROR and clean.status.description is None
    # Everything else is the span's own, drop counts included.
    assert (clean.name, clean.context, clean.parent, clean.kind) == (
        span.name,
        span.context,
        span.parent,
        span.kind,
    )
    assert (clean.start_time, clean.end_time) == (span.start_time, span.end_time)
    assert dict(clean.attributes or {}) == dict(span.attributes or {})
    assert (clean.dropped_attributes, clean.dropped_events, clean.dropped_links) == (
        span.dropped_attributes,
        span.dropped_events,
        span.dropped_links,
    )
    assert "private" not in everything(clean)


def test_an_exception_event_with_no_type_keeps_none() -> None:
    span = span_with([Event("exception", {"exception.message": "private"})])
    (event,) = scrubbed(span).events
    assert dict(event.attributes or {}) == {}


def test_the_exporter_passes_shutdown_and_flush_to_the_real_one() -> None:
    memory = InMemorySpanExporter()
    exporter = ScrubbingExporter(memory)
    assert exporter.force_flush() is True
    exporter.shutdown()
    assert memory._stopped  # InMemorySpanExporter's own flag


def test_the_api_flushes_its_spans_when_its_lifespan_ends_without_main_s_finally(
    fake_config: RagConfig, fake_collector: FakeCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A SIGTERM ends rag-serve once uvicorn has shut down, before any finally of main runs
    # (server.py), so the lifespan's flush is the only thing that sends the last spans. The
    # batch processor would wait 5 s on its own; shutdown() is never called here.
    monkeypatch.setenv(ENDPOINT, fake_collector.url)
    assert tracing.configure("rag-test")
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with TestClient(
        create_app(fake_config, ServerSettings(), serving=serving), base_url="http://127.0.0.1"
    ) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
        assert fake_collector.requests == []  # nothing sent yet: the processor is waiting
    sent = [name for _, _, names in exported(fake_collector) for name in names]
    assert "POST /v1/query" in sent and "chat StreamingFakeChatModel" in sent
