"""Disconnects under real uvicorn (S2-7, DEC-18): a client that leaves stops its answer.

FastAPI's test client cannot drop a connection mid-response. These tests run uvicorn on a
free loopback port, with the fake pipeline, and drop raw sockets at chosen moments:

- during the model's simulated prompt evaluation;
- mid-stream;
- before the first event, while retrieval runs;
- while an event waits to be sent (forced).

Each time, the fake model's ``finally`` (the stand-in for closing the HTTP stream to
Ollama) must run within 1 s, and the slot must be free.

**The forced case** is the window S2-7's spike found. FastAPI sends each event from a
producer task of its own, through an anyio memory stream. A disconnect that lands while an
event waits to be sent cancels that task outside the route's generator, and leaves the
generator paused at a ``yield``, in a reference cycle. The slot's teardown closes the
stream it holds (:meth:`rag_qa.api.state.Slot.aclose`). The negative control shows the
window is real: without that close, the stream stays open.

Only the real daemon shows that Ollama itself stops; that is the story's manual step
(caveat 21).

**With tracing on too** (S3-7b, DEC-22): every drop runs twice, the second time with the
contrib instrumentation wrapping the app and the spans recorded. The same 1 s holds, every
span that started ends, and no context fails to detach: the ASGI wrapper did not break
DEC-18's cancel.
"""

import gc
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from anyio.streams.memory import MemoryObjectSendStream
from conftest import StreamingFakeChatModel, Traced, fake_serving, sse_events
from fastapi import FastAPI, Request
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from opentelemetry.trace import SpanKind, StatusCode

from rag_qa.api.app import create_app
from rag_qa.api.limits import BodyLimit
from rag_qa.api.server import uvicorn_config
from rag_qa.api.state import Serving, Slot
from rag_qa.chain import RetrievalQuery
from rag_qa.observability import metrics, tracing
from rag_qa.schema import RagConfig
from rag_qa.settings import ServerSettings

#: How long a stopped answer may take to close its stream to the model (the story's 1 s).
CLOSE_WITHIN_S = 1.0

Start = Callable[..., int]


@pytest.fixture
def start_server() -> Iterator[Start]:
    """Start uvicorn with an app on a free loopback port, in a thread; returns the port.

    As ``rag-serve`` configures it (:func:`rag_qa.api.server.uvicorn_config`), for ``settings``
    or the defaults, so the forwarded headers are handled as they are in production (S3-8);
    ``config`` is a uvicorn config to run instead, for a control."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(
        app: Any, settings: ServerSettings | None = None, config: uvicorn.Config | None = None
    ) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        if config is None:
            config = uvicorn_config(
                app,
                settings or ServerSettings(),
                "127.0.0.1",
                port,
                log_level="warning",
                timeout_graceful_shutdown=1,
            )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        running.append((server, thread))
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)
        return port

    yield start
    for server, thread in running:
        server.should_exit = True
        thread.join(10)


def serve(start: Start, config: RagConfig, serving: Serving) -> int:
    app = create_app(config, ServerSettings(), serving=serving)
    # Every drop runs with the body limit on: it wraps receive, which carries the disconnect
    # (DEC-23, design review 2).
    assert app.user_middleware[0].cls is BodyLimit
    return start(app)


def ask(port: int, question: str = "How many staff?") -> socket.socket:
    """Send a query over a raw socket, which a test can then drop at any moment."""
    body = f'{{"question": "{question}"}}'.encode()
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(
        b"POST /v1/query HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    return sock


def read_until(sock: socket.socket, needle: bytes, count: int = 1, timeout_s: float = 10) -> bytes:
    sock.settimeout(timeout_s)
    received = b""
    while received.count(needle) < count:
        chunk = sock.recv(65536)
        assert chunk, f"the server closed the stream before {count}x {needle!r}: {received!r}"
        received += chunk
    return received


def answers(outcome: str) -> float:
    """``rag_answers_total`` for ``outcome`` now (S3-7)."""
    return metrics.REGISTRY.get_sample_value("rag_answers_total", {"outcome": outcome}) or 0.0


def abandoned() -> dict[str, float]:
    """``rag_abandoned_seconds_count`` now, by what the answer waited for (S3-7b's fix)."""
    return {
        waiting_for: metrics.REGISTRY.get_sample_value(
            "rag_abandoned_seconds_count", {"waiting_for": waiting_for}
        )
        or 0.0
        for waiting_for in (metrics.WAITING_FOR_SOURCES, metrics.WAITING_FOR_FIRST_TOKEN)
    }


def moved(before: dict[str, float]) -> dict[str, float]:
    return {waiting_for: count - before[waiting_for] for waiting_for, count in abandoned().items()}


def within(seconds: float, condition: Callable[[], bool]) -> bool:
    """Whether ``condition`` holds, polled until ``seconds`` have passed."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


# --- the drops --------------------------------------------------------------------------------


@pytest.mark.usefixtures("either_way")
def test_a_drop_while_the_model_reads_the_prompt_closes_its_stream(
    start_server: Start, fake_config: RagConfig
) -> None:
    # No event is sent during a prefill, so only Starlette's listening for the disconnect
    # can notice it (uvicorn reports ASGI spec 2.3).
    model = StreamingFakeChatModel(prefill_s=10.0)
    serving = fake_serving(fake_config, model)
    before = abandoned()
    sock = ask(serve(start_server, fake_config, serving))
    read_until(sock, b"event: sources")
    assert within(CLOSE_WITHIN_S, lambda: model.log == ["start"])
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: model.closed), (
        "the model was still reading the prompt 1 s after its client left"
    )
    assert model.log == ["start", "closed"]  # no token generated
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)
    # Abandoned while it waited for its first token: the SLI's only trace of it (S3-7b's fix).
    assert within(CLOSE_WITHIN_S, lambda: moved(before) == {"sources": 0, "first_token": 1})


@pytest.mark.usefixtures("either_way")
def test_a_drop_mid_stream_closes_the_models_stream(
    start_server: Start, fake_config: RagConfig
) -> None:
    model = StreamingFakeChatModel(tokens=[f"t{i} " for i in range(400)], gap_s=0.01)
    serving = fake_serving(fake_config, model)
    cancelled = answers("cancelled")
    before = abandoned()
    sock = ask(serve(start_server, fake_config, serving))
    read_until(sock, b"event: token", count=3)
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: model.closed), "generation ran on after the drop"
    assert len(model.log) < 100  # "start", the tokens sent so far, "closed": not 400
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)
    # Counted once, as cancelled (S3-7): the metrics' observation awaits nothing.
    assert within(CLOSE_WITHIN_S, lambda: answers("cancelled") == cancelled + 1)
    assert moved(before) == {"sources": 0, "first_token": 0}  # its first token had come


def test_a_dropped_answer_s_spans_end_its_chat_cancelled_its_server_span_unset(
    start_server: Start, fake_config: RagConfig, traced: Traced
) -> None:
    # S3-7b: the client's choice to leave is no error, on either span.
    model = StreamingFakeChatModel(tokens=[f"t{i} " for i in range(400)], gap_s=0.01)
    sock = ask(serve(start_server, fake_config, fake_serving(fake_config, model)))
    read_until(sock, b"event: token", count=3)
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: model.closed), "generation ran on after the drop"
    assert within(CLOSE_WITHIN_S, lambda: len(traced.spans("POST /v1/query")) == 1)
    server = traced.one("POST /v1/query")
    assert server.kind is SpanKind.SERVER
    assert server.status.status_code is StatusCode.UNSET
    assert within(CLOSE_WITHIN_S, lambda: bool(traced.children(server)))
    (chat,) = traced.children(server)
    assert chat.name == "chat StreamingFakeChatModel"
    assert chat.attributes[tracing.CANCELLED] is True
    assert chat.status.status_code is StatusCode.UNSET


class Refused(StreamingFakeChatModel):
    """A stopped Ollama: its stream fails at once, before any token."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        self.log.append("start")
        raise httpx.ConnectError("All connection attempts failed")
        yield  # an async generator, like every model's stream


def through_the_fallback(config: RagConfig, spare: StreamingFakeChatModel) -> Serving:
    serving = fake_serving(config, Refused())
    serving.pipeline = replace(serving.pipeline, fallback_llm=spare)
    return serving


@pytest.mark.parametrize(
    ("spare", "sent"),
    [
        (StreamingFakeChatModel(prefill_s=10.0), b"event: sources"),
        (
            StreamingFakeChatModel(tokens=[f"t{i} " for i in range(400)], gap_s=0.01),
            b"event: token",
        ),
    ],
    ids=["during-its-prefill", "mid-stream"],
)
@pytest.mark.usefixtures("either_way")
def test_a_drop_while_the_fallback_answers_closes_its_stream(
    start_server: Start, fake_config: RagConfig, spare: StreamingFakeChatModel, sent: bytes
) -> None:
    # S3-6: the fallback's stream is read as the model's is, in the same task, so a client
    # that leaves stops it as fast (DEC-14, DEC-21).
    serving = through_the_fallback(fake_config, spare)
    sock = ask(serve(start_server, fake_config, serving))
    read_until(sock, sent)
    assert within(CLOSE_WITHIN_S, lambda: "start" in spare.log)
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: spare.closed), "the fallback ran on after the drop"
    assert len(spare.log) < 100
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)


@pytest.mark.usefixtures("either_way")
def test_a_drop_before_the_first_event_frees_the_slot_and_the_next_retrieval_waits(
    start_server: Start, fake_config: RagConfig
) -> None:
    # The retrieval thread cannot be stopped (DEC-14). Its slot is freed at once, but the
    # next request's retrieval must wait for it rather than run beside it (S2-4's second
    # review: two reranks would compete for the same CPU).
    spans: list[tuple[str, float, float]] = []
    chunk = Document(page_content="x", metadata={"source": "notes.txt"})

    def slow_retrieval(asked: RetrievalQuery) -> list[Document]:
        started = time.monotonic()
        time.sleep(1.0)
        spans.append((asked.question, started, time.monotonic()))
        return [chunk]

    model = StreamingFakeChatModel()
    serving = fake_serving(fake_config, model, retrieve=RunnableLambda(slow_retrieval))
    port = serve(start_server, fake_config, serving)
    before = abandoned()

    first = ask(port, "first")
    time.sleep(0.2)  # its retrieval is running in a worker thread now
    first.close()
    assert within(0.5, lambda: serving.slots.in_use == 0)  # freed before its retrieval ends
    assert spans == []
    # Abandoned while it waited for its sources (S3-7b's fix).
    assert moved(before) == {"sources": 1, "first_token": 0}

    second = httpx.post(f"http://127.0.0.1:{port}/v1/query", json={"question": "second"})
    assert second.status_code == 200  # a slot, not a 503
    assert sse_events(second.text)[-1][0] == "done"
    (first_q, _, first_end), (second_q, second_start, _) = spans
    assert (first_q, second_q) == ("first", "second")
    assert second_start >= first_end
    assert len(model.prompts) == 1  # the first question never reached the model


def test_a_question_while_another_streams_gets_503_until_it_ends(
    start_server: Start, fake_config: RagConfig
) -> None:
    # The slot is held for the whole stream: its teardown runs after the response
    # (scope="request"), not when the route's generator is created.
    model = StreamingFakeChatModel(prefill_s=0.5)
    port = serve(start_server, fake_config, fake_serving(fake_config, model))
    url = f"http://127.0.0.1:{port}/v1/query"

    first = ask(port)
    read_until(first, b"event: sources")
    busy = httpx.post(url, json={"question": "q"})
    assert busy.status_code == 503
    assert busy.headers["retry-after"] == "30"
    read_until(first, b"event: done")
    first.close()
    assert httpx.post(url, json={"question": "q"}).status_code == 200


def test_uvicorn_reports_asgi_spec_2_3(
    start_server: Start, fake_config: RagConfig, caplog: pytest.LogCaptureFixture
) -> None:
    # The disconnect handling above relies on it: from 2.4, Starlette stops listening for a
    # disconnect while a response streams (DEC-18).
    caplog.set_level("INFO", logger="rag_qa")
    port = serve(start_server, fake_config, fake_serving(fake_config, StreamingFakeChatModel()))
    assert httpx.get(f"http://127.0.0.1:{port}/v1/health/live").status_code == 200
    assert "The server reports ASGI spec 2.3" in caplog.text


# --- the forced window (S2-7's spike) ----------------------------------------------------------


@pytest.fixture
def blocked_sends(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Make FastAPI's SSE producer block when it sends a ``token`` event, as it does when
    the client reads slowly, so a drop lands outside the route's generator, every time.

    Returns a counter. A test asserts it moved: if FastAPI stops sending through an anyio
    memory stream, the window this forces has moved, and S2-7's note on DEC-18 needs a
    fresh look.
    """
    hits = {"blocked": 0}
    send = MemoryObjectSendStream.send

    async def blocking_send(self: MemoryObjectSendStream[Any], item: Any) -> None:
        if isinstance(item, bytes) and item.startswith(b"event: token"):
            hits["blocked"] += 1
            await anyio.sleep_forever()
        await send(self, item)

    monkeypatch.setattr(MemoryObjectSendStream, "send", blocking_send)
    return hits


@pytest.fixture
def no_garbage_collection() -> Iterator[None]:
    """The close must not depend on the garbage collector happening to run."""
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def drop_while_an_event_waits(
    start: Start, config: RagConfig, hits: dict[str, int]
) -> tuple[StreamingFakeChatModel, Serving]:
    model = StreamingFakeChatModel(tokens=[f"t{i} " for i in range(400)], gap_s=0.01)
    serving = fake_serving(config, model)
    sock = ask(serve(start, config, serving))
    read_until(sock, b"event: sources")
    assert within(CLOSE_WITHIN_S, lambda: hits["blocked"] == 1), (
        "FastAPI's SSE producer no longer sent through an anyio memory stream"
    )
    sock.close()
    return model, serving


@pytest.mark.usefixtures("either_way")
def test_a_drop_while_an_event_waits_to_be_sent_still_closes_the_stream(
    start_server: Start,
    fake_config: RagConfig,
    blocked_sends: dict[str, int],
    no_garbage_collection: None,
) -> None:
    cancelled = answers("cancelled")
    before = abandoned()
    model, serving = drop_while_an_event_waits(start_server, fake_config, blocked_sends)
    assert within(CLOSE_WITHIN_S, lambda: model.closed), (
        "the model's stream stayed open after a drop between events"
    )
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)
    # The slot's close ends the stream at its yield (GeneratorExit, not a cancel): it still
    # counts once, as cancelled (S3-7). The blocked event is the first token, which the route
    # had already observed: not abandoned (S3-7b's fix).
    assert within(CLOSE_WITHIN_S, lambda: answers("cancelled") == cancelled + 1)
    assert moved(before) == {"sources": 0, "first_token": 0}


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the window S2-7's spike found: without the slot's close, a drop between events "
    "leaves the stream open until garbage collection. If this passes, FastAPI changed how it "
    "sends events: re-check DEC-18's S2-7 note rather than delete the control.",
)
def test_negative_control_without_the_slots_close_the_stream_stays_open(
    start_server: Start,
    fake_config: RagConfig,
    blocked_sends: dict[str, int],
    no_garbage_collection: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def release_only(self: Slot) -> None:
        self.release()

    monkeypatch.setattr(Slot, "aclose", release_only)
    model, _ = drop_while_an_event_waits(start_server, fake_config, blocked_sends)
    assert within(CLOSE_WITHIN_S, lambda: model.closed)


# --- what reaches the app (S3-8, DEC-23) -------------------------------------------------------


def status_of(sock: socket.socket) -> int:
    """The status code of the response on ``sock``."""
    head = read_until(sock, b"\r\n\r\n")
    return int(head.split(b" ", 2)[1])


def test_a_declared_megabyte_without_a_token_is_413_before_any_of_it_is_sent(
    start_server: Start, fake_config: RagConfig
) -> None:
    # The body is never read: the answer comes before the client sends a byte of it.
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    port = start_server(create_app(fake_config, ServerSettings(token="s3cret"), serving=serving))
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.sendall(
            b"POST /v1/query HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: 1048576\r\n\r\n"
        )
        assert status_of(sock) == 413


def test_a_chunked_body_over_the_limit_without_a_token_is_413_not_400_401_or_422(
    start_server: Start, fake_config: RagConfig
) -> None:
    # No Content-Length: the limit finds out as the chunks arrive, and FastAPI answers its
    # HTTPException; any other exception would have been a 400 (DEC-23, design review 2). The
    # body is sent whole, 80 KiB, so the server closes nothing on unread bytes.
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    port = start_server(create_app(fake_config, ServerSettings(token="s3cret"), serving=serving))
    chunk = b"4000\r\n" + b" " * 0x4000 + b"\r\n"  # 16 KiB a chunk
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.sendall(
            b"POST /v1/query HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n" + chunk * 5 + b"0\r\n\r\n"
        )
        assert status_of(sock) == 413
    assert serving.slots.in_use == 0


def client_probe() -> FastAPI:
    """An app that answers with the client's address as the app sees it."""
    probe = FastAPI()

    @probe.get("/client")
    async def client(request: Request) -> dict[str, str | None]:
        return {"host": None if request.client is None else request.client.host}

    return probe


def seen_as(port: int, forwarded_for: str) -> str:
    response = httpx.get(
        f"http://127.0.0.1:{port}/client", headers={"X-Forwarded-For": forwarded_for}
    )
    return str(response.json()["host"])


def test_a_loopback_client_cannot_name_its_own_address(
    start_server: Start, monkeypatch: pytest.MonkeyPatch
) -> None:
    # uvicorn believes a loopback client's X-Forwarded-For by default, so every request could
    # claim a fresh address, and a rate limit of its own (DEC-23, design review 2). rag-serve's
    # configuration believes nobody's unless RAG_API_TRUSTED_PROXIES names it.
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    ours = start_server(client_probe())
    assert seen_as(ours, "203.0.113.7") == "127.0.0.1"
    # The control: uvicorn's own default, which rag-serve never uses.
    default = start_server(None, config=uvicorn.Config(client_probe(), log_level="warning"))
    assert seen_as(default, "203.0.113.7") == "203.0.113.7"
    # A proxy named by its address is believed.
    trusted = start_server(client_probe(), ServerSettings(trusted_proxies="127.0.0.1"))
    assert seen_as(trusted, "203.0.113.7") == "203.0.113.7"


def test_spoofed_forwarded_addresses_draw_on_the_real_address_s_failure_bucket(
    start_server: Start, fake_config: RagConfig
) -> None:
    # The verification's case (DEC-23, design review 2): eleven wrong tokens within a minute,
    # each naming a fresh X-Forwarded-For, from loopback. Every one counts against 127.0.0.1,
    # so the eleventh is 429; under uvicorn's default, each would have had a bucket of its own.
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    port = start_server(create_app(fake_config, ServerSettings(token="s3cret"), serving=serving))
    codes = [
        httpx.get(
            f"http://127.0.0.1:{port}/v1/ingest/x",
            headers={"Authorization": "Bearer wrong", "X-Forwarded-For": f"203.0.113.{n}"},
        ).status_code
        for n in range(11)
    ]
    assert codes == [401] * 10 + [429]
