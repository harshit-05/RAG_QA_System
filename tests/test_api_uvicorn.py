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
"""

import gc
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import anyio
import httpx
import pytest
import uvicorn
from anyio.streams.memory import MemoryObjectSendStream
from conftest import StreamingFakeChatModel, fake_serving, sse_events
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_qa.api.app import create_app
from rag_qa.api.state import Serving, Slot
from rag_qa.schema import RagConfig
from rag_qa.settings import ServerSettings

#: How long a stopped answer may take to close its stream to the model (the story's 1 s).
CLOSE_WITHIN_S = 1.0

Start = Callable[[FastAPI], int]


@pytest.fixture
def start_server() -> Iterator[Start]:
    """Start uvicorn with an app on a free loopback port, in a thread; returns the port."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(app: FastAPI) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=1)
        )
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
    return start(create_app(config, ServerSettings(), serving=serving))


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


def within(seconds: float, condition: Callable[[], bool]) -> bool:
    """Whether ``condition`` holds, polled until ``seconds`` have passed."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


# --- the drops --------------------------------------------------------------------------------


def test_a_drop_while_the_model_reads_the_prompt_closes_its_stream(
    start_server: Start, fake_config: RagConfig
) -> None:
    # No event is sent during a prefill, so only Starlette's listening for the disconnect
    # can notice it (uvicorn reports ASGI spec 2.3).
    model = StreamingFakeChatModel(prefill_s=10.0)
    serving = fake_serving(fake_config, model)
    sock = ask(serve(start_server, fake_config, serving))
    read_until(sock, b"event: sources")
    assert within(CLOSE_WITHIN_S, lambda: model.log == ["start"])
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: model.closed), (
        "the model was still reading the prompt 1 s after its client left"
    )
    assert model.log == ["start", "closed"]  # no token generated
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)


def test_a_drop_mid_stream_closes_the_models_stream(
    start_server: Start, fake_config: RagConfig
) -> None:
    model = StreamingFakeChatModel(tokens=[f"t{i} " for i in range(400)], gap_s=0.01)
    serving = fake_serving(fake_config, model)
    sock = ask(serve(start_server, fake_config, serving))
    read_until(sock, b"event: token", count=3)
    sock.close()
    assert within(CLOSE_WITHIN_S, lambda: model.closed), "generation ran on after the drop"
    assert len(model.log) < 100  # "start", the tokens sent so far, "closed": not 400
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)


def test_a_drop_before_the_first_event_frees_the_slot_and_the_next_retrieval_waits(
    start_server: Start, fake_config: RagConfig
) -> None:
    # The retrieval thread cannot be stopped (DEC-14). Its slot is freed at once, but the
    # next request's retrieval must wait for it rather than run beside it (S2-4's second
    # review: two reranks would compete for the same CPU).
    spans: list[tuple[str, float, float]] = []
    chunk = Document(page_content="x", metadata={"source": "notes.txt"})

    def slow_retrieval(question: str) -> list[Document]:
        started = time.monotonic()
        time.sleep(1.0)
        spans.append((question, started, time.monotonic()))
        return [chunk]

    model = StreamingFakeChatModel()
    serving = fake_serving(fake_config, model, retrieve=RunnableLambda(slow_retrieval))
    port = serve(start_server, fake_config, serving)

    first = ask(port, "first")
    time.sleep(0.2)  # its retrieval is running in a worker thread now
    first.close()
    assert within(0.5, lambda: serving.slots.in_use == 0)  # freed before its retrieval ends
    assert spans == []

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


def test_a_drop_while_an_event_waits_to_be_sent_still_closes_the_stream(
    start_server: Start,
    fake_config: RagConfig,
    blocked_sends: dict[str, int],
    no_garbage_collection: None,
) -> None:
    model, serving = drop_while_an_event_waits(start_server, fake_config, blocked_sends)
    assert within(CLOSE_WITHIN_S, lambda: model.closed), (
        "the model's stream stayed open after a drop between events"
    )
    assert within(CLOSE_WITHIN_S, lambda: serving.slots.in_use == 0)


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
