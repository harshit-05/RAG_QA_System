"""What bounds a request before the API works on it (S3-8, DEC-23): the body size.

:class:`rag_qa.api.limits.BodyLimit` is driven here over raw ASGI messages, so the chunks of a
streamed body, and a disconnect between them, arrive exactly as a test sets them. Through the
app, ``test_api.py`` checks the refusal's place in the order, and ``test_api_uvicorn.py`` a body
sent over a real socket, with ``Content-Length`` and chunked.
"""

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.types import Message, Receive, Scope, Send

from rag_qa.api.limits import MAX_BODY_BYTES, TOO_LARGE, BodyLimit, declared_length


def scope(*headers: tuple[bytes, bytes], path: str = "/v1/query") -> Scope:
    return {"type": "http", "method": "POST", "path": path, "headers": list(headers)}


def messages(*items: Message) -> Receive:
    """A ``receive`` that hands out ``items`` in order."""
    pending = list(items)

    async def receive() -> Message:
        return pending.pop(0)

    return receive


def chunks(*sizes: int) -> list[Message]:
    """A streamed body: one ``http.request`` per size, the last with ``more_body`` false."""
    return [
        {"type": "http.request", "body": b"x" * size, "more_body": n < len(sizes) - 1}
        for n, size in enumerate(sizes)
    ]


class Reader:
    """An app that reads the whole body, as FastAPI does before any dependency, and records
    what it saw."""

    def __init__(self) -> None:
        self.ran = False
        self.read: list[Message] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.ran = True
        while True:
            message = await receive()
            self.read.append(message)
            if message["type"] != "http.request" or not message.get("more_body", False):
                return


def run(
    app: Any, scope: Scope, incoming: Iterable[Message], **options: Any
) -> tuple[list[Message], list[Scope]]:
    """Run ``app`` behind the limit; returns what it sent and the scopes it refused."""
    sent: list[Message] = []
    refused: list[Scope] = []

    async def send(message: Message) -> None:
        sent.append(message)

    limited = BodyLimit(app, refused=refused.append, **options)
    asyncio.run(limited(scope, messages(*incoming), send))
    return sent, refused


@pytest.mark.parametrize(
    ("value", "length"),
    [(b"10", 10), (b" 65536 ", 65536), (b"", None), (b"-1", None), (b"1e3", None)],
)
def test_the_declared_length_is_a_whole_number_or_nothing(value: bytes, length: int | None) -> None:
    assert declared_length(scope((b"content-length", value))) == length
    assert declared_length(scope()) is None


def test_a_declared_length_over_the_limit_is_413_and_the_app_never_runs() -> None:
    app = Reader()
    headers = (b"content-length", str(MAX_BODY_BYTES + 1).encode())
    sent, refused = run(app, scope(headers), chunks(10))
    assert not app.ran
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 413
    assert TOO_LARGE.encode() in sent[1]["body"]
    assert len(refused) == 1


def test_a_body_at_the_limit_is_read_whole() -> None:
    app = Reader()
    headers = (b"content-length", str(MAX_BODY_BYTES).encode())
    half = MAX_BODY_BYTES // 2
    sent, refused = run(app, scope(headers), chunks(half, MAX_BODY_BYTES - half))
    assert sum(len(m["body"]) for m in app.read) == MAX_BODY_BYTES
    assert (sent, refused) == ([], [])


def test_a_streamed_body_is_cut_at_the_chunk_that_passes_the_limit() -> None:
    # No Content-Length: the chunks are counted as they arrive. The exception is FastAPI's own
    # HTTPException, which FastAPI re-raises for its handler; any other would become a 400.
    app = Reader()
    with pytest.raises(HTTPException) as raised:
        run(app, scope(), chunks(30_000, 30_000, 10_000, 10_000))
    assert (raised.value.status_code, raised.value.detail) == (413, TOO_LARGE)
    assert len(app.read) == 2  # the third chunk passed the limit, and was never handed over


def test_a_lying_content_length_is_still_cut_as_it_streams() -> None:
    app = Reader()
    with pytest.raises(HTTPException):
        run(app, scope((b"content-length", b"10")), chunks(MAX_BODY_BYTES + 1))


def test_a_disconnect_passes_through_untouched() -> None:
    # The cancel path (DEC-14, DEC-18): Starlette learns a client left from this very message.
    app = Reader()
    disconnect: Message = {"type": "http.disconnect"}
    sent, refused = run(app, scope(), [chunks(10, 10)[0], disconnect])
    assert app.read[-1] is disconnect
    assert (sent, refused) == ([], [])


def test_anything_but_http_goes_by_unread() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    run(app, {"type": "lifespan"}, [])
    assert seen == ["lifespan"]
