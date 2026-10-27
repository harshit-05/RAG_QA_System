"""What bounds a request before the API works on it (S3-8, DEC-23): the body size, and the
rate limits' token buckets.

:class:`rag_qa.api.limits.BodyLimit` is driven here over raw ASGI messages, so the chunks of a
streamed body, and a disconnect between them, arrive exactly as a test sets them. Through the
app, ``test_api.py`` checks the refusal's place in the order, and ``test_api_uvicorn.py`` a body
sent over a real socket, with ``Content-Length`` and chunked.
"""

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.types import Message, Receive, Scope, Send

from rag_qa.api.limits import (
    AUTH_FAILURES_PER_MINUTE,
    MAX_BODY_BYTES,
    MAX_BUCKETS,
    TOO_LARGE,
    BodyLimit,
    Buckets,
    Limits,
    client_key,
    declared_length,
)
from rag_qa.api.models import MAX_QUESTION_CHARS, MAX_SOURCE_CHARS, MAX_SOURCES, QueryRequest


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
    [
        (b"10", 10),
        (b" 65536 ", 65536),
        (b"", None),
        (b"-1", None),
        (b"1e3", None),
        (b"\xb2", None),  # '²' is a digit to str.isdigit(), and int() refuses it
    ],
)
def test_the_declared_length_is_a_whole_number_or_nothing(value: bytes, length: int | None) -> None:
    assert declared_length(scope((b"content-length", value))) == length
    assert declared_length(scope()) is None


def test_the_request_model_s_largest_ascii_request_fits_the_limit() -> None:
    # The comment on MAX_BODY_BYTES says so; the model's bounds move on their own.
    body = json.dumps(
        {
            "question": "q" * MAX_QUESTION_CHARS,
            "sources": ["s" * MAX_SOURCE_CHARS] * MAX_SOURCES,
        }
    )
    QueryRequest.model_validate_json(body)
    assert len(body.encode()) < MAX_BODY_BYTES


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


# --- the rate limits ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_bucket_passes_a_minute_s_burst_then_one_request_per_refill() -> None:
    clock = Clock()
    buckets = Buckets(60, clock=clock)
    assert [buckets.take("k") for _ in range(60)] == [None] * 60
    wait = buckets.take("k")
    assert wait == pytest.approx(1.0)  # 60 a minute: one back each second
    clock.now += 0.5
    assert buckets.take("k") == pytest.approx(0.5)
    clock.now += 0.5
    assert buckets.take("k") is None
    clock.now += 3600  # it refills to a minute's worth, never more
    assert [buckets.take("k") for _ in range(61)].count(None) == 60


def test_each_key_has_its_own_bucket() -> None:
    buckets = Buckets(1, clock=Clock())
    assert (buckets.take("a"), buckets.take("b")) == (None, None)
    assert buckets.take("a") == pytest.approx(60.0)


def test_a_full_table_evicts_a_full_bucket_before_the_least_recently_used() -> None:
    # Evicting a full bucket loses nothing: a new one is full. A drained one evicted would hand
    # its caller a fresh minute (DEC-23, design review 2).
    clock = Clock()
    buckets = Buckets(2, size=3, clock=clock)
    for key in ("old", "drained"):
        buckets.take(key)
        buckets.take(key)  # both drained
    clock.now += 1
    buckets.take("recent")  # one token left, a few seconds from full
    clock.now += 30
    # At +31 s: "old" and "drained" have refilled one token of two, "recent" is full again.
    buckets.take("new")
    assert "recent" not in buckets
    assert all(key in buckets for key in ("old", "drained", "new"))
    # With none full, the least recently used goes.
    buckets.take("new")
    buckets.take("drained")
    buckets.take("newer")
    assert "old" not in buckets and len(buckets) == 3


def test_the_table_never_holds_more_than_its_size() -> None:
    clock = Clock()
    buckets = Buckets(10, clock=clock)
    for n in range(MAX_BUCKETS + 500):
        buckets.take(f"10.0.{n // 256}.{n % 256}")
    assert len(buckets) == MAX_BUCKETS == 10_000


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("192.0.2.7", "192.0.2.7"),
        ("2001:db8:1:2:3:4:5:6", "2001:db8:1:2::/64"),
        ("2001:db8:1:2:ffff::1", "2001:db8:1:2::/64"),  # the same /64: one bucket
        ("::ffff:192.0.2.7", "192.0.2.7"),  # IPv4-mapped: the IPv4 client it is
        ("testclient", "testclient"),
        (None, "-"),
    ],
)
def test_a_client_is_counted_by_its_address_and_ipv6_by_its_64(host: str | None, key: str) -> None:
    assert client_key(host) == key


def test_the_api_s_tables_and_their_rates() -> None:
    limits = Limits(60)
    assert (limits.keys.capacity, limits.addresses.capacity) == (60, 60)
    assert limits.failures.capacity == AUTH_FAILURES_PER_MINUTE == 10
