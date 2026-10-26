"""What bounds a request before the API does any work for it (DEC-23, S3-8).

**The body size** (:class:`BodyLimit`). A request body is bounded at :data:`MAX_BODY_BYTES`
before FastAPI reads or parses it, ahead of every dependency: that closes the
pre-authentication gap S2-7's first review found, where FastAPI parsed a body of any size
before the token was checked.

- **By its ``Content-Length``:** a declared size over the limit gets 413 at once, and the app
  never runs, so nothing of the body is read.
- **As it streams** (a chunked body has no ``Content-Length``): the ``receive`` the middleware
  wraps counts the bytes, and raises ``HTTPException(413)`` once they pass the limit. It must be
  that exception: FastAPI turns any other one raised while it reads a body into a 400, "There
  was an error parsing the body" (``fastapi/routing.py``), and re-raises an ``HTTPException``
  for its handler to answer (DEC-23, design review 2).
- **``http.disconnect`` passes through untouched,** so a client that leaves still cancels its
  answer (DEC-14, DEC-18): the wrapper only reads the size of what ``receive`` returns.

A pure ASGI middleware, not Starlette's ``BaseHTTPMiddleware``, which would run the app in a
task of its own and stand between the route and the disconnect.

**The rate limits** (:class:`Limits`): token buckets of our own, in memory, per replica.

- **Per key:** ``RAG_API_RATE_LIMIT`` requests a minute (60 by default).
- **Per client address,** at the same rate, when the server has no credential configured (a
  loopback server with no key and no token), so its requests carry no key to count by.
- **Per address, for failed authentication:** :data:`AUTH_FAILURES_PER_MINUTE`, which slows
  token guessing. It gates only the requests that fail: a valid key is never refused because
  someone else failed from its address (DEC-23, design review 2).

A bucket holds a minute's worth and refills steadily, so a burst of a minute's requests passes
and the rest wait their turn; a refusal says how long in ``Retry-After``. The health routes
take none: Docker's and Caddy's checks poll them from one address, and their probes are cached
(DEC-23, design review 1). **Each table is bounded** at :data:`MAX_BUCKETS`: a new caller
evicts the first full bucket from the least recently used end, which loses nothing (a full
bucket is what a new one is), and only failing that the least recently used. An IPv6 client is
counted by its /64, which one host holds whole (:func:`client_key`); unbounded, one /64 grew
the tables by 36 MB per 200,000 addresses (measured in design review 2). The client's address
is uvicorn's, which believes ``X-Forwarded-For`` from the trusted proxies alone
(:func:`rag_qa.api.server.uvicorn_config`).

The buckets are used on the event loop alone, by the API's async dependencies, so they take
no lock.
"""

import ipaddress
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Final

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The largest request body accepted, in bytes. The request model's largest ASCII request, 2,000
#: characters of question and 50 document names of 1,024, is about 53 KB and fits (a test pins
#: it). Names of multi-byte characters can pass it: a source must be a path the index holds.
MAX_BODY_BYTES: Final = 64 * 1024

#: What a body over the limit is told: fixed text, the same for both ways of finding out.
TOO_LARGE: Final = f"the request body is larger than {MAX_BODY_BYTES // 1024} KiB"


def declared_length(scope: Scope) -> int | None:
    """The request's ``Content-Length``, or ``None`` when it sends none, or one that is not a
    whole number (uvicorn's parser refuses those itself before the app sees them)."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            text = value.decode("latin-1").strip()
            # ASCII only: str.isdigit() also accepts '²', which int() then refuses.
            return int(text) if text.isascii() and text.isdigit() else None
    return None


class BodyLimit:
    """413 for a request body over ``limit`` bytes, before the app reads it (the module
    docstring). ``refused`` is called with the request's scope for each one, before the answer:
    the API counts those of a question."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limit: int = MAX_BODY_BYTES,
        refused: Callable[[Scope], None] | None = None,
    ) -> None:
        self.app = app
        self.limit = limit
        self.refused = refused

    def _refuse(self, scope: Scope) -> None:
        if self.refused is not None:
            self.refused(scope)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = declared_length(scope)
        if declared is not None and declared > self.limit:
            self._refuse(scope)
            response = JSONResponse({"detail": TOO_LARGE}, status.HTTP_413_CONTENT_TOO_LARGE)
            await response(scope, receive, send)
            return
        received = 0

        async def limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    self._refuse(scope)
                    raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, TOO_LARGE)
            return message

        await self.app(scope, limited, send)


# --- the rate limits -----------------------------------------------------------------------------

#: The most buckets one table holds (DEC-23, design review 2). A bucket costs about 180 bytes.
MAX_BUCKETS: Final = 10_000
#: Failed authentications an address may make in a minute.
AUTH_FAILURES_PER_MINUTE: Final = 10


def client_key(host: str | None) -> str:
    """What the limits count a client's address as: an IPv4 address, an IPv4-mapped IPv6
    address as its IPv4 one, any other IPv6 address as its /64, and anything else (a test
    client's name) as it is."""
    if host is None:
        return "-"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


class Buckets:
    """Token buckets by key, ``per_minute`` each and full at first, at most ``size`` of them
    (the module docstring). ``clock`` is for the tests."""

    def __init__(
        self,
        per_minute: int,
        *,
        size: int = MAX_BUCKETS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(per_minute)
        self.refill_per_s = per_minute / 60.0
        self.size = size
        self._clock = clock
        #: ``[tokens, when]`` by key, the least recently used first.
        self._buckets: OrderedDict[str, list[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buckets)

    def __contains__(self, key: str) -> bool:
        return key in self._buckets

    def _level(self, bucket: list[float], now: float) -> float:
        tokens, when = bucket
        return min(self.capacity, tokens + (now - when) * self.refill_per_s)

    def take(self, key: str) -> float | None:
        """Take one request's token from ``key``'s bucket: ``None`` when it had one, or the
        seconds until it will."""
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.size:
                self._evict(now)
            bucket = self._buckets[key] = [self.capacity, now]
        else:
            self._buckets.move_to_end(key)
        tokens = self._level(bucket, now)
        if tokens >= 1.0:
            bucket[:] = [tokens - 1.0, now]
            return None
        bucket[:] = [tokens, now]
        return (1.0 - tokens) / self.refill_per_s

    def _evict(self, now: float) -> None:
        """Make room for one: the first full bucket from the least recently used end, else the
        least recently used. Under a flood of new addresses the oldest is full within a few
        seconds, so the scan stops at once; it reads every bucket only when all of them were
        drawn on in the last minute."""
        for key, bucket in self._buckets.items():
            if self._level(bucket, now) >= self.capacity:
                del self._buckets[key]
                return
        self._buckets.popitem(last=False)


class Limits:
    """The API's three tables: per key, per address without a credential, and per address for
    failed authentication (the module docstring)."""

    def __init__(self, per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.keys = Buckets(per_minute, clock=clock)
        self.addresses = Buckets(per_minute, clock=clock)
        self.failures = Buckets(AUTH_FAILURES_PER_MINUTE, clock=clock)
