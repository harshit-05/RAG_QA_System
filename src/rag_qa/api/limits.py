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
"""

from collections.abc import Callable
from typing import Final

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: The largest request body accepted, in bytes. A question and its sources fit in a fraction of
#: it: the request model allows 2,000 characters of question and 50 document names.
MAX_BODY_BYTES: Final = 64 * 1024

#: What a body over the limit is told: fixed text, the same for both ways of finding out.
TOO_LARGE: Final = f"the request body is larger than {MAX_BODY_BYTES // 1024} KiB"


def declared_length(scope: Scope) -> int | None:
    """The request's ``Content-Length``, or ``None`` when it sends none, or one that is not a
    whole number (uvicorn's parser refuses those itself before the app sees them)."""
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            text = value.decode("latin-1").strip()
            return int(text) if text.isdigit() else None
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
