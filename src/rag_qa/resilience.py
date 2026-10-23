"""Circuit breakers and retries for the network calls (NFR-6, DEC-21, ARCHITECTURE.md §3.1).

Three services are reached over a network: Qdrant, Ollama and the hosted fallback. Each call
has a timeout, set where its client is built, and goes through the service's breaker; a call
is retried only where a repeat is safe.

- **A breaker** (:class:`CircuitBreaker`) counts the consecutive failed attempts against one
  service. After :data:`FAILURES` of them it opens, and refuses every attempt at once with
  :class:`BreakerOpenError` for :data:`RESET_AFTER_S`: the API answers 503 with
  ``Retry-After``, and a command says which service is down. Then it lets one attempt through
  (half-open), a probe whose success closes it and whose failure opens it again. A dead
  service so costs a fast, explicit refusal rather than a timeout per call. It wraps each
  *attempt*, never a whole retried call, so an open breaker also ends a retry loop (design
  review 2).
- **What counts against it** is :func:`classify`'s verdict: a connection that failed, a
  timeout, or an answer that the service cannot serve just now (429, 5xx). Any other answer
  is a success, since the service answered. A cancel says nothing about the service.
- **Retries** wait :func:`backoff_s` between attempts: exponential, with full jitter, so
  callers that failed together do not retry together. :func:`retrying` is tenacity's loop for
  a call that returns once (Qdrant's); the answer stream writes its own loop, since a model's
  stream is an async generator (:func:`rag_qa.answering.stream_answer`).
- **A deadline** (:func:`deadline`) bounds every attempt made inside it, in its thread or
  task: a retrieval holds the API's retrieval lock throughout, so its attempts share one
  budget (design review 2).

The standard library and tenacity only. Errors are told apart by their class names and
modules, so this imports none of the clients that raise them (``tests/test_architecture.py``).
"""

import math
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, Literal

import tenacity

#: Consecutive failed attempts that open a breaker.
FAILURES: Final = 3
#: Seconds an open breaker refuses every attempt before it lets one probe through.
RESET_AFTER_S: Final = 30.0

#: Attempts at one Qdrant call: every call is safe to repeat (DEC-21; ``vectorstore``).
QDRANT_ATTEMPTS: Final = 3
#: Attempts at a model's answer, made only before its first token and only after a failed
#: connection: a repeat after a token would duplicate text.
LLM_ATTEMPTS: Final = 2
#: The hosted fallback's attempts: one. Its client retries nothing either (``max_retries: 0``),
#: so its breaker sees every failure.
FALLBACK_ATTEMPTS: Final = 1

#: The back-off's scale and cap, in seconds: attempt n waits up to ``BACKOFF_S * 2**(n-1)``.
BACKOFF_S: Final = 0.5
BACKOFF_MAX_S: Final = 4.0

#: :func:`classify`'s verdicts.
CONNECTION: Final = "connection"  # never reached, or dropped before an answer: safe to repeat
TIMEOUT: Final = "timeout"  # no answer in time
UNAVAILABLE: Final = "unavailable"  # it answered that it cannot serve now: 429, 5xx, or mid-stream
ANSWERED: Final = "answered"  # it answered, refusing the request itself: any other 4xx
#: The verdicts that count against a breaker.
FAILED: Final = frozenset({CONNECTION, TIMEOUT, UNAVAILABLE})

#: A breaker's states.
type BreakerState = Literal["closed", "open", "half-open"]
CLOSED: Final = "closed"
OPEN: Final = "open"
HALF_OPEN: Final = "half-open"

#: The HTTP clients whose errors :func:`classify` reads: httpx (Ollama, Qdrant) and httpx2,
#: which openai 3 is built on (the fallback).
_HTTP_CLIENTS: Final = frozenset({"httpx", "httpx2"})
#: Their errors for a server that could not be reached, or that dropped the connection
#: before it answered. Not the read, write and pool timeouts (``TimeoutException``).
_UNREACHABLE: Final = frozenset(
    {"ConnectError", "ConnectTimeout", "ReadError", "WriteError", "RemoteProtocolError"}
)


class BreakerOpenError(Exception):
    """A service's breaker refused an attempt: the service failed :data:`FAILURES` times in a
    row, and is not asked again until ``retry_after_s`` have passed."""

    def __init__(self, service: str, retry_after_s: float, failures: int) -> None:
        self.service = service
        self.retry_after_s = retry_after_s
        wait = max(1, math.ceil(retry_after_s))
        super().__init__(
            f"{service} is not answering: {failures} attempts in a row failed, so it is not "
            f"asked again for {wait} s"
        )


class DeadlineError(TimeoutError):
    """The deadline that bounds these attempts (:func:`deadline`) has passed."""


class Attempt:
    """One attempt a breaker let through. Its first recorded outcome counts; later ones are
    ignored, so an answer judged at its first token is never judged twice."""

    def __init__(self, breaker: "CircuitBreaker", *, probe: bool) -> None:
        self._breaker = breaker
        self._probe = probe
        self._ended = False

    def succeeded(self) -> None:
        """The service answered."""
        self._end(succeeded=True)

    def failed(self) -> None:
        """The service failed this attempt (a :data:`FAILED` verdict)."""
        self._end(succeeded=False)

    def abandoned(self) -> None:
        """The attempt ended with no verdict on the service: it was cancelled, or failed for
        a reason of its own. It frees a probe."""
        self._end(succeeded=None)

    def _end(self, *, succeeded: bool | None) -> None:
        if not self._ended:
            self._ended = True
            self._breaker._record(succeeded, probe=self._probe)


class CircuitBreaker:
    """One service's breaker: closed, open after ``failures`` consecutive failed attempts,
    then half-open once ``reset_after_s`` have passed, letting exactly one probe through.

    Thread-safe: retrieval runs in worker threads, while answers stream on the event loop.
    ``clock`` is for tests.
    """

    def __init__(
        self,
        name: str,
        failures: int = FAILURES,
        reset_after_s: float = RESET_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failures < 1:
            raise ValueError(f"failures must be at least 1, not {failures}")
        if not reset_after_s > 0:
            raise ValueError(f"reset_after_s must be a positive number, not {reset_after_s}")
        self.name = name  # the service, as messages name it: "Qdrant", "Ollama"
        self.failures = failures
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failed = 0  # consecutive failed attempts
        self._opened_at: float | None = None
        self._probing = False

    def __repr__(self) -> str:
        return f"CircuitBreaker({self.name!r}, state={self.state!r})"

    @property
    def state(self) -> BreakerState:
        """:data:`CLOSED`, :data:`OPEN`, or :data:`HALF_OPEN` once the wait is over."""
        with self._lock:
            return self._state()

    def _state(self) -> BreakerState:
        if self._opened_at is None:
            return CLOSED
        if self._clock() - self._opened_at < self.reset_after_s:
            return OPEN
        return HALF_OPEN

    def _wait(self) -> float:
        """Seconds left before the probe; zero or less once it may go."""
        if self._opened_at is None:
            return 0.0
        return self.reset_after_s - (self._clock() - self._opened_at)

    def retry_after_s(self) -> float:
        """Seconds until the breaker lets a probe through; 0 unless it is open."""
        with self._lock:
            return self._wait() if self._state() == OPEN else 0.0

    def allow(self) -> Attempt:
        """Let one attempt through, or raise :class:`BreakerOpenError`.

        Half-open, the first caller gets the probe, and every other is refused until it
        ends. The caller records the attempt's outcome on what this returns.
        """
        with self._lock:
            state = self._state()
            if state == CLOSED:
                return Attempt(self, probe=False)
            if state == HALF_OPEN and not self._probing:
                self._probing = True
                return Attempt(self, probe=True)
            wait, failed = self._wait(), self._failed
        raise BreakerOpenError(self.name, wait, failed)

    def _record(self, succeeded: bool | None, *, probe: bool) -> None:
        with self._lock:
            if succeeded:
                self._failed = 0
                self._opened_at = None
                self._probing = False
                return
            if probe:
                self._probing = False
            if succeeded is None:
                return
            self._failed += 1
            # A failed probe opens it again, with a fresh wait; so does the failure that
            # reaches the count. Failures while it is open already change nothing else.
            if probe or (self._opened_at is None and self._failed >= self.failures):
                self._opened_at = self._clock()


def _is_ollama_stream_error(error: BaseException) -> bool:
    """Ollama's error inside a stream: a ``ResponseError`` with no HTTP status (-1)."""
    return type(error).__module__.split(".")[0] == "ollama" and (
        getattr(error, "status_code", None) == -1
    )


def classify(error: BaseException) -> str | None:
    """What ``error`` says about the service that raised it, or ``None`` when it says nothing.

    :data:`CONNECTION`, :data:`TIMEOUT` and :data:`UNAVAILABLE` count against its breaker;
    :data:`ANSWERED` means the service answered, refusing the request itself. Told apart by
    class name across the MRO and the defining module, for httpx and httpx2 (whose errors
    ChatOllama's stream and qdrant-client raise unwrapped), openai and its subclasses in
    langchain-openai, Ollama's ``ResponseError``, and the builtins.
    """
    for kind in type(error).__mro__:
        package = kind.__module__.split(".")[0]
        if package in _HTTP_CLIENTS:
            if kind.__name__ in _UNREACHABLE:
                return CONNECTION
            if kind.__name__ == "TimeoutException":
                return TIMEOUT
        elif package == "openai":
            if kind.__name__ == "APITimeoutError":
                return TIMEOUT
            if kind.__name__ == "APIConnectionError":
                return CONNECTION
    if isinstance(error, ConnectionError):  # ollama wraps httpx's ConnectError in one
        return CONNECTION
    if isinstance(error, TimeoutError):  # asyncio's, and langchain-openai's chunk timeout
        return TIMEOUT
    if _is_ollama_stream_error(error):
        return UNAVAILABLE
    status = getattr(error, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status == 429 or status >= 500:
            return UNAVAILABLE
        if 400 <= status < 500:
            return ANSWERED
    return None


def backoff_s(attempt: int) -> float:
    """Seconds to wait after failed attempt number ``attempt`` (from 1): a random time up to
    ``BACKOFF_S * 2**(attempt-1)``, capped at :data:`BACKOFF_MAX_S` ("full jitter")."""
    # Not for anything secret: spreading out the retries of callers that failed together.
    return random.uniform(0, min(BACKOFF_MAX_S, BACKOFF_S * 2 ** (attempt - 1)))


#: The end of the deadline that bounds the attempts made now, on :func:`time.monotonic`'s
#: clock; ``None`` when there is none. A context variable, so each thread and each task has
#: its own.
_DEADLINE: ContextVar[float | None] = ContextVar("rag_qa_deadline", default=None)


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Bound the attempts made inside to ``seconds`` from now, in this thread or task.

    A deadline already in force that ends sooner stays in force: one budget covers a whole
    retrieval, however many calls it makes.
    """
    end = time.monotonic() + seconds
    current = _DEADLINE.get()
    token = _DEADLINE.set(end if current is None else min(current, end))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining() -> float | None:
    """Seconds left before the deadline in force, or ``None`` when there is none. Negative
    once it has passed."""
    end = _DEADLINE.get()
    return None if end is None else end - time.monotonic()


class _PastDeadline(tenacity.stop.stop_base):
    """Stop when the next wait would end at or past the :func:`deadline` in force."""

    def __call__(self, retry_state: tenacity.RetryCallState) -> bool:
        left = remaining()
        return left is not None and left <= retry_state.upcoming_sleep


def retrying(attempts: int, retry_on: Callable[[BaseException], bool]) -> tenacity.Retrying:
    """tenacity's loop for a call that returns once: up to ``attempts`` tries while it raises
    what ``retry_on`` accepts, :func:`backoff_s` apart, and none that would pass the
    :func:`deadline`. The last error is raised as it was, never wrapped."""
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(attempts) | _PastDeadline(),
        # Looked up at each wait, so a test can make the back-off instant.
        wait=lambda state: backoff_s(state.attempt_number),
        retry=tenacity.retry_if_exception(retry_on),
        reraise=True,
    )
