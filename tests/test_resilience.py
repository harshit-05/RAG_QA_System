"""Circuit breakers, error verdicts, the back-off and the deadline (S3-6, NFR-6, DEC-21).

The breaker runs on a fake clock, so its waits are exact and instant. ``classify`` is checked
against the real errors of httpx, httpx2 and ollama, and stand-ins for openai's, which only
the ``fallback`` extra installs: it reads class names and modules, which the stand-ins share.
"""

import threading
import time

import httpx
import httpx2
import ollama
import pytest
from conftest import openai_error

from rag_qa import resilience
from rag_qa.resilience import (
    ANSWERED,
    CLOSED,
    CONNECTION,
    HALF_OPEN,
    OPEN,
    PROBE_WAIT_S,
    TIMEOUT,
    UNAVAILABLE,
    BreakerOpenError,
    CircuitBreaker,
    classify,
    is_marked_fallback,
    mark_fallback,
)
from rag_qa.resilience import backoff_s as real_backoff_s  # before the conftest's patch


class FakeClock:
    """Seconds that pass only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def breaker(clock: FakeClock, failures: int = 3, reset_after_s: float = 30.0) -> CircuitBreaker:
    return CircuitBreaker("Ollama", failures=failures, reset_after_s=reset_after_s, clock=clock)


def fail(b: CircuitBreaker, times: int = 1) -> None:
    for _ in range(times):
        b.allow().failed()


# --- the state machine -------------------------------------------------------------------


def test_it_opens_after_the_set_number_of_consecutive_failures() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 2)
    assert b.state == CLOSED
    fail(b)
    assert b.state == OPEN
    with pytest.raises(BreakerOpenError) as exc:
        b.allow()
    assert exc.value.service == "Ollama"
    assert exc.value.retry_after_s == pytest.approx(30.0)
    assert str(exc.value) == (
        "Ollama is not answering: 3 attempts in a row failed, so it is not asked again for 30 s"
    )


def test_a_success_resets_the_count() -> None:
    b = breaker(FakeClock())
    fail(b, 2)
    b.allow().succeeded()
    fail(b, 2)
    assert b.state == CLOSED  # 2, 0, 2: never 3 in a row


def test_an_answer_that_says_nothing_does_not_reset_the_count() -> None:
    b = breaker(FakeClock())
    fail(b, 2)
    b.allow().abandoned()  # a cancel: nothing learned about the service
    fail(b)
    assert b.state == OPEN


def test_retry_after_counts_down_and_the_wait_ends_half_open() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 12
    assert b.retry_after_s() == pytest.approx(18.0)
    clock.now += 18
    assert b.state == HALF_OPEN
    assert b.retry_after_s() == 0.0


def test_half_open_lets_exactly_one_probe_through_and_its_success_closes_it() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    probe = b.allow()
    with pytest.raises(BreakerOpenError):
        b.allow()  # a second caller while the probe is out
    probe.succeeded()
    assert b.state == CLOSED
    b.allow().succeeded()


def test_it_refuses_while_open_and_while_its_probe_is_out() -> None:
    clock = FakeClock()
    b = breaker(clock)
    assert b.refusal_s() is None  # closed
    fail(b, 3)
    clock.now += 12
    assert b.refusal_s() == pytest.approx(18.0)  # open: the wait that is left
    clock.now += 18
    assert b.state == HALF_OPEN
    assert b.refusal_s() is None  # half-open, no probe yet: the next attempt is the probe
    probe = b.allow()
    assert b.refusal_s() == PROBE_WAIT_S  # the probe is out: everything else is refused
    probe.succeeded()
    assert b.refusal_s() is None


def test_an_abandoned_probe_stops_the_refusal() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    probe = b.allow()
    assert b.refusal_s() == PROBE_WAIT_S
    probe.abandoned()
    assert b.refusal_s() is None  # half-open still, and its probe free again


def test_an_error_is_marked_as_the_fallback_s_only_when_it_is_marked() -> None:
    error = ConnectionError("x")
    assert not is_marked_fallback(error)
    mark_fallback(error)
    assert is_marked_fallback(error)
    assert not is_marked_fallback(ConnectionError("x"))  # the mark is on the instance


def test_marking_an_error_that_takes_no_attributes_is_harmless() -> None:
    class Frozen(Exception):
        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError(name)

    error = Frozen()
    mark_fallback(error)  # must not raise: the answer's own error is what matters
    assert not is_marked_fallback(error)


def test_a_failed_probe_opens_it_again_for_a_fresh_wait() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 31
    b.allow().failed()
    assert b.state == OPEN
    assert b.retry_after_s() == pytest.approx(30.0)


def test_an_abandoned_probe_frees_the_probe_without_a_verdict() -> None:
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    b.allow().abandoned()  # cancelled mid-probe
    assert b.state == HALF_OPEN
    b.allow().succeeded()  # the next caller probes
    assert b.state == CLOSED


def test_an_attempt_counts_once() -> None:
    clock = FakeClock()
    b = breaker(clock, failures=1)
    tried = b.allow()
    tried.succeeded()  # the first token
    tried.failed()  # a failure after it is not judged again
    assert b.state == CLOSED


def test_failures_while_open_change_nothing_else() -> None:
    clock = FakeClock()
    b = breaker(clock)
    early = [b.allow() for _ in range(4)]  # four attempts let through while closed
    for tried in early[:3]:
        tried.failed()
    clock.now += 10
    early[3].failed()  # a late failure does not restart the wait
    assert b.retry_after_s() == pytest.approx(20.0)


@pytest.mark.parametrize(("failures", "reset"), [(0, 30.0), (3, 0.0), (3, -1.0)])
def test_its_settings_are_checked(failures: int, reset: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        CircuitBreaker("Qdrant", failures=failures, reset_after_s=reset)


def test_half_open_admits_one_probe_among_many_threads() -> None:
    # Retrieval runs in worker threads: the probe must be one, however many race for it.
    clock = FakeClock()
    b = breaker(clock)
    fail(b, 3)
    clock.now += 30
    start = threading.Barrier(50)
    admitted: list[object] = []

    def race() -> None:
        start.wait()
        try:
            admitted.append(b.allow())
        except BreakerOpenError:
            pass

    threads = [threading.Thread(target=race) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(admitted) == 1


def test_concurrent_failures_are_all_counted() -> None:
    b = breaker(FakeClock(), failures=1000)
    tries = [b.allow() for _ in range(200)]
    threads = [threading.Thread(target=t.failed) for t in tries]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert b._failed == 200


# --- what an error says about its service --------------------------------------------------

REQUEST = httpx.Request("GET", "http://127.0.0.1:1/api/chat")


@pytest.mark.parametrize(
    ("error", "verdict"),
    [
        # httpx, as ChatOllama's stream and qdrant-client raise it, unwrapped
        (httpx.ConnectError("All connection attempts failed", request=REQUEST), CONNECTION),
        (httpx.ConnectTimeout("timed out", request=REQUEST), CONNECTION),
        (httpx.ReadError("reset", request=REQUEST), CONNECTION),
        (httpx.WriteError("broken pipe", request=REQUEST), CONNECTION),
        (httpx.RemoteProtocolError("Server disconnected", request=REQUEST), CONNECTION),
        (httpx.ReadTimeout("timed out", request=REQUEST), TIMEOUT),
        (httpx.WriteTimeout("timed out", request=REQUEST), TIMEOUT),
        (httpx.PoolTimeout("timed out", request=REQUEST), TIMEOUT),
        # httpx2, which openai 3 is built on
        (httpx2.ConnectError("refused"), CONNECTION),
        (httpx2.ReadTimeout("timed out"), TIMEOUT),
        # the builtins: ollama wraps httpx's ConnectError in one; asyncio's timeout
        (ConnectionRefusedError("refused"), CONNECTION),
        (ConnectionError("Failed to connect to Ollama"), CONNECTION),
        (TimeoutError("slow"), TIMEOUT),
        # Ollama's answers
        (ollama.ResponseError("model requires more system memory", 500), UNAVAILABLE),
        (ollama.ResponseError("server busy", 503), UNAVAILABLE),
        (ollama.ResponseError("llama runner process has terminated"), UNAVAILABLE),  # mid-stream
        (ollama.ResponseError("model 'x' not found", 404), ANSWERED),
        # openai's, and through their names langchain-openai's subclasses
        (openai_error("APIConnectionError")("Connection error."), CONNECTION),
        (openai_error("APITimeoutError", "APIConnectionError")("Request timed out."), TIMEOUT),
        (
            openai_error("RateLimitError", "APIStatusError", status_code=429)("slow down"),
            UNAVAILABLE,
        ),
        (openai_error("InternalServerError", status_code=503)("down"), UNAVAILABLE),
        (openai_error("AuthenticationError", status_code=401)("Invalid API Key"), ANSWERED),
        # nothing to say about a service
        (ValueError("bad input"), None),
        (BreakerOpenError("Qdrant", 3.0, 3), None),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, BaseException) else str(value),
)
def test_classify(error: BaseException, verdict: str | None) -> None:
    assert classify(error) == verdict


def test_a_class_named_as_httpx_s_elsewhere_is_not_httpx_s() -> None:
    impostor = type("ConnectError", (Exception,), {"__module__": "somewhere"})
    assert classify(impostor("refused")) is None


def test_a_subclass_inherits_its_parents_verdict() -> None:
    # langchain-openai re-raises openai's errors as its own subclasses.
    base = openai_error("APITimeoutError", "APIConnectionError")
    child = type("OpenAITimeoutError", (base,), {"__module__": "langchain_openai.chat_models"})
    assert classify(child("Request timed out.")) == TIMEOUT


# --- the back-off ---------------------------------------------------------------------------


def test_the_back_off_is_full_jitter_under_a_cap() -> None:
    real = real_backoff_s  # the conftest makes the module's instant; this is the real one
    for attempt, cap in [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (9, 4.0)]:
        waits = [real(attempt) for _ in range(300)]
        assert all(0 <= wait <= cap for wait in waits)
        assert max(waits) > cap / 2  # spread over the range, not pinned low


# --- the deadline ---------------------------------------------------------------------------


def test_no_deadline_outside_one() -> None:
    assert resilience.remaining() is None


def test_a_deadline_counts_down_and_ends_with_its_block() -> None:
    with resilience.deadline(5):
        left = resilience.remaining()
        assert left is not None
        assert 4.5 < left <= 5
    assert resilience.remaining() is None


def test_a_nested_deadline_keeps_the_sooner_end() -> None:
    with resilience.deadline(2):
        with resilience.deadline(60):
            left = resilience.remaining()
            assert left is not None
            assert left <= 2
        with resilience.deadline(0.5):
            left = resilience.remaining()
            assert left is not None
            assert left <= 0.5


def test_a_deadline_belongs_to_its_thread() -> None:
    seen: list[float | None] = []
    with resilience.deadline(5):
        thread = threading.Thread(target=lambda: seen.append(resilience.remaining()))
        thread.start()
        thread.join()
    assert seen == [None]


def test_retrying_stops_at_its_attempts_and_raises_the_last_error() -> None:
    calls: list[int] = []

    def refused() -> None:
        calls.append(1)
        raise ConnectionRefusedError(f"attempt {len(calls)}")

    with pytest.raises(ConnectionRefusedError, match="attempt 3"):
        resilience.retrying(3, lambda e: isinstance(e, ConnectionError))(refused)
    assert len(calls) == 3


def test_retrying_repeats_only_what_it_is_told_to() -> None:
    calls: list[int] = []

    def wrong() -> None:
        calls.append(1)
        raise ValueError("not transient")

    with pytest.raises(ValueError, match="not transient"):
        resilience.retrying(3, lambda e: isinstance(e, ConnectionError))(wrong)
    assert len(calls) == 1


def test_retrying_waits_no_further_than_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resilience, "backoff_s", lambda attempt: 0.3)
    calls: list[float] = []

    def refused() -> None:
        calls.append(time.monotonic())
        raise ConnectionRefusedError("refused")

    started = time.monotonic()
    with resilience.deadline(0.5), pytest.raises(ConnectionRefusedError):
        resilience.retrying(5, lambda e: isinstance(e, ConnectionError))(refused)
    # The first wait fits in 0.5 s, the second would end past it: two attempts, not five.
    assert len(calls) == 2
    assert time.monotonic() - started < 0.5
