"""The metrics (NFR-10, DEC-22, S3-7): ``rag_qa.observability.metrics``, and the API moving
them as answers, refusals and ingests happen.

The registry is the process's, so every test reads a metric before and after, and checks the
difference.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import (
    UNLIMITED,
    MakeConfig,
    StreamingFakeChatModel,
    StubCrossEncoder,
    fake_serving,
    sse_events,
    use_fakes,
    use_fakes_and_reranker,
)
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from prometheus_client import generate_latest

from rag_qa.api import state as api_state
from rag_qa.api.app import create_app
from rag_qa.api.state import Served, Serving
from rag_qa.chain import RetrievalQuery, build_retrieve
from rag_qa.components import build_embedder, build_reranker
from rag_qa.config import load_config
from rag_qa.ingest import _embed_with_progress
from rag_qa.observability import logs, metrics
from rag_qa.observability.genai import ModelIdentity, model_identity
from rag_qa.resilience import BreakerOpenError, CircuitBreaker
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ServerSettings
from rag_qa.vectorstore import StoreUnavailableError

USAGE = {"input_tokens": 900, "output_tokens": 120, "total_tokens": 1020}


def value(name: str, **labels: str) -> float:
    """A sample's value now, 0 when it has none yet."""
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


class Counts:
    """Samples read at the start, for the differences a test checks."""

    def __init__(self, *names: tuple[str, dict[str, str]]) -> None:
        self.before = {
            (name, tuple(labels.items())): value(name, **labels) for name, labels in names
        }

    def moved(self, name: str, **labels: str) -> float:
        return value(name, **labels) - self.before[(name, tuple(labels.items()))]


ANSWER_SAMPLES = (
    ("rag_answers_total", {"outcome": "answered"}),
    ("rag_answers_total", {"outcome": "failed"}),
    ("rag_answers_total", {"outcome": "cancelled"}),
    ("rag_fallback_answers_total", {}),
    ("rag_sources_seconds_count", {}),
    ("rag_ttft_seconds_count", {}),
    ("rag_answer_seconds_count", {}),
    ("rag_abandoned_seconds_count", {"waiting_for": "sources"}),
    ("rag_abandoned_seconds_count", {"waiting_for": "first_token"}),
)


def tokens(provider: str, model: str, kind: str) -> tuple[str, dict[str, str]]:
    return (
        "gen_ai_client_token_usage_sum",
        {
            "gen_ai_operation_name": "chat",
            "gen_ai_provider_name": provider,
            "gen_ai_request_model": model,
            "gen_ai_token_type": kind,
        },
    )


@contextmanager
def served(config: RagConfig, serving: Serving) -> Iterator[TestClient]:
    app = create_app(config, ServerSettings(rate_limit=UNLIMITED), serving=serving)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


# --- the module ------------------------------------------------------------------------------


def test_the_token_metric_is_named_as_the_genai_convention_names_it() -> None:
    # Taken from the package's own constants, never typed (S3-7's plan).
    assert metrics.TOKEN_USAGE._name == gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE.replace(".", "_")
    assert list(metrics.TOKEN_USAGE._labelnames) == [
        name.replace(".", "_")
        for name in (
            gen_ai.GEN_AI_OPERATION_NAME,
            gen_ai.GEN_AI_PROVIDER_NAME,
            gen_ai.GEN_AI_REQUEST_MODEL,
            gen_ai.GEN_AI_TOKEN_TYPE,
        )
    ]
    # The convention's buckets: powers of 4 from 1 to 4^13.
    assert list(metrics.TOKEN_USAGE._upper_bounds)[:-1] == [4**n for n in range(14)]


@pytest.mark.parametrize(
    ("histogram", "objective"),
    [
        (metrics.RETRIEVAL_SECONDS, 0.3),  # NFR-1
        (metrics.SOURCES_SECONDS, 2.0),  # NFR-2a
        (metrics.TTFT_SECONDS, 300.0),  # NFR-2b, this CPU's class
        # The abandoned answers count in both SLIs, each at its own objective (S3-7b's fix).
        (metrics.ABANDONED_SECONDS, 2.0),
        (metrics.ABANDONED_SECONDS, 300.0),
    ],
)
def test_each_slo_histogram_has_a_bucket_at_its_objective(histogram: Any, objective: float) -> None:
    # S3-7b's burn-rate rules count the answers under it: le="<objective>".
    assert objective in histogram._upper_bounds


def test_a_slot_counts_an_answer_abandoned_before_its_first_token_once_with_its_wait() -> None:
    # S3-7b's fix: no first token to observe, and availability leaves cancels out, so this is
    # the only trace a hung model's abandoned answers leave in the SLIs.
    def abandoned(waiting_for: str, kind: str = "count") -> float:
        return value(f"rag_abandoned_seconds_{kind}", waiting_for=waiting_for)

    before = {w: (abandoned(w), abandoned(w, "sum")) for w in ("sources", "first_token")}
    pool = api_state.SlotPool(4)

    during_retrieval = pool.take()
    assert during_retrieval is not None
    during_retrieval.begin(time.perf_counter() - 30.0)  # asked 30 s ago
    during_retrieval.release()
    during_retrieval.release()  # once, however many paths release it
    assert abandoned("sources") - before["sources"][0] == 1
    assert 30.0 <= abandoned("sources", "sum") - before["sources"][1] < 31.0

    during_prefill = pool.take()
    assert during_prefill is not None
    during_prefill.begin(time.perf_counter() - 400.0)
    during_prefill.waiting_for = metrics.WAITING_FOR_FIRST_TOKEN  # the route, at sources
    during_prefill.release()
    assert abandoned("first_token") - before["first_token"][0] == 1
    assert 400.0 <= abandoned("first_token", "sum") - before["first_token"][1] < 401.0

    # After a first token, or ended otherwise, or before its stream began: not abandoned.
    for waiting_for, outcome in (
        (None, metrics.CANCELLED),
        (metrics.WAITING_FOR_FIRST_TOKEN, metrics.FAILED),
        (metrics.WAITING_FOR_FIRST_TOKEN, metrics.ANSWERED),
    ):
        slot = pool.take()
        assert slot is not None
        slot.begin(time.perf_counter())
        slot.waiting_for, slot.outcome = waiting_for, outcome
        slot.release()
    refused = pool.take()  # its slot taken, then refused before its answer began
    assert refused is not None
    refused.release()
    assert abandoned("sources") - before["sources"][0] == 1
    assert abandoned("first_token") - before["first_token"][0] == 1
    assert pool.in_use == 0


def test_every_label_set_is_there_at_zero_and_no_created_series() -> None:
    text = generate_latest(metrics.REGISTRY).decode()
    for line in (
        'rag_answers_total{outcome="cancelled"}',
        'rag_refusals_total{code="503",reason="busy"}',
        'rag_refusals_total{code="503",reason="dependency_down"}',
        'rag_refusals_total{code="503",reason="no_index"}',
        'rag_refusals_total{code="413",reason="too_large"}',
        'rag_refusals_total{code="500",reason="error"}',
        'rag_refusals_total{code="401",reason="unauthenticated"}',
        'rag_refusals_total{code="403",reason="scope"}',
        'rag_refusals_total{code="429",reason="rate_limited"}',
        'rag_refusals_total{code="429",reason="auth_limited"}',
        'rag_refusals_total{code="503",reason="keys_unreadable"}',
        'rag_ingest_runs_total{status="failed"}',
        'rag_abandoned_seconds_count{waiting_for="sources"}',
        'rag_abandoned_seconds_count{waiting_for="first_token"}',
    ):
        assert line in text
    assert "_created" not in text


def test_the_breakers_and_the_documents_are_read_at_each_scrape() -> None:
    states = {"qdrant": "closed", "llm": "open"}
    metrics.watch(lambda: states, lambda: 3)
    assert value("rag_breaker_state", breaker="llm", state="open") == 1.0
    assert value("rag_breaker_state", breaker="llm", state="closed") == 0.0
    assert value("rag_index_documents") == 3.0
    states["llm"] = "half-open"  # read again at the next scrape
    assert value("rag_breaker_state", breaker="llm", state="half-open") == 1.0
    metrics.unwatch()
    assert (
        metrics.REGISTRY.get_sample_value("rag_breaker_state", {"breaker": "llm", "state": "open"})
        is None
    )


@pytest.mark.parametrize(
    ("module", "attributes", "identity"),
    [
        ("langchain_ollama.chat_models", {"model": "mistral"}, ModelIdentity("ollama", "mistral")),
        (
            "langchain_openai.chat_models.base",
            {"model_name": "openai/gpt-oss-120b"},
            ModelIdentity("openai", "openai/gpt-oss-120b"),
        ),
        ("tests.fakes", {}, ModelIdentity("tests", "Fake")),
    ],
)
def test_a_models_identity_comes_from_its_class_and_its_model(
    module: str, attributes: dict[str, str], identity: ModelIdentity
) -> None:
    fake = type("Fake", (), {"__module__": module, **attributes})()
    assert model_identity(fake) == identity


# --- the API ---------------------------------------------------------------------------------


def test_an_answer_moves_its_times_its_tokens_and_its_outcome(
    fake_config: RagConfig, caplog: pytest.LogCaptureFixture
) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel(usage=USAGE))
    # The fake's identity: its package, and its class for want of a model name.
    sent_in, sent_out = (
        tokens("conftest", "StreamingFakeChatModel", k) for k in ("input", "output")
    )
    counts = Counts(*ANSWER_SAMPLES, sent_in, sent_out)
    caplog.set_level(logging.INFO, logger="rag_qa")
    with served(fake_config, serving) as client:
        response = client.post("/v1/query", json={"question": "How many staff?"})
    done = sse_events(response.text)[-1]
    assert done == (
        "done",
        {
            "ttft_ms": done[1]["ttft_ms"],
            "total_ms": done[1]["total_ms"],
            "model": "llm",
            "input_tokens": 900,
            "output_tokens": 120,
        },
    )
    assert counts.moved("rag_answers_total", outcome="answered") == 1
    for name in ("rag_sources_seconds_count", "rag_ttft_seconds_count", "rag_answer_seconds_count"):
        assert counts.moved(name) == 1
    for waiting_for in ("sources", "first_token"):
        assert counts.moved("rag_abandoned_seconds_count", waiting_for=waiting_for) == 0
    assert counts.moved(sent_in[0], **sent_in[1]) == 900
    assert counts.moved(sent_out[0], **sent_out[1]) == 120
    assert counts.moved("rag_fallback_answers_total") == 0
    (line,) = [r for r in caplog.records if r.getMessage().startswith("Answered by")]
    assert line.getMessage() == (
        f"Answered by llm in {done[1]['total_ms'] / 1000:.1f} s (first token at "
        f"{done[1]['ttft_ms'] / 1000:.1f} s), 900 input and 120 output tokens."
    )
    assert (line.input_tokens, line.output_tokens, line.model) == (900, 120, "llm")  # type: ignore[attr-defined]
    assert "How many staff?" not in caplog.text  # never the question


def test_a_fallbacks_answer_counts_as_one_that_left_the_host(
    fake_config: RagConfig, caplog: pytest.LogCaptureFixture
) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    breaker = CircuitBreaker("Ollama")
    for _ in range(breaker.failures):
        breaker.allow().failed()
    spare = StreamingFakeChatModel(usage=USAGE)
    serving.pipeline = replace(serving.pipeline, fallback_llm=spare, llm_breaker=breaker)
    counts = Counts(*ANSWER_SAMPLES)
    with served(fake_config, serving) as client:
        response = client.post("/v1/query", json={"question": "q"})
    assert sse_events(response.text)[-1][1]["model"] == "fallback"
    assert counts.moved("rag_fallback_answers_total") == 1
    assert counts.moved("rag_answers_total", outcome="answered") == 1
    assert "Answered by the hosted fallback fallback: the question and its retrieved" in caplog.text


class TimesOut(StreamingFakeChatModel):
    """A model that times out before its first token: the answer's error (DEC-21)."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:
        raise httpx.ReadTimeout("no answer in time")
        yield  # an async generator, like every model's stream


def test_an_answer_that_fails_after_its_200_counts_as_failed(fake_config: RagConfig) -> None:
    serving = fake_serving(fake_config, TimesOut())
    counts = Counts(*ANSWER_SAMPLES)
    with served(fake_config, serving) as client:
        events = sse_events(client.post("/v1/query", json={"question": "q"}).text)
    assert [name for name, _ in events] == ["sources", "error"]
    assert counts.moved("rag_answers_total", outcome="failed") == 1
    assert counts.moved("rag_answers_total", outcome="answered") == 0
    assert counts.moved("rag_answer_seconds_count") == 0
    # Failed before its first token, but not abandoned: its client was still there.
    assert counts.moved("rag_abandoned_seconds_count", waiting_for="first_token") == 0


def test_an_open_breaker_after_the_stream_began_is_one_warning_line_not_a_traceback(
    fake_config: RagConfig, caplog: pytest.LogCaptureFixture
) -> None:
    # S3-6's backlog line: the breaker opened between the refusal before the first byte and the
    # model's turn (here, during the retrieval), so the stream had begun.
    breaker = CircuitBreaker("Ollama")
    docs = [Document(page_content="Station Kestrel.", metadata={"source": "notes.txt"})]

    def retrieve(asked: RetrievalQuery) -> list[Document]:
        for _ in range(breaker.failures):
            breaker.allow().failed()
        return docs

    serving = fake_serving(fake_config, StreamingFakeChatModel(), retrieve=RunnableLambda(retrieve))
    serving.pipeline = replace(serving.pipeline, llm_breaker=breaker)
    with served(fake_config, serving) as client:
        events = sse_events(client.post("/v1/query", json={"question": "q"}).text)
    assert events[-1][0] == "error"
    (line,) = [r for r in caplog.records if "after its stream began" in r.getMessage()]
    assert line.levelno == logging.WARNING
    assert line.exc_info is None  # one line: no traceback
    assert line.getMessage().startswith(
        "An answer was refused after its stream began: Ollama is not answering, and is asked "
        "again in "
    )
    assert (line.service, line.retry_after_s) == ("Ollama", 30)  # type: ignore[attr-defined]


def test_a_store_refused_by_its_breaker_mid_answer_is_one_warning_line_too(
    fake_config: RagConfig, caplog: pytest.LogCaptureFixture
) -> None:
    def retrieve(asked: RetrievalQuery) -> list[Document]:
        raise StoreUnavailableError("Qdrant is not answering") from BreakerOpenError(
            "Qdrant", 12.0, 3
        )

    serving = fake_serving(fake_config, StreamingFakeChatModel(), retrieve=RunnableLambda(retrieve))
    with served(fake_config, serving) as client:
        client.post("/v1/query", json={"question": "q"})
    (line,) = [r for r in caplog.records if "after its stream began" in r.getMessage()]
    assert (line.levelno, line.exc_info, line.service) == (logging.WARNING, None, "Qdrant")  # type: ignore[attr-defined]


def test_each_refusal_counts_with_its_reason(fake_config: RagConfig) -> None:
    reasons = ("busy", "dependency_down", "no_index")
    counts = Counts(*(("rag_refusals_total", {"code": "503", "reason": r}) for r in reasons))

    busy = fake_serving(fake_config, StreamingFakeChatModel())
    busy.slots.in_use = busy.slots.limit
    down = fake_serving(fake_config, StreamingFakeChatModel())
    for _ in range(down.store.breaker.failures):
        down.store.breaker.allow().failed()
    empty = fake_serving(fake_config, StreamingFakeChatModel())
    empty.served = Served(generation=None, manifest=None, problem="no index yet")
    empty.refresh = lambda: _done()  # type: ignore[method-assign]  # keep the problem

    for serving in (busy, down, empty):
        with served(fake_config, serving) as client:
            assert client.post("/v1/query", json={"question": "q"}).status_code == 503
    for reason in reasons:
        assert counts.moved("rag_refusals_total", code="503", reason=reason) == 1


async def _done() -> None:
    return None


def test_a_question_whose_body_is_too_large_counts_as_a_413_and_no_other_route_does(
    fake_config: RagConfig,
) -> None:
    counts = Counts(("rag_refusals_total", {"code": "413", "reason": "too_large"}))
    big = b"x" * (64 * 1024 + 1)
    as_json = {"content-type": "application/json"}
    with served(fake_config, fake_serving(fake_config, StreamingFakeChatModel())) as client:
        assert client.post("/v1/query", content=big, headers=as_json).status_code == 413
        assert client.post("/v1/ingest", content=big, headers=as_json).status_code == 413
    assert counts.moved("rag_refusals_total", code="413", reason="too_large") == 1


def test_an_unexpected_error_before_the_answer_counts_as_a_500_once(
    fake_config: RagConfig,
) -> None:
    # No refusal of ours makes a 500, so nothing else counts it (S3-7b's second review): the
    # availability SLI would not see it.
    counts = Counts(
        ("rag_refusals_total", {"code": "500", "reason": "error"}),
        *(("rag_refusals_total", {"code": "503", "reason": r}) for r in ("busy", "no_index")),
    )
    serving = fake_serving(fake_config, StreamingFakeChatModel())

    async def broken() -> None:
        raise RuntimeError("a bug")

    serving.refresh = broken  # type: ignore[method-assign]
    app = create_app(fake_config, ServerSettings(rate_limit=UNLIMITED), serving=serving)
    with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 500
        assert client.get("/v1/documents").status_code == 500  # not a question
    assert counts.moved("rag_refusals_total", code="500", reason="error") == 1
    assert counts.moved("rag_refusals_total", code="503", reason="busy") == 0
    assert counts.moved("rag_refusals_total", code="503", reason="no_index") == 0


def test_an_answer_that_fails_after_its_200_is_no_500(fake_config: RagConfig) -> None:
    # It counts as a failed answer (rag_answers_total), once: never as a 500 besides.
    counts = Counts(("rag_refusals_total", {"code": "500", "reason": "error"}))

    def retrieve(asked: RetrievalQuery) -> list[Document]:
        raise RuntimeError("retrieval broke")

    serving = fake_serving(fake_config, StreamingFakeChatModel(), retrieve=RunnableLambda(retrieve))
    with served(fake_config, serving) as client:
        response = client.post("/v1/query", json={"question": "q"})
    assert response.status_code == 200 and "event: error" in response.text
    assert counts.moved("rag_refusals_total", code="500", reason="error") == 0


def answer_outcomes() -> dict[str, float]:
    """``rag_answers_total`` by outcome: every series the registry holds, not only the three
    started at zero."""
    return {
        sample.labels["outcome"]: sample.value
        for family in metrics.REGISTRY.collect()
        if family.name == "rag_answers"
        for sample in family.samples
        if sample.name == "rag_answers_total"
    }


def test_a_question_refused_after_taking_its_slot_counts_no_answer(
    fake_config: RagConfig,
) -> None:
    # No index (503) and a malformed body (422) are refused by dependencies that run after
    # take_slot: the slot is released with no answer begun, and no outcome may count, nor a
    # series for none (S3-7's second review: the guard survived its removal).
    empty = fake_serving(fake_config, StreamingFakeChatModel())
    empty.served = Served(generation=None, manifest=None, problem="no index yet")
    empty.refresh = lambda: _done()  # type: ignore[method-assign]  # keep the problem
    before = answer_outcomes()
    with served(fake_config, empty) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 503
    with served(fake_config, fake_serving(fake_config, StreamingFakeChatModel())) as client:
        assert client.post("/v1/query", json={}).status_code == 422
    assert answer_outcomes() == before
    assert set(before) == {"answered", "failed", "cancelled"}


#: How long the alias read takes in the next test, as when Qdrant answers slowly or is retried.
SLOW_REFRESH_S = 0.3


def test_the_answer_times_start_when_the_request_arrives_not_when_the_route_does(
    fake_config: RagConfig,
) -> None:
    # FastAPI starts the route's generator only after its dependencies, served_pipeline's
    # alias read among them, and the client waits for that read too. Timed from the route,
    # a slow read was invisible: 1.5 s waited, 0.006 s recorded (S3-7's second review).
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    refresh = serving.refresh

    async def slow_refresh() -> None:
        await asyncio.sleep(SLOW_REFRESH_S)
        await refresh()

    serving.refresh = slow_refresh  # type: ignore[method-assign]
    sums = ("rag_sources_seconds_sum", "rag_ttft_seconds_sum", "rag_answer_seconds_sum")
    counts = Counts(*((name, {}) for name in sums))
    with served(fake_config, serving) as client:
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
    for name in sums:
        assert counts.moved(name) >= SLOW_REFRESH_S * 0.9, name


def test_an_ingest_job_moves_the_runs_and_the_chunks(
    make_config: MakeConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    config = load_config(make_config(use_fakes))
    counts = Counts(
        ("rag_ingest_runs_total", {"status": "succeeded"}),
        ("rag_ingest_chunks_total", {}),
        ("rag_index_documents", {}),
    )
    with TestClient(
        create_app(config, ServerSettings(rate_limit=UNLIMITED)), base_url="http://127.0.0.1"
    ) as client:
        started = client.post("/v1/ingest", json={"rebuild": False}).json()
        job = _wait(client, started["job_id"])
        # The job ends, then the server reopens what it published: wait for that.
        documents = _until(lambda: value("rag_index_documents") or None) or 0.0
    assert job["status"] == "succeeded"
    assert counts.moved("rag_ingest_runs_total", status="succeeded") == 1
    assert counts.moved("rag_ingest_chunks_total") == job["report"]["chunks"] > 0
    assert documents == 5  # read from what the server serves, once the job published
    assert value("rag_index_documents") == 0  # the lifespan stopped reporting it


def _until(read: Callable[[], Any], seconds: float = 10) -> Any:
    """``read()`` once it returns something other than ``None``, else ``None`` after ``seconds``."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        found = read()
        if found is not None:
            return found
        time.sleep(0.02)
    return None


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    end = time.monotonic() + 30
    while time.monotonic() < end:
        job = client.get(f"/v1/ingest/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return dict(job)
        time.sleep(0.02)
    raise AssertionError("the ingest job did not end")


# --- review findings: what the first review's mutations left alive (S3-7) ----------------------


def test_retrieval_and_rerank_are_timed_apart(
    fake_rag: Path,
    make_config: MakeConfig,
    stub_cross_encoder: type[StubCrossEncoder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rag_retrieval_seconds is what NFR-1 bounds: the search alone. A slow reranker must land
    # in rag_rerank_seconds and nowhere else.
    def slow(self: StubCrossEncoder, inputs: list[tuple[str, str]], **kw: Any) -> Any:
        time.sleep(0.2)
        return [0.0] * len(inputs)

    monkeypatch.setattr(stub_cross_encoder, "predict", slow)
    config = load_config(make_config(use_fakes_and_reranker))
    retrieve = build_retrieve(config, build_embedder(config), reranker=build_reranker(config))
    counts = Counts(("rag_retrieval_seconds_sum", {}), ("rag_rerank_seconds_sum", {}))
    retrieve.invoke(RetrievalQuery("Staff?"))
    assert counts.moved("rag_rerank_seconds_sum") >= 0.2
    assert counts.moved("rag_retrieval_seconds_sum") < 0.15


QUESTION = "WHO-IS-THE-SECRET-EMPLOYEE-QUESTION"


@pytest.mark.parametrize(
    ("chosen", "detailed"),
    [("console", True), ("console", False), ("json", True), ("json", False)],
)
def test_a_failed_answers_log_never_holds_the_question_or_the_chunks(
    fake_config: RagConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    chosen: str,
    detailed: bool,
) -> None:
    # rich's traceback prints every frame's locals, the question and the prompt among them:
    # rag-serve's console lines (detailed) once did. The log goes wherever the operator sends it.
    monkeypatch.setenv("RAG_LOG_FORMAT", chosen)
    logs.configure_logging(detailed=detailed)
    serving = fake_serving(fake_config, TimesOut())
    with served(fake_config, serving) as client:
        events = sse_events(client.post("/v1/query", json={"question": QUESTION}).text)
    assert [name for name, _ in events] == ["sources", "error"]
    err = capsys.readouterr().err
    assert "An answer failed after its stream began" in err
    assert "ReadTimeout" in err  # the traceback is there...
    assert QUESTION not in err and "Station Kestrel" not in err  # ...without the locals


def test_the_answer_line_carries_the_requests_id_in_json(
    fake_config: RagConfig, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The first dependency must be async: FastAPI runs a plain one in a worker thread, and the
    # request_id it binds there never reaches the request's own task (ARCHITECTURE.md §3.1).
    monkeypatch.setenv("RAG_LOG_FORMAT", "json")
    logs.configure_logging()
    serving = fake_serving(fake_config, StreamingFakeChatModel(usage=USAGE))
    with served(fake_config, serving) as client:
        client.post("/v1/query", json={"question": "q"})
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    (answered,) = [x for x in lines if x["event"].startswith("Answered by")]
    assert len(answered["request_id"]) == 32


def test_an_ingest_jobs_log_lines_carry_its_job_id(
    make_config: MakeConfig,
    sample_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv("RAG_LOG_FORMAT", "json")
    logs.configure_logging()
    config = load_config(make_config(use_fakes))
    with TestClient(
        create_app(config, ServerSettings(rate_limit=UNLIMITED)), base_url="http://127.0.0.1"
    ) as client:
        started = client.post("/v1/ingest", json={"rebuild": False}).json()
        _wait(client, started["job_id"])
    lines = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    (published,) = [x for x in lines if x["event"].startswith("Published ")]
    assert published["job_id"] == started["job_id"]


@pytest.mark.parametrize(("chosen", "bar"), [("json", False), ("console", True)])
def test_the_embedding_progress_bar_is_off_in_json_logs(
    monkeypatch: pytest.MonkeyPatch, chosen: str, bar: bool
) -> None:
    class Embedder:
        show_progress = False

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    monkeypatch.setenv("RAG_LOG_FORMAT", chosen)
    logs.configure_logging()
    embedder = Embedder()
    _embed_with_progress(embedder, ["a"])  # type: ignore[arg-type]
    assert embedder.show_progress is bar
