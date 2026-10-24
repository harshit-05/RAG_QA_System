"""The numbers the SLOs are written in, for Prometheus to scrape (NFR-10, DEC-22, S3-7).

prometheus-client, not the OpenTelemetry metrics SDK: Prometheus scraping is what the SRS
names, and the SDK's Prometheus exporter is still a beta (DEC-22). The metrics live in a
registry of their own, :data:`REGISTRY`, with the process's own (memory, CPU, open files),
and ``rag-serve`` serves it on a listener of its own, ``RAG_API_METRICS_PORT``, off unless set
(:func:`serve`).

**Who records what.** The two retrieval histograms are recorded where retrieval runs
(``chain``), so tier 1 and the benchmark record them too, into a registry nobody scrapes. The
rest are the API's: the query route, its refusals, its ingest jobs, and the breakers and the
index read from what it serves at scrape time (:func:`watch`). A ``rag-ingest`` run has no
listener: it reports in its log (DEC-22).

**The SLOs** (S3-7b's alert rules). Each latency histogram has a bucket at its objective:
``rag_retrieval_seconds`` at 0.3 s (NFR-1), ``rag_sources_seconds`` at 2 s (NFR-2a),
``rag_ttft_seconds`` at 300 s (NFR-2b, this CPU's class). Availability counts the 503s
(:data:`REFUSALS`) and the answers that failed after their 200 (:data:`ANSWERS`).

**The GenAI convention's metric**, ``gen_ai_client_token_usage``: its name, labels and values
from opentelemetry-semantic-conventions 0.66b1, never typed (:mod:`.genai` says why only its
members are read); its buckets and required labels as the convention gives them (semantic
conventions 1.37.0, ``gen-ai-metrics``).

Loads no model stack (``tests/test_architecture.py``).
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Final

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    ProcessCollector,
    disable_created_metrics,
    start_http_server,
)
from prometheus_client.metrics_core import GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from rag_qa.observability.genai import CHAT, ModelIdentity
from rag_qa.resilience import CLOSED, HALF_OPEN, OPEN

# A `_created` series beside every counter and histogram, which nothing here reads.
disable_created_metrics()


def prometheus_name(name: str) -> str:
    """A semantic convention's name as Prometheus spells it: dots become underscores."""
    return name.replace(".", "_")


#: Every metric below, and the process's own.
REGISTRY: Final = CollectorRegistry()
ProcessCollector(registry=REGISTRY)

# --- latency ---------------------------------------------------------------------------------

RETRIEVAL_SECONDS: Final = Histogram(
    "rag_retrieval_seconds",
    "One retrieval's query embedding and vector search, the rerank left out: what NFR-1 "
    "bounds (p95 under 0.3 s).",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
    registry=REGISTRY,
)
RERANK_SECONDS: Final = Histogram(
    "rag_rerank_seconds",
    "One retrieval's rerank: the cross-encoder over the retriever's candidates.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)
SOURCES_SECONDS: Final = Histogram(
    "rag_sources_seconds",
    "From a question to its sources event, retrieval and the rerank: what NFR-2a bounds "
    "(p95 under 2 s, with the model idle).",
    buckets=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)
TTFT_SECONDS: Final = Histogram(
    "rag_ttft_seconds",
    "From a question to its answer's first token, answers cancelled later included: NFR-2b, "
    "reported per class of hardware.",
    buckets=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 450.0, 600.0),
    registry=REGISTRY,
)
ANSWER_SECONDS: Final = Histogram(
    "rag_answer_seconds",
    "From a question to the end of its complete answer.",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 450.0, 600.0, 900.0),
    registry=REGISTRY,
)

# --- tokens: the GenAI convention's metric ---------------------------------------------------

#: The labels the convention requires, and the request model, which it requires when known.
_TOKEN_LABELS: Final = tuple(
    prometheus_name(attribute)
    for attribute in (
        gen_ai.GEN_AI_OPERATION_NAME,
        gen_ai.GEN_AI_PROVIDER_NAME,
        gen_ai.GEN_AI_REQUEST_MODEL,
        gen_ai.GEN_AI_TOKEN_TYPE,
    )
)
TOKEN_USAGE: Final = Histogram(
    prometheus_name(gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE),
    "Number of input and output tokens used.",
    labelnames=_TOKEN_LABELS,
    buckets=(
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ),
    registry=REGISTRY,
)
INPUT: Final = gen_ai.GenAiTokenTypeValues.INPUT.value
OUTPUT: Final = gen_ai.GenAiTokenTypeValues.OUTPUT.value

# --- answers and refusals --------------------------------------------------------------------

#: How an answer whose stream began ended: complete, an ``error`` event after the 200, or its
#: client gone.
ANSWERED: Final = "answered"
FAILED: Final = "failed"
CANCELLED: Final = "cancelled"
ANSWERS: Final = Counter(
    "rag_answers",
    "Answers whose stream began, by how they ended: answered, failed (an error event after "
    "the 200), or cancelled (the client left).",
    labelnames=("outcome",),
    registry=REGISTRY,
)
FALLBACK_ANSWERS: Final = Counter(
    "rag_fallback_answers",
    "Answers the hosted fallback gave: their question and retrieved chunks left this host "
    "(DEC-21).",
    registry=REGISTRY,
)

#: Why a question was refused before its first byte (DEC-18): no free slot, a breaker open on
#: a dependency it needs (DEC-21), or no index the config can use.
BUSY: Final = "busy"
DEPENDENCY_DOWN: Final = "dependency_down"
NO_INDEX: Final = "no_index"
REFUSALS: Final = Counter(
    "rag_refusals",
    "Questions refused before their first byte, by status code and reason: 503 busy, "
    "dependency_down or no_index. S3-8 adds the 429s.",
    labelnames=("code", "reason"),
    registry=REGISTRY,
)

# --- ingest jobs -----------------------------------------------------------------------------

INGEST_RUNS: Final = Counter(
    "rag_ingest_runs",
    "Ingest jobs the API ran, by how they ended: succeeded or failed.",
    labelnames=("status",),
    registry=REGISTRY,
)
INGEST_CHUNKS: Final = Counter(
    "rag_ingest_chunks",
    "Chunks the API's ingest jobs embedded.",
    registry=REGISTRY,
)

# Every label set starts at zero, so a rate over a quiet window is 0, never absent.
for _outcome in (ANSWERED, FAILED, CANCELLED):
    ANSWERS.labels(outcome=_outcome)
for _reason in (BUSY, DEPENDENCY_DOWN, NO_INDEX):
    REFUSALS.labels(code="503", reason=_reason)
for _status in ("succeeded", "failed"):
    INGEST_RUNS.labels(status=_status)


def observe_tokens(
    identity: ModelIdentity, input_tokens: int | None, output_tokens: int | None
) -> None:
    """One answer's token counts, those its provider reported, under the model that answered."""
    for kind, count in ((INPUT, input_tokens), (OUTPUT, output_tokens)):
        if count is not None:
            TOKEN_USAGE.labels(CHAT, identity.provider, identity.model, kind).observe(count)


# --- what the API serves, read at scrape time ------------------------------------------------

#: A breaker's states, as :mod:`rag_qa.resilience` names them.
BREAKER_STATES: Final = (CLOSED, OPEN, HALF_OPEN)


class _Watched:
    """What :func:`watch` was given: the breakers' states by role, and the documents served."""

    def __init__(
        self, breakers: Callable[[], Mapping[str, str]], documents: Callable[[], int]
    ) -> None:
        self.breakers = breakers
        self.documents = documents


_watched: _Watched | None = None


def watch(breakers: Callable[[], Mapping[str, str]], documents: Callable[[], int]) -> None:
    """Report, at every scrape, each breaker's state (``breakers``: the state by role,
    ``qdrant``, ``llm`` and ``fallback`` as ``/v1/health`` names them) and the number of
    documents in the index served (``documents``, 0 when none). The API's lifespan calls it
    once its pipeline is built, and :func:`unwatch` at shutdown."""
    global _watched
    _watched = _Watched(breakers, documents)


def unwatch() -> None:
    """Stop reporting what :func:`watch` was given."""
    global _watched
    _watched = None


def _breaker_family() -> GaugeMetricFamily:
    return GaugeMetricFamily(
        "rag_breaker_state",
        "Each dependency's circuit breaker (DEC-21): 1 for its current state, 0 for the others.",
        labels=("breaker", "state"),
    )


def _documents_family() -> GaugeMetricFamily:
    return GaugeMetricFamily(
        "rag_index_documents",
        "Documents in the index generation served; 0 while none is.",
        labels=(),
    )


class _ServingCollector(Collector):
    """The breakers and the documents, read from what the API serves when Prometheus asks."""

    def describe(self) -> Iterable[Metric]:
        return [_breaker_family(), _documents_family()]

    def collect(self) -> Iterator[Metric]:
        watched = _watched
        breakers, documents = _breaker_family(), _documents_family()
        if watched is not None:
            for role, state in watched.breakers().items():
                for each in BREAKER_STATES:
                    breakers.add_metric([role, each], 1.0 if state == each else 0.0)
            documents.add_metric([], float(watched.documents()))
        yield breakers
        yield documents


REGISTRY.register(_ServingCollector())


def serve(host: str, port: int) -> Callable[[], None]:
    """Serve :data:`REGISTRY` at ``http://host:port/metrics``, from a thread of its own, and
    return what stops it. Raises ``OSError`` when the address cannot be bound."""
    server, thread = start_http_server(port, addr=host, registry=REGISTRY)

    def stop() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return stop
