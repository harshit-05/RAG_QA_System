"""What the API serves: the pipeline, the index generation behind it, and the slots.

Built once, at start-up (:meth:`Serving.build`), and shared by every request (DEC-18).

**The index can change under the server.** An ingest job, or a ``rag-ingest`` run outside
the server, publishes a new generation by switching the Qdrant alias (DEC-19). Before
reading what is served, a request calls :meth:`Serving.refresh`, which compares the
alias's target (one ``GET /aliases``) with the generation being served. On a change, it
reopens the index in a thread and swaps only the retrieval half (``dataclasses.replace``):
an answer in flight keeps the one it started with, bound to its generation, and the LLM
and its clients are never rebuilt.

- A reopen runs the same checks as start-up. An index from another embedder is never
  served: the server goes not-ready until an ingest fixes it.
- An open that fails for another reason (Qdrant unreachable, a failing embedder) is not a
  verdict on the index. The server goes not-ready with a fixed message, logs the
  exception, and tries again on the next request, rather than answer 500 (S2-7's first
  review). So does a refresh that cannot read the alias.
- The embedder and the reranker are built once and reused on every reopen. The reranker
  alone takes seconds and about 90 MB to load (S2-4).

**One retrieval at a time** (S2-4's second review). Retrieval runs in a worker thread that
a cancel cannot stop: a client that disconnects mid-rerank frees its slot at once, while
its rerank runs on for a second or two. Every retrieval half served is therefore wrapped
in :func:`serialized`. A lock is taken inside the worker thread, so the next request's
retrieval waits for the orphaned one instead of competing with it for the CPU, whatever
``RAG_API_MAX_CONCURRENT`` is.

**Slots** (DEC-18). A CPU host generates one answer at a time. A request that finds no
free slot gets 503 rather than a place in a queue. A :class:`Slot` is released by
whichever comes first, the answer stream's ``finally`` or the request's teardown, and
only once.

**Breakers** (DEC-21, S3-6). The store and each model have one, shared by every request.
While one that a question needs refuses, open or half-open with its probe out,
:meth:`Serving.unavailable` says so and the route answers 503 with ``Retry-After``, before
the first byte; with a hosted fallback whose own breaker is not refusing, one on the model
is not one of them. While Qdrant's is
open, :meth:`Serving.refresh` does not ask Qdrant at all: the back-off after a failed
reopen.
"""

import asyncio
import contextvars
import functools
import inspect
import logging
import math
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda

from rag_qa.api.jobs import IngestJobs
from rag_qa.chain import (
    IncompatibleIndexError,
    NoIndexError,
    QueryPipeline,
    Retrieval,
    RetrievalQuery,
    assemble,
    live_generation,
    open_retrieval,
)
from rag_qa.components import (
    build_embedder,
    build_fallback_llm,
    build_llm,
    build_reranker,
    build_store,
)
from rag_qa.manifest import Manifest
from rag_qa.observability import metrics
from rag_qa.resilience import OPEN, BreakerOpenError, CircuitBreaker
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import QdrantStore, StoreError

logger = logging.getLogger(__name__)

#: Why nothing is served. Fixed text, because it goes over HTTP; the exceptions behind
#: it name the index's location, so they go to the server log only (DEC-18).
NO_INDEX_YET = "no index yet: run an ingest (POST /v1/ingest)"
UNAVAILABLE = (
    "the index could not be opened just now; the server log says why, and the next "
    "request tries again"
)
REINGEST = (
    "the index cannot be used with this configuration: re-ingest with --rebuild "
    '(POST /v1/ingest with {"rebuild": true}); the server log says why'
)
#: Why a question is refused while a breaker it needs is open (DEC-21): fixed text that names
#: the service, never its address (DEC-18). ``Retry-After`` says when to ask again.
STORE_DOWN = "Qdrant is not answering; ask again after Retry-After"


def breaker_refusal(error: BaseException) -> BreakerOpenError | None:
    """The breaker refusal behind ``error``, or ``None``: the error itself, or the cause of a
    store error, since the store turns a refused request into one (DEC-21)."""
    if isinstance(error, BreakerOpenError):
        return error
    if isinstance(error, StoreError) and isinstance(error.__cause__, BreakerOpenError):
        return error.__cause__
    return None


def _refused(current: str | None | StoreError) -> bool:
    """Whether reading the alias was refused by Qdrant's breaker, open or half-open with a
    probe out, rather than failed against Qdrant: :meth:`Serving.refresh` learns nothing
    from such a refusal."""
    return isinstance(current, StoreError) and breaker_refusal(current) is not None


def wait_s(seconds: float) -> int:
    """A ``Retry-After`` in whole seconds, at least 1."""
    return max(1, math.ceil(seconds))


@dataclass(frozen=True)
class Served:
    """The index generation served and its manifest, or why there is none.

    ``generation`` is what :func:`rag_qa.chain.live_generation` said when the index was
    last opened, or tried, so a refresh reopens only when the alias has moved since.
    ``unseen`` marks a refresh that could not read the alias at all: the next one reopens,
    whatever the alias says then.
    """

    generation: str | None
    manifest: Manifest | None
    problem: str | None  # None while an index is served
    unseen: bool = False

    @property
    def ready(self) -> bool:
        return self.problem is None


#: What is served while Qdrant cannot be asked: nothing, and the next request asks again.
NOT_SEEN = Served(generation=None, manifest=None, problem=UNAVAILABLE, unseen=True)


def open_served(
    config: RagConfig,
    store: QdrantStore,
    embeddings: Embeddings,
    reranker: BaseDocumentCompressor | None,
) -> tuple[Runnable[RetrievalQuery, list[Document]] | None, Served]:
    """Open the live index as a retrieval half, or say why it cannot be served."""
    try:
        seen = live_generation(store)
        opened = _try_open(config, store, embeddings, reranker)
        if isinstance(opened, NoIndexError) and live_generation(store) != seen:
            # A switch landed while it opened. Only the live generation and the previous
            # one are kept, so the one it checked may be gone already: look once more
            # rather than report a missing index that is there (S2-6's third review).
            seen = live_generation(store)
            opened = _try_open(config, store, embeddings, reranker)
    except StoreError:
        logger.exception("Could not read the index's alias")
        return None, NOT_SEEN
    if isinstance(opened, NoIndexError):
        logger.warning("Not serving an index: %s", opened)  # the server log may name hosts
        problem = REINGEST if isinstance(opened, IncompatibleIndexError) else NO_INDEX_YET
        return None, Served(generation=seen, manifest=None, problem=problem)
    if isinstance(opened, Exception):
        # Not a verdict on the index (S2-7's first review): Qdrant gone, a failing embedder.
        # Nothing is recorded as seen, so the next request tries again.
        return None, NOT_SEEN
    logger.info(
        "Serving index %s (%d documents).",
        opened.generation,
        len(opened.manifest.documents),
    )
    return opened.retrieve, Served(opened.generation, opened.manifest, problem=None)


def _try_open(
    config: RagConfig,
    store: QdrantStore,
    embeddings: Embeddings,
    reranker: BaseDocumentCompressor | None,
) -> Retrieval | Exception:
    """The index opened, or why not: a :class:`NoIndexError` (``IncompatibleIndexError``
    too) when the index is missing or unusable, any other exception when opening it failed
    for a reason that says nothing about the index. That one is logged here, in full."""
    try:
        return open_retrieval(config, embeddings, reranker=reranker, store=store)
    except NoIndexError as e:  # IncompatibleIndexError too
        return e
    # Deliberately broad: a reopen runs inside a request (and inside health), and an error
    # escaping it would answer 500 to each of them, again on every request.
    except Exception as e:
        logger.exception("Could not open the index")  # the server log may name paths
        return e


def serialized(
    retrieve: Runnable[RetrievalQuery, list[Document]], lock: threading.Lock
) -> Runnable[RetrievalQuery, list[Document]]:
    """``retrieve``, run under ``lock``, taken inside the worker thread.

    The thread holds the lock until its retrieval ends, even after the request awaiting it
    was cancelled. The next retrieval therefore waits for an orphaned one rather than
    overlapping it. A waiting retrieval whose own request is cancelled before it gets the
    lock does no work at all.

    **The thread runs in a copy of the request's context** (S3-7b). ``run_in_executor``
    carries no context variables, unlike LangChain's own executor helper on ``rag-query``'s
    path. With the copy, the retrieval's spans are the request's server span's children,
    and its log lines carry the request's ``request_id``.
    """

    def run(asked: RetrievalQuery, abandoned: threading.Event | None = None) -> list[Document]:
        with lock:
            if abandoned is not None and abandoned.is_set():
                return []  # its client left while it waited: nobody will read the result
            return retrieve.invoke(asked)

    async def arun(asked: RetrievalQuery) -> list[Document]:
        abandoned = threading.Event()
        in_context = functools.partial(contextvars.copy_context().run, run, asked, abandoned)
        try:
            return await asyncio.get_running_loop().run_in_executor(None, in_context)
        except asyncio.CancelledError:
            abandoned.set()
            raise

    return RunnableLambda(run, afunc=arun)


class Slot:
    """One answer's claim on a slot, and the answer stream it holds.

    Released once, however many paths release it: the stream's ``finally``
    (:meth:`release`) and the request's teardown (:meth:`aclose`).

    **How the answer ended is counted at that one release** (S3-7): ``outcome`` is set by
    the route once its answer begins, and moved as it ends. A drop between events ends the
    route's own generator nowhere but in garbage collection, while the teardown always
    releases, so this is the one place every ending passes exactly once.
    """

    def __init__(self, pool: "SlotPool") -> None:
        self._pool = pool
        self._released = False
        self._answer: AsyncGenerator[Any, None] | None = None
        #: ``rag_answers_total``'s outcome, once the answer began: ``None`` before.
        self.outcome: str | None = None

    def hold(self, answer: AsyncGenerator[Any, None]) -> None:
        """The answer stream to close when the request ends (:meth:`aclose`)."""
        self._answer = answer

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._pool.in_use -= 1
            if self.outcome is not None:
                metrics.ANSWERS.labels(outcome=self.outcome).inc()

    async def aclose(self) -> None:
        """The request's teardown: close the answer stream if it is still open, then
        release the slot.

        **Why the close** (S2-7's spike). FastAPI iterates a streaming route in a producer
        task of its own, and every event it sends passes a checkpoint outside the route's
        generator. A disconnect that lands there cancels the producer and leaves the
        generator paused at a ``yield``. Nothing resumes it, and it sits in a reference
        cycle until garbage collection happens to run, with the stream to Ollama open. The
        spike forced that moment: 5 of 5 streams were still open 5 s later, and 5 of 5 were
        closed within 2 ms once this close was added. When the cancel lands inside the
        model's read instead, the stream has closed already and this does nothing.
        """
        answer, self._answer = self._answer, None
        state = None if answer is None else inspect.getasyncgenstate(answer)
        # Suspended: left at a yield, the case above. Created: never started, nothing to
        # unwind, closed for tidiness. Closed: ended already. Running cannot happen here
        # (the producer task has ended), and aclose() would refuse it.
        if answer is not None and state in (inspect.AGEN_SUSPENDED, inspect.AGEN_CREATED):
            if state == inspect.AGEN_SUSPENDED:
                logger.info("Closing an answer stream that its disconnected client left open")
            try:
                await answer.aclose()
            # Deliberately broad: the slot must be released whatever the close raises.
            except Exception:
                logger.exception("Could not close an answer stream")
        self.release()


class SlotPool:
    """``limit`` slots. Taking one never waits: no free slot means 503 (DEC-18)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_use = 0

    def take(self) -> Slot | None:
        """A slot, or ``None`` when all are taken. One synchronous step, so no other request
        can run between the check and the take."""
        if self.in_use >= self.limit:
            return None
        self.in_use += 1
        return Slot(self)


class Serving:
    """Everything the API answers from: built once, shared by every request."""

    def __init__(
        self,
        config: RagConfig,
        pipeline: QueryPipeline,
        served: Served,
        *,
        store: QdrantStore,
        embeddings: Embeddings,
        reranker: BaseDocumentCompressor | None,
        max_concurrent: int,
    ) -> None:
        self.config = config
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.slots = SlotPool(max_concurrent)
        self.jobs = IngestJobs(config, after=self.refresh)
        self.retrieval_lock = threading.Lock()
        self._reopen_lock = asyncio.Lock()
        self._reopens = 0  # attempts finished: a request that waited for one takes its result
        self.served = served
        self.pipeline = replace(pipeline, retrieve=self._serialized(pipeline.retrieve))

    @classmethod
    def build(cls, config: RagConfig, *, max_concurrent: int) -> "Serving":
        """Build the LLM half, load the embedder and the reranker, and open the index.

        Blocking, for seconds: run it in a thread. The LLM comes first, since it is the
        part whose failure stops the server (Ollama down); ``rag-serve`` has checked Qdrant
        before that. A missing or unusable index only makes the server not-ready. With a
        hosted fallback configured, it is built too, and the model without asking Ollama
        anything: Ollama down is then the fallback's case, not a failure (DEC-21).
        """
        spare = build_fallback_llm(config)  # needs its key; asks nothing of the provider
        # Without a fallback, validate_model_on_init asks Ollama for its models.
        llm = build_llm(config) if spare is None else build_llm(config, validate=False)
        store = build_store(config)
        embeddings = build_embedder(config)
        reranker = build_reranker(config)
        retrieve, served = open_served(config, store, embeddings, reranker)
        pipeline = assemble(config, retrieve, llm, spare)
        return cls(
            config,
            pipeline,
            served,
            store=store,
            embeddings=embeddings,
            reranker=reranker,
            max_concurrent=max_concurrent,
        )

    def _serialized(
        self, retrieve: Runnable[RetrievalQuery, list[Document]] | None
    ) -> Runnable[RetrievalQuery, list[Document]] | None:
        return None if retrieve is None else serialized(retrieve, self.retrieval_lock)

    async def _current(self) -> str | None | StoreError:
        """The alias's target now, read in a thread (a request with a timeout must not hold
        the event loop), or the error that stopped it being read."""
        try:
            return await asyncio.to_thread(live_generation, self.store)
        except StoreError as e:
            return e

    def _unchanged(self, current: str | None) -> bool:
        return not self.served.unseen and current == self.served.generation

    async def refresh(self) -> None:
        """Serve the generation the alias names now, if it is not the one served.

        One ``GET /aliases`` when nothing changed. A reopen runs in a thread, under the
        retrieval lock, so it never overlaps a retrieval that uses the same embedder and
        reranker. Concurrent requests reopen once, whether the open succeeds or fails: a
        failed one records no generation as seen, so without the attempt count each request
        queued behind it would try again in turn (S2-7's second review). An alias that
        cannot be read makes the server not-ready until one can.

        While Qdrant's breaker is open, it returns at once and asks nothing: the back-off
        after a failed reopen (DEC-21). The first refresh after the wait is the probe. A read
        the breaker refuses, half-open with another request's probe out, changes nothing
        either: the refusal says nothing new about Qdrant, so what is served stays served
        (S3-6's second review: a readiness poll during the probe made the server not-ready).
        """
        if self.store.breaker.state == OPEN:
            return
        first = await self._current()
        if _refused(first):
            return
        if not isinstance(first, StoreError) and self._unchanged(first):
            return
        attempts = self._reopens
        async with self._reopen_lock:
            if self._reopens != attempts:
                return  # another request reopened it, or tried, while this one waited
            current = await self._current()
            if _refused(current):
                # The first read's own failures may have opened the breaker: then that
                # failure, Qdrant's, decides. Otherwise nothing new is known.
                if not isinstance(first, StoreError):
                    return
                current = first
            if isinstance(current, StoreError):
                if not self.served.unseen:
                    logger.warning("Not serving an index: %s", current)
                retrieve, served = None, NOT_SEEN
            elif self._unchanged(current):
                return
            else:
                retrieve, served = await asyncio.to_thread(self._reopen)
            self._reopens += 1
            self.served = served
            self.pipeline = replace(self.pipeline, retrieve=self._serialized(retrieve))

    def _reopen(self) -> tuple[Runnable[RetrievalQuery, list[Document]] | None, Served]:
        with self.retrieval_lock:
            return open_served(self.config, self.store, self.embeddings, self.reranker)

    def breakers(self) -> dict[str, CircuitBreaker]:
        """Each dependency's breaker, by what ``GET /v1/health`` calls it: ``qdrant``,
        ``llm``, and ``fallback`` when one is configured."""
        pipeline = self.pipeline
        found = {"qdrant": self.store.breaker, "llm": pipeline.llm_breaker}
        if pipeline.fallback_llm is not None:
            found["fallback"] = pipeline.fallback_breaker
        return found

    def unavailable(self) -> tuple[str, int] | None:
        """Why no question can be answered now, and the seconds before one may be, or
        ``None``: a breaker the answer needs is open (DEC-21).

        Qdrant's always counts. The model's counts unless the hosted fallback can answer in
        its place, its own breaker refusing nothing; then both do, and the sooner wait is the
        wait. A breaker refuses while it is open, and while it is half-open with its probe
        out (:meth:`~rag_qa.resilience.CircuitBreaker.refusal_s`): a second question then
        gets this 503 rather than a retrieval and an ``error`` event. The first question of a
        half-open breaker goes through: it is the probe. The text is fixed, and names the
        service, never its address (DEC-18).
        """
        qdrant = self.store.breaker
        if (refused := qdrant.refusal_s()) is not None:
            return STORE_DOWN, wait_s(refused)
        pipeline = self.pipeline
        llm = pipeline.llm_breaker
        if (refused := llm.refusal_s()) is None:
            return None
        if pipeline.fallback_llm is None:
            return f"{llm.name} is not answering; ask again after Retry-After", wait_s(refused)
        spare = pipeline.fallback_breaker
        if (spare_refused := spare.refusal_s()) is None:
            return None  # the fallback answers in its place
        return (
            f"neither {llm.name} nor {spare.name} is answering; ask again after Retry-After",
            wait_s(min(refused, spare_refused)),
        )

    async def aclose(self) -> None:
        """Close the model's and the store's clients, on the server's loop."""
        try:
            await self.pipeline.aclose()
        finally:
            self.store.close()
