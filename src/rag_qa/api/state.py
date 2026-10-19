"""What the API serves: the pipeline, the index generation behind it, and the slots.

Built once, at start-up (:meth:`Serving.build`), and shared by every request (DEC-18).

**The index can change under the server.** An ingest job, or a ``rag-ingest`` run outside
the server, publishes a new generation by flipping the index path's symlink (DEC-17).
Before reading what is served, a request calls :meth:`Serving.refresh`, which compares the
link with the generation being served. On a change, it reopens the index in a thread and
swaps only the retrieval half (``dataclasses.replace``): an answer in flight keeps the one
it started with, and the LLM and its clients are never rebuilt.

- A reopen runs the same checks as start-up. An index from another embedder is never
  served: the server goes not-ready until an ingest fixes it.
- An open that fails for another reason (a disk error, a failing embedder) is not a verdict
  on the index. The server goes not-ready with a fixed message, logs the exception, and
  tries again on the next request, rather than answer 500 (S2-7's first review).
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
"""

import asyncio
import inspect
import logging
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace
from pathlib import Path
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
    build_prompt,
    live_generation,
    open_retrieval,
)
from rag_qa.components import build_embedder, build_llm, build_reranker
from rag_qa.manifest import Manifest
from rag_qa.schema import RagConfig

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


@dataclass(frozen=True)
class Served:
    """The index generation served and its manifest, or why there is none.

    ``generation`` is what :func:`rag_qa.chain.live_generation` said when the index was
    last opened, or tried, so a refresh reopens only when the link has moved since.
    """

    generation: Path | None
    manifest: Manifest | None
    problem: str | None  # None while an index is served

    @property
    def ready(self) -> bool:
        return self.problem is None


def open_served(
    config: RagConfig, embeddings: Embeddings, reranker: BaseDocumentCompressor | None
) -> tuple[Runnable[str, list[Document]] | None, Served]:
    """Open the live index as a retrieval half, or say why it cannot be served."""
    seen = live_generation(config)
    opened = _try_open(config, embeddings, reranker)
    if isinstance(opened, NoIndexError) and live_generation(config) != seen:
        # A flip landed while it opened. Only the live generation and the previous one are
        # kept, so the one it checked may be gone already: look once more rather than
        # report a missing index that is there (S2-6's third review).
        seen = live_generation(config)
        opened = _try_open(config, embeddings, reranker)
    if isinstance(opened, NoIndexError):
        logger.warning("Not serving an index: %s", opened)  # the server log may name paths
        problem = REINGEST if isinstance(opened, IncompatibleIndexError) else NO_INDEX_YET
        return None, Served(generation=seen, manifest=None, problem=problem)
    if isinstance(opened, Exception):
        # Not a verdict on the index (S2-7's first review): a disk error, a failing embedder.
        # No generation is recorded as seen, so the next request tries again.
        return None, Served(generation=None, manifest=None, problem=UNAVAILABLE)
    logger.info(
        "Serving index %s (%d documents).",
        opened.generation.name,
        len(opened.manifest.documents),
    )
    return opened.retrieve, Served(opened.generation, opened.manifest, problem=None)


def _try_open(
    config: RagConfig, embeddings: Embeddings, reranker: BaseDocumentCompressor | None
) -> Retrieval | Exception:
    """The index opened, or why not: a :class:`NoIndexError` (``IncompatibleIndexError``
    too) when the index is missing or unusable, any other exception when opening it failed
    for a reason that says nothing about the index. That one is logged here, in full."""
    try:
        return open_retrieval(config, embeddings, reranker=reranker)
    except NoIndexError as e:  # IncompatibleIndexError too
        return e
    # Deliberately broad: a reopen runs inside a request (and inside health), and an error
    # escaping it would answer 500 to each of them, again on every request.
    except Exception as e:
        logger.exception("Could not open the index")  # the server log may name paths
        return e


def serialized(
    retrieve: Runnable[str, list[Document]], lock: threading.Lock
) -> Runnable[str, list[Document]]:
    """``retrieve``, run under ``lock``, taken inside the worker thread.

    The thread holds the lock until its retrieval ends, even after the request awaiting it
    was cancelled. The next retrieval therefore waits for an orphaned one rather than
    overlapping it. A waiting retrieval whose own request is cancelled before it gets the
    lock does no work at all.
    """

    def run(question: str, abandoned: threading.Event | None = None) -> list[Document]:
        with lock:
            if abandoned is not None and abandoned.is_set():
                return []  # its client left while it waited: nobody will read the result
            return retrieve.invoke(question)

    async def arun(question: str) -> list[Document]:
        abandoned = threading.Event()
        try:
            return await asyncio.get_running_loop().run_in_executor(None, run, question, abandoned)
        except asyncio.CancelledError:
            abandoned.set()
            raise

    return RunnableLambda(run, afunc=arun)


class Slot:
    """One answer's claim on a slot, and the answer stream it holds.

    Released once, however many paths release it: the stream's ``finally``
    (:meth:`release`) and the request's teardown (:meth:`aclose`).
    """

    def __init__(self, pool: "SlotPool") -> None:
        self._pool = pool
        self._released = False
        self._answer: AsyncGenerator[Any, None] | None = None

    def hold(self, answer: AsyncGenerator[Any, None]) -> None:
        """The answer stream to close when the request ends (:meth:`aclose`)."""
        self._answer = answer

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._pool.in_use -= 1

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
        embeddings: Embeddings,
        reranker: BaseDocumentCompressor | None,
        max_concurrent: int,
    ) -> None:
        self.config = config
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
        part whose failure stops the server (Ollama down). A missing or unusable index only
        makes the server not-ready.
        """
        llm = build_llm(config)  # validate_model_on_init asks Ollama for its models
        embeddings = build_embedder(config)
        reranker = build_reranker(config)
        retrieve, served = open_served(config, embeddings, reranker)
        pipeline = QueryPipeline(
            retrieve=retrieve,
            prompt=build_prompt(config),
            llm=llm,
            corpus_root=config.paths.data,
        )
        return cls(
            config,
            pipeline,
            served,
            embeddings=embeddings,
            reranker=reranker,
            max_concurrent=max_concurrent,
        )

    def _serialized(
        self, retrieve: Runnable[str, list[Document]] | None
    ) -> Runnable[str, list[Document]] | None:
        return None if retrieve is None else serialized(retrieve, self.retrieval_lock)

    async def refresh(self) -> None:
        """Serve the generation the index path links to now, if it is not the one served.

        A few system calls when nothing changed. A reopen runs in a thread, under the
        retrieval lock, so it never overlaps a retrieval that uses the same embedder and
        reranker. Concurrent requests reopen once, whether the open succeeds or fails: a
        failed one records no generation as seen, so without the attempt count each request
        queued behind it would try again in turn (S2-7's second review).
        """
        if live_generation(self.config) == self.served.generation:
            return
        attempts = self._reopens
        async with self._reopen_lock:
            if self._reopens != attempts or live_generation(self.config) == self.served.generation:
                return  # another request reopened it, or tried, while this one waited
            retrieve, served = await asyncio.to_thread(self._reopen)
            self._reopens += 1
            self.served = served
            self.pipeline = replace(self.pipeline, retrieve=self._serialized(retrieve))

    def _reopen(self) -> tuple[Runnable[str, list[Document]] | None, Served]:
        with self.retrieval_lock:
            return open_served(self.config, self.embeddings, self.reranker)

    async def aclose(self) -> None:
        """Close the model's HTTP clients, on the server's loop."""
        await self.pipeline.aclose()
