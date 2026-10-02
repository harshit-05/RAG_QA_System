"""One answer as a stream of typed events, for every front end (DEC-14).

:func:`stream_answer` turns a question into :data:`AnswerEvent`\\ s: one
:class:`Sources`, then a :class:`Token` per chunk of answer text, then :class:`Done`.
The CLI renders them, the API (S2-7) encodes them as SSE, and the evaluation harness
(S2-5) records them. Sources come first, so a client can show citations before the
first token (the §0.4 order).

**Stopping an answer means cancelling the task that consumes the stream.** That works
because the answer half is streamed *directly*, never through
``RunnablePassthrough.assign``: ``RunnableParallel`` waits on its step tasks with
``asyncio.wait`` and never cancels them, so under ``assign`` a cancelled consumer
returns while the generation runs on (ARCHITECTURE.md §2.1, the verified trap). Here a
cancel lands inside the model's await and unwinds every frame down to its HTTP request
before the cancelled consumer returns. ``tests/test_answering.py`` holds both shapes to
that.

A consumer that merely *stops* at a ``yield`` is different: langchain-core 1.6.3 never
closes a sequence's inner generators, so the model's stream closes about 20 loop
iterations after ``aclose()``, not before it returns (S2-1, Discovered). Front ends that
must stop generation therefore cancel the consuming task. In the API that is Starlette's
own disconnect handling, which cancels the response stream that ``stream_answer`` runs
inside (DEC-18): never a separate task, which a disconnect would not reach.

Dependency direction: ``answering → chain``. Imports no front end.
"""

import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.documents import Document

from rag_qa.chain import NoIndexError, QueryPipeline, citation, format_docs, page_label


@dataclass(frozen=True)
class SourceRef:
    """One retrieved chunk, as a citation a client can show."""

    n: int  # the [n] the prompt used
    citation: str  # "2412.14140v2.pdf, p. 7"
    source: str  # corpus-relative path; the bare file name when outside corpus_root
    page: str | None  # printed page label, as the citation shows it
    score: float | None  # rerank_score when reranked


@dataclass(frozen=True)
class Sources:
    """The retrieved chunks, numbered as the prompt numbers them. Always first."""

    sources: list[SourceRef]


@dataclass(frozen=True)
class Token:
    """A chunk of answer text, in order."""

    text: str


@dataclass(frozen=True)
class Done:
    """The answer is complete. Times are from the question to the first token
    (``None`` when no token came) and to the end, in milliseconds."""

    ttft_ms: float | None
    total_ms: float


AnswerEvent = Sources | Token | Done


def corpus_relative(source: str, corpus_root: Path) -> str:
    """``source`` relative to the corpus, so no absolute host path leaves the stream.

    Falls back to the bare file name when ``source`` is not under ``corpus_root`` (an
    index built on another machine) instead of raising. A path that is already
    relative, as S2-6 will store it, is kept unless it climbs out with ``..``.
    """
    path = Path(source)
    if path.is_absolute():
        if not path.is_relative_to(corpus_root):
            return path.name
        path = path.relative_to(corpus_root)
    if ".." in path.parts:
        return path.name
    return path.as_posix()


def source_refs(docs: list[Document], corpus_root: Path) -> list[SourceRef]:
    """Citations for ``docs``, numbered from 1 exactly as :func:`format_docs` numbers them."""
    return [
        SourceRef(
            n=n,
            citation=citation(doc),
            source=corpus_relative(str(doc.metadata.get("source", "unknown source")), corpus_root),
            page=page_label(doc),
            score=doc.metadata.get("rerank_score"),
        )
        for n, doc in enumerate(docs, 1)
    ]


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


async def stream_answer(pipeline: QueryPipeline, question: str) -> AsyncIterator[AnswerEvent]:
    """Answer ``question``: :class:`Sources`, then :class:`Token`\\ s, then :class:`Done`.

    Raises :class:`~rag_qa.chain.NoIndexError` when the pipeline has no retrieval half
    yet. A cancelled stream ends with ``CancelledError``, not ``Done``; other errors
    propagate to the front end, which renders them. Empty text chunks are dropped, so
    ``ttft_ms`` times the first visible token (the figure NFR-2 sets a target for).
    """
    if pipeline.retrieve is None:
        raise NoIndexError("no index yet: ingest the corpus first")
    start = time.perf_counter()

    # Retrieval is awaited, not streamed. It runs in an executor thread that a cancel
    # cannot stop: the awaiting task returns at once and the thread finishes on its own
    # (measured, S2-1 review). It takes milliseconds, so that is not worth machinery (DEC-14).
    docs = await pipeline.retrieve.ainvoke(question)
    yield Sources(source_refs(docs, pipeline.corpus_root))

    ttft_ms: float | None = None
    # Streamed directly — never through RunnablePassthrough.assign (module docstring).
    # astream is an async generator; the cast exposes aclose(), which the
    # AsyncIterator annotation hides.
    stream = cast(
        AsyncGenerator[str, None],
        pipeline.answer.astream({"context": format_docs(docs), "question": question}),
    )
    try:
        async for text in stream:
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = _ms_since(start)
            yield Token(text)
    finally:
        # A cancel lands inside the model's await and unwinds every frame down to the
        # HTTP response before it reaches us: that is the path that stops generation at
        # once (the CLI's Ctrl-C, Starlette's disconnect). This aclose() covers the other
        # ending, a consumer that stops at a `yield`. It closes the sequence's own
        # generator only: langchain-core 1.6.3 iterates the inner ones with `async for`
        # and never closes them, so the loop's async-generator finalizer closes the rest
        # over the next few iterations (about 20, measured in S2-1).
        # A failure to close must not replace the error that ended the stream (the
        # model's own, say). BaseException, so a cancel, still gets through.
        with suppress(Exception):
            await stream.aclose()
    yield Done(ttft_ms=ttft_ms, total_ms=_ms_since(start))
