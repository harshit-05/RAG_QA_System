"""One answer as a stream of typed events, for every front end (DEC-14).

:func:`stream_answer` turns a question into :data:`AnswerEvent`\\ s: one
:class:`Sources`, then a :class:`Token` per chunk of answer text, then :class:`Done`.
The CLI renders them, the API (S2-7) encodes them as SSE, and the evaluation harness
(S2-5) records them. Sources come first, so a client can show citations before the
first token (the §0.4 order).

**Stopping an answer means cancelling the task that consumes the stream.** That works
because the prompt is rendered here and the *model itself* is streamed, in the
consumer's own task (ARCHITECTURE.md §2.1, DEC-14). Two shapes would break it:

- ``RunnablePassthrough.assign``: ``RunnableParallel`` waits on its step tasks with
  ``asyncio.wait`` and never cancels them, so a cancelled consumer returns while the
  generation runs on (the first verified trap).
- ``prompt | llm | StrOutputParser()``, even streamed directly: langchain-core runs each
  chunk of a sequence in its own task, and the parser hops to a thread per chunk. A
  cancel that lands at one of those boundaries ends the chunk's task instead of reaching
  the model's await, and the model's stream is left paused at a ``yield``, open until the
  loop's finalizer closes it. In the CLI the loop is idle at the prompt, so that waited
  for the next question (the second trap; found as a flaky test, fixed after S2-3).

Streamed itself, the model's stream is only ever paused in its own read, so a cancel
lands there and unwinds its HTTP response before the cancelled consumer returns.
``tests/test_answering.py`` holds all three shapes to that.

A consumer that merely *stops* at a ``yield`` is different: langchain-core 1.6.3 never
closes a chat model's inner generator, so the model's stream closes a loop iteration or
two after ``aclose()``, not before it returns (S2-1, Discovered). Front ends that must
stop generation therefore cancel the consuming task. In the API that is Starlette's own
disconnect handling, which cancels the response stream that ``stream_answer`` runs inside
(DEC-18): never a separate task, which a disconnect would not reach.

Dependency direction: ``answering → chain``. Imports no front end.
"""

import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

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
    # (measured, S2-1 review). With the reranker that is up to about 2 s of CPU, which
    # front ends that limit concurrency must count (DEC-14; the API's slot, S2-7).
    docs = await pipeline.retrieve.ainvoke(question)
    yield Sources(source_refs(docs, pipeline.corpus_root))

    ttft_ms: float | None = None
    # The prompt is rendered here and the model itself is streamed, in this task: never
    # through a Runnable sequence, whose per-chunk tasks would let a cancel miss the
    # model's read (module docstring). format_messages is what the sequence's prompt step
    # runs, so the model gets the same messages, as test_chain checks. astream is an async
    # generator; the cast exposes aclose(), which the AsyncIterator annotation hides.
    messages = pipeline.prompt.format_messages(context=format_docs(docs), question=question)
    stream = cast(AsyncGenerator[AIMessageChunk, None], pipeline.llm.astream(messages))
    try:
        async for chunk in stream:
            # What StrOutputParser yields for a chunk: its text blocks, joined. str(),
            # because .text is a str subclass kept callable for compatibility.
            text = str(chunk.text)
            if not text:
                continue
            if ttft_ms is None:
                ttft_ms = _ms_since(start)
            yield Token(text)
    finally:
        # A cancel lands inside the model's own read and unwinds every frame down to the
        # HTTP response before it reaches us: that is the path that stops generation at
        # once (the CLI's Ctrl-C, Starlette's disconnect). This aclose() covers the other
        # ending, a consumer that stops at a `yield`. It closes the chat model's own
        # generator; langchain-core 1.6.3 iterates the provider's generator with
        # `async for` and never closes it, so the loop's finalizer closes that one a
        # loop iteration or two later.
        # A failure to close must not replace the error that ended the stream (the
        # model's own, say). BaseException, so a cancel, still gets through.
        with suppress(Exception):
            await stream.aclose()
    yield Done(ttft_ms=ttft_ms, total_ms=_ms_since(start))
