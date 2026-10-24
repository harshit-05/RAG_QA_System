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

**Retries, breakers and the hosted fallback** (NFR-6, DEC-21, S3-6). Each model has a
breaker (:class:`~rag_qa.resilience.CircuitBreaker`), and the first token judges an attempt:
once one has arrived, the service answered, and nothing after it is retried or routed, since
a repeat would duplicate text. Before it:

- a failed connection is tried once more, after a back-off, then the next model;
- a timeout, or an answer that the service cannot serve now (5xx), is the answer's error: it
  counts against the breaker, and is neither repeated nor routed;
- an open breaker sends the answer to the next model at once.

The next model is the hosted fallback, when the pipeline has one. Its stream is read exactly
as the first one's, in this same task, behind the same ``finally``: never through
``.with_fallbacks()``, whose wrapper would put back the Runnable machinery this module
avoids. :class:`Done` names the model that answered.

**Token counts** (NFR-10, S3-7). :class:`Done` carries the provider's counts, from the last
chunk that reports ``usage_metadata``. Ollama reports them on its final chunk, which has no
text, so they are read before the empty-text skip. Reading a chunk awaits nothing, so the
cancel path is unchanged.

Dependency direction: ``answering → {chain, resilience}``. Imports no front end.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import UsageMetadata

from rag_qa import resilience
from rag_qa.chain import (
    NoIndexError,
    QueryPipeline,
    RetrievalQuery,
    citation,
    format_docs,
    page_label,
)
from rag_qa.resilience import Attempt, BreakerOpenError, CircuitBreaker


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
    (``None`` when no token came) and to the end, in milliseconds. ``model`` is the config
    reference of the model that answered: ``pipeline.query.llm``'s, or the hosted
    fallback's (S3-6). The token counts are the provider's own (S3-7), ``None`` when it
    reported none: the hosted fallback reports them only with ``stream_usage`` set."""

    ttft_ms: float | None
    total_ms: float
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


AnswerEvent = Sources | Token | Done


def answered_line(done: Done) -> str:
    """One line on a complete answer, for the logs (NFR-10, S3-7): the model that answered,
    the times and the token counts. Never the question's text, nor the answer's."""
    first = "no text" if done.ttft_ms is None else f"first token at {done.ttft_ms / 1000:.1f} s"
    if done.input_tokens is None and done.output_tokens is None:
        tokens = "no token counts reported"
    else:
        counts = [
            f"{count:,} {kind}" if count is not None else f"no {kind} count"
            for count, kind in ((done.input_tokens, "input"), (done.output_tokens, "output"))
        ]
        tokens = f"{counts[0]} and {counts[1]} tokens"
    return f"Answered by {done.model} in {done.total_ms / 1000:.1f} s ({first}), {tokens}."


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


@dataclass(frozen=True)
class _Model:
    """A model an answer may stream from, as :func:`stream_answer` tries them."""

    llm: BaseChatModel
    ref: str  # its config reference, which Done names when it answers
    breaker: CircuitBreaker
    attempts: int  # tries before its first token, each after a failed connection
    fallback: bool = False  # the hosted fallback: its errors are marked as such


def _models(pipeline: QueryPipeline) -> list[_Model]:
    """The pipeline's model, then its hosted fallback when it has one (DEC-21)."""
    models = [_Model(pipeline.llm, pipeline.llm_ref, pipeline.llm_breaker, resilience.LLM_ATTEMPTS)]
    if pipeline.fallback_llm is not None:
        models.append(
            _Model(
                pipeline.fallback_llm,
                pipeline.fallback_ref,
                pipeline.fallback_breaker,
                resilience.FALLBACK_ATTEMPTS,
                fallback=True,
            )
        )
    return models


def _judge(tried: Attempt, verdict: str | None) -> None:
    """Record what an attempt that failed before its first token says about its service."""
    if verdict in resilience.FAILED:
        tried.failed()
    elif verdict == resilience.ANSWERED:  # it answered, refusing the request itself
        tried.succeeded()
    else:  # an error of the request's own, or ours: nothing learned about the service
        tried.abandoned()


async def stream_answer(
    pipeline: QueryPipeline, question: str, *, sources: Sequence[str] = ()
) -> AsyncIterator[AnswerEvent]:
    """Answer ``question``: :class:`Sources`, then :class:`Token`\\ s, then :class:`Done`.

    With ``sources``, corpus-relative paths, only those documents are searched (S3-5,
    FR-3); the front end has checked them against the served manifest. Raises
    :class:`~rag_qa.chain.NoIndexError` when the pipeline has no retrieval half yet. A
    cancelled stream ends with ``CancelledError``, not ``Done``; other errors propagate to
    the front end, which renders them. Empty text chunks are dropped, so ``ttft_ms`` times
    the first visible token (the figure NFR-2 sets a target for).

    Before its first token, an answer is retried after a failed connection, then routed to
    the hosted fallback, as the module docstring says. When no model can answer, the error
    raised is the last one, or :class:`~rag_qa.resilience.BreakerOpenError` when the last
    model's breaker refused it.
    """
    if pipeline.retrieve is None:
        raise NoIndexError("no index yet: ingest the corpus first")
    start = time.perf_counter()

    # Retrieval is awaited, not streamed. It runs in an executor thread that a cancel
    # cannot stop: the awaiting task returns at once and the thread finishes on its own
    # (measured, S2-1 review). With the reranker that is up to about 2 s of CPU, which
    # front ends that limit concurrency must count (DEC-14; the API's slot, S2-7).
    docs = await pipeline.retrieve.ainvoke(RetrievalQuery(question, tuple(sources)))
    yield Sources(source_refs(docs, pipeline.corpus_root))

    ttft_ms: float | None = None
    # The prompt is rendered here and the model itself is streamed, in this task: never
    # through a Runnable sequence, whose per-chunk tasks would let a cancel miss the
    # model's read (module docstring). format_messages is what the sequence's prompt step
    # runs, so the model gets the same messages, as test_chain checks.
    messages = pipeline.prompt.format_messages(context=format_docs(docs), question=question)
    failure: Exception = RuntimeError("no model was asked")
    for model in _models(pipeline):
        for attempt in range(1, model.attempts + 1):
            if attempt > 1:  # after a failed connection: the same model, once more
                await asyncio.sleep(resilience.backoff_s(attempt - 1))
            try:
                tried = model.breaker.allow()
            except BreakerOpenError as e:
                failure = e
                break  # this model is out for now: the next one, if there is one
            # astream is an async generator; the cast exposes aclose(), which the
            # AsyncIterator annotation hides.
            stream = cast(AsyncGenerator[AIMessageChunk, None], model.llm.astream(messages))
            usage: UsageMetadata | None = None  # this attempt's, from its provider
            try:
                async for chunk in stream:
                    if chunk.usage_metadata:
                        usage = chunk.usage_metadata  # the last chunk that reports it wins
                    # What StrOutputParser yields for a chunk: its text blocks, joined.
                    # str(), because .text is a str subclass kept callable for compatibility.
                    text = str(chunk.text)
                    if not text:
                        continue
                    if ttft_ms is None:
                        ttft_ms = _ms_since(start)
                        tried.succeeded()  # the first token: the service answered
                    yield Token(text)
            except Exception as e:
                if model.fallback:  # its HTTP errors look like Ollama's: say whose they are
                    resilience.mark_fallback(e)
                if ttft_ms is not None:
                    raise  # after the first token: never repeated, never routed
                verdict = resilience.classify(e)
                _judge(tried, verdict)
                if verdict != resilience.CONNECTION:
                    raise  # a timeout or an error answer: no retry, no fallback (DEC-21)
                failure = e
                continue  # the same model once more, or the next one
            except BaseException:
                tried.abandoned()  # a cancel: nothing learned about the service
                raise
            finally:
                # A cancel lands inside the model's own read and unwinds every frame down
                # to the HTTP response before it reaches us: that is the path that stops
                # generation at once (the CLI's Ctrl-C, Starlette's disconnect). This
                # aclose() covers the other ending, a consumer that stops at a `yield`. It
                # closes the chat model's own generator; langchain-core 1.6.3 iterates the
                # provider's generator with `async for` and never closes it, so the loop's
                # finalizer closes that one a loop iteration or two later.
                # A failure to close must not replace the error that ended the stream (the
                # model's own, say). BaseException, so a cancel, still gets through.
                with suppress(Exception):
                    await stream.aclose()
            tried.succeeded()  # an answer with no text at all; a no-op after a token
            yield Done(
                ttft_ms=ttft_ms,
                total_ms=_ms_since(start),
                model=model.ref,
                input_tokens=None if usage is None else usage["input_tokens"],
                output_tokens=None if usage is None else usage["output_tokens"],
            )
            return
    raise failure
