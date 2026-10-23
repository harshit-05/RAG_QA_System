"""The answer-event stream (S2-1, DEC-14), checked hermetically.

Two claims are under test:

* **The events.** ``stream_answer`` yields ``Sources``, then ``Token``\\ s, then ``Done``,
  numbered exactly as the prompt numbers its chunks, with no absolute host path.
* **Cancelling works.** Cancelling the task that consumes the stream closes the model's
  stream *before* that task returns: during the simulated prefill, mid-stream, and when
  the cancel arrives while a chunk is being processed (forced, so it is deterministic).
  The fake model's ``finally`` stands in for ChatOllama closing its HTTP response.

**Read the negative controls first.** The same assertion must fail against the two
shapes ``stream_answer`` avoids (ARCHITECTURE.md §2.1, DEC-14):

* ``RunnablePassthrough.assign`` (today's ``build_rag_chain``): ``RunnableParallel``
  waits on its step tasks with ``asyncio.wait`` and never cancels them;
* ``prompt | llm | StrOutputParser()``, streamed directly, as ``stream_answer`` did until
  S2-3's second review: each chunk runs in its own task, so a cancel that arrives while a
  chunk is in flight ends that task and leaves the model's stream paused at a ``yield``.
  That made the mid-stream test flaky (66 failures in 300 on one core).

Both are ``xfail(strict=True)``: if a langchain-core release fixes either, the suite goes
red and says the trap is gone, instead of a control quietly passing and no longer telling
the bug from the fix.
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from operator import itemgetter
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest
from conftest import StreamingFakeChatModel, openai_error
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag_qa.answering import (
    Done,
    Sources,
    Token,
    corpus_relative,
    source_refs,
    stream_answer,
)
from rag_qa.chain import NoIndexError, QueryPipeline, build_query_pipeline, format_docs
from rag_qa.config import load_config
from rag_qa.resilience import (
    FALLBACK_ATTEMPTS,
    LLM_ATTEMPTS,
    BreakerOpenError,
    CircuitBreaker,
    is_marked_fallback,
)

CORPUS = Path("/corpus")
DOCS = [
    Document(page_content="Preface.", metadata={"source": "/corpus/guide.pdf", "page_label": "i"}),
    Document(page_content="Notes.", metadata={"source": "/corpus/sub/notes.txt"}),
]


PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Answer from the context."), ("human", "{context}\n\n{question}")]
)


def fake_pipeline(model: StreamingFakeChatModel, docs: list[Document] = DOCS) -> QueryPipeline:
    return QueryPipeline(
        retrieve=RunnableLambda(lambda question: docs),
        prompt=PROMPT,
        llm=model,
        corpus_root=CORPUS,
    )


async def collect(events: AsyncIterator[Any]) -> list[Any]:
    return [event async for event in events]


# --- the events --------------------------------------------------------------------------


def test_events_are_sources_then_tokens_then_done(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real pipeline over the real (fake-embedded) index; only the model is a fake,
    # so the prompt it records is the production prompt.
    model = StreamingFakeChatModel()
    monkeypatch.setattr("rag_qa.chain.build_llm", lambda config: model)
    pipeline = build_query_pipeline(load_config(fake_rag))

    sources, *tokens, done = asyncio.run(collect(stream_answer(pipeline, "How many staff?")))

    assert isinstance(sources, Sources)
    assert all(isinstance(token, Token) for token in tokens)
    assert "".join(token.text for token in tokens) == "".join(model.tokens)
    assert isinstance(done, Done)
    assert done.ttft_ms is not None
    assert 0 <= done.ttft_ms <= done.total_ms

    refs = sources.sources
    assert 0 < len(refs) <= 5  # retriever k
    assert [ref.n for ref in refs] == list(range(1, len(refs) + 1))
    for ref in refs:
        # The [n] the CLI prints is the [n] the model was shown.
        assert f"[{ref.n}] ({ref.citation})" in model.prompts[0]
    # Corpus-relative, never a host path (the sample corpus has nested files).
    corpus_files = {"guide.pdf", "notes.txt", "README.md", "sub/report.docx", "sub/deeper/LOUD.TXT"}
    assert {ref.source for ref in refs} <= corpus_files


def test_an_answer_without_text_has_no_time_to_first_token() -> None:
    model = StreamingFakeChatModel(tokens=["", ""])
    events = asyncio.run(collect(stream_answer(fake_pipeline(model), "q")))
    assert [type(event) for event in events] == [Sources, Done]  # empty chunks are not tokens
    assert events[-1].ttft_ms is None


def test_a_pipeline_without_an_index_raises_no_index_error() -> None:
    model = StreamingFakeChatModel()
    pipeline = QueryPipeline(retrieve=None, prompt=PROMPT, llm=model, corpus_root=CORPUS)
    with pytest.raises(NoIndexError):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert model.prompts == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param("/corpus/guide.pdf", "guide.pdf", id="top level"),
        pytest.param("/corpus/sub/report.docx", "sub/report.docx", id="nested"),
        pytest.param("/home/someone-else/corpus/guide.pdf", "guide.pdf", id="other machine"),
        pytest.param("/corpus-old/guide.pdf", "guide.pdf", id="sibling, not under"),
        pytest.param("sub/report.docx", "sub/report.docx", id="already relative"),
        pytest.param("../private/guide.pdf", "guide.pdf", id="relative, climbing out"),
    ],
)
def test_sources_are_reported_relative_to_the_corpus(source: str, expected: str) -> None:
    assert corpus_relative(source, CORPUS) == expected


def test_source_refs_carry_citation_page_and_rerank_score() -> None:
    docs = [
        Document(page_content="a", metadata={"source": "/elsewhere/old.pdf", "page": 4}),
        Document(page_content="b", metadata={"source": "/corpus/x.txt", "rerank_score": 0.75}),
    ]
    first, second = source_refs(docs, CORPUS)
    assert (first.n, first.citation, first.source, first.page, first.score) == (
        1,
        "old.pdf, p. 5",
        "old.pdf",
        "5",
        None,
    )
    assert (second.n, second.citation, second.source, second.page, second.score) == (
        2,
        "x.txt",
        "x.txt",
        None,
        0.75,
    )


# --- cancellation ------------------------------------------------------------------------

# Where the cancel lands: during the simulated prompt evaluation, or between tokens.
MOMENTS: dict[str, tuple[dict[str, Any], Callable[[StreamingFakeChatModel], bool]]] = {
    "prefill": ({"prefill_s": 5.0}, lambda model: model.log == ["start"]),
    "mid-stream": (
        {"tokens": [f"t{i} " for i in range(40)], "gap_s": 0.05},
        lambda model: len(model.log) >= 3,  # "start" and two tokens
    ),
}


def sequence(pipeline: QueryPipeline) -> Runnable[dict[str, str], str]:
    """``prompt | llm | StrOutputParser()`` over the pipeline's parts."""
    return pipeline.prompt | pipeline.llm | StrOutputParser()


def assign_shape(pipeline: QueryPipeline) -> Runnable[dict[str, str], dict[str, Any]]:
    """The shape ``build_rag_chain`` composes, over the same parts: the first trap."""
    assert pipeline.retrieve is not None
    to_prompt_inputs: Runnable[dict[str, Any], dict[str, str]] = RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )
    return RunnablePassthrough.assign(
        context=itemgetter("question") | pipeline.retrieve
    ) | RunnablePassthrough.assign(answer=to_prompt_inputs | sequence(pipeline))


def sequence_shape(pipeline: QueryPipeline, question: str) -> AsyncIterator[str]:
    """What ``stream_answer`` streamed until S2-3's second review: the second trap."""
    assert pipeline.retrieve is not None
    docs = pipeline.retrieve.invoke(question)
    return sequence(pipeline).astream({"context": format_docs(docs), "question": question})


async def closed_when_cancelled(
    events: AsyncIterator[Any],
    model: StreamingFakeChatModel,
    ready: Callable[[StreamingFakeChatModel], bool],
) -> bool:
    """Consume ``events`` in a task, cancel it once ``ready``, and report whether the
    model's stream was already closed at the moment the cancelled task returned.

    Answered inside the running loop: after ``asyncio.run`` returns, its own cleanup
    would have closed everything and the answer would mean nothing.
    """

    async def consume() -> None:
        async for _ in events:
            pass

    consumer = asyncio.create_task(consume())
    while not ready(model):
        assert not consumer.done(), "the answer ended before the cancel point"
        await asyncio.sleep(0.01)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    return model.closed


@pytest.mark.parametrize("moment", MOMENTS)
def test_cancel_closes_the_stream_before_the_consumer_returns(moment: str) -> None:
    settings, ready = MOMENTS[moment]
    model = StreamingFakeChatModel(**settings)
    events = stream_answer(fake_pipeline(model), "q")
    assert asyncio.run(closed_when_cancelled(events, model, ready)), (
        "the consumer returned while the model was still streaming: generation would run on"
    )


@pytest.mark.xfail(
    strict=True,
    # Only the assertion below may satisfy it: a TypeError from a broken test, say,
    # would otherwise count as the expected failure.
    raises=AssertionError,
    reason="the trap DEC-14 avoids: RunnableParallel never cancels its step tasks, so "
    "under RunnablePassthrough.assign the cancelled consumer returns first. If this "
    "passes, langchain-core changed: re-check §2.1 rather than delete the control.",
)
@pytest.mark.parametrize("moment", MOMENTS)
def test_negative_control_cancel_under_assign_leaves_the_stream_open(moment: str) -> None:
    settings, ready = MOMENTS[moment]
    model = StreamingFakeChatModel(**settings)
    events = assign_shape(fake_pipeline(model)).astream({"question": "q"})
    assert asyncio.run(closed_when_cancelled(events, model, ready))


class CancelsAtAChunk(StreamingFakeChatModel):
    """Cancels the consuming task while a chunk is being processed: just before it yields
    token ``at``, instead of while it waits on Ollama.

    That is where Ctrl-C lands when it arrives as a chunk comes in, and it is the moment
    that made the mid-stream test flaky. Forcing it makes the test deterministic.
    ``consumer`` is the task to cancel, set by the test once it exists.
    """

    at: int = 0
    consumer: Any = None

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._record(messages)
        self.log.append("start")
        try:
            for i, token in enumerate(self.tokens):
                if i == self.at:
                    self.consumer.cancel()
                self.log.append(token)
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
                await asyncio.sleep(0)  # the next read from Ollama
        finally:
            self.log.append("closed")


async def closed_on_return(events: AsyncIterator[Any], model: CancelsAtAChunk) -> bool:
    """Consume ``events`` in a task the model cancels, and report whether the model's
    stream was closed when that task returned."""

    async def consume() -> None:
        async for _ in events:
            pass

    model.consumer = asyncio.create_task(consume())
    with pytest.raises(asyncio.CancelledError):
        await model.consumer
    return model.closed


TOKENS = [f"t{i} " for i in range(6)]


@pytest.mark.parametrize("at", [0, 1, 3, 5])
def test_a_cancel_while_a_chunk_is_processed_closes_the_stream_before_returning(at: int) -> None:
    model = CancelsAtAChunk(tokens=TOKENS, at=at)
    assert asyncio.run(closed_on_return(stream_answer(fake_pipeline(model), "q"), model)), (
        f"the consumer returned with the model's stream still open: {model.log}"
    )
    assert model.log == ["start", *TOKENS[: at + 1], "closed"]  # nothing after the cancel


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the second trap DEC-14 avoids: a sequence runs each chunk in its own task, so "
    "a cancel while a chunk is in flight ends that task and leaves the model's stream "
    "paused at a yield. If this passes, langchain-core changed: re-check §2.1.",
)
@pytest.mark.parametrize("at", [0, 3])
def test_negative_control_the_same_cancel_under_a_sequence_leaves_the_stream_open(
    at: int,
) -> None:
    model = CancelsAtAChunk(tokens=TOKENS, at=at)
    assert asyncio.run(closed_on_return(sequence_shape(fake_pipeline(model), "q"), model))


def test_a_consumer_that_stops_early_closes_the_stream_while_the_loop_runs() -> None:
    # The other ending: the consumer stops at a yield and closes the generator, instead
    # of being cancelled inside the model's read. Not immediate: langchain-core 1.6.3
    # never aclose()s the provider's generator inside BaseChatModel.astream, so the
    # loop's async-generator finalizer closes it a loop iteration or two later. Prompt
    # while a loop runs, so no more tokens are generated; cancellation remains the path
    # that closes the stream *before* control returns (the tests above).
    model = StreamingFakeChatModel(tokens=[f"t{i} " for i in range(40)], gap_s=0.01)

    async def iterations_until_closed() -> int | None:
        events = stream_answer(fake_pipeline(model), "q")
        async for event in events:
            if isinstance(event, Token):
                break
        await events.aclose()
        for iteration in range(100):
            if model.closed:
                return iteration
            await asyncio.sleep(0)
        return None

    assert asyncio.run(iterations_until_closed()) is not None
    assert model.log == ["start", "t0 ", "closed"]  # nothing generated after the stop


def test_a_failure_to_close_does_not_replace_the_streams_own_error() -> None:
    class Boom(RuntimeError): ...

    class FailingClose:
        """An async generator that fails, and whose close fails as well."""

        def __aiter__(self) -> "FailingClose":
            return self

        async def __anext__(self) -> str:
            raise Boom("the model failed")

        async def aclose(self) -> None:
            raise RuntimeError("close failed too")

    class FailingModel:
        def astream(self, messages: Any) -> FailingClose:
            return FailingClose()

    pipeline = QueryPipeline(
        retrieve=RunnableLambda(lambda question: DOCS),
        prompt=PROMPT,
        llm=FailingModel(),  # type: ignore[arg-type]
        corpus_root=CORPUS,
    )
    with pytest.raises(Boom):
        asyncio.run(collect(stream_answer(pipeline, "q")))


# --- retries, breakers and the hosted fallback (S3-6, DEC-21) ------------------------------
#
# The models below fail as the real clients do, with the real errors: ChatOllama's stream
# raises httpx's unwrapped, and Ollama's own answers are ollama's ResponseError. Every stream
# logs "start" and, from a finally, "closed": the stand-in for its HTTP response closing.

PRIMARY = "components.llms.primary"
SPARE = "components.llms.spare"


class Refusing(StreamingFakeChatModel):
    """Fails before its first token with ``error``, as a stopped Ollama refuses the
    connection; counts its attempts."""

    error: Any = None

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._record(messages)
        self.log.append("start")
        try:
            await asyncio.sleep(self.prefill_s)
            raise self.error or httpx.ConnectError("All connection attempts failed")
            yield  # an async generator, like every model's stream
        finally:
            self.log.append("closed")

    @property
    def attempts(self) -> int:
        return self.log.count("start")


class BreaksAfterAToken(StreamingFakeChatModel):
    """Yields one token, then loses its connection."""

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.log.append("start")
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Station "))
            raise httpx.RemoteProtocolError("peer closed connection")
        finally:
            self.log.append("closed")


def two_models(
    primary: BaseChatModel, fallback: BaseChatModel | None, **breakers: Any
) -> QueryPipeline:
    return QueryPipeline(
        retrieve=RunnableLambda(lambda question: DOCS),
        prompt=PROMPT,
        llm=primary,
        corpus_root=CORPUS,
        llm_ref=PRIMARY,
        fallback_llm=fallback,
        fallback_ref=SPARE,
        **breakers,
    )


def opened(name: str, clock: Callable[[], float] = time.monotonic) -> CircuitBreaker:
    breaker = CircuitBreaker(name, clock=clock)
    for _ in range(breaker.failures):
        breaker.allow().failed()
    return breaker


def test_a_refused_connection_is_tried_once_more_then_the_fallback_answers() -> None:
    primary = Refusing()
    spare = StreamingFakeChatModel(tokens=["Spare ", "answer."])
    pipeline = two_models(primary, spare)
    sources, *tokens, done = asyncio.run(collect(stream_answer(pipeline, "q")))
    assert isinstance(sources, Sources)
    assert [token.text for token in tokens] == ["Spare ", "answer."]
    assert isinstance(done, Done)
    assert done.model == SPARE
    assert primary.attempts == LLM_ATTEMPTS
    assert primary.log == ["start", "closed"] * LLM_ATTEMPTS  # each stream closed in turn
    assert spare.log[0] == "start" and spare.closed
    assert pipeline.llm_breaker._failed == LLM_ATTEMPTS  # each attempt counted
    assert pipeline.fallback_breaker._failed == 0
    assert spare.prompts == primary.prompts[:1]  # the same messages, rendered once


def test_the_model_that_answers_is_named_by_done() -> None:
    pipeline = two_models(StreamingFakeChatModel(), StreamingFakeChatModel())
    *_, done = asyncio.run(collect(stream_answer(pipeline, "q")))
    assert done.model == PRIMARY
    assert pipeline.fallback_llm.log == []  # type: ignore[union-attr]  # never asked


def test_without_a_fallback_a_refused_connection_is_the_answer_s_error() -> None:
    primary = Refusing()
    pipeline = two_models(primary, None)
    with pytest.raises(httpx.ConnectError):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert primary.attempts == LLM_ATTEMPTS


def test_an_error_from_the_fallback_s_stream_is_marked_as_its_own() -> None:
    # Its HTTP client's errors are httpx's (httpx2's, under openai 3): nothing in their class
    # tells them from Ollama's, so the answer that read the stream says whose they are.
    spare = BreaksAfterAToken()
    pipeline = two_models(StreamingFakeChatModel(), spare, llm_breaker=opened("Ollama"))
    with pytest.raises(httpx.RemoteProtocolError) as after_a_token:
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert is_marked_fallback(after_a_token.value)

    refusing = two_models(
        Refusing(), Refusing(error=httpx.ConnectError("refused")), llm_breaker=opened("Ollama")
    )
    with pytest.raises(httpx.ConnectError) as before_one:
        asyncio.run(collect(stream_answer(refusing, "q")))
    assert is_marked_fallback(before_one.value)


def test_the_model_s_own_errors_are_not_marked_as_the_fallback_s() -> None:
    after_a_token = two_models(BreaksAfterAToken(), StreamingFakeChatModel())
    with pytest.raises(httpx.RemoteProtocolError) as caught:
        asyncio.run(collect(stream_answer(after_a_token, "q")))
    assert not is_marked_fallback(caught.value)
    # Refused by the model, then by the fallback: the error raised is the fallback's, and
    # the model's own, kept apart, was never marked.
    primary_error = httpx.ConnectError("Ollama refused")
    refusing = two_models(
        Refusing(error=primary_error), Refusing(error=httpx.ConnectError("spare refused"))
    )
    with pytest.raises(httpx.ConnectError):
        asyncio.run(collect(stream_answer(refusing, "q")))
    assert not is_marked_fallback(primary_error)


def test_a_failure_after_the_first_token_is_neither_repeated_nor_routed() -> None:
    # A repeat after a token would duplicate text (DEC-21).
    primary = BreaksAfterAToken()
    spare = StreamingFakeChatModel()
    pipeline = two_models(primary, spare)
    events: list[Any] = []

    async def run() -> None:
        async for event in stream_answer(pipeline, "q"):
            events.append(event)

    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(run())
    assert [type(event) for event in events] == [Sources, Token]
    assert primary.log == ["start", "closed"]
    assert spare.log == []
    assert pipeline.llm_breaker._failed == 0  # the first token judged it: the service answered


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),  # a silent prefill past the 600 s read timeout
        ollama.ResponseError("model requires more system memory", 500),
        ollama.ResponseError("llama runner process has terminated"),  # inside the stream
    ],
    ids=["timeout", "5xx", "mid-stream error"],
)
def test_a_timeout_or_an_error_answer_counts_but_is_neither_repeated_nor_routed(
    error: Exception,
) -> None:
    primary = Refusing(error=error)
    spare = StreamingFakeChatModel()
    pipeline = two_models(primary, spare)
    with pytest.raises(type(error)):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert primary.attempts == 1
    assert spare.log == []
    assert pipeline.llm_breaker._failed == 1


def test_a_refusal_of_the_request_itself_counts_as_an_answer() -> None:
    primary = Refusing(error=ollama.ResponseError("model 'x' not found", 404))
    pipeline = two_models(primary, StreamingFakeChatModel())
    with pytest.raises(ollama.ResponseError):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert pipeline.llm_breaker._failed == 0  # Ollama is up: nothing to route around


def test_with_the_model_s_breaker_open_the_fallback_answers_and_the_model_is_not_asked() -> None:
    primary = Refusing()
    spare = StreamingFakeChatModel(tokens=["Spare."])
    pipeline = two_models(primary, spare, llm_breaker=opened("Ollama"))
    *_, done = asyncio.run(collect(stream_answer(pipeline, "q")))
    assert done.model == SPARE
    assert primary.attempts == 0


def test_with_both_breakers_open_the_answer_is_refused_and_no_model_is_asked() -> None:
    primary, spare = Refusing(), StreamingFakeChatModel()
    pipeline = two_models(
        primary,
        spare,
        llm_breaker=opened("Ollama"),
        fallback_breaker=opened("the hosted fallback"),
    )
    with pytest.raises(BreakerOpenError, match="the hosted fallback is not answering"):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert primary.attempts == 0
    assert spare.log == []


def test_a_fallback_that_refuses_too_is_the_answer_s_error() -> None:
    primary = Refusing()
    spare = Refusing(error=openai_error("APIConnectionError")("Connection error."))
    pipeline = two_models(primary, spare)
    with pytest.raises(Exception, match="Connection error"):
        asyncio.run(collect(stream_answer(pipeline, "q")))
    assert spare.attempts == FALLBACK_ATTEMPTS  # never retried: max_retries 0, one attempt
    assert pipeline.fallback_breaker._failed == 1


def test_a_cancelled_probe_frees_the_probe() -> None:
    now = [1000.0]
    breaker = opened("Ollama", clock=lambda: now[0])
    now[0] += breaker.reset_after_s  # half-open: the next attempt is the probe
    primary = StreamingFakeChatModel(prefill_s=5.0)
    pipeline = two_models(primary, None, llm_breaker=breaker)
    assert asyncio.run(
        closed_when_cancelled(stream_answer(pipeline, "q"), primary, MOMENTS["prefill"][1])
    )
    assert breaker.state == "half-open"
    breaker.allow().succeeded()  # another probe may go; the cancel judged nothing


def test_the_fallback_streams_in_the_consuming_task() -> None:
    # No new task near the stream (DEC-14): a cancel of the consumer must reach its read.
    seen: list[asyncio.Task[Any] | None] = []

    class Spy(StreamingFakeChatModel):
        async def _astream(self, *args: Any, **kw: Any) -> AsyncIterator[ChatGenerationChunk]:
            seen.append(asyncio.current_task())
            async for chunk in super()._astream(*args, **kw):
                yield chunk

    async def consume() -> asyncio.Task[Any] | None:
        await collect(stream_answer(two_models(Refusing(), Spy()), "q"))
        return asyncio.current_task()

    consumer = asyncio.run(consume())
    assert seen == [consumer]


@pytest.mark.parametrize("at", [0, 1, 3, 5])
def test_a_cancel_while_a_fallback_chunk_is_processed_closes_it_before_returning(at: int) -> None:
    primary = Refusing()
    spare = CancelsAtAChunk(tokens=TOKENS, at=at)
    events = stream_answer(two_models(primary, spare), "q")
    assert asyncio.run(closed_on_return(events, spare)), (
        f"the consumer returned with the fallback's stream still open: {spare.log}"
    )
    assert spare.log == ["start", *TOKENS[: at + 1], "closed"]  # nothing after the cancel
    assert primary.log == ["start", "closed"] * LLM_ATTEMPTS


def test_a_cancel_during_the_fallback_s_prefill_closes_it_before_returning() -> None:
    spare = StreamingFakeChatModel(prefill_s=5.0)
    events = stream_answer(two_models(Refusing(), spare), "q")
    assert asyncio.run(closed_when_cancelled(events, spare, MOMENTS["prefill"][1]))


def test_a_cancel_during_the_back_off_starts_no_other_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("rag_qa.resilience.backoff_s", lambda attempt: 5.0)
    primary = Refusing()
    spare = StreamingFakeChatModel()
    events = stream_answer(two_models(primary, spare), "q")
    # Ready once the first attempt's stream has closed: the answer then waits out its back-off.
    assert asyncio.run(closed_when_cancelled(events, primary, lambda m: m.closed))
    assert primary.log == ["start", "closed"]
    assert spare.log == []


def test_a_cancel_as_the_fallback_starts_closes_its_stream_before_returning() -> None:
    # The switch between the two models has no await of its own: the first place a cancel
    # can land is the fallback's own read, which closes as any model's does.
    class CancelsAtStart(StreamingFakeChatModel):
        consumer: Any = None

        async def _astream(self, *args: Any, **kw: Any) -> AsyncIterator[ChatGenerationChunk]:
            self.consumer.cancel()
            async for chunk in super()._astream(*args, **kw):
                yield chunk

    spare = CancelsAtStart()
    events = stream_answer(two_models(Refusing(), spare), "q")

    async def run() -> bool:
        async def consume() -> None:
            async for _ in events:
                pass

        spare.consumer = asyncio.create_task(consume())
        with pytest.raises(asyncio.CancelledError):
            await spare.consumer
        return spare.closed

    assert asyncio.run(run())
    assert spare.log == ["start", "closed"]


def test_a_cancel_after_the_fallback_s_first_token_judges_nothing_twice() -> None:
    breaker = CircuitBreaker("the hosted fallback")
    spare = CancelsAtAChunk(tokens=TOKENS, at=2)
    pipeline = two_models(Refusing(), spare, fallback_breaker=breaker)
    assert asyncio.run(closed_on_return(stream_answer(pipeline, "q"), spare))
    assert breaker._failed == 0  # its first token said it answered; the cancel adds nothing


# --- the real ChatOllama, refused, refusing and dropped (S3-6) ------------------------------


def dead_url() -> str:
    """A local port nothing listens on, so a connection there is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


@contextmanager
def ollama_answering(chat: Callable[[BaseHTTPRequestHandler], None]) -> Iterator[str]:
    """A local server whose ``/api/chat`` is ``chat``: the real client reads what it sends."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # as Ollama: a body ends by its length or last chunk

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            chat(self)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def real_ollama(base_url: str) -> ChatOllama:
    return ChatOllama(
        model="mistral",
        base_url=base_url,
        client_kwargs={"timeout": httpx.Timeout(connect=5, read=10, write=5, pool=5)},
    )


def answered(pipeline: QueryPipeline) -> list[Any]:
    async def run() -> list[Any]:
        try:
            return await collect(stream_answer(pipeline, "q"))
        finally:
            await pipeline.aclose()

    return asyncio.run(run())


def test_the_real_client_refused_is_tried_once_more_then_the_fallback_answers() -> None:
    primary = real_ollama(dead_url())
    pipeline = two_models(primary, StreamingFakeChatModel(tokens=["Spare."]))
    *_, done = answered(pipeline)
    assert done.model == SPARE
    assert pipeline.llm_breaker._failed == LLM_ATTEMPTS


def test_the_real_client_answered_500_is_the_answer_s_error() -> None:
    def busy(handler: BaseHTTPRequestHandler) -> None:
        body = json.dumps({"error": "model requires more system memory"}).encode()
        handler.send_response(500)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    spare = StreamingFakeChatModel()
    with ollama_answering(busy) as url:
        pipeline = two_models(real_ollama(url), spare)
        with pytest.raises(ollama.ResponseError) as exc:
            answered(pipeline)
    assert exc.value.status_code == 500
    assert spare.log == []
    assert pipeline.llm_breaker._failed == 1


def test_the_real_client_dropped_after_a_token_is_not_routed() -> None:
    def one_token_then_gone(handler: BaseHTTPRequestHandler) -> None:
        line = {
            "model": "mistral",
            "message": {"role": "assistant", "content": "Station "},
            "done": False,
        }
        data = (json.dumps(line) + "\n").encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")  # one chunk
        handler.wfile.flush()
        # The runner died mid-answer: the connection closes with no last chunk.
        handler.connection.shutdown(socket.SHUT_RDWR)

    spare = StreamingFakeChatModel()
    events: list[Any] = []
    with ollama_answering(one_token_then_gone) as url:
        pipeline = two_models(real_ollama(url), spare)

        async def run() -> None:
            try:
                async for event in stream_answer(pipeline, "q"):
                    events.append(event)
            finally:
                await pipeline.aclose()

        with pytest.raises(httpx.HTTPError):
            asyncio.run(run())
    assert [type(event) for event in events] == [Sources, Token]
    assert spare.log == []
