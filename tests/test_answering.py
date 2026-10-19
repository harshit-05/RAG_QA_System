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
from collections.abc import AsyncIterator, Callable
from operator import itemgetter
from pathlib import Path
from typing import Any

import pytest
from conftest import StreamingFakeChatModel
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

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
        1, "old.pdf, p. 5", "old.pdf", "5", None
    )
    assert (second.n, second.citation, second.source, second.page, second.score) == (
        2, "x.txt", "x.txt", None, 0.75
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
    return (
        RunnablePassthrough.assign(context=itemgetter("question") | pipeline.retrieve)
        | RunnablePassthrough.assign(answer=to_prompt_inputs | sequence(pipeline))
    )


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
