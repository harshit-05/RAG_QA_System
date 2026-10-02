"""The query chain's contract (ARCHITECTURE.md §0.4, S0-5), checked hermetically.

``fake_rag`` gives a real index built by the real ingest, and these tests build the
real chain with ``build_rag_chain``. Only the embedder and the chat model are fakes,
so the retrieval and prompt plumbing under test is the production code.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from conftest import FAKE_ANSWER, MakeConfig, StreamingFakeChatModel, use_fakes
from langchain_core.documents import Document

from rag_qa.answering import stream_answer
from rag_qa.chain import (
    NoIndexError,
    build_query_pipeline,
    build_rag_chain,
    citation,
    format_docs,
)
from rag_qa.config import load_config
from rag_qa.settings import ENV_VECTOR_STORE_PATH


def test_invoke_returns_question_context_and_answer(fake_rag: Path) -> None:
    result = build_rag_chain(load_config(fake_rag)).invoke({"question": "How many staff?"})

    assert sorted(result) == ["answer", "context", "question"]
    assert result["question"] == "How many staff?"
    assert result["answer"] == FAKE_ANSWER
    assert 0 < len(result["context"]) <= 5  # retriever k
    assert all(isinstance(doc, Document) for doc in result["context"])


def test_stream_sends_context_before_the_first_answer_token(fake_rag: Path) -> None:
    # The order a streaming client relies on (S0-5): citations can be shown before
    # the answer starts. Phase 2's SSE endpoint emits `sources` from this chunk.
    chunks = list(build_rag_chain(load_config(fake_rag)).stream({"question": "Staff?"}))
    kinds = [next(iter(chunk)) for chunk in chunks]

    assert kinds[0] == "question"
    assert kinds[1] == "context"
    assert set(kinds[2:]) == {"answer"} and len(kinds[2:]) > 1  # token by token
    assert "".join(chunk["answer"] for chunk in chunks[2:]) == FAKE_ANSWER


def test_build_does_not_touch_the_config(fake_rag: Path) -> None:
    # Phase 0's chain wrote a live retriever object into the config dict. The config
    # is frozen now; this checks that building leaves it equal, not just unassigned.
    config = load_config(fake_rag)
    before = config.model_dump()
    build_rag_chain(config)
    assert config.model_dump() == before


def test_format_docs_numbers_chunks_the_way_the_cli_lists_sources() -> None:
    docs = [
        Document(
            page_content="Preface.",
            metadata={"source": "/c/guide.pdf", "page": 0, "page_label": "i"},
        ),
        Document(page_content="Notes.", metadata={"source": "/c/notes.txt"}),
    ]
    assert format_docs(docs) == "[1] (guide.pdf, p. i)\nPreface.\n\n[2] (notes.txt)\nNotes."


def test_citation_falls_back_to_page_plus_one_without_a_label() -> None:
    doc = Document(page_content="x", metadata={"source": "/c/old.pdf", "page": 4})
    assert citation(doc) == "old.pdf, p. 5"


# --- the query pipeline (S2-1, ARCHITECTURE.md §2.4) -------------------------------------


def test_no_index_fails_before_any_model_is_built(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "no-index"))
    config = load_config(make_config(use_fakes))

    def no_model(config: Any) -> None:
        raise AssertionError("an LLM was built for a pipeline that cannot answer")

    monkeypatch.setattr("rag_qa.chain.build_llm", no_model)
    with pytest.raises(NoIndexError, match="Run rag-ingest first"):
        build_query_pipeline(config)


def test_without_require_index_the_llm_half_is_built_anyway(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The API's start before its first ingest (S2-7).
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "no-index"))
    pipeline = build_query_pipeline(load_config(make_config(use_fakes)), require_index=False)
    assert pipeline.retrieve is None
    assert pipeline.answer.invoke({"context": "c", "question": "q"}) == FAKE_ANSWER


@pytest.mark.parametrize("require_index", [True, False])
def test_with_an_index_both_halves_are_built(fake_rag: Path, require_index: bool) -> None:
    pipeline = build_query_pipeline(load_config(fake_rag), require_index=require_index)
    assert pipeline.retrieve is not None
    assert 0 < len(pipeline.retrieve.invoke("How many staff?")) <= 5  # retriever k
    assert pipeline.answer.invoke({"context": "c", "question": "q"}) == FAKE_ANSWER


def test_build_rag_chain_and_stream_answer_give_the_model_the_same_prompt(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two compositions of the same parts must not drift apart, or the CLI and the API
    # (stream_answer) would answer differently from the invoke path (build_rag_chain).
    models: list[StreamingFakeChatModel] = []

    def recording_llm(config: Any) -> StreamingFakeChatModel:
        models.append(StreamingFakeChatModel())
        return models[-1]

    monkeypatch.setattr("rag_qa.chain.build_llm", recording_llm)
    config = load_config(fake_rag)
    build_rag_chain(config).invoke({"question": "How many staff?"})

    async def answer() -> None:
        async for _ in stream_answer(build_query_pipeline(config), "How many staff?"):
            pass

    asyncio.run(answer())
    via_chain, via_stream = (model.prompts for model in models)
    assert len(via_chain) == 1
    assert via_chain == via_stream
