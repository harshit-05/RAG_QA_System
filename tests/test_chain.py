"""The query chain's contract (ARCHITECTURE.md §0.4, S0-5), checked hermetically.

``fake_rag`` gives a real index built by the real ingest, and these tests build the
real chain with ``build_rag_chain``. Only the embedder and the chat model are fakes,
so the retrieval and prompt plumbing under test is the production code.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FAKE_ANSWER,
    RERANK_CANDIDATES,
    RERANK_TOP_N,
    MakeConfig,
    StreamingFakeChatModel,
    StubCrossEncoder,
    edit_manifest,
    live_generation,
    use_fakes,
    use_fakes_and_reranker,
)
from langchain_core.documents import Document

from rag_qa.answering import Sources, stream_answer
from rag_qa.chain import (
    IncompatibleIndexError,
    NoIndexError,
    QueryPipeline,
    build_query_pipeline,
    build_rag_chain,
    build_retrieve,
    citation,
    format_docs,
)
from rag_qa.components import build_embedder
from rag_qa.config import load_config
from rag_qa.settings import ENV_VECTOR_STORE_PATH


def answer_from(pipeline: QueryPipeline) -> str:
    """The LLM half on its own: the prompt rendered, then the model invoked."""
    messages = pipeline.prompt.format_messages(context="c", question="q")
    return str(pipeline.llm.invoke(messages).text)


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
    assert answer_from(pipeline) == FAKE_ANSWER


@pytest.mark.parametrize("require_index", [True, False])
def test_with_an_index_both_halves_are_built(fake_rag: Path, require_index: bool) -> None:
    pipeline = build_query_pipeline(load_config(fake_rag), require_index=require_index)
    assert pipeline.retrieve is not None
    assert 0 < len(pipeline.retrieve.invoke("How many staff?")) <= 5  # retriever k
    assert answer_from(pipeline) == FAKE_ANSWER


# --- an index the config cannot use (S2-6, DEC-17) ---------------------------------------


@pytest.fixture
def no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building any model fails the test: refusals must come first. The mpnet download, or
    Ollama for Verification 4, would otherwise come before the error."""

    def no_model(*args: Any) -> None:
        raise AssertionError("a model was built for an index the config cannot use")

    for name in ("build_llm", "build_embedder", "build_reranker"):
        monkeypatch.setattr(f"rag_qa.chain.{name}", no_model)


def other_embedder(c: dict[str, Any]) -> None:
    """Config edit: the fakes, with an embedder whose vectors the fake_rag index lacks."""
    use_fakes(c)
    c["components"]["embedders"]["fake"]["size"] = 16


def test_a_v02_index_is_refused_before_any_model_is_built(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_models: None
) -> None:
    store = tmp_path / "index"  # v0.2's layout: a real folder, no manifest
    store.mkdir()
    (store / "index.faiss").write_bytes(b"")
    (store / "index.pkl").write_bytes(b"")
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(store))
    with pytest.raises(IncompatibleIndexError, match=r"built by v0\.2.*Re-run rag-ingest"):
        build_query_pipeline(load_config(make_config(use_fakes)))


def test_an_index_without_a_manifest_is_refused(fake_rag: Path, no_models: None) -> None:
    config = load_config(fake_rag)
    (live_generation(config.paths.vector_store) / "manifest.json").unlink()
    with pytest.raises(IncompatibleIndexError, match="has no manifest"):
        build_query_pipeline(config)


def test_an_index_with_an_unusable_manifest_is_refused(fake_rag: Path, no_models: None) -> None:
    config = load_config(fake_rag)
    (live_generation(config.paths.vector_store) / "manifest.json").write_text("{oops")
    with pytest.raises(IncompatibleIndexError, match="cannot be used: it is not valid JSON"):
        build_query_pipeline(config)


def test_an_index_from_another_embedder_is_refused_before_any_model_is_built(
    fake_rag: Path, make_config: MakeConfig, no_models: None
) -> None:
    with pytest.raises(IncompatibleIndexError, match=r"different embedder.*--rebuild"):
        build_query_pipeline(load_config(make_config(other_embedder)))


def test_without_require_index_an_unusable_index_counts_as_none(
    fake_rag: Path, make_config: MakeConfig
) -> None:
    # The API starts anyway, and reports not-ready (DEC-18).
    pipeline = build_query_pipeline(load_config(make_config(other_embedder)), require_index=False)
    assert pipeline.retrieve is None


def test_build_rag_chain_refuses_an_unusable_index_before_any_model_is_built(
    make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_models: None
) -> None:
    # The invoke path (the RAGAs harness) built the LLM and the embedder first (second
    # review): with Ollama down, the Ollama error hid the real one.
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "no-index"))
    with pytest.raises(NoIndexError, match="Run rag-ingest first"):
        build_rag_chain(load_config(make_config(use_fakes)))


@pytest.fixture
def no_llm_or_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model(*args: Any) -> None:
        raise AssertionError("built the LLM or the reranker before refusing the index")

    monkeypatch.setattr("rag_qa.chain.build_llm", no_model)
    monkeypatch.setattr("rag_qa.chain.build_reranker", no_model)


def test_a_dimension_mismatch_is_refused_before_the_llm(
    fake_rag: Path, no_llm_or_reranker: None
) -> None:
    # It needs the embedder, but not Ollama: retrieval is built before the LLM.
    config = load_config(fake_rag)
    edit_manifest(config.paths.vector_store, lambda m: m["embedder"].update(dimension=9))
    with pytest.raises(IncompatibleIndexError, match="9-dimension"):
        build_query_pipeline(config)


def test_an_index_that_will_not_open_is_refused_not_a_traceback(
    fake_rag: Path, no_llm_or_reranker: None
) -> None:
    # A corrupt pickle crashed rag-query with UnpicklingError (second review).
    config = load_config(fake_rag)
    (live_generation(config.paths.vector_store) / "index.pkl").write_bytes(b"not a pickle")
    with pytest.raises(IncompatibleIndexError, match=r"cannot open the index.*UnpicklingError"):
        build_query_pipeline(config)


def test_without_require_index_a_dimension_mismatch_counts_as_none(fake_rag: Path) -> None:
    # The dimension is checked once the embedder is built, after the LLM: without
    # require_index, that refusal must still leave retrieve as None (first review).
    config = load_config(fake_rag)
    edit_manifest(config.paths.vector_store, lambda m: m["embedder"].update(dimension=9))
    pipeline = build_query_pipeline(config, require_index=False)
    assert pipeline.retrieve is None
    assert answer_from(pipeline) == FAKE_ANSWER


def test_a_dimension_the_index_does_not_hold_is_refused_before_the_reranker_loads(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The identity's backstop: the same spec, but a model that now makes other vectors.
    config = load_config(fake_rag)
    edit_manifest(config.paths.vector_store, lambda m: m["embedder"].update(dimension=9))

    def no_model(*args: Any) -> None:
        raise AssertionError("the reranker loaded before the dimension was checked")

    monkeypatch.setattr("rag_qa.chain.build_reranker", no_model)
    with pytest.raises(IncompatibleIndexError, match="9-dimension vectors.*makes 8-dimension"):
        build_retrieve(config, build_embedder(config))


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


# --- the reranker in the retrieval half (S2-4, DEC-16) ------------------------------------


def best_first(model: StubCrossEncoder, pairs: list[tuple[str, str]]) -> list[str]:
    """The chunk texts the stub was given, by its scores, ties in the retriever's order."""
    return [text for _, text in sorted(pairs, key=lambda p: model.score_of(p[1]), reverse=True)]


def test_with_a_reranker_the_retriever_fetches_reranker_candidates(
    fake_rag: Path, make_config: MakeConfig, stub_cross_encoder: type[StubCrossEncoder]
) -> None:
    config = load_config(make_config(use_fakes_and_reranker))
    chunks = build_retrieve(config, build_embedder(config)).invoke("How many staff?")

    (model,) = stub_cross_encoder.built
    (pairs,) = model.seen
    assert len(pairs) == RERANK_CANDIDATES  # not the retriever's own k of 5
    assert {question for question, _ in pairs} == {"How many staff?"}
    assert [chunk.page_content for chunk in chunks] == best_first(model, pairs)[:RERANK_TOP_N]
    assert all(type(chunk.metadata["rerank_score"]) is float for chunk in chunks)
    # The override went to a copy: the config still holds the retriever's own k.
    retriever = config.retriever(config.pipeline.query.retriever)
    assert retriever.kwargs()["search_kwargs"]["k"] == 5


def test_without_a_reranker_the_retriever_keeps_its_own_k(
    fake_rag: Path, stub_cross_encoder: type[StubCrossEncoder]
) -> None:
    config = load_config(fake_rag)  # use_fakes: no reranker
    chunks = build_retrieve(config, build_embedder(config)).invoke("How many staff?")
    assert len(chunks) == 5
    assert stub_cross_encoder.built == []
    assert not any("rerank_score" in chunk.metadata for chunk in chunks)


def test_sources_carry_the_rerank_score_in_reranked_order(
    fake_rag: Path, make_config: MakeConfig, stub_cross_encoder: type[StubCrossEncoder]
) -> None:
    # The async path every front end takes: stream_answer awaits retrieve.ainvoke.
    pipeline = build_query_pipeline(load_config(make_config(use_fakes_and_reranker)))

    async def first_event() -> Sources:
        async for event in stream_answer(pipeline, "How many staff?"):
            assert isinstance(event, Sources)
            return event
        raise AssertionError("no events")

    sources = asyncio.run(first_event()).sources
    (model,) = stub_cross_encoder.built
    (pairs,) = model.seen
    expected = best_first(model, pairs)[:RERANK_TOP_N]
    assert [ref.n for ref in sources] == [1, 2]
    assert [ref.score for ref in sources] == [model.score_of(text) for text in expected]
