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
    generations,
    live_generation,
    use_fakes,
    use_fakes_and_reranker,
)
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from rag_qa import vectorstore
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
    open_retrieval,
)
from rag_qa.chain import live_generation as chain_live_generation
from rag_qa.components import build_embedder, build_reranker, build_store
from rag_qa.config import load_config
from rag_qa.ingest import ingest
from rag_qa.manifest import Manifest


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
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(make_config(use_fakes))  # the test's store holds no index

    def no_model(config: Any) -> None:
        raise AssertionError("an LLM was built for a pipeline that cannot answer")

    monkeypatch.setattr("rag_qa.chain.build_llm", no_model)
    with pytest.raises(NoIndexError, match="Run rag-ingest first"):
        build_query_pipeline(config)


def test_with_an_index_both_halves_are_built(fake_rag: Path) -> None:
    pipeline = build_query_pipeline(load_config(fake_rag))
    assert pipeline.retrieve is not None
    assert 0 < len(pipeline.retrieve.invoke("How many staff?")) <= 5  # retriever k
    assert answer_from(pipeline) == FAKE_ANSWER


def test_open_retrieval_names_the_generation_it_opened_and_its_manifest(
    fake_rag: Path, qdrant: QdrantClient
) -> None:
    # What the API serves, lists and reports (S2-7): the generation the alias named, with
    # that generation's own manifest.
    config = load_config(fake_rag)
    opened = open_retrieval(config, build_embedder(config))
    assert opened.generation == live_generation(qdrant)
    assert opened.generation == chain_live_generation(build_store(config))
    assert opened.manifest.generation == opened.generation
    assert "guide.pdf" in opened.manifest.documents
    assert opened.retrieve.invoke("How many staff?")


def test_without_an_index_there_is_no_live_generation(make_config: MakeConfig) -> None:
    assert chain_live_generation(build_store(load_config(make_config(use_fakes)))) is None


def test_a_reranker_passed_in_is_used_and_none_is_built(
    fake_rag: Path, make_config: MakeConfig, stub_cross_encoder: type[StubCrossEncoder]
) -> None:
    # The API builds its reranker once and reuses it on every reopen (S2-4: 2.3 s and
    # about 90 MB per build), so a reopen must never build another.
    config = load_config(make_config(use_fakes_and_reranker))
    reranker = build_reranker(config)
    embeddings = build_embedder(config)
    for _ in range(3):
        chunks = build_retrieve(config, embeddings, reranker=reranker).invoke("Staff?")
        assert len(chunks) == RERANK_TOP_N
    assert len(stub_cross_encoder.built) == 1


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


@pytest.mark.parametrize("what", ["a collection", "an alias of something else"])
def test_what_rag_ingest_will_not_touch_is_refused_without_sending_the_user_there(
    make_config: MakeConfig, qdrant: QdrantClient, no_models: None, what: str
) -> None:
    # "Re-run rag-ingest" would only meet rag-ingest's own exit 2 here (S2-6's review).
    qdrant.create_collection("theirs" if what != "a collection" else "rag_qa", vectors_config={})
    if what != "a collection":
        qdrant.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name="theirs", alias_name="rag_qa")
                )
            ]
        )
    with pytest.raises(IncompatibleIndexError) as refused:
        build_query_pipeline(load_config(make_config(use_fakes)))
    assert "Set RAG_QDRANT_COLLECTION" in str(refused.value)
    assert "Re-run rag-ingest" not in str(refused.value)


def test_an_index_without_a_manifest_is_refused(
    fake_rag: Path, qdrant: QdrantClient, no_models: None
) -> None:
    config = load_config(fake_rag)
    qdrant.update_collection(live_generation(qdrant), metadata={"manifest": None})
    with pytest.raises(IncompatibleIndexError, match="its manifest cannot be used: there is none"):
        build_query_pipeline(config)


def test_an_index_with_an_unusable_manifest_is_refused(
    fake_rag: Path, qdrant: QdrantClient, no_models: None
) -> None:
    config = load_config(fake_rag)
    edit_manifest(qdrant, lambda m: m.update(version=99))
    with pytest.raises(IncompatibleIndexError, match="unknown manifest version 99"):
        build_query_pipeline(config)


def test_a_live_collection_lost_names_the_complete_generation_to_point_the_alias_at(
    fake_rag: Path, sample_corpus: Path, qdrant: QdrantClient, no_models: None
) -> None:
    # Design review 2: deleting the alias's target deletes the alias. The previous
    # generation survives (recovery keeps complete ones), and "no index" names it.
    config = load_config(fake_rag)
    previous = live_generation(qdrant)
    (sample_corpus / "late.txt").write_text("A second generation.\n")
    ingest(config)
    qdrant.delete_collection(live_generation(qdrant))
    with pytest.raises(NoIndexError) as refused:
        build_query_pipeline(config)
    message = str(refused.value)
    assert f"its generation {previous} is complete" in message
    assert 'README, "Pointing the alias at a generation"' in message
    assert not isinstance(refused.value, IncompatibleIndexError)


def test_an_index_from_another_embedder_is_refused_before_any_model_is_built(
    fake_rag: Path, make_config: MakeConfig, no_models: None
) -> None:
    with pytest.raises(IncompatibleIndexError, match=r"different embedder.*--rebuild"):
        build_query_pipeline(load_config(make_config(other_embedder)))


def test_build_rag_chain_refuses_an_unusable_index_before_any_model_is_built(
    make_config: MakeConfig, no_models: None
) -> None:
    # The invoke path (the RAGAs harness) built the LLM and the embedder first (S2-6's
    # second review): with Ollama down, the Ollama error hid the real one.
    with pytest.raises(NoIndexError, match="Run rag-ingest first"):
        build_rag_chain(load_config(make_config(use_fakes)))


@pytest.fixture
def no_llm_or_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_model(*args: Any) -> None:
        raise AssertionError("built the LLM or the reranker before refusing the index")

    monkeypatch.setattr("rag_qa.chain.build_llm", no_model)
    monkeypatch.setattr("rag_qa.chain.build_reranker", no_model)


def test_a_dimension_mismatch_is_refused_before_the_llm(
    fake_rag: Path, qdrant: QdrantClient, no_llm_or_reranker: None
) -> None:
    # It needs the embedder, but not Ollama: retrieval is built before the LLM.
    config = load_config(fake_rag)
    edit_manifest(qdrant, lambda m: m["embedder"].update(dimension=9))
    with pytest.raises(IncompatibleIndexError, match="9-dimension"):
        build_query_pipeline(config)


def test_an_incomplete_live_generation_is_refused_not_served(
    fake_rag: Path, qdrant: QdrantClient, no_llm_or_reranker: None
) -> None:
    # The alias only ever names a complete generation; one pointed at another by hand is
    # refused with the reason, rather than answered from (DEC-19).
    config = load_config(fake_rag)
    qdrant.update_collection(live_generation(qdrant), metadata={"complete": False})
    with pytest.raises(IncompatibleIndexError, match="it is incomplete"):
        build_query_pipeline(config)


def test_a_dimension_the_index_does_not_hold_is_refused_before_the_reranker_loads(
    fake_rag: Path, qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The identity's backstop: the same spec, but a model that now makes other vectors.
    config = load_config(fake_rag)
    edit_manifest(qdrant, lambda m: m["embedder"].update(dimension=9))

    def no_model(*args: Any) -> None:
        raise AssertionError("the reranker loaded before the dimension was checked")

    monkeypatch.setattr("rag_qa.chain.build_reranker", no_model)
    with pytest.raises(IncompatibleIndexError, match="9-dimension vectors.*makes 8-dimension"):
        build_retrieve(config, build_embedder(config))


# --- readers and the alias (DEC-19, story step 8) --------------------------------------------


def test_the_retrieval_half_resolves_the_alias_before_every_question(
    fake_rag: Path, sample_corpus: Path, qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rag-query before each question, rag-eval before each item: a session that outlives
    # an ingest searches the new generation, and one that outlives two still answers,
    # although the generation it started on is gone by then.
    config = load_config(fake_rag)
    retrieve = build_retrieve(config, build_embedder(config))
    searched: list[str] = []
    real = vectorstore.search

    def spy(store: Any, generation: str, vector: Any, k: int) -> Any:
        searched.append(generation)
        return real(store, generation, vector, k)

    monkeypatch.setattr(vectorstore, "search", spy)
    names = [live_generation(qdrant)]
    assert retrieve.invoke("Staff?")
    for name in ("penguins.txt", "terns.txt"):
        (sample_corpus / name).write_text(f"{name} nest on Kestrel Island.\n")
        ingest(config)
        names.append(live_generation(qdrant))
        assert retrieve.invoke("Staff?")
    assert names[0] not in generations(qdrant)  # pruned two switches later
    assert searched == names  # each question searched the generation live at the time


def test_a_question_reads_the_alias_once_and_opens_nothing_when_it_has_not_moved(
    fake_rag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(fake_rag)
    retrieve = build_retrieve(config, build_embedder(config))
    opens: list[str] = []
    real = open_retrieval

    def counting(*args: Any, **kwargs: Any) -> Any:
        opens.append("open")
        return real(*args, **kwargs)

    monkeypatch.setattr("rag_qa.chain.open_retrieval", counting)
    for _ in range(3):
        retrieve.invoke("Staff?")
    assert opens == []


def test_a_new_generation_is_heard_before_it_is_searched_and_may_be_refused(
    fake_rag: Path, sample_corpus: Path
) -> None:
    # rag-eval generate holds a generation it moves to against the corpus it records.
    config = load_config(fake_rag)
    heard: list[Manifest] = []
    retrieve = build_retrieve(config, build_embedder(config), on_new_generation=heard.append)
    retrieve.invoke("q")
    assert heard == []  # the first generation is the one it opened with
    (sample_corpus / "late.txt").write_text("Late.\n")
    ingest(config)
    retrieve.invoke("q")
    assert len(heard) == 1 and "late.txt" in heard[0].documents

    def refuse(manifest: Manifest) -> None:
        raise RuntimeError("not the corpus this run records")

    refusing = build_retrieve(config, build_embedder(config), on_new_generation=refuse)
    (sample_corpus / "later.txt").write_text("Later.\n")
    ingest(config)
    with pytest.raises(RuntimeError, match="not the corpus"):
        refusing.invoke("q")


def test_a_bound_half_whose_generation_is_gone_resolves_the_alias_once_more(
    fake_rag: Path, sample_corpus: Path, qdrant: QdrantClient
) -> None:
    # The API's case: its half is bound to the generation it served, and two switches can
    # delete that generation between the refresh and the search (design review 2).
    config = load_config(fake_rag)
    opened = open_retrieval(config, build_embedder(config))
    for name in ("one.txt", "two.txt"):
        (sample_corpus / name).write_text(f"{name}\n")
        ingest(config)
    assert opened.generation not in generations(qdrant)
    assert opened.retrieve.invoke("Staff?")  # from the live generation, not an error


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
