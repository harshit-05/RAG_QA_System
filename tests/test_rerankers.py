"""Our own cross-encoder reranker (S2-4, DEC-16), with a stub in place of its model.

``stub_cross_encoder`` stands in for ``sentence_transformers.CrossEncoder``, so nothing
downloads (DEC-11), and its scores come from a table. What is ours is what these tests
check: the order, the ``top_n`` cut, ``min_score``, copies rather than mutation, and the
empty case.
"""

import json
from typing import Any

import pytest
from conftest import StubCrossEncoder
from langchain_core.documents import Document
from pydantic import ValidationError

from rag_qa.rerankers import CrossEncoderReranker


def chunks(*texts: str) -> list[Document]:
    """One chunk per text, in the retriever's order."""
    return [
        Document(page_content=text, metadata={"source": f"/corpus/{text}.txt", "page": n})
        for n, text in enumerate(texts)
    ]


def texts(docs: list[Document]) -> list[str]:
    return [doc.page_content for doc in docs]


def reranker(**settings: Any) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name="stub/model", **settings)


def test_orders_by_score_best_first(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    stub_cross_encoder.scores = {"a": 0.5, "b": 3.0, "c": -2.0, "d": 1.5}
    kept = reranker(top_n=4).compress_documents(chunks("a", "b", "c", "d"), "q")
    assert texts(kept) == ["b", "d", "a", "c"]
    assert [doc.metadata["rerank_score"] for doc in kept] == [3.0, 1.5, 0.5, -2.0]


def test_keeps_the_best_top_n(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    stub_cross_encoder.scores = {"a": 1.0, "b": 4.0, "c": 3.0, "d": 2.0}
    assert texts(reranker(top_n=2).compress_documents(chunks("a", "b", "c", "d"), "q")) == [
        "b",
        "c",
    ]


def test_equal_scores_keep_the_retrievers_order(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    stub_cross_encoder.scores = {"a": 1.0, "b": 2.0, "c": 1.0, "d": 2.0}
    kept = reranker(top_n=4).compress_documents(chunks("a", "b", "c", "d"), "q")
    assert texts(kept) == ["b", "d", "a", "c"]


@pytest.mark.parametrize(
    ("min_score", "expected"),
    [
        pytest.param(None, ["c", "a", "d", "b"], id="off"),
        pytest.param(0.0, ["c", "a", "d"], id="a chunk at the cut stays"),
        pytest.param(10.0, [], id="everything below"),
    ],
)
def test_min_score_drops_chunks_below_it(
    stub_cross_encoder: type[StubCrossEncoder], min_score: float | None, expected: list[str]
) -> None:
    stub_cross_encoder.scores = {"a": 0.5, "b": -1.0, "c": 2.0, "d": 0.0}
    kept = reranker(top_n=5, min_score=min_score).compress_documents(
        chunks("a", "b", "c", "d"), "q"
    )
    assert texts(kept) == expected


def test_returns_copies_and_never_modifies_the_chunks_given(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    # FAISS returns the objects its docstore holds: a write here would change the index.
    given = chunks("a", "b")
    before = [doc.model_dump() for doc in given]
    kept = reranker().compress_documents(given, "q")

    assert [doc.model_dump() for doc in given] == before
    assert not any(k is g or k.metadata is g.metadata for k in kept for g in given)
    # The copy keeps everything the citation needs, plus the score.
    by_text = {doc.page_content: doc for doc in kept}
    assert by_text["a"].metadata == {"source": "/corpus/a.txt", "page": 0, "rerank_score": 1.0}


def test_scores_are_python_floats(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    # The model returns numpy float32s; the score travels on to SourceRef and JSON.
    (kept,) = reranker().compress_documents(chunks("a"), "q")
    assert type(kept.metadata["rerank_score"]) is float
    json.dumps(kept.metadata)


def test_no_chunks_means_no_model_call(stub_cross_encoder: type[StubCrossEncoder]) -> None:
    assert reranker().compress_documents([], "q") == []
    assert stub_cross_encoder.built[0].seen == []


def test_the_model_is_loaded_once_at_construction(
    stub_cross_encoder: type[StubCrossEncoder],
) -> None:
    rerank = reranker(device="cpu")
    assert [(m.model_name_or_path, m.device) for m in stub_cross_encoder.built] == [
        ("stub/model", "cpu")
    ]

    rerank.compress_documents(chunks("a", "b"), "first?")
    rerank.compress_documents(chunks("c"), "second?")
    (model,) = stub_cross_encoder.built  # still the one model
    assert model.seen == [[("first?", "a"), ("first?", "b")], [("second?", "c")]]


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        # BaseDocumentCompressor alone would ignore the typo and keep top_n at 5.
        pytest.param({"topn": 3}, "topn", id="a misspelled setting"),
        pytest.param({"top_n": 0}, "greater than or equal to 1", id="top_n of 0"),
    ],
)
def test_a_bad_setting_is_refused_before_the_model_loads(
    stub_cross_encoder: type[StubCrossEncoder], settings: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValidationError, match=problem):
        reranker(**settings)
    assert stub_cross_encoder.built == []


def test_a_model_with_several_scores_per_pair_is_refused(
    stub_cross_encoder: type[StubCrossEncoder], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A classification cross-encoder (an NLI one, say) gives one score per label.
    monkeypatch.setattr(stub_cross_encoder, "num_labels", 3)
    with pytest.raises(ValueError, match="num_labels=1"):
        reranker()
