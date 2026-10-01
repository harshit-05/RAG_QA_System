"""The query chain's contract (ARCHITECTURE.md §0.4, S0-5), checked hermetically.

``fake_rag`` gives a real index built by the real ingest, and these tests build the
real chain with ``build_rag_chain``. Only the embedder and the chat model are fakes,
so the retrieval and prompt plumbing under test is the production code.
"""

from pathlib import Path

from conftest import FAKE_ANSWER
from langchain_core.documents import Document

from rag_qa.chain import build_rag_chain, citation, format_docs
from rag_qa.config import load_config


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
