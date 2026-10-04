"""The shared assembly layer (S1-3): one way to build each component."""

import asyncio
from itertools import pairwise
from pathlib import Path

import pytest
from conftest import (
    RERANK_TOP_N,
    MakeConfig,
    StubCrossEncoder,
    use_fakes,
    use_fakes_and_reranker,
)
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_ollama import ChatOllama

from rag_qa.components import (
    aclose_llm,
    build_embedder,
    build_loader,
    build_reranker,
    build_splitter,
)
from rag_qa.config import load_config
from rag_qa.loaders import DocxLoader, PdfLoader, TextLoader
from rag_qa.rerankers import CrossEncoderReranker


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.pdf", PdfLoader),
        ("b.docx", DocxLoader),
        ("c.txt", TextLoader),
        ("d.md", TextLoader),
        ("E.PDF", PdfLoader),  # matched on the lowercased suffix
    ],
)
def test_build_loader_follows_the_extension_map(
    make_config: MakeConfig, tmp_path: Path, filename: str, expected: type
) -> None:
    loader = build_loader(load_config(make_config()), tmp_path / filename)
    assert type(loader) is expected
    assert loader.file_path == str(tmp_path / filename)


@pytest.mark.parametrize("filename", ["image.png", "old.doc", "no_suffix", "archive.tar.gz"])
def test_unmapped_files_get_no_loader(
    make_config: MakeConfig, tmp_path: Path, filename: str
) -> None:
    assert build_loader(load_config(make_config()), tmp_path / filename) is None


def test_loaders_are_built_through_the_allowlist(make_config: MakeConfig, tmp_path: Path) -> None:
    # The old ingest path called import_from_string directly; now loaders go through
    # build_object like everything else. A non-class target shows the checks apply.
    config = load_config(make_config(lambda c: c["components"]["loaders"]["pdf"].update(
        _target_="rag_qa.registry.build_object"
    )))
    with pytest.raises(ImportError, match="not a class"):
        build_loader(config, tmp_path / "a.pdf")


def test_build_embedder_always_reads_the_ingestion_embedder(make_config: MakeConfig) -> None:
    def use_fake_embedder(c: dict) -> None:
        c["components"]["embedders"]["fake"] = {
            "_target_": "langchain_core.embeddings.DeterministicFakeEmbedding", "size": 8
        }
        c["pipeline"]["ingestion"]["embedder"] = "components.embedders.fake"

    embedder = build_embedder(load_config(make_config(use_fake_embedder)))
    assert isinstance(embedder, DeterministicFakeEmbedding)
    assert len(embedder.embed_query("same model for index and query")) == 8


def test_build_splitter_uses_the_configured_splitter(make_config: MakeConfig) -> None:
    # Behaviour, not private fields: config.yaml says chunk_size 1000, overlap 150.
    splitter = build_splitter(load_config(make_config()))
    assert type(splitter).__name__ == "RecursiveCharacterTextSplitter"

    words = " ".join(f"word{i:04d}" for i in range(600))  # ~5,400 characters
    chunks = splitter.split_text(words)
    assert len(chunks) > 1
    # Words are 9 characters with their space, so the splitter lands within one word
    # of each bound. Both bounds, so a smaller chunk_size or overlap fails too.
    assert all(900 < len(chunk) <= 1000 for chunk in chunks[:-1])
    # Each chunk starts with the tail of the previous one: a shared run of text close
    # to the configured 150 characters.
    for left, right in pairwise(chunks):
        shared = max((n for n in range(1, len(right) + 1) if left.endswith(right[:n])), default=0)
        assert 100 < shared <= 150


def test_build_reranker_is_none_without_a_reranker(make_config: MakeConfig) -> None:
    assert build_reranker(load_config(make_config(use_fakes))) is None


def test_build_reranker_builds_the_configured_entry(
    make_config: MakeConfig, stub_cross_encoder: type[StubCrossEncoder]
) -> None:
    # Through build_object, like every component: the settings come from the config entry.
    reranker = build_reranker(load_config(make_config(use_fakes_and_reranker)))
    assert isinstance(reranker, CrossEncoderReranker)
    assert (reranker.top_n, reranker.device, reranker.min_score) == (RERANK_TOP_N, "cpu", None)
    assert [(m.model_name_or_path, m.device) for m in stub_cross_encoder.built] == [
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cpu")
    ]


def test_aclose_llm_closes_chat_ollamas_http_clients() -> None:
    # No server needed: ChatOllama creates both clients at construction, and only
    # validate_model_on_init would contact Ollama. The httpx clients sit under ollama's.
    llm = ChatOllama(model="unused", validate_model_on_init=False)
    clients = [llm._client._client, llm._async_client._client]
    assert not any(client.is_closed for client in clients)
    asyncio.run(aclose_llm(llm))
    assert all(client.is_closed for client in clients)


def test_aclose_llm_is_a_no_op_for_a_model_without_clients() -> None:
    asyncio.run(aclose_llm(FakeListChatModel(responses=["x"])))
