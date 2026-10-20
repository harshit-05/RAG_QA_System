"""The shared assembly layer (S1-3): one way to build each component."""

import asyncio
import socket
import time
from itertools import pairwise
from pathlib import Path

import pytest
from conftest import (
    RERANK_TOP_N,
    FakeOllama,
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
    build_store,
    llm_reachable,
)
from rag_qa.config import ConfigError, load_config
from rag_qa.loaders import DocxLoader, PdfLoader, TextLoader
from rag_qa.rerankers import CrossEncoderReranker
from rag_qa.vectorstore import QdrantStore


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
    config = load_config(
        make_config(
            lambda c: c["components"]["loaders"]["pdf"].update(
                _target_="rag_qa.registry.build_object"
            )
        )
    )
    with pytest.raises(ImportError, match="not a class"):
        build_loader(config, tmp_path / "a.pdf")


def test_build_embedder_always_reads_the_ingestion_embedder(make_config: MakeConfig) -> None:
    def use_fake_embedder(c: dict) -> None:
        c["components"]["embedders"]["fake"] = {
            "_target_": "langchain_core.embeddings.DeterministicFakeEmbedding",
            "size": 8,
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


def markdown_for_md(c: dict) -> None:
    c["pipeline"]["ingestion"]["splitters"] = {".md": "components.splitters.markdown"}


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (".md", "MarkdownTextSplitter"),  # its own (S3-4)
        (".pdf", "RecursiveCharacterTextSplitter"),  # the default
        (None, "RecursiveCharacterTextSplitter"),  # no extension: the default
    ],
)
def test_build_splitter_follows_the_per_extension_map(
    make_config: MakeConfig, extension: str | None, expected: str
) -> None:
    splitter = build_splitter(load_config(make_config(markdown_for_md)), extension)
    assert type(splitter).__name__ == expected


def test_the_markdown_splitter_starts_a_chunk_at_a_heading(make_config: MakeConfig) -> None:
    # Its heading separator is a pattern, "\n#{1,6} ". Without is_separator_regex, which
    # langchain-text-splitters 1.1.2 leaves off, it is matched literally and never splits
    # a real heading: "## Second" then rides at the end of the first chunk, as the default
    # splitter leaves it.
    paragraph = " ".join(f"word{i:04d}" for i in range(60))  # 539 characters, no newline
    text = f"# First\n{paragraph}\n## Second\n{paragraph}"
    config = load_config(make_config(markdown_for_md))
    chunks = build_splitter(config, ".md").split_text(text)
    assert [chunk.split("\n")[0] for chunk in chunks] == ["# First", "## Second"]
    default = build_splitter(config).split_text(text)
    assert default[0].endswith("## Second")  # the case the setting is there for


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


# --- llm_reachable: the API's readiness check and rag-serve's start-up check (S2-7) ---------


def test_llm_reachable_asks_ollama_for_its_models(fake_ollama: FakeOllama) -> None:
    llm = ChatOllama(model="mistral", base_url=fake_ollama.url)
    assert asyncio.run(llm_reachable(llm)) is True
    assert fake_ollama.requests == ["/api/tags"]


def test_llm_reachable_is_false_when_nothing_listens() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    assert asyncio.run(llm_reachable(ChatOllama(model="mistral", base_url=url))) is False


def test_llm_reachable_is_false_when_the_server_does_not_answer_in_time() -> None:
    # A socket that accepts connections (the kernel's backlog) and never replies.
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        llm = ChatOllama(model="mistral", base_url=f"http://127.0.0.1:{silent.getsockname()[1]}")
        started = time.monotonic()
        assert asyncio.run(llm_reachable(llm, timeout_s=0.3)) is False
        assert time.monotonic() - started < 2


def test_llm_reachable_is_none_for_a_model_with_no_server() -> None:
    assert asyncio.run(llm_reachable(FakeListChatModel(responses=["x"]))) is None


# --- the vector store (S3-2, DEC-19) ----------------------------------------------------------


def test_the_store_is_built_from_pipeline_vector_store(make_config: MakeConfig) -> None:
    store = build_store(load_config(make_config()))
    assert isinstance(store, QdrantStore)
    assert (store.url, store.collection, store.timeout_s) == ("http://localhost:6333", "rag_qa", 10)


def test_an_api_key_in_the_config_is_refused_as_a_build_error(make_config: MakeConfig) -> None:
    # The key comes from RAG_QDRANT_API_KEY only (§9): the store takes no such kwarg.
    path = make_config(lambda c: c["components"]["vector_stores"]["qdrant"].update(api_key="k"))
    with pytest.raises(ConfigError, match="Cannot build the vector store .*'api_key'"):
        build_store(load_config(path))


def test_a_bad_collection_set_after_load_is_still_refused(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A long-lived process reads the environment when it builds the store, not only at load.
    config = load_config(make_config())
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "Not-Allowed")
    with pytest.raises(ConfigError, match="RAG_QDRANT_COLLECTION: collection 'Not-Allowed'"):
        build_store(config)


def test_a_target_that_is_not_a_store_is_refused(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_qa.components.build_object", lambda spec: object())
    with pytest.raises(ConfigError, match="builds a object, not a store"):
        build_store(load_config(make_config()))
