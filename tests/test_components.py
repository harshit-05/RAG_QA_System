"""The shared assembly layer (S1-3): one way to build each component."""

import asyncio
import socket
import sys
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import (
    REAL_CONFIG,
    RERANK_TOP_N,
    FakeOllama,
    MakeConfig,
    StubCrossEncoder,
    openai_error,
    use_fakes,
    use_fakes_and_reranker,
)
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_ollama import ChatOllama

from rag_qa.components import (
    OLLAMA_HINT,
    Preflight,
    aclose_llm,
    build_embedder,
    build_fallback_llm,
    build_llm,
    build_loader,
    build_reranker,
    build_sparse_encoder,
    build_splitter,
    build_store,
    configured_sparse,
    llm_reachable,
    preflight,
    raised_by_fallback,
    service_of,
)
from rag_qa.config import ConfigError, load_config
from rag_qa.loaders import DocxLoader, PdfLoader, TextLoader
from rag_qa.manifest import SparseRecord, spec_identity
from rag_qa.rerankers import CrossEncoderReranker
from rag_qa.sparse import SPARSE_VERSION, Bm25Encoder
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


def test_an_api_key_in_the_store_entry_is_refused_at_load(make_config: MakeConfig) -> None:
    # The key comes from RAG_QDRANT_API_KEY only (§9). Until S3-6 the store's missing kwarg
    # refused it at build; now any component key named like an API key is refused at load,
    # naming where and the variables, never the value.
    path = make_config(
        lambda c: c["components"]["vector_stores"]["qdrant"].update(api_key="sk-sentinel")
    )
    with pytest.raises(ConfigError, match="vector_stores.qdrant.api_key: an API key") as exc:
        load_config(path)
    assert "RAG_QDRANT_API_KEY" in str(exc.value)
    assert "sk-sentinel" not in str(exc.value)


def test_a_bad_collection_set_after_load_is_still_refused(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A long-lived process reads the environment when it builds the store, not only at load.
    config = load_config(make_config())
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "Not-Allowed")
    with pytest.raises(ConfigError, match="RAG_QDRANT_COLLECTION: collection 'Not-Allowed'"):
        build_store(config)


# --- the sparse encoder (S3-5, DEC-20) -------------------------------------------------------

BM25 = "components.sparse_encoders.bm25"


def with_bm25(c: dict[str, Any]) -> None:
    c["pipeline"]["ingestion"]["sparse_encoder"] = BM25


def test_no_sparse_encoder_is_built_or_recorded_while_none_is_set(make_config: MakeConfig) -> None:
    config = load_config(make_config())  # the real config: set by nothing yet
    assert build_sparse_encoder(config) is None
    assert configured_sparse(config) is None


def test_the_sparse_encoder_is_the_ingestion_one_with_its_record(make_config: MakeConfig) -> None:
    config = load_config(make_config(with_bm25))
    encoder = build_sparse_encoder(config)
    assert isinstance(encoder, Bm25Encoder)
    assert (encoder.k1, encoder.b, encoder.avg_len) == (1.2, 0.75, 100.76)
    assert configured_sparse(config) == SparseRecord(
        ref=BM25, identity=spec_identity(config.component(BM25).spec()), version=SPARSE_VERSION
    )


@pytest.mark.parametrize(
    ("settings", "problem"),
    [
        ({"avg_len": 0}, "avg_len must be a finite number above 0"),
        ({"k1": "1.2"}, "k1 must be a number"),
        ({"stemming": True}, "unexpected keyword argument 'stemming'"),
    ],
)
def test_a_sparse_encoder_that_cannot_be_built_is_a_config_error(
    make_config: MakeConfig, settings: dict[str, Any], problem: str
) -> None:
    def edit(c: dict[str, Any]) -> None:
        with_bm25(c)
        c["components"]["sparse_encoders"]["bm25"].update(settings)

    with pytest.raises(ConfigError, match=f"Cannot build the sparse encoder {BM25}: .*{problem}"):
        build_sparse_encoder(load_config(make_config(edit)))


def test_a_target_that_is_not_a_sparse_encoder_is_refused(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(make_config(with_bm25))
    monkeypatch.setattr("rag_qa.components.build_object", lambda spec: object())
    with pytest.raises(ConfigError, match="builds a object, not a sparse encoder"):
        build_sparse_encoder(config)


def test_a_target_that_is_not_a_store_is_refused(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_qa.components.build_object", lambda spec: object())
    with pytest.raises(ConfigError, match="builds a object, not a store"):
        build_store(load_config(make_config()))


# --- timeouts and the hosted fallback (S3-6, DEC-21) ----------------------------------------


def closed_port() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def test_an_ollama_model_s_timeout_phases_reach_both_of_its_http_clients() -> None:
    # YAML cannot spell an httpx.Timeout; build_llm makes one from the phases (D7). Built
    # without validation, so no Ollama is asked.
    llm = build_llm(load_config(REAL_CONFIG), validate=False)
    assert isinstance(llm, ChatOllama)
    expected = httpx.Timeout(connect=5, read=600, write=30, pool=30)
    assert llm._async_client._client.timeout == expected
    assert llm._client._client.timeout == expected
    asyncio.run(aclose_llm(llm))


def test_built_without_validation_a_model_asks_its_server_nothing(make_config: MakeConfig) -> None:
    # A pipeline with a fallback builds the model so: Ollama down is the fallback's case.
    def edit(c: dict[str, Any]) -> None:
        c["components"]["llms"]["mistral_ollama"]["base_url"] = closed_port()

    config = load_config(make_config(edit))
    # validate_model_on_init asks it. ollama's client wraps httpx's ConnectError in the
    # builtin ConnectionError, which langchain-ollama 1.1.0 then lets through.
    with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
        build_llm(config)
    asyncio.run(aclose_llm(build_llm(config, validate=False)))  # and this does not


def test_a_number_as_a_timeout_is_every_phase(make_config: MakeConfig) -> None:
    def edit(c: dict[str, Any]) -> None:
        c["components"]["llms"]["mistral_ollama"]["client_kwargs"] = {"timeout": 42}

    llm = build_llm(load_config(make_config(edit)), validate=False)
    assert llm._async_client._client.timeout == httpx.Timeout(42)  # type: ignore[attr-defined]
    asyncio.run(aclose_llm(llm))


def test_the_service_behind_the_model_is_named_for_its_breaker(make_config: MakeConfig) -> None:
    assert service_of(load_config(REAL_CONFIG)) == "Ollama"
    assert service_of(load_config(make_config(use_fakes))) == "the language model"


def with_fallback(c: dict[str, Any]) -> None:
    c["pipeline"]["query"]["fallback_llm"] = "components.llms.groq_fallback"


def test_no_fallback_configured_builds_none() -> None:
    assert build_fallback_llm(load_config(REAL_CONFIG)) is None


def test_a_fallback_without_its_key_is_refused_naming_the_variable(make_config: MakeConfig) -> None:
    with pytest.raises(ConfigError) as exc:
        build_fallback_llm(load_config(make_config(with_fallback)))
    assert str(exc.value).startswith(
        "the hosted fallback components.llms.groq_fallback needs its API key in "
        "RAG_FALLBACK_API_KEY, which is not set."
    )


def test_a_fallback_without_its_extra_says_how_to_install_it(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_FALLBACK_API_KEY", "gsk-sentinel")
    monkeypatch.setitem(sys.modules, "openai", None)  # as without `uv sync --extra fallback`
    with pytest.raises(ConfigError, match="install it with `uv sync --extra fallback`") as exc:
        build_fallback_llm(load_config(make_config(with_fallback)))
    assert "gsk-sentinel" not in str(exc.value)


def ollama_with_fallback(base_url: str) -> Any:
    def edit(c: dict[str, Any]) -> None:
        c["components"]["llms"]["mistral_ollama"]["base_url"] = base_url
        with_fallback(c)

    return edit


def test_preflight_with_a_fallback_warns_about_ollama_down_and_lets_the_start_go_on(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    spare = FakeListChatModel(responses=["x"])
    monkeypatch.setattr("rag_qa.components.build_fallback_llm", lambda config: spare)
    check = preflight(load_config(make_config(ollama_with_fallback(closed_port()))))
    assert check.problem is None
    assert check.warning is not None
    assert check.warning.startswith(
        "the language model components.llms.mistral_ollama did not answer within 2 s. Until "
        "it does, the hosted fallback components.llms.groq_fallback answers, and each "
        "question and its retrieved chunks are sent to https://api.groq.com/openai/v1."
    )
    assert OLLAMA_HINT in check.warning


def test_preflight_without_the_fallback_s_key_refuses_the_start(make_config: MakeConfig) -> None:
    check = preflight(load_config(make_config(ollama_with_fallback(closed_port()))))
    assert check.problem is not None and "RAG_FALLBACK_API_KEY" in check.problem
    assert check.warning is None


def test_preflight_for_tier_2_ignores_the_fallback_and_refuses_ollama_down(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_qa.components.build_fallback_llm", lambda config: 1 / 0)
    check = preflight(load_config(make_config(ollama_with_fallback(closed_port()))), fallback=False)
    assert check.problem is not None
    assert check.problem.startswith(
        "cannot build the language model components.llms.mistral_ollama"
    )
    assert check.warning is None


def test_preflight_with_ollama_up_finds_nothing(
    make_config: MakeConfig, fake_ollama: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_qa.components.build_fallback_llm", lambda config: None)
    assert preflight(load_config(make_config(ollama_with_fallback(fake_ollama.url)))) == Preflight()


@pytest.mark.parametrize(
    ("error", "fallback"),
    [
        (openai_error("APIConnectionError")("x"), True),
        (type("OpenAITimeoutError", (openai_error("APITimeoutError"),), {})("x"), True),
        (httpx.ConnectError("refused"), False),
        (ConnectionRefusedError("refused"), False),
        (ValueError("x"), False),
    ],
)
def test_an_error_is_the_fallback_s_when_openai_raised_it(error: Exception, fallback: bool) -> None:
    assert raised_by_fallback(error) is fallback


def test_a_fallback_whose_class_will_not_import_says_how_to_install_it(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # openai present (the eval extra's), langchain_openai not: the allowlist's import fails.
    monkeypatch.setenv("RAG_FALLBACK_API_KEY", "gsk-sentinel")

    def no_langchain_openai(spec: Any) -> None:
        raise ImportError("Could not import 'langchain_openai.ChatOpenAI': No module named it")

    monkeypatch.setattr("rag_qa.components._build_model", no_langchain_openai)
    with pytest.raises(ConfigError, match="It needs the fallback extra: install it with"):
        build_fallback_llm(load_config(make_config(with_fallback)))


def test_preflight_reports_a_fallback_that_fails_to_build_as_the_reason(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(config: Any) -> None:
        raise RuntimeError("bad entry")

    monkeypatch.setattr("rag_qa.components.build_fallback_llm", broken)
    check = preflight(load_config(make_config(ollama_with_fallback(closed_port()))))
    assert check == Preflight(
        problem="cannot build the hosted fallback components.llms.groq_fallback: "
        "RuntimeError: bad entry"
    )


def test_preflight_closes_the_fallback_when_the_model_will_not_build(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[Any] = []
    spare = FakeListChatModel(responses=["x"])

    async def record(model: Any) -> None:
        closed.append(model)

    def broken(config: Any, **kwargs: Any) -> None:
        raise RuntimeError("bad model")

    monkeypatch.setattr("rag_qa.components.build_fallback_llm", lambda config: spare)
    monkeypatch.setattr("rag_qa.components.build_llm", broken)
    monkeypatch.setattr("rag_qa.components.aclose_llm", record)
    check = preflight(load_config(make_config(ollama_with_fallback(closed_port()))))
    assert check.problem is not None
    assert check.problem.startswith(
        "cannot build the language model components.llms.mistral_ollama: RuntimeError: bad model"
    )
    assert closed == [spare]  # built, never used, and not left open
