"""Shared fixtures. The suite is hermetic: no network, no Ollama, no model download.

Binary fixtures (PDF, DOCX) are generated at test time rather than committed —
CLAUDE.md: never commit binaries — and the generators are small enough to read.

**Qdrant runs in process** (DEC-19, ARCHITECTURE.md §3.5). Every test gets one
``QdrantClient(":memory:")``, the :func:`qdrant` fixture, which every store the test
builds connects to: ingestion and the query side see the same data, as they would on one
server. Tests that need the real server carry the ``qdrant_server`` marker and are skipped
unless ``RAG_TEST_QDRANT_URL`` names one (CI's eval-retrieval job does).
"""

import asyncio
import copy
import json
import os
import threading
import zipfile
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from qdrant_client import QdrantClient

from rag_qa.settings import ENV_CONFIG, ENV_DATA_PATH, ENV_STATE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.yaml"

#: The collection the real config names, and so every test's store unless it says otherwise.
COLLECTION = "rag_qa"

#: The server the ``qdrant_server`` tests run against. Read here, at import, because the
#: autouse ``isolate_env`` fixture clears every ``RAG_*`` variable before a test runs.
QDRANT_TEST_URL = os.environ.get("RAG_TEST_QDRANT_URL") or None

MakeConfig = Callable[..., Path]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the ``qdrant_server`` tests unless ``RAG_TEST_QDRANT_URL`` names a server."""
    if QDRANT_TEST_URL is not None:
        return
    skip = pytest.mark.skip(reason="needs a Qdrant server: set RAG_TEST_QDRANT_URL")
    for item in items:
        if item.get_closest_marker("qdrant_server"):
            item.add_marker(skip)


def in_memory_client() -> QdrantClient:
    """A local-mode client whose ``close`` does nothing, so one store closing it (an ingest
    closes the store it built) leaves it open for the next."""
    client = QdrantClient(":memory:")
    client.close = lambda **kwargs: None  # type: ignore[method-assign]
    return client


@pytest.fixture(autouse=True)
def qdrant(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> QdrantClient:
    """The test's Qdrant: every store it builds connects to this one in-memory client.

    Autouse, so no test reaches a real server by accident. The ``qdrant_server`` tests
    connect to the real one themselves, and are left unpatched.
    """
    client = in_memory_client()
    if request.node.get_closest_marker("qdrant_server") is None:
        monkeypatch.setattr("rag_qa.vectorstore.connect", lambda *args, **kwargs: client)
    return client


def write_pdf(path: Path, texts: list[str], page_labels: str | None = None) -> Path:
    """A minimal valid PDF: one line of Helvetica text per page.

    ``page_labels`` is the body of the catalog's ``/PageLabels /Nums`` array, e.g.
    ``"0 << /S /r >> 2 << /S /D >>"`` labels pages i, ii, 1 — the case where the
    printed label and ``page + 1`` disagree, which is what citations depend on.
    Hand-built because pypdf can only add objects through a private API.
    """
    first_page = 4  # objects 1-3 are catalog, page tree, font; then page/content pairs
    page_refs = " ".join(f"{first_page + 2 * i} 0 R" for i in range(len(texts)))
    labels = f" /PageLabels << /Nums [{page_labels}] >>" if page_labels else ""
    bodies = [
        f"<< /Type /Catalog /Pages 2 0 R{labels} >>",
        f"<< /Type /Pages /Kids [{page_refs}] /Count {len(texts)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(texts):
        content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        bodies.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {first_page + 2 * i + 1} 0 R >>"
        )
        bodies.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref_at = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def write_docx(path: Path, paragraphs: list[str]) -> Path:
    """A minimal .docx: the three parts docx2txt and Word need, one run per paragraph."""
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    parts = {
        "[Content_Types].xml": (
            f'{xml}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/'
            'vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
        ),
        "_rels/.rels": (
            f'{xml}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": (
            f'{xml}<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>"
        ),
    }
    with zipfile.ZipFile(path, "w") as docx:
        for name, content in parts.items():
            docx.writestr(name, content)
    return path


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """A small nested corpus covering every loader, recursion and the skip path::

        corpus/guide.pdf               3 pages, labelled i, ii, 1
        corpus/notes.txt
        corpus/README.md
        corpus/diagram.png             no loader mapped -> skipped
        corpus/sub/report.docx         one level down
        corpus/sub/deeper/LOUD.TXT     two levels down, uppercase suffix

    and files a real working folder accumulates, which discovery must ignore::

        corpus/.ipynb_checkpoints/notes-checkpoint.txt   hidden folder (a duplicate)
        corpus/.draft.md                                 hidden file
        corpus/sub/~$report.docx                         Word's lock file, not a docx
    """
    root = tmp_path / "corpus"
    (root / "sub" / "deeper").mkdir(parents=True)
    write_pdf(
        root / "guide.pdf",
        ["Preface text", "Contents text", "Chapter one text"],
        page_labels="0 << /S /r >> 2 << /S /D >>",
    )
    (root / "notes.txt").write_text("Plain text notes.\n", encoding="utf-8")
    (root / "README.md").write_text("# Heading\n\nMarkdown body.\n", encoding="utf-8")
    (root / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    write_docx(root / "sub" / "report.docx", ["Report title", "Report body"])
    (root / "sub" / "deeper" / "LOUD.TXT").write_text("Shouted text.\n", encoding="utf-8")

    (root / ".ipynb_checkpoints").mkdir()
    (root / ".ipynb_checkpoints" / "notes-checkpoint.txt").write_text(
        "Plain text notes.\n", encoding="utf-8"
    )
    (root / ".draft.md").write_text("Not ready.\n", encoding="utf-8")
    (root / "sub" / "~$report.docx").write_bytes(b"\x00" * 162)  # what Word writes
    return root


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test waits between retries (S3-6): the back-off is random seconds, which would make
    a failing call slow and a timing flaky. ``tests/test_resilience.py`` checks the real one."""
    monkeypatch.setattr("rag_qa.resilience.backoff_s", lambda attempt: 0.0)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A developer's exported RAG_* variables must not leak into any test: the paths, since
    S2-7 the server's RAG_API_* settings, a token included, and since S3-2 the store's
    RAG_QDRANT_* ones, a key included. ``paths.state`` goes to the test's own folder, so
    no test takes the repository's writer lock."""
    for name in {ENV_CONFIG, ENV_DATA_PATH, *os.environ}:
        if name.startswith("RAG_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ENV_STATE_PATH, str(tmp_path / "var"))


@pytest.fixture
def make_config(tmp_path: Path) -> MakeConfig:
    """Write a scratch config and return its path.

    By default it is the repo's real ``config.yaml``, optionally changed by ``edit``
    (a function mutating the loaded dict), so every malformed case is one small change
    to the real file rather than a hand-written config that drifts from it. ``text``
    writes raw content instead, for cases YAML can't express as a dict.
    """

    def make(
        edit: Callable[[dict[str, Any]], None] | None = None, *, text: str | None = None
    ) -> Path:
        path = tmp_path / "config.yaml"
        if text is not None:
            path.write_text(text)
            return path
        data = yaml.safe_load(REAL_CONFIG.read_text())
        if edit is not None:
            edit(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return make


# --- fakes at the model boundary: everything else in the pipeline is real ----------------

#: Allowlisted classes defined in langchain_core (S1-2 checks pass), so a config using
#: them goes through the same build path as the real components.
FAKE_EMBEDDER = {"_target_": "langchain_core.embeddings.DeterministicFakeEmbedding", "size": 8}
FAKE_ANSWER = "Station Kestrel houses forty staff."
FAKE_LLM = {
    "_target_": "langchain_core.language_models.fake_chat_models.FakeListChatModel",
    "responses": [FAKE_ANSWER],
}


class StreamingFakeChatModel(BaseChatModel):
    """A chat model fake that streams slowly and records how its stream ended.

    Waits ``prefill_s`` (a simulated prompt evaluation), then yields ``tokens``
    ``gap_s`` apart. ``log`` gets ``"start"``, each token as it is yielded, and
    ``"closed"`` from a ``finally``: the stand-in for ChatOllama's HTTP response being
    closed, which is what cancelling an answer must achieve (DEC-14). ``prompts`` holds
    every prompt it was given, rendered as text.
    """

    tokens: list[str] = Field(default_factory=lambda: ["Station ", "Kestrel ", "staff."])
    prefill_s: float = 0.0
    gap_s: float = 0.0
    prompts: list[str] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    @property
    def closed(self) -> bool:
        return "closed" in self.log

    def _record(self, messages: list[BaseMessage]) -> None:
        self.prompts.append("\n".join(f"{m.type}: {m.content}" for m in messages))

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self._record(messages)
        message = AIMessage(content="".join(self.tokens))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._record(messages)
        self.log.append("start")
        try:
            await asyncio.sleep(self.prefill_s)
            for token in self.tokens:
                self.log.append(token)
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
                await asyncio.sleep(self.gap_s)
        finally:
            self.log.append("closed")


def openai_error(name: str, *parents: str, status_code: int | None = None) -> type[Exception]:
    """A stand-in for one of openai's errors: ``name``, under ``parents`` (nearest first),
    each defined "in" the ``openai`` package, as the real ones say they are. That is all
    :func:`rag_qa.resilience.classify` reads, so the hosted fallback's errors can be tested
    without the ``fallback`` extra, which CI does not install. With ``status_code``, its
    instances carry one, as openai's ``APIStatusError`` does (S3-6)."""
    base: type[Exception] = Exception
    for parent in reversed(parents):
        base = type(parent, (base,), {"__module__": "openai"})
    attrs: dict[str, Any] = {"__module__": "openai"}
    if status_code is not None:
        attrs["status_code"] = status_code
    return type(name, (base,), attrs)


class StubCrossEncoder:
    """``sentence_transformers.CrossEncoder`` without the download (S2-4).

    ``predict`` scores each (question, text) pair as ``scores[text]``, or as the text's
    length when the table does not name it, and returns them as float32, as the real model
    does. Every stub built is kept in ``built``, and each one records in ``seen`` the pairs
    it was asked to score. The ``stub_cross_encoder`` fixture resets both.
    """

    num_labels = 1
    built: ClassVar[list["StubCrossEncoder"]] = []
    scores: ClassVar[dict[str, float]] = {}

    def __init__(self, model_name_or_path: str, *, device: str | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.seen: list[list[tuple[str, str]]] = []
        self.built.append(self)

    def score_of(self, text: str) -> float:
        return self.scores.get(text, float(len(text)))

    def predict(self, inputs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
        self.seen.append(list(inputs))
        return np.array([self.score_of(text) for _, text in inputs], dtype=np.float32)


@pytest.fixture
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> type[StubCrossEncoder]:
    """Build every reranker on :class:`StubCrossEncoder`, so none downloads its model."""
    monkeypatch.setattr(StubCrossEncoder, "built", [])
    monkeypatch.setattr(StubCrossEncoder, "scores", {})
    monkeypatch.setattr("rag_qa.rerankers.CrossEncoder", StubCrossEncoder)
    return StubCrossEncoder


def use_fake_embedder(c: dict[str, Any]) -> None:
    """Config edit: a deterministic 8-dim embedder, so nothing downloads MiniLM.

    A copy each time: a later edit that changes the entry must not change it for every
    test after it.
    """
    c["components"]["embedders"]["fake"] = copy.deepcopy(FAKE_EMBEDDER)
    c["pipeline"]["ingestion"]["embedder"] = "components.embedders.fake"


def use_fakes(c: dict[str, Any]) -> None:
    """Config edit: fake embedder and a fake chat model, and no reranker.

    So nothing needs Ollama or downloads a model: the reranker's cross-encoder is a
    download too. Tests of the reranker switch it on with ``use_fakes_and_reranker``.
    """
    use_fake_embedder(c)
    c["components"]["llms"]["fake"] = copy.deepcopy(FAKE_LLM)
    c["pipeline"]["query"]["llm"] = "components.llms.fake"
    c["pipeline"]["query"].pop("reranker", None)
    c["pipeline"]["query"].pop("reranker_candidates", None)


#: ``use_fakes_and_reranker``'s settings. sample_corpus indexes as 7 chunks and the
#: retriever's own k is 5, so a reranker seeing 6 and keeping 2 shows both settings at work.
RERANK_CANDIDATES = 6
RERANK_TOP_N = 2


def use_fakes_and_reranker(c: dict[str, Any]) -> None:
    """Config edit: ``use_fakes``, with the real config's CPU reranker switched on.

    Build it only under ``stub_cross_encoder``, which stands in for its model.
    """
    use_fakes(c)
    c["components"]["rerankers"]["ms_marco_minilm_cpu"]["top_n"] = RERANK_TOP_N
    c["pipeline"]["query"]["reranker"] = "components.rerankers.ms_marco_minilm_cpu"
    c["pipeline"]["query"]["reranker_candidates"] = RERANK_CANDIDATES


def live_generation(client: QdrantClient, collection: str = COLLECTION) -> str:
    """The generation the alias names (DEC-19); fails the test when there is none."""
    targets = [
        a.collection_name for a in client.get_aliases().aliases if a.alias_name == collection
    ]
    assert targets, f"no alias {collection!r}"
    return targets[0]


def published(client: QdrantClient, collection: str = COLLECTION) -> bool:
    """Whether an index is published under ``collection``: its alias exists."""
    return any(a.alias_name == collection for a in client.get_aliases().aliases)


def generations(client: QdrantClient, collection: str = COLLECTION) -> list[str]:
    """Every generation of ``collection`` on the client, oldest first."""
    prefix = f"{collection}--gen-"
    return sorted(c.name for c in client.get_collections().collections if c.name.startswith(prefix))


def metadata(client: QdrantClient, generation: str) -> dict[str, Any]:
    """A generation's metadata: its manifest under ``manifest``, and ``complete``."""
    return dict(client.get_collection(generation).config.metadata or {})


def edit_manifest(
    client: QdrantClient, edit: Callable[[dict[str, Any]], None], collection: str = COLLECTION
) -> None:
    """Rewrite the live generation's manifest through ``edit``, whole, as tampering would."""
    generation = live_generation(client, collection)
    manifest = copy.deepcopy(metadata(client, generation)["manifest"])
    edit(manifest)
    client.update_collection(generation, metadata={"manifest": manifest})


@pytest.fixture
def fake_rag(make_config: MakeConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hermetic RAG setup: the real ingest built an index of ``sample_corpus`` with
    the fakes, in the test's in-memory Qdrant. Returns the config path; the corpus and
    ``paths.state`` are scratch folders."""
    from rag_qa.config import load_config
    from rag_qa.ingest import ingest

    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    path = make_config(use_fakes)
    ingest(load_config(path))
    return path


# --- a fake Ollama, for the tier-2 commands (S2-5) ----------------------------------------

JUDGE_DIGEST = "a" * 64
BASE_DIGEST = "b" * 64
MISTRAL_DIGEST = "c" * 64
#: The chat template the fake judge inherits from its base, as gemma2's real one is.
FAKE_TEMPLATE = "<start_of_turn>user\n{{ .Prompt }}<end_of_turn>"


@dataclass
class FakeOllama:
    """Ollama's native API as the tier-2 commands read it: ``/api/tags``, ``/api/show`` and
    ``/api/ps``, served on a free local port so no test reaches the real daemon. Tests
    change its tables to make a judge go missing, a context differ or a digest move."""

    url: str = ""
    digests: dict[str, str] = field(
        default_factory=lambda: {
            "rag-judge:latest": JUDGE_DIGEST,
            "gemma2:9b": BASE_DIGEST,
            "mistral:latest": MISTRAL_DIGEST,
        }
    )
    #: /api/show's ``parameters`` per model, as Ollama formats them. A created model serves
    #: its base's (``stop``) with its Modelfile's added, as the real rag-judge does.
    parameters: dict[str, str] = field(
        default_factory=lambda: {
            "rag-judge:latest": "\n".join(
                f"{name:<30} {value}"
                for name, value in [
                    ("num_ctx", "8192"),
                    ("num_predict", "2048"),
                    ("seed", "42"),
                    ("stop", '"<end>"'),
                    ("temperature", "0"),
                ]
            ),
            "gemma2:9b": f'{"stop":<30} "<end>"',
        }
    )
    #: /api/show's ``details.parent_model``: what each created model was built FROM.
    parents: dict[str, str] = field(default_factory=lambda: {"rag-judge:latest": "gemma2:9b"})
    #: /api/show's ``system`` (absent when unset, as Ollama does) and ``template``.
    systems: dict[str, str] = field(default_factory=dict)
    templates: dict[str, str] = field(
        default_factory=lambda: {"rag-judge:latest": FAKE_TEMPLATE, "gemma2:9b": FAKE_TEMPLATE}
    )
    loaded: list[dict[str, Any]] = field(default_factory=list)  # /api/ps
    requests: list[str] = field(default_factory=list)


def _tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


@pytest.fixture
def fake_ollama() -> Iterator[FakeOllama]:
    state = FakeOllama()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            state.requests.append(self.path)
            if self.path == "/api/tags":
                models = [{"name": n, "model": n, "digest": d} for n, d in state.digests.items()]
                self._reply(200, {"models": models})
            elif self.path == "/api/ps":
                self._reply(200, {"models": state.loaded})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            state.requests.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = _tag(body.get("model", ""))
            if self.path == "/api/show" and name in state.parameters:
                details = {"parent_model": state.parents.get(name, "")}
                shown = {"parameters": state.parameters[name], "details": details}
                if name in state.systems:
                    shown["system"] = state.systems[name]
                if name in state.templates:
                    shown["template"] = state.templates[name]
                self._reply(200, shown)
            else:
                self._reply(404, {"error": f"model '{name}' not found"})

        def log_message(self, *args: Any) -> None:  # keep the test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


# --- the HTTP API (S2-7) --------------------------------------------------------------------


@pytest.fixture
def fake_config(make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fakes' config, with an empty corpus, in tmp, and an empty in-memory Qdrant.

    For a server over :func:`fake_serving`: with no index in the store, it never reopens one.
    """
    from rag_qa.config import load_config

    (tmp_path / "corpus").mkdir()
    monkeypatch.setenv(ENV_DATA_PATH, str(tmp_path / "corpus"))
    return load_config(make_config(use_fakes))


def fake_serving(
    config: Any, model: BaseChatModel, *, retrieve: Any = None, max_concurrent: int = 1
) -> Any:
    """What the API serves, over a fake pipeline: ``retrieve`` (by default, one fixed chunk)
    and ``model``. The chunk's ``source`` is an absolute path under the corpus, as a v0.2
    index stored it, so it must come out corpus-relative."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    from rag_qa.api.state import Served, Serving
    from rag_qa.chain import QueryPipeline
    from rag_qa.components import build_store

    docs = [
        Document(
            page_content="Station Kestrel.",
            metadata={"source": str(config.paths.data / "notes.txt")},
        )
    ]
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Answer from the context."), ("human", "{context}\n\n{question}")]
    )
    pipeline = QueryPipeline(
        retrieve=retrieve if retrieve is not None else RunnableLambda(lambda question: docs),
        prompt=prompt,
        llm=model,
        corpus_root=config.paths.data,
    )
    return Serving(
        config,
        pipeline,
        Served(generation=None, manifest=None, problem=None),
        store=build_store(config),
        embeddings=DeterministicFakeEmbedding(size=8),
        reranker=None,
        max_concurrent=max_concurrent,
    )


def sse_events(body: str) -> list[tuple[str, Any]]:
    """``(event, data)`` for each event in an SSE body; comments (``: ping``) skipped."""
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if lines:
            (name,) = [line.removeprefix("event: ") for line in lines if line.startswith("event: ")]
            data = "\n".join(
                line.removeprefix("data: ") for line in lines if line.startswith("data: ")
            )
            events.append((name, json.loads(data)))
    return events
