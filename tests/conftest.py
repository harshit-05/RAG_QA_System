"""Shared fixtures. The suite is hermetic: no network, no Ollama, no model download.

Binary fixtures (PDF, DOCX) are generated at test time rather than committed —
CLAUDE.md: never commit binaries — and the generators are small enough to read.
"""

import asyncio
import copy
import json
import os
import zipfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from rag_qa.settings import ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.yaml"

MakeConfig = Callable[..., Path]


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
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported RAG_* variables must not leak into any test."""
    for name in (ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH):
        monkeypatch.delenv(name, raising=False)


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


def live_generation(store: Path) -> Path:
    """The generation folder an index's symlink points to (DEC-17)."""
    return store.parent / os.readlink(store)


def edit_manifest(store: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the live generation's ``manifest.json`` through ``edit``, as tampering would."""
    path = live_generation(store) / "manifest.json"
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.fixture
def fake_rag(
    make_config: MakeConfig, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A hermetic RAG setup: the real ingest built an index of ``sample_corpus`` with
    the fakes. Returns the config path; both data and index paths point at scratch."""
    from rag_qa.config import load_config
    from rag_qa.ingest import ingest

    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "index"))
    path = make_config(use_fakes)
    ingest(load_config(path))
    return path
