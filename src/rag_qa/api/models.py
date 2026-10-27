"""What the HTTP API accepts and returns (ARCHITECTURE.md §2.4).

**A request never supplies a component** (DEC-18). Both request bodies are closed:
``extra="forbid"`` turns a body that names a component (``"llm": …``) into a 422, so
nothing a client sends reaches :func:`rag_qa.registry.build_object`. They are strict as
well, so ``"rebuild": "yes"`` is refused instead of read as true.

Responses carry corpus-relative paths and generation names only, never a host path.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

#: The longest question accepted, in characters.
MAX_QUESTION_CHARS = 2000
#: The most documents one question may be limited to (S3-5, DEC-20).
MAX_SOURCES = 50
#: The longest document path accepted, in characters: far beyond any corpus path, and a
#: bound on what a 422 repeats back.
MAX_SOURCE_CHARS = 1024

JobStatus = Literal["queued", "running", "succeeded", "failed"]


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class QueryRequest(_Request):
    """``POST /v1/query``: the question, and optionally the documents to answer from."""

    #: Surrounding whitespace is dropped, as the CLI drops it, so a blank question is empty.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]
    #: The corpus documents to answer from, by their paths relative to the corpus, as
    #: ``GET /v1/documents`` lists them (S3-5, FR-3); none means the whole corpus. Each must
    #: be in the served index, or the request is refused with 422 naming the unknown ones.
    #: Matched exactly, so not stripped: they are names, not prose.
    sources: list[Annotated[str, StringConstraints(min_length=1, max_length=MAX_SOURCE_CHARS)]] = (
        Field(default_factory=list, max_length=MAX_SOURCES)
    )


class IngestRequest(_Request):
    """``POST /v1/ingest``. ``rebuild`` is required: a POST with no JSON body must fail,
    because a JSON content type is what a web page cannot send cross-site without a
    preflight (DEC-18)."""

    rebuild: bool


class IndexHealth(BaseModel):
    ready: bool
    generation: str | None  # the index generation served, e.g. rag_qa--gen-…
    documents: int | None
    detail: str


class LlmHealth(BaseModel):
    ready: bool
    detail: str


class BreakerHealth(BaseModel):
    """One dependency's circuit breaker (DEC-21, S3-6). ``retry_after_s`` is how long an
    open one waits before it lets a probe through, and ``null`` unless it is open."""

    state: Literal["closed", "open", "half-open"]
    retry_after_s: int | None


class KeysHealth(BaseModel):
    """The API keys file (DEC-23, S3-8): whether it can be used, and if not why, never its
    path."""

    ready: bool
    detail: str


class Health(BaseModel):
    """``GET /v1/health``: 200 when ready, 503 when not. ``keys`` is ``null`` unless a keys
    file is set. ``breakers`` holds ``qdrant``, ``llm`` and, with a hosted fallback
    configured, ``fallback``: named by role, never by address (DEC-18)."""

    status: Literal["ready", "not ready"]
    index: IndexHealth
    llm: LlmHealth
    keys: KeysHealth | None
    breakers: dict[str, BreakerHealth]


class DocumentInfo(BaseModel):
    source: str  # corpus-relative
    sha256: str
    chunks: int
    loader: str  # the config reference, e.g. components.loaders.pdf
    ingested_at: str


class Documents(BaseModel):
    """``GET /v1/documents``: what the served index generation holds, from its manifest."""

    documents: list[DocumentInfo]


class IngestReportOut(BaseModel):
    """:class:`rag_qa.ingest.IngestReport` as JSON. Its names are corpus-relative, and
    its errors path-free (S2-6)."""

    documents: int
    chunks: int
    found: int
    added: int
    changed: int
    removed: int
    unchanged: int
    kept: int
    rebuild: str | None
    published: str | None
    skipped: list[str]
    #: ``[name, size in bytes]`` for each file above ``max_file_mb``, the limit applied.
    too_large: list[tuple[str, int]]
    max_file_mb: float | None
    symlinks: list[str]
    ignored: int
    failed: list[tuple[str, str]]
    #: The run's phases, in seconds (NFR-3, S3-7): loading and splitting, embedding, and
    #: writing the new generation; and the points it holds.
    load_s: float
    embed_s: float
    write_s: float
    points: int


class IngestAccepted(BaseModel):
    """``POST /v1/ingest``'s 202."""

    job_id: str
    status: JobStatus


class IngestJobOut(BaseModel):
    """``GET /v1/ingest/{job_id}``. ``exit_code`` is what ``rag-ingest`` would have
    returned, and ``log`` the run's log lines, path-free. ``abandoned`` is true for a job whose
    process died before it ended (S3-8, DEC-23): its ``status`` is then ``failed``, and its
    ``exit_code`` and ``report`` are ``null``."""

    job_id: str
    status: JobStatus
    started_at: datetime | None
    finished_at: datetime | None
    exit_code: int | None
    report: IngestReportOut | None
    log: list[str]
    abandoned: bool
