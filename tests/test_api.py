"""The HTTP API (S2-7, DEC-18), checked hermetically with FastAPI's test client.

Two kinds of server are used:

- **a fake pipeline** (:func:`fake_serving`): a retrieval half that returns fixed chunks,
  and a streaming fake model. It serves the SSE protocol, the refusals and the slot.
- **the real build** over the fakes in the config (``use_fakes``), used for the index, the
  ingest jobs, the documents and the health checks. Everything but the models is
  production code there.

**Every response either kind gives is scanned** for the test's temporary folder, which
holds the corpus and the index. No host path may leave the server (DEC-18).

The test client cannot drop a connection mid-response. Disconnects run under real uvicorn,
in ``test_api_uvicorn.py``.
"""

import asyncio
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import (
    FAKE_ANSWER,
    FakeOllama,
    MakeConfig,
    StreamingFakeChatModel,
    edit_manifest,
    fake_serving,
    sse_events,
    use_fakes,
)
from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import ValidationError
from qdrant_client import QdrantClient
from starlette.datastructures import State
from starlette.requests import Request

from rag_qa.api import jobs as api_jobs
from rag_qa.api import routes
from rag_qa.api import state as api_state
from rag_qa.api.app import create_app
from rag_qa.api.jobs import KEEP
from rag_qa.api.models import IngestReportOut, QueryRequest
from rag_qa.api.state import NO_INDEX_YET, REINGEST, Served, Serving, serialized
from rag_qa.chain import IncompatibleIndexError, NoIndexError, RetrievalQuery
from rag_qa.components import build_store
from rag_qa.config import load_config
from rag_qa.ingest import CorpusUnreadableError, IngestionRunningError, IngestReport, ingest
from rag_qa.manifest import STORE, DocumentRecord, EmbedderRecord, Manifest, SplitterRecord
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ServerSettings
from rag_qa.vectorstore import StoreUnavailableError

TOKEN = "s3cret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
#: The sample corpus's files that a loader is mapped to.
CORPUS_FILES = ["README.md", "guide.pdf", "notes.txt", "sub/deeper/LOUD.TXT", "sub/report.docx"]


# --- harness ------------------------------------------------------------------------------


@contextmanager
def serve(
    config: RagConfig,
    private: Path,
    serving: Serving | None = None,
    *,
    token: str | None = None,
) -> Iterator[tuple[TestClient, Serving]]:
    """A running app, its lifespan included, and a client that fails any response that
    names ``private`` (the folder holding the corpus and the index)."""
    app = create_app(config, ServerSettings(token=token), serving=serving)

    def scan(response: httpx.Response) -> None:
        response.read()
        assert str(private) not in response.text, f"a host path left the server: {response.text}"

    with TestClient(app, base_url="http://127.0.0.1") as client:
        client.event_hooks["response"] = [scan]
        yield client, app.state.serving


@pytest.fixture
def scratch_config(
    make_config: MakeConfig, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RagConfig:
    """The fakes' config over the sample corpus, with no index yet in the test's store."""
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    return load_config(make_config(use_fakes))


def wait_for_job(client: TestClient, job_id: str, timeout_s: float = 30) -> dict[str, Any]:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        job = client.get(f"/v1/ingest/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return dict(job)
        time.sleep(0.02)
    raise AssertionError(f"ingest job {job_id} did not finish within {timeout_s} s")


def run_job(client: TestClient, *, rebuild: bool = False) -> dict[str, Any]:
    """Start an ingest job and wait for it to end."""
    started = client.post("/v1/ingest", json={"rebuild": rebuild})
    assert started.status_code == 202, started.text
    return wait_for_job(client, started.json()["job_id"])


def refused(client: TestClient) -> tuple[httpx.Response, httpx.Response]:
    """A question and the document list: what a server with no usable index refuses."""
    return client.post("/v1/query", json={"question": "q"}), client.get("/v1/documents")


def bigger_embedder(c: dict[str, Any]) -> None:
    """Config edit: the fakes, with an embedder whose vectors another index cannot hold."""
    use_fakes(c)
    c["components"]["embedders"]["fake"]["size"] = 16


def free_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# --- the answer stream ----------------------------------------------------------------------


def test_an_answer_streams_sources_then_tokens_then_done(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
        response = client.post("/v1/query", json={"question": "How many staff?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}
    texts = [data["text"] for name, data in events if name == "token"]
    assert "".join(texts) == "".join(model.tokens)
    assert events[0][1] == {
        "sources": [
            {"n": 1, "citation": "notes.txt", "source": "notes.txt", "page": None, "score": None}
        ]
    }
    done = events[-1][1]
    assert 0 <= done["ttft_ms"] <= done["total_ms"]
    assert model.closed
    assert serving.slots.in_use == 0


def test_the_question_is_what_the_model_is_asked(fake_config: RagConfig, tmp_path: Path) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, _):
        client.post("/v1/query", json={"question": "  How many staff?  "})
    assert model.prompts[0].endswith("How many staff?")  # stripped, as the CLI strips it


class ConnectError(Exception):
    """Named as httpx's: the error event recognises a model's errors by class name."""


class BreaksMidStream(StreamingFakeChatModel):
    """Yields one token, then fails as an unreachable Ollama would. The error's own text
    names ``secret``, which must reach the server log only."""

    secret: str = ""

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.log.append("start")
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="Station "))
            raise ConnectError(f"[Errno 111] Connect call failed, see {self.secret}")
        finally:
            self.log.append("closed")


def test_a_failure_after_the_stream_began_is_an_error_event_with_a_fixed_message(
    fake_config: RagConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = BreaksMidStream(secret=str(tmp_path / "corpus" / "notes.txt"))
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
        response = client.post("/v1/query", json={"question": "q"})

    assert response.status_code == 200  # sent before the failure
    events = sse_events(response.text)
    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"type": "ConnectError", "message": "the language model is unreachable"}
    assert str(tmp_path) in caplog.text  # the whole exception, in the server log only
    assert serving.slots.in_use == 0


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (NoIndexError("no index at '/somewhere'"), NO_INDEX_YET),
        (IncompatibleIndexError("built with another embedder at '/somewhere'"), REINGEST),
        (ConnectError("refused"), "the language model is unreachable"),
        (TimeoutError("slow"), "the language model timed out"),
        (
            type("ResponseError", (Exception,), {})("model not found"),
            "the language model returned an error",
        ),
        (ValueError("/home/someone/x"), "the answer failed; the server log has the detail"),
    ],
)
def test_every_error_event_message_is_fixed_text(error: Exception, message: str) -> None:
    assert routes.error_message(error) == message


def test_any_character_in_a_token_round_trips(fake_config: RagConfig, tmp_path: Path) -> None:
    # U+2028, U+0085 and \r split lines for str.splitlines(), which fastapi 0.136.1 split SSE
    # data with: inside a JSON string, that would hand the client broken JSON (S2-7).
    tokens = ["a b", "c\u0085d", "e\nf", "g\r\nh", "é ", " "]
    model = StreamingFakeChatModel(tokens=tokens)
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, _):
        response = client.post("/v1/query", json={"question": "q"})
    assert [data["text"] for name, data in sse_events(response.text) if name == "token"] == tokens


# --- the slot -------------------------------------------------------------------------------


def test_with_every_slot_taken_a_query_is_refused_with_retry_after(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
        held = serving.slots.take()
        assert held is not None
        busy = client.post("/v1/query", json={"question": "q"})
        assert busy.status_code == 503
        assert busy.headers["retry-after"] == str(routes.RETRY_AFTER_S)
        assert "busy" in busy.json()["detail"]
        assert model.prompts == []
        held.release()
        held.release()  # idempotent: a second release must not free a slot twice
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
        assert serving.slots.in_use == 0


def test_a_slot_is_free_after_an_answer_ends_either_way(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    for model in (StreamingFakeChatModel(), BreaksMidStream()):
        with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
            for _ in range(3):  # one leaked slot would make the second a 503
                assert client.post("/v1/query", json={"question": "q"}).status_code == 200
            assert serving.slots.in_use == 0


# --- what a request may say -----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"question": ""}, id="empty"),
        pytest.param({"question": "   \n"}, id="blank"),
        pytest.param({"question": "x" * 2001}, id="overlong"),
        pytest.param({"question": "x", "llm": "components.llms.qwen2_ollama"}, id="choose an llm"),
        pytest.param(
            {"question": "x", "retriever": {"search_kwargs": {"k": 100}}}, id="override a retriever"
        ),
        pytest.param({"question": 7}, id="not a string"),
        pytest.param({}, id="no question"),
        pytest.param(["x"], id="not an object"),
        pytest.param({"question": "x", "sources": None}, id="sources null"),
        pytest.param({"question": "x", "sources": "notes.txt"}, id="sources as one string"),
        pytest.param({"question": "x", "sources": [""]}, id="an empty source"),
        pytest.param({"question": "x", "sources": [7]}, id="a source not a string"),
        pytest.param({"question": "x", "sources": ["s"] * 51}, id="51 sources"),
        pytest.param({"question": "x", "sources": ["s" * 1025]}, id="an overlong source"),
    ],
)
def test_a_query_outside_the_contract_is_refused_with_422(
    fake_config: RagConfig, tmp_path: Path, body: Any
) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
        assert client.post("/v1/query", json=body).status_code == 422
        assert serving.slots.in_use == 0
    assert model.prompts == []


# --- the sources filter (S3-5, FR-3) --------------------------------------------------------


def test_a_query_with_sources_is_answered_from_those_documents_only(
    scratch_config: RagConfig, tmp_path: Path
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, _):
        every = sse_events(client.post("/v1/query", json={"question": "Station Kestrel"}).text)
        assert len({ref["source"] for ref in every[0][1]["sources"]}) > 1
        body = {"question": "Station Kestrel", "sources": ["sub/report.docx", "notes.txt"]}
        answer = client.post("/v1/query", json=body)
        assert answer.status_code == 200
        events = sse_events(answer.text)
        assert {ref["source"] for ref in events[0][1]["sources"]} == {
            "sub/report.docx",
            "notes.txt",
        }
        assert events[-1][0] == "done"


def test_sources_the_index_does_not_hold_are_refused_with_422_naming_them(
    scratch_config: RagConfig, tmp_path: Path
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, serving):
        body = {"question": "q", "sources": ["notes.txt", "nope.pdf", "../etc/passwd", "nope.pdf"]}
        response = client.post("/v1/query", json=body)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail.startswith("unknown sources, not in the index: 'nope.pdf', '../etc/passwd'.")
        assert "notes.txt" not in detail
        assert "GET /v1/documents" in detail
        assert serving.slots.in_use == 0
        # Matched exactly: a path is a name, so neither stripped nor folded.
        for near in (" notes.txt", "NOTES.TXT", "corpus/notes.txt"):
            assert (
                client.post("/v1/query", json={"question": "q", "sources": [near]}).status_code
                == 422
            )


def test_without_an_index_a_query_with_sources_gets_the_503_first(
    scratch_config: RagConfig, tmp_path: Path
) -> None:
    # The refusal order (routes.py): the index's 503 before any 422 about the body.
    with serve(scratch_config, tmp_path) as (client, _):
        for body in (
            {"question": "q", "sources": ["nope.pdf"]},
            {"question": "q", "sources": None},
        ):
            response = client.post("/v1/query", json=body)
            assert (response.status_code, response.json()["detail"]) == (503, NO_INDEX_YET)


def test_the_retrieval_half_is_asked_with_the_sources_once_each(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    asked: list[RetrievalQuery] = []
    chunk = Document(page_content="x", metadata={"source": "notes.txt"})

    def recording(query: RetrievalQuery) -> list[Document]:
        asked.append(query)
        return [chunk]

    serving = fake_serving(
        fake_config, StreamingFakeChatModel(), retrieve=RunnableLambda(recording)
    )
    serving.served = Served(generation=None, manifest=held("notes.txt", "a/b.md"), problem=None)
    with serve(fake_config, tmp_path, serving) as (client, _):
        body = {"question": " q ", "sources": ["a/b.md", "notes.txt", "a/b.md"]}
        assert client.post("/v1/query", json=body).status_code == 200
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200
    assert asked == [RetrievalQuery("q", ("a/b.md", "notes.txt")), RetrievalQuery("q")]


def held(*names: str) -> Manifest:
    """A manifest whose index holds ``names``, for a server that serves no real index."""
    record = DocumentRecord(
        sha256="0" * 64,
        loader="components.loaders.txt",
        loader_identity="l",
        splitter="s",
        chunks=1,
        ingested_at="2026-10-21T00:00:00Z",
    )
    return Manifest(
        generation="g",
        chunking=1,
        embedder=EmbedderRecord(ref="e", identity="e", dimension=8),
        splitter=SplitterRecord(ref="s", identity="s"),
        sparse=None,
        store=STORE,
        documents=dict.fromkeys(names, record),
    )


def test_the_longest_question_allowed_is_answered(fake_config: RagConfig, tmp_path: Path) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, _):
        assert client.post("/v1/query", json={"question": "x" * 2000}).status_code == 200


def test_a_request_model_has_no_room_for_a_component() -> None:
    # Nothing a client sends can reach build_object (DEC-18): a question, and since S3-5 the
    # documents to answer from, a list of strings that only ever filters a search.
    assert set(QueryRequest.model_fields) == {"question", "sources"}
    assert QueryRequest.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"question": "q", "sources": [{"_target_": "os.system"}]})


@pytest.mark.parametrize(
    ("path", "body"),
    [("/v1/query", b'{"question": "q"}'), ("/v1/ingest", b'{"rebuild": false}')],
)
@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", None],
)
def test_a_post_that_is_not_json_is_refused(
    fake_config: RagConfig, tmp_path: Path, path: str, body: bytes, content_type: str | None
) -> None:
    # The content types a web page can send cross-site without a CORS preflight (DEC-18).
    model = StreamingFakeChatModel()
    headers = {} if content_type is None else {"content-type": content_type}
    with serve(fake_config, tmp_path, fake_serving(fake_config, model)) as (client, serving):
        assert client.post(path, content=body, headers=headers).status_code in (415, 422)
        assert client.post(path).status_code in (415, 422)  # no body at all
        assert serving.jobs.running is None
        assert serving.slots.in_use == 0
    assert model.prompts == []


def test_a_body_that_is_not_json_is_refused_before_the_dependencies(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    # FastAPI parses the body before it runs any dependency, so malformed JSON gets its 422
    # ahead of the 401 and the 403. A body that parses but breaks the model is validated
    # after them. Nothing runs either way (S2-7's first review). If FastAPI changes this
    # order, ARCHITECTURE.md §2.4 and routes.py's docstring need the same change.
    as_json = {"content-type": "application/json"}
    from_a_page = {"origin": "https://example.com"}
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model), token=TOKEN) as (
        client,
        serving,
    ):
        for path in ("/v1/query", "/v1/ingest"):
            for headers in (as_json, {**as_json, **from_a_page}):
                assert client.post(path, content=b"{not json", headers=headers).status_code == 422
        # A valid body gets the documented order: Origin, then the token.
        valid = {"question": "q"}
        assert client.post("/v1/query", json=valid).status_code == 401
        assert client.post("/v1/query", json=valid, headers=from_a_page).status_code == 403
        # A JSON body that breaks the model is validated after the token.
        assert client.post("/v1/query", json={"question": ""}).status_code == 401
        assert client.post("/v1/query", json={"question": ""}, headers=AUTH).status_code == 422
        assert serving.jobs.running is None
        assert serving.slots.in_use == 0
    assert model.prompts == []


@pytest.mark.parametrize(
    "body",
    [{}, {"rebuild": "true"}, {"rebuild": 1}, {"rebuild": None}, {"rebuild": False, "config": "x"}],
)
def test_an_ingest_body_outside_the_contract_is_refused(
    fake_config: RagConfig, tmp_path: Path, body: Any
) -> None:
    with serve(fake_config, tmp_path, fake_serving(fake_config, StreamingFakeChatModel())) as (
        client,
        serving,
    ):
        assert client.post("/v1/ingest", json=body).status_code == 422
        assert serving.jobs.running is None


# --- who may ask: the token, Origin and Host --------------------------------------------------


def route_requests(app: FastAPI) -> list[tuple[str, str]]:
    """``(method, path)`` for every route the app serves, path parameters filled in.

    FastAPI 0.143 keeps included routers as such in ``app.routes``; the routes are walked
    the way its own OpenAPI generation walks them.
    """
    found = []
    for context in iter_route_contexts(app.routes):
        path = context.path
        assert isinstance(context.original_route, APIRoute), f"{path} bypasses the dependencies"
        assert path is not None
        for method in sorted(context.methods or ()):
            found.append((method, re.sub(r"\{[^}]+\}", "x", path)))
    return found


HEALTH = {"/v1/health", "/v1/health/live"}


def test_every_route_but_the_health_probes_needs_the_token(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    # Fail closed: a route added later without the token's dependency turns this red.
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving, token=TOKEN) as (client, _):
        requests = route_requests(client.app)  # type: ignore[arg-type]
        for method, path in requests:
            status = client.request(method, path).status_code
            if path in HEALTH:
                assert status != 401, path
            else:
                assert status == 401, (method, path)
    assert {path for _, path in requests} == {
        "/v1/query",
        "/v1/ingest",
        "/v1/ingest/x",
        "/v1/documents",
        "/docs",
        "/redoc",
        "/openapi.json",
        *HEALTH,
    }


@pytest.mark.parametrize(
    "authorization",
    [
        pytest.param(None, id="missing"),
        pytest.param("Basic czNjcmV0", id="wrong scheme"),
        pytest.param(f"Token {TOKEN}", id="another scheme"),
        pytest.param("Bearer wrong", id="wrong token"),
        pytest.param("Bearer S3CRET", id="wrong case"),
        pytest.param("Bearer ", id="empty token"),
        pytest.param("Bearer", id="no token"),
        pytest.param(f"Bearer  {TOKEN}", id="extra space before"),
        pytest.param(f"Bearer {TOKEN} ", id="space after"),
        pytest.param(f"Bearer {TOKEN}{TOKEN}", id="longer"),
        pytest.param(TOKEN, id="no scheme"),
        pytest.param("Bearer sécret".encode(), id="non-ASCII"),
        pytest.param("Bearer \xff\xfe".encode("latin-1"), id="not UTF-8"),
    ],
)
def test_a_missing_or_wrong_token_is_refused_with_401(
    fake_config: RagConfig, tmp_path: Path, authorization: str | bytes | None
) -> None:
    model = StreamingFakeChatModel()
    headers = {} if authorization is None else {"Authorization": authorization}
    with serve(fake_config, tmp_path, fake_serving(fake_config, model), token=TOKEN) as (
        client,
        serving,
    ):
        response = client.post("/v1/query", json={"question": "q"}, headers=headers)
        assert response.status_code == 401  # never 500: a non-ASCII token is compared as bytes
        assert response.headers["www-authenticate"] == "Bearer"
        assert serving.slots.in_use == 0
    assert model.prompts == []


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_the_right_token_is_accepted_in_any_scheme_case(
    fake_config: RagConfig, tmp_path: Path, scheme: str
) -> None:
    model = StreamingFakeChatModel()
    with serve(fake_config, tmp_path, fake_serving(fake_config, model), token=TOKEN) as (client, _):
        headers = {"Authorization": f"{scheme} {TOKEN}"}
        assert client.post("/v1/query", json={"question": "q"}, headers=headers).status_code == 200


def test_compare_digest_on_a_non_ascii_str_raises_which_is_why_bytes_are_compared() -> None:
    # The trap the token check avoids: on str, a hostile header would become a 500.
    with pytest.raises(TypeError):
        secrets.compare_digest("sécret", TOKEN)
    assert secrets.compare_digest("sécret".encode(), TOKEN.encode()) is False


def test_the_health_probes_stay_open_with_a_token(fake_config: RagConfig, tmp_path: Path) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving, token=TOKEN) as (client, _):
        assert client.get("/v1/health/live").json() == {"status": "alive"}
        assert client.get("/v1/health").status_code in (200, 503)


@pytest.mark.parametrize("token", [None, TOKEN])
def test_any_request_from_a_web_page_is_refused_on_every_route(
    fake_config: RagConfig, tmp_path: Path, token: str | None
) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    headers = {"Origin": "https://example.com", **(AUTH if token else {})}
    with serve(fake_config, tmp_path, serving, token=token) as (client, _):
        for method, path in route_requests(client.app):  # type: ignore[arg-type]
            assert client.request(method, path, headers=headers).status_code == 403, path
        # Even the null origin of a sandboxed frame, and with a valid JSON query.
        response = client.post(
            "/v1/query", json={"question": "q"}, headers={**headers, "Origin": "null"}
        )
        assert response.status_code == 403
        assert serving.slots.in_use == 0


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.1:8000", "localhost:8000", "LOCALHOST", "[::1]:8000", "127.0.0.2:9"],
)
def test_without_a_token_a_loopback_host_is_served(
    fake_config: RagConfig, tmp_path: Path, host: str
) -> None:
    with serve(fake_config, tmp_path, fake_serving(fake_config, StreamingFakeChatModel())) as (
        client,
        _,
    ):
        assert client.get("/v1/health/live", headers={"host": host}).status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8000",
        "localhost.evil.example",
        "127.0.0.1.nip.io:8000",
        "[2001:db8::1]:8000",
        "0.0.0.0:8000",
        "192.168.1.10",
    ],
)
def test_without_a_token_any_other_host_is_refused(
    fake_config: RagConfig, tmp_path: Path, host: str
) -> None:
    # DNS rebinding: a page that points its own name at 127.0.0.1 still sends that name.
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving) as (client, _):
        for method, path in route_requests(client.app):  # type: ignore[arg-type]
            assert client.request(method, path, headers={"host": host}).status_code == 403


def test_with_a_token_the_host_is_not_checked(fake_config: RagConfig, tmp_path: Path) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving, token=TOKEN) as (client, _):
        headers = {"host": "rag.lan:8000", **AUTH}
        assert client.get("/v1/ingest/x", headers=headers).status_code == 404


@pytest.mark.parametrize(
    ("header", "host"),
    [
        ("127.0.0.1:8000", "127.0.0.1"),
        ("localhost", "localhost"),
        ("[::1]:8000", "::1"),
        ("[::1]", "::1"),
        ("::1", "::1"),
        ("", ""),
    ],
)
def test_the_host_is_read_without_its_port(header: str, host: str) -> None:
    assert routes.host_of(header) == host


def test_with_a_token_the_docs_and_the_schema_need_it(
    fake_config: RagConfig, tmp_path: Path
) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving, token=TOKEN) as (client, _):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers=AUTH).status_code == 200
        assert "/v1/query" in client.get("/openapi.json", headers=AUTH).json()["paths"]


def test_without_a_token_the_docs_are_served(fake_config: RagConfig, tmp_path: Path) -> None:
    serving = fake_serving(fake_config, StreamingFakeChatModel())
    with serve(fake_config, tmp_path, serving) as (client, _):
        assert client.get("/docs").status_code == 200
        assert "/v1/ingest/{job_id}" in client.get("/openapi.json").json()["paths"]


def test_the_asgi_spec_version_is_logged_once(
    fake_config: RagConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="rag_qa")
    with serve(fake_config, tmp_path, fake_serving(fake_config, StreamingFakeChatModel())) as (
        client,
        _,
    ):
        client.get("/v1/health/live")
        client.get("/v1/health/live")
    logged = [r.getMessage() for r in caplog.records if "ASGI spec" in r.getMessage()]
    assert logged == ["The server reports ASGI spec 2.0: a disconnect cancels its answer."]


def test_a_server_reporting_spec_2_4_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    # From 2.4 Starlette stops listening for disconnects while a response streams.
    app = SimpleNamespace(state=State())
    request = Request({"type": "http", "asgi": {"spec_version": "2.4"}, "app": app})
    routes.log_spec_version(request)
    assert "keep-alive ping" in caplog.text
    assert [r.levelname for r in caplog.records] == ["WARNING"]


# --- the index, the ingest jobs and the documents (the real build) --------------------------


def test_a_server_with_no_index_starts_refuses_queries_and_serves_after_an_ingest_job(
    scratch_config: RagConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[BaseChatModel] = []
    real_build_llm = api_state.build_llm

    def counting_build_llm(config: RagConfig) -> BaseChatModel:
        built.append(real_build_llm(config))
        return built[-1]

    monkeypatch.setattr(api_state, "build_llm", counting_build_llm)
    with serve(scratch_config, tmp_path) as (client, _):
        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["index"] == {
            "ready": False,
            "generation": None,
            "documents": None,
            "detail": NO_INDEX_YET,
        }
        assert health.json()["llm"]["ready"] is True  # a fake model: nothing to ask
        for response in refused(client):
            assert response.status_code == 503
            assert response.json()["detail"] == NO_INDEX_YET

        started = client.post("/v1/ingest", json={"rebuild": False})
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        assert started.headers["location"] == f"/v1/ingest/{job_id}"
        job = wait_for_job(client, job_id)
        assert (job["status"], job["exit_code"]) == ("succeeded", 0)
        assert job["report"]["added"] == len(CORPUS_FILES)
        assert job["report"]["published"] is not None
        assert job["started_at"] <= job["finished_at"]
        assert "Full rebuild: there is no index yet." in job["log"]
        assert job["log"][-1].startswith(f"Published {job['report']['published']}")

        health = client.get("/v1/health").json()
        assert health["status"] == "ready"
        assert health["index"]["generation"] == job["report"]["published"]
        assert health["index"]["documents"] == len(CORPUS_FILES)
        answer = client.post("/v1/query", json={"question": "How many staff?"})
        assert answer.status_code == 200
        events = sse_events(answer.text)
        assert {ref["source"] for ref in events[0][1]["sources"]} <= set(CORPUS_FILES)
        assert "".join(d["text"] for name, d in events if name == "token") == FAKE_ANSWER
        documents = client.get("/v1/documents").json()["documents"]
        assert [d["source"] for d in documents] == CORPUS_FILES
        assert documents[0]["loader"] == "components.loaders.txt"
        assert all(d["chunks"] >= 1 and len(d["sha256"]) == 64 for d in documents)
    assert len(built) == 1  # only the retrieval half was swapped in: the LLM was not rebuilt


def test_an_ingest_job_reports_rag_ingests_counts_in_its_report(
    scratch_config: RagConfig, tmp_path: Path
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, _):
        job = run_job(client)
    assert (job["status"], job["exit_code"]) == ("succeeded", 0)
    assert job["report"]["unchanged"] == len(CORPUS_FILES)
    assert job["report"]["published"] is None  # nothing changed
    assert job["report"]["skipped"] == ["diagram.png"]
    assert job["log"] == [
        "  - Skipped diagram.png (no loader configured for this file type)",
        "Nothing changed: the index is up to date.",
    ]


def test_the_job_report_mirrors_rag_ingests_report() -> None:
    assert list(IngestReportOut.model_fields) == list(IngestReport.__dataclass_fields__)


def test_an_ingest_job_reports_a_file_above_max_file_mb_with_its_size(
    make_config: MakeConfig, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S3-4: the same as rag-ingest's report and line, path-free.
    (sample_corpus / "big.txt").write_text("x" * 4999 + "\n")  # 5,000 bytes
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))

    def with_a_limit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["pipeline"]["ingestion"]["max_file_mb"] = 0.004

    with serve(load_config(make_config(with_a_limit)), tmp_path) as (client, _):
        job = run_job(client)
    assert (job["status"], job["exit_code"]) == ("succeeded", 0)
    assert (job["report"]["too_large"], job["report"]["max_file_mb"]) == (
        [["big.txt", 5000]],
        0.004,
    )
    assert "  - Skipped big.txt (0.005 MB, above the 0.004 MB limit)" in job["log"]


def test_a_real_job_over_an_unreadable_file_reports_it_without_a_host_path(
    scratch_config: RagConfig, sample_corpus: Path, tmp_path: Path
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads any file, so chmod 000 proves nothing")
    secret = sample_corpus / "secret.txt"
    secret.write_text("Not for anyone.\n")
    secret.chmod(0)
    try:
        with serve(scratch_config, tmp_path) as (client, _):
            started = client.post("/v1/ingest", json={"rebuild": False})
            job = wait_for_job(client, started.json()["job_id"])
            assert (job["status"], job["exit_code"]) == ("failed", 1)
            ((name, error),) = job["report"]["failed"]
            assert name == "secret.txt"
            assert error.startswith("PermissionError:")
            assert any("secret.txt" in line for line in job["log"])
            assert job["log"][-1].startswith("Error: 1 item(s) could not be read")
            # The rest was indexed, and is served.
            assert job["report"]["published"] is not None
            assert client.get("/v1/health").status_code == 200
    finally:
        secret.chmod(0o644)


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (IngestionRunningError("another ingestion is running (index.lock is locked)"), 2),
        (FileNotFoundError("Corpus directory not found: {corpus} (set by paths.data)"), 2),
        (CorpusUnreadableError("Corpus directory not readable: {corpus} (Permission denied)"), 2),
        (RuntimeError("unexpected, at {corpus}/x.pdf"), 1),
    ],
)
def test_a_job_that_cannot_run_fails_with_rag_ingests_exit_code(
    scratch_config: RagConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
) -> None:
    # An ingest the CLI holds the lock for fails at once (exit 2), not a 409: the route
    # cannot hold the flock on ingest()'s behalf.
    error.args = (str(error.args[0]).format(corpus=scratch_config.paths.data),)

    def failing_ingest(config: RagConfig, *, rebuild: bool) -> IngestReport:
        raise error

    monkeypatch.setattr("rag_qa.api.jobs.ingest", failing_ingest)
    with serve(scratch_config, tmp_path) as (client, _):
        job = run_job(client)
    assert (job["status"], job["exit_code"], job["report"]) == ("failed", exit_code, None)
    assert job["log"][-1].startswith(f"Error: {type(error).__name__}: ")


def test_a_second_ingest_job_while_one_runs_is_refused_with_409(
    scratch_config: RagConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def slow_ingest(config: RagConfig, *, rebuild: bool) -> IngestReport:
        release.wait(10)
        return IngestReport(found=1)

    monkeypatch.setattr("rag_qa.api.jobs.ingest", slow_ingest)
    with serve(scratch_config, tmp_path) as (client, serving):
        first = client.post("/v1/ingest", json={"rebuild": True}).json()
        second = client.post("/v1/ingest", json={"rebuild": False})
        assert second.status_code == 409
        assert first["job_id"] in second.json()["detail"]
        assert client.get(f"/v1/ingest/{first['job_id']}").json()["status"] in ("queued", "running")
        release.set()
        assert wait_for_job(client, first["job_id"])["status"] == "succeeded"
        assert serving.jobs.running is None
        third = client.post("/v1/ingest", json={"rebuild": False})  # one may start again
        assert third.status_code == 202
        wait_for_job(client, third.json()["job_id"])


def test_an_unknown_job_is_404_and_only_the_last_jobs_are_kept(
    scratch_config: RagConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rag_qa.api.jobs.ingest", lambda config, *, rebuild: IngestReport(found=1))
    with serve(scratch_config, tmp_path) as (client, _):
        assert client.get("/v1/ingest/nope").status_code == 404
        ids = []
        for _ in range(KEEP + 1):
            ids.append(client.post("/v1/ingest", json={"rebuild": False}).json()["job_id"])
            wait_for_job(client, ids[-1])
        assert client.get(f"/v1/ingest/{ids[0]}").status_code == 404
        assert client.get(f"/v1/ingest/{ids[1]}").status_code == 200


def test_an_ingest_run_outside_the_server_is_served_on_the_next_request(
    scratch_config: RagConfig, sample_corpus: Path, tmp_path: Path
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, _):
        before = client.get("/v1/health").json()["index"]["generation"]
        (sample_corpus / "added.txt").write_text("A new note about penguins.\n")
        ingest(scratch_config)  # as rag-ingest would: a new generation flipped in
        documents = [d["source"] for d in client.get("/v1/documents").json()["documents"]]
        assert "added.txt" in documents
        after = client.get("/v1/health").json()["index"]
        assert after["generation"] != before
        assert after["documents"] == len(CORPUS_FILES) + 1
        assert client.post("/v1/query", json={"question": "penguins?"}).status_code == 200


def test_an_index_rebuilt_with_another_embedder_makes_the_server_not_ready(
    scratch_config: RagConfig, make_config: MakeConfig, tmp_path: Path
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, _):
        assert client.get("/v1/health").status_code == 200
        ingest(load_config(make_config(bigger_embedder)), rebuild=True)  # a CLI --rebuild
        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["index"]["detail"] == REINGEST
        for response in refused(client):
            assert response.status_code == 503
            assert response.json()["detail"] == REINGEST


@pytest.mark.parametrize("damage", ["another embedder", "another dimension"])
def test_an_unusable_index_at_start_up_counts_as_none(
    scratch_config: RagConfig,
    make_config: MakeConfig,
    tmp_path: Path,
    damage: str,
    qdrant: QdrantClient,
) -> None:
    # The cases build_query_pipeline(require_index=False) covered until S2-7.
    if damage == "another embedder":
        ingest(load_config(make_config(bigger_embedder)))
    else:
        ingest(scratch_config)
        edit_manifest(qdrant, lambda m: m["embedder"].update(dimension=9))
    with serve(scratch_config, tmp_path) as (client, _):
        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["index"]["detail"] == REINGEST
        assert client.post("/v1/query", json={"question": "q"}).status_code == 503


def test_health_asks_the_models_server(
    scratch_config: RagConfig,
    tmp_path: Path,
    fake_ollama: FakeOllama,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest(scratch_config)
    up = ChatOllama(model="mistral", base_url=fake_ollama.url)
    monkeypatch.setattr("rag_qa.api.state.build_llm", lambda config: up)
    with serve(scratch_config, tmp_path) as (client, _):
        health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["llm"] == {"ready": True, "detail": "the model's server answered"}
    assert "/api/tags" in fake_ollama.requests


def test_health_is_503_when_the_models_server_does_not_answer(
    scratch_config: RagConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(scratch_config)
    down = ChatOllama(model="mistral", base_url=f"http://127.0.0.1:{free_port()}")
    monkeypatch.setattr("rag_qa.api.state.build_llm", lambda config: down)
    with serve(scratch_config, tmp_path) as (client, _):
        health = client.get("/v1/health")
    assert health.status_code == 503
    assert health.json()["status"] == "not ready"
    assert health.json()["index"]["ready"] is True
    assert health.json()["llm"]["ready"] is False


# --- one retrieval at a time (S2-4's second review) -----------------------------------------


def test_retrievals_run_one_at_a_time_even_after_a_cancel() -> None:
    # A disconnect frees the slot at once, but the retrieval thread runs on: the next
    # request's retrieval must wait for it rather than rerank beside it.
    lock = threading.Lock()
    spans: list[tuple[str, float, float]] = []

    def slow(question: str) -> list[Document]:
        start = time.monotonic()
        time.sleep(0.3)
        spans.append((question, start, time.monotonic()))
        return []

    retrieve = serialized(RunnableLambda(slow), lock)

    async def scenario() -> None:
        first = asyncio.create_task(retrieve.ainvoke("first"))
        await asyncio.sleep(0.05)  # its thread holds the lock by now
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await retrieve.ainvoke("second")

    asyncio.run(scenario())
    (first, _, first_end), (second, second_start, _) = spans
    assert (first, second) == ("first", "second")
    assert second_start >= first_end


def test_a_waiting_retrieval_whose_request_was_cancelled_does_no_work() -> None:
    lock = threading.Lock()
    asked: list[str] = []
    retrieve = serialized(RunnableLambda(lambda q: asked.append(q) or []), lock)

    async def scenario() -> None:
        lock.acquire()  # an orphaned retrieval holds it
        waiter = asyncio.create_task(retrieve.ainvoke("abandoned"))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0.05)
        lock.release()
        await retrieve.ainvoke("next")

    asyncio.run(scenario())
    assert asked == ["next"]


# --- the served state: races and failures --------------------------------------------------


def test_an_open_that_a_flip_raced_is_tried_once_more(
    scratch_config: RagConfig, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S2-6's third review: only the live generation and the previous one are kept, so two
    # flips during an open can delete the one it checked. Looking again finds the index.
    ingest(scratch_config)
    real_open = api_state.open_retrieval
    calls: list[str] = []

    def raced_open(config: RagConfig, embeddings: Any, *, reranker: Any, store: Any) -> Any:
        calls.append("open")
        if len(calls) == 1:
            (sample_corpus / "late.txt").write_text("A late addition.\n")
            ingest(config)  # the flip lands mid-open
            raise IncompatibleIndexError("the generation it checked is gone")
        return real_open(config, embeddings, reranker=reranker, store=store)

    monkeypatch.setattr(api_state, "open_retrieval", raced_open)
    embeddings = DeterministicFakeEmbedding(size=8)
    retrieve, served = api_state.open_served(
        scratch_config, build_store(scratch_config), embeddings, None
    )
    assert calls == ["open", "open"]
    assert served.ready and retrieve is not None
    assert served.manifest is not None and "late.txt" in served.manifest.documents


def test_an_open_that_fails_with_no_flip_is_not_retried(
    scratch_config: RagConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def failing_open(config: RagConfig, embeddings: Any, *, reranker: Any, store: Any) -> Any:
        calls.append("open")
        raise IncompatibleIndexError("built by another embedder")

    monkeypatch.setattr(api_state, "open_retrieval", failing_open)
    embeddings = DeterministicFakeEmbedding(size=8)
    retrieve, served = api_state.open_served(
        scratch_config, build_store(scratch_config), embeddings, None
    )
    assert (calls, retrieve, served.problem) == (["open"], None, REINGEST)


def test_an_open_that_fails_for_another_reason_is_not_ready_and_tried_again(
    scratch_config: RagConfig,
    sample_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # S2-7's first review: only NoIndexError was caught, so an OSError or an embedder
    # failure during a reopen made every route that refreshes answer 500, health included.
    # It is not ready instead, says so without the error's text (which names paths), and
    # the next request tries again, since such a failure is likelier to pass than not.
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, _):
        real_open = api_state.open_retrieval
        failing = [True]

        def flaky_open(config: RagConfig, embeddings: Any, *, reranker: Any, store: Any) -> Any:
            if failing[0]:
                raise OSError(f"cannot read {tmp_path}/index/manifest.json: the disk went away")
            return real_open(config, embeddings, reranker=reranker, store=store)

        monkeypatch.setattr(api_state, "open_retrieval", flaky_open)
        (sample_corpus / "added.txt").write_text("A new note.\n")
        ingest(scratch_config)  # a rag-ingest outside the server flips a new generation in

        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["index"]["detail"] == api_state.UNAVAILABLE
        for response in refused(client):
            assert response.status_code == 503
            assert response.json()["detail"] == api_state.UNAVAILABLE
        assert "the disk went away" in caplog.text  # in the server log only

        failing[0] = False
        assert client.get("/v1/health").status_code == 200
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200


def test_qdrant_unreachable_on_a_refresh_is_not_ready_and_tried_again(
    scratch_config: RagConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # DEC-19: every request reads the alias. With Qdrant gone it answers 503, never 500,
    # and the first request after Qdrant is back serves the index again.
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, serving):
        assert client.get("/v1/health").status_code == 200
        real = serving.store.alias_target

        def down() -> str | None:
            raise StoreUnavailableError("cannot use Qdrant at http://q (ConnectError)")

        monkeypatch.setattr(serving.store, "alias_target", down)
        health = client.get("/v1/health")
        assert health.status_code == 503
        assert health.json()["index"]["detail"] == api_state.UNAVAILABLE
        for response in refused(client):
            assert response.status_code == 503
            assert response.json()["detail"] == api_state.UNAVAILABLE
        assert caplog.text.count("Not serving an index: cannot use Qdrant") == 1  # said once

        monkeypatch.setattr(serving.store, "alias_target", real)
        assert client.get("/v1/health").status_code == 200
        assert client.post("/v1/query", json={"question": "q"}).status_code == 200


def test_qdrant_unreachable_at_start_up_is_not_ready_not_a_crash(
    scratch_config: RagConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_store(scratch_config)

    def down() -> str | None:
        raise StoreUnavailableError("cannot use Qdrant at http://q (ConnectError)")

    monkeypatch.setattr(store, "alias_target", down)
    retrieve, served = api_state.open_served(
        scratch_config, store, DeterministicFakeEmbedding(size=8), None
    )
    assert (retrieve, served) == (None, api_state.NOT_SEEN)


def test_concurrent_requests_share_one_failed_reopen(
    scratch_config: RagConfig, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S2-7's second review: a failed open records no generation as seen, so every request
    # queued behind it tried again in turn, one after another: five concurrent refreshes
    # made five opens. A request that waited for another's attempt now takes its result,
    # and a later request still tries again.
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, serving):
        opens: list[str] = []

        def failing_open(config: RagConfig, embeddings: Any, *, reranker: Any, store: Any) -> Any:
            opens.append("open")
            time.sleep(0.2)  # long enough for the other refreshes to queue on the lock
            raise OSError("the disk went away")

        monkeypatch.setattr(api_state, "open_retrieval", failing_open)
        (sample_corpus / "added.txt").write_text("A new note.\n")
        ingest(scratch_config)

        async def five_refreshes() -> None:
            await asyncio.gather(*(serving.refresh() for _ in range(5)))

        client.portal.call(five_refreshes)  # type: ignore[union-attr]
        assert serving.served.problem == api_state.UNAVAILABLE
        assert opens == ["open"]
        assert client.get("/v1/health").status_code == 503
        assert opens == ["open", "open"]


def test_concurrent_requests_reopen_a_new_generation_once(
    scratch_config: RagConfig, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest(scratch_config)
    with serve(scratch_config, tmp_path) as (client, serving):
        real_open = api_state.open_retrieval
        opens: list[str] = []

        def counting_open(config: RagConfig, embeddings: Any, *, reranker: Any, store: Any) -> Any:
            opens.append("open")
            time.sleep(0.2)  # long enough for the other refreshes to queue on the lock
            return real_open(config, embeddings, reranker=reranker, store=store)

        monkeypatch.setattr(api_state, "open_retrieval", counting_open)
        (sample_corpus / "added.txt").write_text("A new note.\n")
        ingest(scratch_config)

        async def three_refreshes() -> None:
            await asyncio.gather(serving.refresh(), serving.refresh(), serving.refresh())

        client.portal.call(three_refreshes)  # type: ignore[union-attr]
    assert opens == ["open"]


async def stream_that_fails_to_close() -> AsyncIterator[int]:
    try:
        yield 1
    finally:
        raise RuntimeError("the close failed")


def test_a_slot_is_released_even_when_closing_its_stream_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool = api_state.SlotPool(1)

    async def scenario() -> None:
        slot = pool.take()
        assert slot is not None
        stream = stream_that_fails_to_close()
        await anext(stream)  # paused at its yield, as a disconnect between events leaves it
        slot.hold(stream)  # type: ignore[arg-type]
        await slot.aclose()

    asyncio.run(scenario())
    assert pool.in_use == 0
    assert "Could not close an answer stream" in caplog.text


def test_a_failure_to_serve_what_a_job_published_never_fails_the_job(
    scratch_config: RagConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("rag_qa.api.jobs.ingest", lambda config, *, rebuild: IngestReport(found=1))

    async def failing_refresh() -> None:
        raise RuntimeError("cannot reopen")

    async def scenario() -> api_jobs.IngestJob:
        jobs = api_jobs.IngestJobs(scratch_config, after=failing_refresh)
        job = jobs.start(rebuild=False)
        while jobs.running is not None or jobs._tasks:
            await asyncio.sleep(0.01)
        return job

    job = asyncio.run(scenario())
    assert (job.status, job.exit_code) == ("succeeded", 0)
    assert f"Could not serve what ingest job {job.job_id} published" in caplog.text


def test_stopping_the_server_abandons_a_running_job_and_says_so(
    scratch_config: RagConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An ingest cannot be interrupted, and DEC-17 makes an abandoned one safe: shutdown
    # does not wait for it.
    release = threading.Event()

    def stuck_ingest(config: RagConfig, *, rebuild: bool) -> IngestReport:
        release.wait(10)
        return IngestReport(found=1)

    monkeypatch.setattr("rag_qa.api.jobs.ingest", stuck_ingest)
    try:
        with serve(scratch_config, tmp_path) as (client, _):
            job_id = client.post("/v1/ingest", json={"rebuild": False}).json()["job_id"]
    finally:
        release.set()
    assert f"Ingest job {job_id} is still running and is abandoned" in caplog.text
