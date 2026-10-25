"""The S0-5 deprecation gate as a pytest (S1-5, ISS-17's guard).

Records warnings while importing every module **and** running the whole pipeline
(ingest, build the chain, invoke, stream, and ``stream_answer``, the path every front
end answers through since S2-1) with the fakes. It fails on any
``LangChainDeprecationWarning``, and on any other ``DeprecationWarning``. Until S3-3 it
allowed one, DEC-5's ``langchain-community`` sunset notice; that package is gone, so
nothing is allowed.

Two design points, both from S0-5's Discovered section:

* **Never a command-line ``-W error`` gate.** ``langchain_core`` calls
  ``warnings.filterwarnings("default", ...)`` on import and overrides it, so such
  a gate is structurally unable to fail on a deprecated LangChain API.
* **A negative control.** The same predicate, run over an API LangChain deprecates,
  must report a violation. A check with no known-bad case run against it is an
  assumption, not a check. Since S3-3 that API is ``ChatOllama.dict()``; before, it was
  the legacy ``langchain_community.llms.Ollama``, which left with its package.

Each check runs in a **fresh interpreter**: inside one pytest session other tests
have already imported LangChain, and import-time warnings fire once per process,
so an in-process check would see nothing.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_HARNESS = textwrap.dedent(
    """
    import json, sys, warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(sys.argv[1])
    print("__WARNINGS__" + json.dumps([
        {"mro": [c.__name__ for c in w.category.__mro__], "message": str(w.message)}
        for w in caught
    ]))
    """
)


def record_warnings(body: str, env: dict[str, str] | None = None) -> list[dict]:
    """Run ``body`` in a fresh interpreter and return every warning it raised."""
    result = subprocess.run(
        [sys.executable, "-c", _HARNESS, textwrap.dedent(body)],
        env={**os.environ, "HF_HUB_OFFLINE": "1", **(env or {})},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the returncode is asserted below, with the child's stderr
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("__WARNINGS__"))
    return json.loads(line.removeprefix("__WARNINGS__"))


def violations(records: list[dict]) -> list[dict]:
    """The gate's predicate: every deprecation, LangChain's pending kind included.

    ``LangChainDeprecationWarning`` subclasses ``DeprecationWarning``, so the second clause
    covers it; the pending one subclasses ``PendingDeprecationWarning`` and needs its own.
    """
    return [
        r
        for r in records
        if "LangChainPendingDeprecationWarning" in r["mro"] or "DeprecationWarning" in r["mro"]
    ]


def test_no_deprecated_api_across_imports_and_the_whole_pipeline(fake_rag: Path) -> None:
    records = record_warnings(
        """
        import asyncio
        from qdrant_client import QdrantClient
        import rag_qa.vectorstore
        client = QdrantClient(":memory:")  # one store for the whole child, as in conftest
        client.close = lambda **kwargs: None
        rag_qa.vectorstore.connect = lambda *args, **kwargs: client
        import rag_qa.answering, rag_qa.cli, rag_qa.evaluation.cli
        import rag_qa.evaluation.fingerprint, rag_qa.evaluation.gate
        import rag_qa.evaluation.generation, rag_qa.evaluation.ragas_scoring
        import rag_qa.rerankers
        from rag_qa.config import load_config, check_imports
        from rag_qa.ingest import ingest
        from rag_qa.chain import build_query_pipeline, build_rag_chain
        config = load_config()
        check_imports(config, config.references().values())
        ingest(config)
        chain = build_rag_chain(config)
        result = chain.invoke({"question": "How many staff?"})
        assert sorted(result) == ["answer", "context", "question"]
        assert list(chain.stream({"question": "How many staff?"}))
        pipeline = build_query_pipeline(config)
        async def answer():
            return [e async for e in rag_qa.answering.stream_answer(pipeline, "How many staff?")]
        assert asyncio.run(answer())
        asyncio.run(pipeline.aclose())
        """,
        env={
            "RAG_CONFIG": str(fake_rag),
            "RAG_DATA_PATH": os.environ["RAG_DATA_PATH"],
            "RAG_STATE_PATH": os.environ["RAG_STATE_PATH"],
        },
    )
    assert violations(records) == [], violations(records)


def test_no_deprecated_api_with_tracing_on(fake_rag: Path) -> None:
    # S3-7b: the SDK, the instrumented app, every span helper and the semantic conventions'
    # GenAI constants, which 0.66b1 marks deprecated: reading a member must warn of nothing.
    records = record_warnings(
        """
        import asyncio
        from qdrant_client import QdrantClient
        import rag_qa.vectorstore
        client = QdrantClient(":memory:")
        client.close = lambda **kwargs: None
        rag_qa.vectorstore.connect = lambda *args, **kwargs: client
        from fastapi.testclient import TestClient
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        import rag_qa.answering
        from rag_qa.api.app import create_app
        from rag_qa.chain import build_query_pipeline
        from rag_qa.config import load_config
        from rag_qa.ingest import ingest
        from rag_qa.observability import tracing
        from rag_qa.settings import ServerSettings
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracing.install(provider)
        config = load_config()
        ingest(config)
        pipeline = build_query_pipeline(config)
        async def answer():
            with tracing.span(tracing.QUERY):
                events = rag_qa.answering.stream_answer(pipeline, "How many staff?")
                return [e async for e in events]
        assert asyncio.run(answer())
        asyncio.run(pipeline.aclose())
        app = create_app(config, ServerSettings())
        with TestClient(app, base_url="http://127.0.0.1") as http:
            assert http.post("/v1/query", json={"question": "How many staff?"}).status_code == 200
        tracing.uninstall()
        names = {span.name for span in exporter.get_finished_spans()}
        assert {"rag.ingest", "rag.query", "rag.retrieve", "POST /v1/query"} <= names, names
        """,
        env={
            "RAG_CONFIG": str(fake_rag),
            "RAG_DATA_PATH": os.environ["RAG_DATA_PATH"],
            "RAG_STATE_PATH": os.environ["RAG_STATE_PATH"],
        },
    )
    assert violations(records) == [], violations(records)


def test_negative_control_the_gate_flags_a_deprecated_langchain_api() -> None:
    # A method of the chat model class we answer with, deprecated in langchain-core 1.4.2
    # and removed in 2.0, which pyproject.toml's bound keeps out. It replaced the legacy
    # langchain_community.llms.Ollama class when that package left (S3-3).
    records = record_warnings(
        """
        from langchain_ollama import ChatOllama
        ChatOllama(model="x").dict()
        """
    )
    flagged = violations(records)
    assert flagged, "the gate failed to flag ChatOllama.dict(): it is blind"
    assert any("LangChainDeprecationWarning" in r["mro"] for r in flagged)
