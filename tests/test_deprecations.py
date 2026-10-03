"""The S0-5 deprecation gate as a pytest (S1-5, ISS-17's guard).

Records warnings while importing every module **and** running the whole pipeline
(ingest, build the chain, invoke, stream, and ``stream_answer``, the path every front
end answers through since S2-1) with the fakes. It fails on any
``LangChainDeprecationWarning``, and on any ``DeprecationWarning`` except DEC-5's
``langchain-community`` sunset notice.

Two design points, both from S0-5's Discovered section:

* **Never a command-line ``-W error`` gate.** ``langchain_core`` calls
  ``warnings.filterwarnings("default", ...)`` on import and overrides it, so such
  a gate is structurally unable to fail on a deprecated LangChain API.
* **A negative control.** The same predicate, run over the legacy
  ``langchain_community.llms.Ollama``, must report a violation. A check with no
  known-bad case run against it is an assumption, not a check.

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

SUNSET_NOTICE = "`langchain-community` is being sunset"  # DEC-5: expected, allowed

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
    """The gate's predicate: LangChain deprecations, or deprecations other than DEC-5's."""
    return [
        r
        for r in records
        if "LangChainDeprecationWarning" in r["mro"]
        or "LangChainPendingDeprecationWarning" in r["mro"]
        or ("DeprecationWarning" in r["mro"] and SUNSET_NOTICE not in r["message"])
    ]


def test_no_deprecated_api_across_imports_and_the_whole_pipeline(fake_rag: Path) -> None:
    records = record_warnings(
        """
        import asyncio
        import rag_qa.answering, rag_qa.cli, rag_qa.evaluate, rag_qa.evaluation.cli
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
            "RAG_VECTOR_STORE_PATH": os.environ["RAG_VECTOR_STORE_PATH"],
        },
    )
    assert violations(records) == [], violations(records)
    # The one warning we expect is still the sunset notice. If it disappears,
    # DEC-5's staged exit has moved (or the predicate went blind): look, don't ignore.
    assert any(SUNSET_NOTICE in r["message"] for r in records)


def test_negative_control_the_gate_flags_a_deprecated_langchain_api() -> None:
    records = record_warnings(
        """
        from langchain_community.llms import Ollama
        Ollama(model="x")
        """
    )
    flagged = violations(records)
    assert flagged, "the gate failed to flag the legacy Ollama class: it is blind"
    assert any("LangChainDeprecationWarning" in r["mro"] for r in flagged)
