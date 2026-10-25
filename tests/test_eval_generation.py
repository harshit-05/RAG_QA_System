"""``rag-eval generate`` (S2-5): every golden question through ``stream_answer``, recorded.

Runs against ``fake_rag``: the real ingest built an index of ``sample_corpus`` with the
fake embedder, and the llm is a fake chat model, so nothing needs Ollama or a download.
"""

import asyncio
import dataclasses
import json
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import (
    FAKE_ANSWER,
    MISTRAL_DIGEST,
    REAL_CONFIG,
    FakeCollector,
    FakeOllama,
    MakeConfig,
    StreamingFakeChatModel,
    use_fakes,
)

from rag_qa.chain import build_query_pipeline
from rag_qa.components import OLLAMA_HINT
from rag_qa.config import load_config
from rag_qa.evaluation import fingerprint as fp
from rag_qa.evaluation.cli import EXIT_CANNOT_RUN, EXIT_OK, main
from rag_qa.evaluation.dataset import load_golden
from rag_qa.evaluation.gate import Generator, load_answers
from rag_qa.evaluation.generation import answer, generator_record
from rag_qa.observability import tracing
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ENV_QDRANT_COLLECTION

GOLDEN = [
    {
        "id": "notes",
        "question": "What do the notes say?",
        "ground_truth": "Plain text notes.",
        "answerable": True,
        "expected_sources": [{"source": "notes.txt"}],
    },
    {
        "id": "report",
        "question": "What is the report about?",
        "ground_truth": "A report.",
        "answerable": True,
        "expected_sources": [{"source": "sub/report.docx"}],
    },
    {
        "id": "capital",
        "question": "What is the capital of Australia?",
        "ground_truth": "I could not find the answer in the provided documents.",
        "answerable": False,
        "must_not_contain": ["Canberra"],
    },
]


def write_golden(config_file: Path) -> Path:
    eval_dir = config_file.parent / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "eval_dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in GOLDEN))
    return eval_dir


@pytest.fixture
def evals(fake_rag: Path) -> Path:
    write_golden(fake_rag)
    return fake_rag


def generate(config_file: Path, *args: str) -> int:
    return main(["generate", "--config", str(config_file), *args])


def test_generate_records_every_answer_with_its_fingerprint(
    evals: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate(evals) == EXIT_OK
    run = load_answers(evals.parent / "eval" / "runs" / "answers-latest.json")
    assert [item.id for item in run.items] == ["notes", "report", "capital"]
    assert run.limit is None
    golden = load_golden(evals.parent / "eval" / "eval_dataset.jsonl")
    assert run.fingerprint == fp.fingerprint(load_config(evals), golden, parts=fp.GENERATION_PARTS)
    for item in run.items:
        assert item.answer == FAKE_ANSWER
        assert item.contexts and len(item.contexts) == len(item.sources)
        assert [s["n"] for s in item.sources] == list(range(1, len(item.sources) + 1))
        assert item.ttft_ms is not None and item.total_ms >= item.ttft_ms
    # Not an Ollama model: there is no digest to record, and no Ollama was asked.
    assert (run.generator.ref, run.generator.model, run.generator.digest) == (
        "components.llms.fake",
        None,
        None,
    )
    out = capsys.readouterr().out
    assert "[3/3] capital: ttft" in out and "Wrote 3 answer(s)" in out


def test_the_contexts_are_the_chunks_the_prompt_got(evals: Path) -> None:
    llm = StreamingFakeChatModel()
    pipeline = dataclasses.replace(build_query_pipeline(load_config(evals)), llm=llm)
    item = load_golden(evals.parent / "eval" / "eval_dataset.jsonl")[0]
    answered = asyncio.run(answer(pipeline, item))
    assert answered.answer == "".join(llm.tokens)
    prompt = llm.prompts[-1]
    for n, context in enumerate(answered.contexts, 1):
        assert f"[{n}] (" in prompt and context in prompt


def test_an_answer_records_its_providers_token_counts(evals: Path) -> None:
    # S3-7: from Done, so tier 2's answers file has them for every run from now on.
    llm = StreamingFakeChatModel(
        usage={"input_tokens": 700, "output_tokens": 42, "total_tokens": 742}
    )
    pipeline = dataclasses.replace(build_query_pipeline(load_config(evals)), llm=llm)
    item = load_golden(evals.parent / "eval" / "eval_dataset.jsonl")[0]
    answered = asyncio.run(answer(pipeline, item))
    assert (answered.input_tokens, answered.output_tokens) == (700, 42)


def test_generate_traces_nothing_with_an_endpoint_left_in_the_shell(
    evals: Path, fake_collector: FakeCollector, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S3-7b: rag-eval never turns tracing on, and the shared code (retrieval, the answer
    # stream) gets a no-op tracer unless a command did, so an endpoint exported for a
    # rag-serve run cannot trace a tier-2 run.
    installed: list[Any] = []
    monkeypatch.setattr(tracing, "install", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", fake_collector.url)
    assert generate(evals) == EXIT_OK
    assert installed == []
    assert not tracing.enabled()
    assert fake_collector.requests == []


def test_limit_needs_out(evals: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        generate(evals, "--limit", "1")
    assert exc.value.code == 2
    assert "--limit needs --out" in capsys.readouterr().err


def test_a_limited_run_goes_where_out_says(evals: Path, tmp_path: Path) -> None:
    out = tmp_path / "smoke" / "answers.json"
    assert generate(evals, "--limit", "1", "--out", str(out)) == EXIT_OK
    run = load_answers(out)
    assert (run.limit, [item.id for item in run.items]) == (1, ["notes"])
    assert not (evals.parent / "eval" / "runs").exists()  # the committed file is untouched


def test_no_index_cannot_run(
    make_config: MakeConfig,
    sample_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_QDRANT_COLLECTION, "no_index")  # a collection nothing published
    config = make_config(use_fakes)
    write_golden(config)
    assert generate(config) == EXIT_CANNOT_RUN
    assert "no index" in capsys.readouterr().err


def test_an_index_that_lags_the_corpus_is_refused(
    evals: Path, sample_corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Answers from old chunks, under a fingerprint claiming the new ones, would pass check.
    (sample_corpus / "notes.txt").write_text("Edited after the last ingest.\n")
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "not up to date with the corpus" in err and "1 changed (e.g. notes.txt)" in err
    assert "Run rag-ingest first" in err


@pytest.mark.parametrize(
    ("edit", "stale"),
    [
        pytest.param(
            lambda d: d["components"]["splitters"]["english_recursive"].update(chunk_size=500),
            "documents 5 changed (e.g. README.md, guide.pdf, notes.txt)",
            id="the default splitter",
        ),
        # S3-4: each document's own splitter decides, so only the .md file is stale.
        pytest.param(
            lambda d: d["pipeline"]["ingestion"].update(
                splitters={".md": "components.splitters.markdown"}
            ),
            "documents 1 changed (e.g. README.md)",
            id="a per-extension splitter",
        ),
    ],
)
def test_an_index_built_by_another_splitter_is_refused(
    evals: Path, capsys: pytest.CaptureFixture[str], edit: Any, stale: str
) -> None:
    data: dict[str, Any] = yaml.safe_load(evals.read_text())
    edit(data)
    evals.write_text(yaml.safe_dump(data, sort_keys=False))
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert f"not up to date with the corpus and config: {stale}." in err


def test_an_index_without_the_configured_sparse_vectors_is_refused(
    evals: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # S3-5: rag-ingest would rebuild it with them, so the answers would come from an index
    # the run's fingerprint does not describe. A dense retriever would not even notice.
    data: dict[str, Any] = yaml.safe_load(evals.read_text())
    data["pipeline"]["ingestion"]["sparse_encoder"] = "components.sparse_encoders.bm25"
    evals.write_text(yaml.safe_dump(data, sort_keys=False))
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert (
        "not up to date with the corpus and config: its sparse vectors are not the "
        "configured sparse encoder's." in err
    )


def test_an_index_whose_top_level_splitter_alone_differs_is_current(
    fake_rag: Path, sample_corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Every file of the corpus has its own splitter, so the default chunks nothing: its
    # edit leaves the index as rag-ingest would build it, and generate must not refuse it.
    data: dict[str, Any] = yaml.safe_load(fake_rag.read_text())
    ingestion = data["pipeline"]["ingestion"]
    splitters = data["components"]["splitters"]
    splitters["own"] = dict(splitters["english_recursive"])
    ingestion["splitters"] = {ext: "components.splitters.own" for ext in ingestion["loaders"]}
    splitters["english_recursive"]["chunk_size"] = 500
    fake_rag.write_text(yaml.safe_dump(data, sort_keys=False))
    write_golden(fake_rag)
    assert generate(fake_rag) == EXIT_OK, capsys.readouterr().err


def test_a_generation_error_cannot_run_and_shows_its_traceback(
    evals: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Ollama went away")

    monkeypatch.setattr("rag_qa.evaluation.generation.answer", broken)
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Traceback" in err and "generation could not run: ConnectionError" in err


@pytest.fixture
def flaky(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``answer`` fails the first time it is asked "report", as Ollama dropping out would.
    Returns the ids asked, in order."""
    asked: list[str] = []

    async def answer_once_broken(pipeline: Any, item: Any) -> Any:
        asked.append(item.id)
        if item.id == "report" and asked.count("report") == 1:
            raise ConnectionError("Ollama went away")
        return await answer(pipeline, item)

    monkeypatch.setattr("rag_qa.evaluation.generation.answer", answer_once_broken)
    return asked


def test_an_interrupted_run_resumes_where_it_stopped(
    evals: Path, flaky: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    runs = evals.parent / "eval" / "runs"
    kept = runs / "answers-latest.partial.jsonl"
    assert generate(evals) == EXIT_CANNOT_RUN
    assert not (runs / "answers-latest.json").exists()
    header, *records = [json.loads(line) for line in kept.read_text().splitlines()]
    assert [r["id"] for r in records] == ["notes"]  # what finished before the failure
    err = capsys.readouterr().err
    assert "ConnectionError: Ollama went away" in err
    assert "answers-latest.partial.jsonl: once the problem above is fixed" in err

    assert generate(evals) == EXIT_OK
    assert flaky == ["notes", "report", "report", "capital"]  # "notes" was not asked again
    run = load_answers(runs / "answers-latest.json")
    assert [item.id for item in run.items] == ["notes", "report", "capital"]
    assert run.generated_at == header["started_at"]  # the run is the one that started
    assert run.items[0].model_dump(mode="json") == records[0]
    assert not kept.exists()  # deleted once the run file is written
    assert "Resuming" in capsys.readouterr().out


def test_a_finished_checkpoint_writes_without_loading_a_model(
    evals: Path, flaky: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = evals.parent / "eval" / "runs" / "answers-latest.json"
    assert generate(evals) == EXIT_CANNOT_RUN
    assert generate(evals) == EXIT_OK
    run = load_answers(out)
    # Put every answer back in the checkpoint, as if the run file's write had failed.
    kept = out.with_name("answers-latest.partial.jsonl")
    identity = {
        "fingerprint": run.fingerprint,
        "generator": run.generator.model_dump(),
        "limit": None,
    }
    header = {"kind": "answers", "started_at": run.generated_at, "identity": identity}
    kept.write_text(
        "".join(
            json.dumps(line) + "\n"
            for line in [header, *[item.model_dump(mode="json") for item in run.items]]
        )
    )

    def no_model(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing was left to ask, so no model should load")

    monkeypatch.setattr("rag_qa.evaluation.generation.build_query_pipeline", no_model)
    assert generate(evals) == EXIT_OK
    assert load_answers(out) == run and not kept.exists()


def test_a_checkpoint_of_another_setup_is_refused(
    evals: Path, flaky: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert generate(evals) == EXIT_CANNOT_RUN
    capsys.readouterr()
    data: dict[str, Any] = yaml.safe_load(evals.read_text())
    data["components"]["retrievers"]["vector_search"]["search_kwargs"]["k"] = 4  # moves query
    evals.write_text(yaml.safe_dump(data, sort_keys=False))
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "cannot be resumed: it is of a run with another fingerprint" in err
    assert "Delete it to start this run over" in err
    assert "the same command resumes" not in err
    assert flaky == ["notes", "report"]  # nothing was asked under the new setup


def test_a_limited_run_neither_resumes_nor_deletes_a_full_runs_checkpoint(
    evals: Path, flaky: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out = evals.parent / "eval" / "runs" / "answers-latest.json"
    assert generate(evals) == EXIT_CANNOT_RUN  # a full run, interrupted
    kept = out.with_name("answers-latest.partial.jsonl")
    before = kept.read_text()
    capsys.readouterr()
    assert generate(evals, "--limit", "1", "--out", str(out)) == EXIT_CANNOT_RUN
    assert "it is of a run with another limit" in capsys.readouterr().err
    assert kept.read_text() == before


def test_an_unknown_digest_is_no_reason_to_delete_the_answers(
    evals: Path,
    flaky: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    digest = {"now": MISTRAL_DIGEST}

    def mistral(config: RagConfig) -> Generator:
        return Generator(ref="components.llms.fake", model="mistral", digest=digest["now"])

    monkeypatch.setattr("rag_qa.evaluation.generation.generator_record", mistral)
    assert generate(evals) == EXIT_CANNOT_RUN
    capsys.readouterr()
    digest["now"] = None  # Ollama still coming back: /api/tags did not answer
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Ollama did not report mistral's digest" in err and "Delete" not in err
    digest["now"] = MISTRAL_DIGEST
    assert generate(evals) == EXIT_OK
    assert flaky == ["notes", "report", "report", "capital"]


def test_ctrl_c_says_where_the_finished_answers_are(
    evals: Path,
    flaky: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def interrupted(pipeline: Any, item: Any) -> Any:
        if item.id == "report":
            raise KeyboardInterrupt
        return await answer(pipeline, item)

    monkeypatch.setattr("rag_qa.evaluation.generation.answer", interrupted)
    with pytest.raises(KeyboardInterrupt):
        generate(evals)
    assert "answers-latest.partial.jsonl: once the problem above" in capsys.readouterr().err


def test_a_run_file_that_cannot_be_written_keeps_its_checkpoint(
    evals: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def full_disk(*args: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("rag_qa.evaluation.cli.write_run", full_disk)
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "No space left on device" in err and "the same command resumes" in err
    kept = evals.parent / "eval" / "runs" / "answers-latest.partial.jsonl"
    assert len(kept.read_text().splitlines()) == 4  # the header and all three answers


def load_config_from(data: dict[str, Any]) -> RagConfig:
    return RagConfig.model_validate(data)


def test_an_ollama_generator_records_its_digest(fake_ollama: FakeOllama) -> None:
    data = yaml.safe_load(REAL_CONFIG.read_text())
    data["components"]["llms"]["mistral_ollama"]["base_url"] = fake_ollama.url
    config = load_config_from(data)
    record = generator_record(config)
    assert (record.model, record.digest) == ("mistral", MISTRAL_DIGEST)


def test_an_unreachable_ollama_records_no_digest() -> None:
    data = yaml.safe_load(REAL_CONFIG.read_text())
    data["components"]["llms"]["mistral_ollama"]["base_url"] = "http://127.0.0.1:9"
    assert generator_record(load_config_from(data)).digest is None


# --- tier 2 and the hosted fallback (S3-6, DEC-21) ------------------------------------------


def refuse_the_fallback(config: Any) -> None:
    raise AssertionError("tier 2 built the hosted fallback")


def test_generate_never_builds_the_fallback_even_with_one_configured(
    evals: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Tier 2 measures the model it names: no answer of a run may come from elsewhere, and its
    # key (unset here) is not needed.
    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["pipeline"]["query"]["fallback_llm"] = "components.llms.groq_fallback"

    make_config(edit)  # over the same file as evals
    monkeypatch.setattr("rag_qa.components.build_fallback_llm", refuse_the_fallback)
    monkeypatch.setattr("rag_qa.chain.build_fallback_llm", refuse_the_fallback)
    assert generate(evals) == EXIT_OK
    run = load_answers(evals.parent / "eval" / "runs" / "answers-latest.json")
    assert {item.answer for item in run.items} == {FAKE_ANSWER}


def test_generate_with_ollama_down_cannot_run_and_names_it(
    evals: Path, make_config: MakeConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    # Its start-up check, as rag-query's (S3-6): one reason and exit 2, not a traceback.
    def edit(c: dict[str, Any]) -> None:
        use_fakes(c)
        c["components"]["llms"]["probe"] = {
            "_target_": "langchain_ollama.ChatOllama",
            "model": "mistral",
            "base_url": dead_url(),
            "validate_model_on_init": True,
        }
        c["pipeline"]["query"]["llm"] = "components.llms.probe"

    make_config(edit)
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Error: cannot build the language model components.llms.probe" in err
    assert OLLAMA_HINT in err
    assert "Traceback" not in err


def dead_url() -> str:
    """A local port nothing listens on, so a connection there is refused at once."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"
