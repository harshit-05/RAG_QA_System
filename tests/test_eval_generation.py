"""``rag-eval generate`` (S2-5): every golden question through ``stream_answer``, recorded.

Runs against ``fake_rag``: the real ingest built an index of ``sample_corpus`` with the
fake embedder, and the llm is a fake chat model, so nothing needs Ollama or a download.
"""

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import (
    FAKE_ANSWER,
    MISTRAL_DIGEST,
    REAL_CONFIG,
    FakeOllama,
    MakeConfig,
    StreamingFakeChatModel,
    use_fakes,
)

from rag_qa.chain import build_query_pipeline
from rag_qa.config import load_config
from rag_qa.evaluation import fingerprint as fp
from rag_qa.evaluation.cli import EXIT_CANNOT_RUN, EXIT_OK, main
from rag_qa.evaluation.dataset import load_golden
from rag_qa.evaluation.gate import load_answers
from rag_qa.evaluation.generation import answer, generator_record
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

GOLDEN = [
    {"id": "notes", "question": "What do the notes say?", "ground_truth": "Plain text notes.",
     "answerable": True, "expected_sources": [{"source": "notes.txt"}]},
    {"id": "report", "question": "What is the report about?", "ground_truth": "A report.",
     "answerable": True, "expected_sources": [{"source": "sub/report.docx"}]},
    {"id": "capital", "question": "What is the capital of Australia?",
     "ground_truth": "I could not find the answer in the provided documents.",
     "answerable": False, "must_not_contain": ["Canberra"]},
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
        "components.llms.fake", None, None
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
    make_config: MakeConfig, sample_corpus: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "no-index"))
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


def test_an_index_built_by_another_splitter_is_refused(
    evals: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data: dict[str, Any] = yaml.safe_load(evals.read_text())
    data["components"]["splitters"]["english_recursive"]["chunk_size"] = 500
    evals.write_text(yaml.safe_dump(data, sort_keys=False))
    assert generate(evals) == EXIT_CANNOT_RUN
    assert "the splitter is not the one that built it" in capsys.readouterr().err


def test_a_generation_error_cannot_run_and_shows_its_traceback(
    evals: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Ollama went away")

    monkeypatch.setattr("rag_qa.evaluation.generation.answer", broken)
    assert generate(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Traceback" in err and "generation could not run: ConnectionError" in err


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
