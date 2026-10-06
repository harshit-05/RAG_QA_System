"""``rag-eval score`` (S2-5) without ragas: its refusals, the decline rate, and the guards.

CI never installs the ``eval`` extra, so everything here runs without it: ragas is only
imported once every refusal has passed, and the conftest's fake Ollama stands in for the
judge's server. The few tests that need ragas or instructor themselves skip without them.
"""

import importlib.metadata
import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import BASE_DIGEST, JUDGE_DIGEST, REAL_CONFIG, REPO_ROOT, FakeOllama

from rag_qa.config import load_config
from rag_qa.evaluation import fingerprint as fp
from rag_qa.evaluation import ragas_scoring as rs
from rag_qa.evaluation.cli import EXIT_CANNOT_RUN, main
from rag_qa.evaluation.dataset import GoldenItem, load_golden
from rag_qa.evaluation.gate import AnswerItem, AnswersRun, Generator, write_run
from rag_qa.settings import ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

MARKER = "could not find the answer"
REFUSAL = "I could not find the answer in the provided documents."
GOLDEN = [
    {"id": "notes", "question": "What do the notes say?", "ground_truth": "Plain text notes.",
     "answerable": True, "expected_sources": [{"source": "notes.txt"}]},
    {"id": "capital", "question": "What is the capital of Australia?", "ground_truth": REFUSAL,
     "answerable": False, "must_not_contain": ["Canberra"]},
]


# ---- the decline rate ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "declined", "leaked"),
    [
        pytest.param(REFUSAL, True, [], id="the refusal sentence"),
        pytest.param("Sorry, I COULD NOT FIND THE ANSWER here.", True, [], id="any case"),
        pytest.param("I could not find the answer to that in these papers.", True, [],
                     id="a paraphrase that keeps the marker"),
        pytest.param("The documents do not say.", False, [], id="a paraphrase that loses it"),
        pytest.param("It is Canberra.", False, ["Canberra"], id="an answer from prior knowledge"),
        pytest.param(f"{REFUSAL} It may be Canberra.", False, ["Canberra"],
                     id="a refusal that leaks"),
        pytest.param(f"{REFUSAL} (canberra?)", False, ["Canberra"], id="a leak in another case"),
        pytest.param(f"{REFUSAL} Canberrans live there.", True, [], id="not as a whole word"),
    ],
)
def test_declines(answer: str, declined: bool, leaked: list[str]) -> None:
    assert rs.declines(answer, MARKER, ["Canberra"]) == (declined, leaked)


@pytest.mark.parametrize(
    ("word", "answer", "leaks"),
    [
        ("COCO", "trained on coco", True),  # a leak in another case still counts
        ("COCO", "the cocoa bean", False),  # whole words only
        ("H100", "on an H100.", True),  # at the end of a sentence
        ("H100", "on an H1000", False),
        ("U.S.", "in the U.S. army", True),  # \b would miss a word ending in punctuation
        ("C++", "written in c++, mostly", True),
    ],
)
def test_leaks_match_whole_words_in_any_case(word: str, answer: str, leaks: bool) -> None:
    assert bool(rs.leak_pattern(word).search(answer.lower())) is leaks


def item(record: dict[str, Any]) -> GoldenItem:
    return GoldenItem.model_validate_json(json.dumps(record))


def test_an_unanswerable_ground_truth_without_the_marker_is_refused() -> None:
    golden = [item(GOLDEN[0]), item(dict(GOLDEN[1], ground_truth="No idea."))]
    with pytest.raises(rs.ScoringError, match="unanswerable record.s. capital does not contain"):
        rs.check_references(golden, MARKER)
    rs.check_references([item(r) for r in GOLDEN], MARKER.upper())  # compared in any case


def test_the_mean_leaves_out_what_was_not_scored() -> None:
    assert rs.mean([1.0, None, 0.5]) == 0.75
    assert rs.mean([None, None]) is None


def test_an_empty_answer_or_no_context_is_never_sent_to_the_judge() -> None:
    answered = AnswerItem(id="q", question="?", answerable=True, answer="", contexts=["c"],
                          sources=[], ttft_ms=None, total_ms=1.0)
    inputs = rs.metric_inputs(answered, "ref")
    assert [name for name, i in inputs.items() if rs.unscorable(i)] == [
        "faithfulness", "answer_relevancy"
    ]
    no_context = answered.model_copy(update={"answer": "a", "contexts": []})
    inputs = rs.metric_inputs(no_context, "ref")
    assert [name for name, i in inputs.items() if rs.unscorable(i)] == [
        "faithfulness", "context_precision", "context_recall"
    ]


# ---- the context guard -------------------------------------------------------------------


def response(prompt: int | None, completion: int | None) -> Any:
    usage = types.SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    return types.SimpleNamespace(usage=usage)


def test_the_call_log_keeps_the_largest_counts() -> None:
    log = rs.CallLog(num_ctx=8192, metric="faithfulness")
    log.on_response(response(593, 84))
    log.on_response(response(2753, 155))
    assert (log.calls, log.max_prompt, log.max_completion, log.overflow) == (2, 2753, 155, None)


@pytest.mark.parametrize(
    ("prompt", "completion"),
    [
        pytest.param(8192 - 255, 10, id="a prompt within 256 of num_ctx"),
        pytest.param(6000, 2192, id="a prompt and answer that fill it"),
        pytest.param(None, None, id="no usage reported"),
    ],
)
def test_a_judgement_that_may_be_truncated_is_kept_as_an_overflow(
    prompt: int | None, completion: int | None
) -> None:
    log = rs.CallLog(num_ctx=8192, metric="context_recall")
    log.on_response(response(prompt, completion))
    assert log.overflow is not None and "context_recall" in log.overflow


def test_the_largest_safe_prompt_passes() -> None:
    log = rs.CallLog(num_ctx=8192)
    log.on_response(response(8192 - 256, 100))
    assert log.overflow is None


def test_parse_retries_are_counted_per_metric() -> None:
    log = rs.CallLog(num_ctx=8192, metric="faithfulness")
    log.on_parse_error(ValueError("bad JSON"), is_last_attempt=False)
    log.metric = "context_recall"
    log.on_parse_error(ValueError("bad JSON"), is_last_attempt=False)
    log.on_parse_error(ValueError("bad JSON"), is_last_attempt=False)
    # A last attempt's parse error is that call's failure, counted as `parse`, not here.
    log.on_parse_error(ValueError("bad JSON"), is_last_attempt=True)
    assert dict(log.retried) == {"faithfulness": 1, "context_recall": 2}


# ---- the refusals, through the command ------------------------------------------------------


@pytest.fixture
def scoring(
    tmp_path: Path, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch, fake_ollama: FakeOllama
) -> Path:
    """A checkout whose answers file is fresh and whose judge is served right, by the fake
    Ollama: every refusal passes until the one a test breaks. Returns the config file."""
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "index"))
    data = yaml.safe_load(REAL_CONFIG.read_text())
    data["evaluation"]["judge"]["base_url"] = f"{fake_ollama.url}/v1"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(data, sort_keys=False))
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "eval_dataset.jsonl").write_text("".join(json.dumps(r) + "\n" for r in GOLDEN))
    (eval_dir / "judge.Modelfile").write_text((REPO_ROOT / "eval" / "judge.Modelfile").read_text())
    golden = load_golden(eval_dir / "eval_dataset.jsonl")
    parts = fp.fingerprint(load_config(config_file), golden, parts=fp.GENERATION_PARTS)
    answers = AnswersRun(
        generated_at="2026-10-07T00:00:00Z", limit=None, fingerprint=parts,
        generator=Generator(ref="components.llms.mistral_ollama", model="mistral", digest=None),
        items=[AnswerItem(id=r["id"], question=r["question"], answerable=r["answerable"],
                          answer=REFUSAL, contexts=["c"], sources=[], ttft_ms=1.0, total_ms=2.0)
               for r in GOLDEN],
    )
    write_run(answers, eval_dir / "runs" / "answers-latest.json")
    return config_file


def score(config_file: Path, *args: str) -> int:
    out = config_file.parent / "scratch-scores.json"
    return main(["score", "--config", str(config_file), "--out", str(out), *args])


@pytest.fixture
def no_ragas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop where ragas would be imported, as on CI, whether or not this machine has it."""
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)


def test_everything_before_ragas_passes_then_the_extra_is_required(
    scoring: Path, no_ragas: None, fake_ollama: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    assert score(scoring) == EXIT_CANNOT_RUN
    assert "needs the eval extra" in capsys.readouterr().err
    assert "/api/show" in fake_ollama.requests  # the judge was checked first


def test_another_ragas_version_is_refused(
    scoring: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.4.2")
    assert score(scoring) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "ragas 0.4.2 is installed, but this checkout scores with 0.4.3" in err


def test_answers_from_another_checkout_are_refused(
    scoring: Path, sample_corpus: Path, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    (sample_corpus / "notes.txt").write_text("Changed since the answers were generated.\n")
    assert score(scoring) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "another checkout (corpus changed: re-run rag-ingest, then generate and score" in err


def test_a_judge_served_with_another_context_is_refused(
    scoring: Path, fake_ollama: FakeOllama, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # The one that fails silently if missed: a 4k judge truncates RAGAs' prompts.
    fake_ollama.parameters["rag-judge:latest"] = "num_ctx                        4096"
    assert score(scoring) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "served with num_ctx 4096, but its Modelfile says 8192" in err
    assert "ollama create rag-judge -f " in err and "eval/judge.Modelfile" in err


def test_a_judge_without_a_context_setting_is_refused(
    scoring: Path, fake_ollama: FakeOllama, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_ollama.parameters["rag-judge:latest"] = "temperature                    0"
    assert score(scoring) == EXIT_CANNOT_RUN
    assert "num_ctx unset (Ollama defaults to 4k)" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("served", "parent", "problem"),
    [
        pytest.param("num_ctx 8192\nnum_predict 1024\nseed 42\ntemperature 0", "gemma2:9b",
                     "num_predict 1024 (the Modelfile says 2048)", id="another parameter"),
        pytest.param("num_ctx 8192\nnum_predict 2048\nseed 42", "gemma2:9b",
                     "temperature unset (the Modelfile says 0)", id="a parameter missing"),
        pytest.param("num_ctx 8192\nnum_predict 2048\nseed 42\ntemperature 0", "llama3:8b",
                     "built on llama3:8b (the Modelfile says FROM gemma2:9b)", id="another base"),
    ],
)
def test_a_judge_not_built_from_its_modelfile_is_refused(
    scoring: Path, fake_ollama: FakeOllama, no_ragas: None, capsys: pytest.CaptureFixture[str],
    served: str, parent: str, problem: str,
) -> None:
    # An edited Modelfile, never re-created: the run would hash a judge that did not run.
    fake_ollama.parameters["rag-judge:latest"] = served
    fake_ollama.parents["rag-judge:latest"] = parent
    assert score(scoring) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert problem in err and "ollama create rag-judge" in err


def test_a_served_value_written_another_way_is_the_same() -> None:
    assert rs._same("0", "0.0") and rs._same("42", "42") and not rs._same(None, "0")


def test_a_missing_judge_is_refused(
    scoring: Path, fake_ollama: FakeOllama, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    del fake_ollama.parameters["rag-judge:latest"]
    assert score(scoring) == EXIT_CANNOT_RUN
    assert "Ollama has no model 'rag-judge'. Run: ollama create" in capsys.readouterr().err


def test_an_unreachable_ollama_is_refused(
    scoring: Path, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(scoring.read_text())
    data["evaluation"]["judge"]["base_url"] = "http://127.0.0.1:9/v1"
    scoring.write_text(yaml.safe_dump(data, sort_keys=False))
    assert score(scoring) == EXIT_CANNOT_RUN
    assert "cannot reach Ollama at http://127.0.0.1:9" in capsys.readouterr().err


def test_an_unanswerable_ground_truth_without_the_marker_exits_2(
    scoring: Path, no_ragas: None, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = scoring.parent / "eval" / "eval_dataset.jsonl"
    records = [dict(GOLDEN[0]), dict(GOLDEN[1], ground_truth="Unknown.")]
    golden.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert score(scoring) == EXIT_CANNOT_RUN
    assert "capital does not contain the decline marker" in capsys.readouterr().err


def test_scoring_another_answers_file_needs_out(
    scoring: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A smoke run's answers must never be scored over the committed 13-hour baseline.
    answers = scoring.parent / "eval" / "runs" / "answers-latest.json"
    status = main(["score", "--config", str(scoring), "--answers", str(answers)])
    assert status == EXIT_CANNOT_RUN
    assert "needs --out" in capsys.readouterr().err
    assert not (scoring.parent / "eval" / "runs" / "generation-latest.json").exists()


def test_parse_errors_reach_the_log_through_instructors_own_hooks() -> None:
    # instructor calls a handler with attempt_number, max_attempts and is_last_attempt, or,
    # when it cannot take all three, with the error alone (S2-5's second review).
    if importlib.util.find_spec("instructor") is None:
        pytest.skip("the eval extra is not installed")
    from instructor.core.hooks import Hooks

    log = rs.CallLog(num_ctx=8192, metric="faithfulness")
    hooks = Hooks()
    hooks.on("parse:error", log.on_parse_error)
    hooks.emit_parse_error(ValueError("x"), attempt_number=1, max_attempts=2, is_last_attempt=False)
    hooks.emit_parse_error(ValueError("x"), attempt_number=2, max_attempts=2, is_last_attempt=True)
    assert dict(log.retried) == {"faithfulness": 1}


def test_a_missing_answers_file_exits_2(scoring: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert score(scoring, "--answers", str(scoring.parent / "nope.json")) == EXIT_CANNOT_RUN
    assert "cannot read the answers file" in capsys.readouterr().err


def test_the_judge_digests_come_from_ollama(fake_ollama: FakeOllama) -> None:
    settings = fp.parse_modelfile((REPO_ROOT / "eval" / "judge.Modelfile").read_text())
    digests = rs.check_judge(fake_ollama.url, "rag-judge", "judge.Modelfile", settings)
    assert digests == (JUDGE_DIGEST, BASE_DIGEST)


# ---- what needs the eval extra: skipped without it, as on CI -----------------------------


def test_failures_are_classified_by_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("the eval extra is not installed")
    from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException

    monkeypatch.delenv("RAGAS_DO_NOT_TRACK", raising=False)  # import_ragas sets it
    ragas = rs.import_ragas()

    def wrapped(cause: BaseException, attempts: int) -> BaseException:
        # As instructor raises it: the final error as the cause.
        error = InstructorRetryException(str(cause), n_attempts=attempts, total_usage=0,
                                         failed_attempts=[object()] * (attempts - 1))
        error.__cause__ = cause
        return error

    bad_json = json.JSONDecodeError("Expecting value", "not json", 0)
    assert rs.classify(wrapped(bad_json, 2), ragas) == "parse"
    assert rs.classify(IncompleteOutputException(), ragas) == "truncated"
    # A dead or slow Ollama aborts the run: it is no judgement. Even on the retry of an
    # attempt that failed to parse (S2-5's first review).
    assert rs.classify(wrapped(ConnectionError("refused"), 1), ragas) is None
    assert rs.classify(wrapped(TimeoutError("2400 s"), 2), ragas) is None
    assert rs.classify(TimeoutError(), ragas) is None


def test_the_removed_module_shim_is_still_needed() -> None:
    # When this fails, a ragas release no longer imports the module langchain-community
    # 0.4.2 removed, or langchain-community has it again: delete the shim in import_ragas.
    spec = importlib.util.find_spec("ragas")
    if spec is None or spec.origin is None:
        pytest.skip("the eval extra is not installed")
    # The file, not find_spec: once import_ragas has run, the stand-in is in sys.modules.
    chat_models = importlib.util.find_spec("langchain_community.chat_models")
    assert chat_models is not None and chat_models.origin is not None
    assert not (Path(chat_models.origin).parent / "vertexai.py").exists()
    base = (Path(spec.origin).parent / "llms" / "base.py").read_text()
    assert f"from {rs.REMOVED_MODULE} import ChatVertexAI" in base


def test_ragas_imports_with_the_shim_and_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("the eval extra is not installed")
    monkeypatch.delenv("RAGAS_DO_NOT_TRACK", raising=False)  # restored after the test
    ragas = rs.import_ragas()
    from ragas import _analytics

    assert ragas.llm_factory and ragas.collections.Faithfulness
    assert _analytics.do_not_track() is True
