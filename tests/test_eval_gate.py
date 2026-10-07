"""Tier 2's fingerprint and gate (S2-5): what moves which part, and ``rag-eval check``.

The fingerprint is computed from a scratch world: the real ``config.yaml``, a two-record
golden set, the real judge Modelfile and ``sample_corpus``. Nothing is built: no model,
no index, no Ollama. ``check --with-ollama`` meets the conftest's fake Ollama.
"""

import copy
import json
import os
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import BASE_DIGEST, JUDGE_DIGEST, MISTRAL_DIGEST, REAL_CONFIG, REPO_ROOT, FakeOllama

from rag_qa.config import load_config
from rag_qa.evaluation import fingerprint as fp
from rag_qa.evaluation.cli import EXIT_CANNOT_RUN, EXIT_FLOOR_MISSED, EXIT_OK, main
from rag_qa.evaluation.dataset import load_golden
from rag_qa.evaluation.gate import (
    GENERATION_METRICS,
    AnswerItem,
    AnswersRun,
    Failures,
    GenerationFloors,
    Generator,
    JudgeRecord,
    ScoreItem,
    ScoresRun,
    ThresholdsError,
    answers_identity,
    below_generation_floors,
    digest_drift,
    load_thresholds,
    ollama_root,
    too_many_unscored,
    write_run,
)
from rag_qa.settings import ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

GOLDEN: list[dict[str, Any]] = [
    {
        "id": "kestrel-notes",
        "question": "What do the notes say?",
        "ground_truth": "They are plain text notes.",
        "answerable": True,
        "expected_sources": [{"source": "notes.txt"}],
        "notes": "notes.txt: 'Plain text notes.'",
    },
    {
        "id": "capital",
        "question": "What is the capital of Australia?",
        "ground_truth": "I could not find the answer in the provided documents.",
        "answerable": False,
        "must_not_contain": ["Canberra"],
    },
]

THRESHOLDS = {
    "retrieval": {"hit_rate": 0.5, "mrr": 0.5, "recall": 0.5},
    "generation": {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.7,
        "context_recall": 0.7,
        "decline_rate": 0.8,
        "max_unscored": 0.1,
    },
}


@dataclass
class World:
    """Everything a fingerprint is computed from, as editable data, written on demand."""

    root: Path
    corpus: Path
    config: dict[str, Any]
    golden: list[dict[str, Any]]
    modelfile: str

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    def write(self) -> Path:
        self.config_file.write_text(yaml.safe_dump(self.config, sort_keys=False))
        self.eval_dir.mkdir(exist_ok=True)
        (self.eval_dir / "eval_dataset.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in self.golden)
        )
        (self.eval_dir / "judge.Modelfile").write_text(self.modelfile)
        (self.eval_dir / "thresholds.yaml").write_text(yaml.safe_dump(THRESHOLDS))
        return self.config_file

    def parts(self) -> dict[str, str]:
        config = load_config(self.write())
        golden = load_golden(self.eval_dir / "eval_dataset.jsonl")
        return fp.fingerprint(config, golden, self.eval_dir)


@pytest.fixture
def world(tmp_path: Path, sample_corpus: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "index"))
    return World(
        root=tmp_path,
        corpus=sample_corpus,
        config=yaml.safe_load(REAL_CONFIG.read_text()),
        golden=copy.deepcopy(GOLDEN),
        modelfile=(REPO_ROOT / "eval" / "judge.Modelfile").read_text(),
    )


# ---- each input moves its own part, and only that one --------------------------------------

Change = Callable[[World, pytest.MonkeyPatch], None]


def config_edit(edit: Callable[[dict[str, Any]], None]) -> Change:
    return lambda w, _: edit(w.config)


def golden_edit(edit: Callable[[list[dict[str, Any]]], None]) -> Change:
    return lambda w, _: edit(w.golden)


def query(c: dict[str, Any]) -> dict[str, Any]:
    return c["pipeline"]["query"]


def llm(c: dict[str, Any]) -> dict[str, Any]:
    return c["components"]["llms"]["mistral_ollama"]


def reranker(c: dict[str, Any]) -> dict[str, Any]:
    return c["components"]["rerankers"]["ms_marco_minilm_cpu"]


def setter(get: Callable[[dict[str, Any]], dict[str, Any]], key: str, value: Any) -> Change:
    return config_edit(lambda c: get(c).__setitem__(key, value))


def rename_llm(c: dict[str, Any]) -> None:
    c["components"]["llms"]["renamed"] = c["components"]["llms"].pop("mistral_ollama")
    query(c)["llm"] = "components.llms.renamed"


def append_to_file(name: str, text: str) -> Change:
    def change(w: World, _: pytest.MonkeyPatch) -> None:
        with open(w.corpus / name, "a", encoding="utf-8") as f:
            f.write(text)

    return change


def add_file(name: str, data: bytes = b"x") -> Change:
    def change(w: World, _: pytest.MonkeyPatch) -> None:
        path = w.corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return change


def add_symlink(w: World, _: pytest.MonkeyPatch) -> None:
    (w.corpus / "linked.txt").symlink_to(w.corpus / "notes.txt")


def move_corpus(w: World, monkeypatch: pytest.MonkeyPatch) -> None:
    copy_ = w.root / "elsewhere" / "corpus"
    shutil.copytree(w.corpus, copy_, symlinks=True)
    monkeypatch.setenv(ENV_DATA_PATH, str(copy_))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(w.root / "another" / "index"))


def first(edit: Callable[[dict[str, Any]], None]) -> Change:
    return golden_edit(lambda g: edit(g[0]))


def second(edit: Callable[[dict[str, Any]], None]) -> Change:
    return golden_edit(lambda g: edit(g[1]))


def judge(c: dict[str, Any]) -> dict[str, Any]:
    return c["evaluation"]["judge"]


#: A Modelfile SYSTEM instruction: it changes the judge's verdicts, so it moves `judge`.
SYSTEM_LINE = 'SYSTEM """Judge strictly."""\n'


def one_word(prompt: dict[str, Any]) -> None:
    """S2-5b's one-word prompt change."""
    prompt["system"] = prompt["system"].replace("helpful and precise", "helpful, precise")


MOVES: list[Any] = [
    # query
    pytest.param(setter(llm, "temperature", 0.2), "query", id="llm setting"),
    pytest.param(setter(llm, "num_thread", 6), "query", id="llm num_thread (S2-5b)"),
    pytest.param(
        config_edit(lambda c: c["components"]["retrievers"]["vector_search"]["search_kwargs"]
                    .__setitem__("k", 4)),
        "query", id="retriever k",
    ),
    pytest.param(setter(reranker, "top_n", 4), "query", id="reranker top_n"),
    pytest.param(setter(query, "reranker_candidates", 30), "query", id="reranker candidates"),
    pytest.param(config_edit(lambda c: one_word(query(c)["prompt"])), "query", id="a prompt word"),
    pytest.param(
        lambda w, mp: mp.setattr("rag_qa.chain.citation", lambda doc: "cited differently"),
        "query", id="citation code (the probe)",
    ),
    # ingestion
    pytest.param(
        config_edit(lambda c: c["components"]["embedders"]["minilm_cpu"].__setitem__(
            "model_name", "sentence-transformers/all-mpnet-base-v2")),
        "ingestion", id="embedder model",
    ),
    pytest.param(
        config_edit(lambda c: c["components"]["splitters"]["english_recursive"].__setitem__(
            "chunk_size", 800)),
        "ingestion", id="splitter",
    ),
    pytest.param(
        config_edit(lambda c: c["components"]["loaders"]["txt"].__setitem__("encoding", "utf-8")),
        "ingestion", id="a loader's spec",
    ),
    pytest.param(
        lambda w, mp: mp.setattr(fp, "CHUNKING_VERSION", 2), "ingestion", id="chunking version"
    ),
    # corpus
    pytest.param(append_to_file("notes.txt", "More.\n"), "corpus", id="a file's bytes"),
    pytest.param(add_file("added.txt"), "corpus", id="a new indexed file"),
    # questions
    pytest.param(first(lambda r: r.__setitem__("question", "What do notes say?")), "questions",
                 id="a question"),
    pytest.param(second(lambda r: r.__setitem__("must_not_contain", ["Sydney"])), "questions",
                 id="must_not_contain"),
    # references
    pytest.param(first(lambda r: r.__setitem__("ground_truth", "Notes, in plain text.")),
                 "references", id="a ground_truth"),
    # judge
    pytest.param(lambda w, _: setattr(w, "modelfile", w.modelfile.replace("seed 42", "seed 7")),
                 "judge", id="the Modelfile"),
    pytest.param(lambda w, _: setattr(w, "modelfile", w.modelfile + SYSTEM_LINE),
                 "judge", id="a SYSTEM line in the Modelfile"),
    pytest.param(setter(judge, "model", "other-judge"), "judge", id="the judge model"),
    pytest.param(setter(judge, "embedding_model", "sentence-transformers/all-mpnet-base-v2"),
                 "judge", id="the embedder"),
    pytest.param(config_edit(lambda c: c["evaluation"].__setitem__("decline_marker",
                                                                     "could not find")),
                 "judge", id="the decline marker"),
    pytest.param(lambda w, mp: mp.setattr(fp, "RAGAS_VERSION", "0.4.4"), "judge",
                 id="the ragas version"),
    pytest.param(lambda w, mp: mp.setitem(fp.METRICS, "answer_relevancy",
                                          {"class": "AnswerRelevancy", "strictness": 3}),
                 "judge", id="a metric setting"),
]


@pytest.mark.parametrize(("change", "part"), MOVES)
def test_each_input_moves_its_own_part_only(
    world: World, monkeypatch: pytest.MonkeyPatch, change: Change, part: str
) -> None:
    before = world.parts()
    change(world, monkeypatch)
    assert fp.moved(before, world.parts()) == [part]


STAYS: list[Any] = [
    pytest.param(move_corpus, id="the corpus and index paths"),
    pytest.param(setter(judge, "base_url", "http://gpu-box:11434/v1"), id="the judge base_url"),
    pytest.param(setter(judge, "timeout_s", 60), id="the judge timeout"),
    pytest.param(setter(llm, "base_url", "http://gpu-box:11434"), id="the llm base_url"),
    # Runtime-only llm keys (S2-5's first review): none changes an answer.
    pytest.param(setter(llm, "validate_model_on_init", False), id="the start-up model check"),
    pytest.param(setter(llm, "keep_alive", "10m"), id="keep_alive"),
    pytest.param(setter(llm, "client_kwargs", {"timeout": 600}), id="the client's timeout"),
    pytest.param(lambda w, _: setattr(w, "modelfile", "# a note\n\n" + w.modelfile + "# end\n"),
                 id="a comment in the Modelfile"),
    pytest.param(setter(query, "reranker", "components.rerankers.ms_marco_minilm_cuda"),
                 id="the reranker's device"),
    pytest.param(
        config_edit(lambda c: c["pipeline"]["ingestion"].__setitem__(
            "embedder", "components.embedders.minilm_cuda")),
        id="the embedder's device",
    ),
    pytest.param(config_edit(rename_llm), id="a renamed component entry"),
    pytest.param(first(lambda r: r.__setitem__("notes", "edited")), id="notes"),
    pytest.param(first(lambda r: r.__setitem__("expected_sources", [{"source": "README.md"}])),
                 id="expected_sources"),
    pytest.param(golden_edit(lambda g: g.reverse()), id="the golden set's order"),
    pytest.param(add_file("photo.png"), id="a file no loader takes"),
    pytest.param(add_file(".hidden/notes.txt"), id="a hidden folder"),
    pytest.param(add_file("~$draft.docx", b"\x00" * 10), id="a Word lock file"),
    pytest.param(add_symlink, id="a symlink"),
]


@pytest.mark.parametrize("change", STAYS)
def test_machine_specific_and_unread_inputs_move_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, change: Change
) -> None:
    before = world.parts()
    change(world, monkeypatch)
    assert fp.moved(before, world.parts()) == []


def test_the_corpus_part_is_exactly_what_rag_ingest_indexes(world: World) -> None:
    # The same walk as rag-ingest: so a name ingest cannot index refuses the fingerprint too.
    world.write()
    scan = fp.corpus_scan(load_config(world.config_file))
    assert sorted(f.name for f in scan.files) == [
        "README.md", "guide.pdf", "notes.txt", "sub/deeper/LOUD.TXT", "sub/report.docx"
    ]


def test_a_corpus_rag_ingest_cannot_read_has_no_fingerprint(world: World) -> None:
    os.mkfifo(world.corpus / "pipe.txt")  # not a regular file: skipped, like any special file
    bad = world.corpus / "unreadable.txt"
    bad.write_text("x")
    bad.chmod(0)
    try:
        if os.access(bad, os.R_OK):
            pytest.skip("running as root: chmod 000 is still readable")
        with pytest.raises(fp.FingerprintError, match="unreadable.txt"):
            world.parts()
    finally:
        bad.chmod(0o644)


def test_a_missing_modelfile_has_no_judge_part(world: World) -> None:
    world.write()
    (world.eval_dir / "judge.Modelfile").unlink()
    config = load_config(world.config_file)
    golden = load_golden(world.eval_dir / "eval_dataset.jsonl")
    with pytest.raises(fp.FingerprintError, match="cannot read the judge's Modelfile"):
        fp.fingerprint(config, golden, world.eval_dir)
    # The generation parts never need it, so generate runs without an evaluation section.
    generation = fp.fingerprint(config, golden, parts=fp.GENERATION_PARTS)
    assert set(generation) == set(fp.GENERATION_PARTS)


def test_ragas_version_is_the_one_uv_lock_pins() -> None:
    # CI has no ragas to ask, so the fingerprint names a version: it must be the locked one.
    # A lock bump fails here until RAGAS_VERSION follows, which then moves `judge`.
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"]}
    assert fp.RAGAS_VERSION == locked["ragas"], "bump fingerprint.RAGAS_VERSION: a re-score follows"


@pytest.mark.parametrize("part", fp.PARTS)
def test_every_stale_message_names_its_rerun(part: str) -> None:
    message = fp.stale_message(part)
    assert message.startswith(f"{part} changed: re-run {fp.RERUN[part]} (")
    expected = {
        "query": "generate and score",
        # generate refuses an index that lags these, so the named fix starts with rag-ingest
        "ingestion": "rag-ingest, then generate and score",
        "corpus": "rag-ingest, then generate and score",
        "questions": "generate and score",
        "references": "score",
        "judge": "score",
    }
    assert fp.RERUN[part] == expected[part]


def test_the_modelfile_is_parsed_for_what_score_needs() -> None:
    settings = fp.parse_modelfile((REPO_ROOT / "eval" / "judge.Modelfile").read_text())
    assert (settings.base, settings.num_ctx, settings.num_predict) == ("gemma2:9b", 8192, 2048)
    assert (settings.temperature, settings.seed) == (0.0, 42)
    assert dict(settings.parameters) == {
        "num_ctx": "8192", "num_predict": "2048", "temperature": "0", "seed": "42"
    }


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("PARAMETER num_ctx 8192\n", "no FROM"),
        ("FROM gemma2:9b\nPARAMETER temperature 0\n", "sets no num_ctx"),
        ("FROM gemma2:9b\nPARAMETER num_ctx big\n", "not a number"),
        ('FROM x\nSYSTEM """\nPARAMETER num_ctx 8192\n"""\n', "sets no num_ctx"),
    ],
)
def test_a_modelfile_without_what_score_needs_is_refused(text: str, problem: str) -> None:
    with pytest.raises(fp.FingerprintError, match=problem):
        fp.parse_modelfile(text)


def test_modelfile_keywords_are_case_insensitive() -> None:
    settings = fp.parse_modelfile("# judge\nfrom gemma2:9b\nparameter NUM_CTX 4096\n")
    assert (settings.base, settings.num_ctx) == ("gemma2:9b", 4096)


# ---- the floors --------------------------------------------------------------------------


def test_generation_floors_may_be_null_but_never_missing(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    floors = dict(THRESHOLDS["generation"], faithfulness=None)
    path.write_text(yaml.safe_dump({**THRESHOLDS, "generation": floors}))
    loaded = load_thresholds(path).generation
    assert loaded is not None and loaded.faithfulness is None
    del floors["faithfulness"]
    path.write_text(yaml.safe_dump({**THRESHOLDS, "generation": floors}))
    with pytest.raises(ThresholdsError, match="generation.faithfulness"):
        load_thresholds(path)


def test_a_negative_generation_floor_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.yaml"
    path.write_text(yaml.safe_dump(
        {**THRESHOLDS, "generation": dict(THRESHOLDS["generation"], decline_rate=-0.1)}
    ))
    with pytest.raises(ThresholdsError, match="decline_rate"):
        load_thresholds(path)


def test_the_committed_thresholds_still_load_without_a_generation_section() -> None:
    # Until S2-5b sets the floors, tier 1 reads the file as before.
    assert load_thresholds(REPO_ROOT / "eval" / "thresholds.yaml").retrieval


def test_every_scored_metric_has_a_floor_key() -> None:
    # A metric added to the set but not to the floors would be scored and never gated.
    assert set(GenerationFloors.model_fields) - {"max_unscored"} == set(GENERATION_METRICS)


def test_below_generation_floors() -> None:
    floors = GenerationFloors.model_validate(dict(THRESHOLDS["generation"], decline_rate=None))
    aggregate = {
        "faithfulness": 0.7,  # at the floor: meets it
        "answer_relevancy": 0.69,
        "context_precision": None,  # nothing scored: a gate never passes on no evidence
        "context_recall": 0.9,
        "decline_rate": 0.0,  # not gated
    }
    assert below_generation_floors(aggregate, floors) == ["answer_relevancy", "context_precision"]


# ---- rag-eval check -----------------------------------------------------------------------


def runs_for(world: World, **aggregate: float | None) -> tuple[AnswersRun, ScoresRun]:
    """A committed run made from ``world`` as it is now: fresh, covering every question."""
    parts = world.parts()
    generator = Generator(ref="components.llms.mistral_ollama", model="mistral",
                          digest=MISTRAL_DIGEST)
    answers = AnswersRun(
        generated_at="2026-10-07T00:00:00Z",
        limit=None,
        fingerprint={p: parts[p] for p in fp.GENERATION_PARTS},
        generator=generator,
        items=[
            AnswerItem(id=r["id"], question=r["question"], answerable=r["answerable"],
                       answer="An answer.", contexts=["A chunk."], sources=[], ttft_ms=1.0,
                       total_ms=2.0)
            for r in world.golden
        ],
    )
    values = {
        "faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.9,
        "context_recall": 0.9, "decline_rate": 1.0, **aggregate,
    }
    scores = ScoresRun(
        generated_at=answers.generated_at,
        scored_at="2026-10-07T01:00:00Z",
        limit=None,
        answers=answers_identity(answers),
        fingerprint=parts,
        generator=generator,
        judge=JudgeRecord(
            model="rag-judge", digest=JUDGE_DIGEST, base="gemma2:9b", base_digest=BASE_DIGEST,
            num_ctx=8192, served_context=8192, max_prompt_tokens=2753, max_completion_tokens=258,
            mode="json", embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        versions={"ragas": "0.4.3", "instructor": "1.17.0", "openai": "3.3.0"},
        failures={name: Failures() for name in fp.METRICS},
        aggregate=values,
        items=[ScoreItem(id=r["id"], answerable=r["answerable"]) for r in world.golden],
    )
    return answers, scores


def commit(world: World, answers: AnswersRun, scores: ScoresRun) -> None:
    write_run(answers, world.eval_dir / "runs" / "answers-latest.json")
    write_run(scores, world.eval_dir / "runs" / "generation-latest.json")


def check(world: World, *args: str) -> int:
    return main(["check", "--config", str(world.config_file), *args])


def test_a_fresh_run_above_its_floors_passes(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(world, *runs_for(world))
    assert check(world) == EXIT_OK
    out = capsys.readouterr().out
    assert "every floor" in out and "CHANGED" not in out


def test_a_prompt_edit_makes_the_run_stale(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(world, *runs_for(world))
    one_word(query(world.config)["prompt"])
    world.write()
    assert check(world) == EXIT_FLOOR_MISSED
    captured = capsys.readouterr()
    assert "query changed: re-run generate and score" in captured.err
    assert "references" not in captured.err


def test_a_ground_truth_edit_needs_a_rescore_only(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(world, *runs_for(world))
    world.golden[0]["ground_truth"] = "Notes in plain text."
    world.write()
    assert check(world) == EXIT_FLOOR_MISSED
    err = capsys.readouterr().err
    assert "references changed: re-run score" in err and "generate" not in err


def test_check_reads_another_eval_dir(world: World, tmp_path: Path) -> None:
    # S2-5b's stale check: a scratch config elsewhere, against the repo's eval folder.
    commit(world, *runs_for(world))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "stale.yaml").write_text(world.config_file.read_text())
    assert main(["check", "--config", str(scratch / "stale.yaml"),
                 "--eval-dir", str(world.eval_dir)]) == EXIT_OK


def test_a_floor_missed_fails(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    commit(world, *runs_for(world, faithfulness=0.5))
    assert check(world) == EXIT_FLOOR_MISSED
    assert "faithfulness 0.5 < 0.7" in capsys.readouterr().err


def test_a_limited_run_never_passes(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    answers, scores = runs_for(world)
    answers = answers.model_copy(update={"items": answers.items[:1]})
    scores = scores.model_copy(update={"items": scores.items[:1]})
    commit(world, answers, scores)
    assert check(world) == EXIT_FLOOR_MISSED
    assert "covers 1 of the golden set's 2" in capsys.readouterr().err


def test_scores_from_another_answers_file_fail(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    answers, scores = runs_for(world)
    answers = answers.model_copy(
        update={"fingerprint": dict(answers.fingerprint, corpus="0" * 64)}
    )
    commit(world, answers, scores)
    assert check(world) == EXIT_FLOOR_MISSED
    err = capsys.readouterr().err
    assert "not computed from the committed answers file (corpus differ)" in err


def test_a_regenerated_answers_file_with_the_same_inputs_fails(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    # A re-pulled model or unhashed code changes answers without moving any part, so the
    # scores record which answers file they judged (S2-5's first review).
    answers, scores = runs_for(world)
    items = [item.model_copy(update={"answer": "Another answer."}) for item in answers.items]
    commit(world, answers.model_copy(update={"items": items}), scores)
    assert check(world) == EXIT_FLOOR_MISSED
    err = capsys.readouterr().err
    assert "not computed from the committed answers file: re-run score on it" in err


def test_an_unexpected_error_exits_2_never_1(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # CI reads 1 as a verdict; a crash must never look like one (S2-5's first review).
    commit(world, *runs_for(world))

    def broken(*args: Any) -> list[str]:
        raise KeyError("surprise")

    monkeypatch.setattr("rag_qa.evaluation.cli.stale", broken)
    assert check(world) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Traceback" in err and "rag-eval check could not run: KeyError" in err


@pytest.mark.parametrize(
    ("damage", "problem"),
    [
        (lambda w: (w.eval_dir / "runs" / "generation-latest.json").unlink(), "cannot read"),
        (lambda w: (w.eval_dir / "runs" / "answers-latest.json").write_text("{}"),
         "is not the answers file this version writes"),
        (lambda w: (w.eval_dir / "thresholds.yaml").write_text(
            yaml.safe_dump({"retrieval": THRESHOLDS["retrieval"]})), "no generation: section"),
        (lambda w: (w.eval_dir / "judge.Modelfile").unlink(), "cannot read the judge's Modelfile"),
    ],
)
def test_check_cannot_run_without_its_inputs(
    world: World, capsys: pytest.CaptureFixture[str], damage: Callable[[World], Any], problem: str
) -> None:
    commit(world, *runs_for(world))
    damage(world)
    assert check(world) == EXIT_CANNOT_RUN
    assert problem in capsys.readouterr().err


def test_check_without_an_evaluation_section_cannot_run(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    commit(world, *runs_for(world))
    del world.config["evaluation"]
    world.write()
    assert check(world) == EXIT_CANNOT_RUN
    assert "no 'evaluation' section" in capsys.readouterr().err


def use_fake_ollama(world: World, ollama: FakeOllama) -> None:
    judge(world.config)["base_url"] = f"{ollama.url}/v1"
    llm(world.config)["base_url"] = ollama.url


def test_with_ollama_passes_when_the_digests_match(world: World, fake_ollama: FakeOllama) -> None:
    use_fake_ollama(world, fake_ollama)
    commit(world, *runs_for(world))
    assert check(world, "--with-ollama") == EXIT_OK
    assert "/api/tags" in fake_ollama.requests


def test_with_ollama_catches_a_repulled_model(
    world: World, fake_ollama: FakeOllama, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_ollama(world, fake_ollama)
    commit(world, *runs_for(world))
    fake_ollama.digests["mistral:latest"] = "d" * 64
    assert check(world, "--with-ollama") == EXIT_FLOOR_MISSED
    assert "mistral changed since the run" in capsys.readouterr().err


def test_with_ollama_unreachable_cannot_run(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    judge(world.config)["base_url"] = "http://127.0.0.1:9/v1"  # nothing listens there
    llm(world.config)["base_url"] = "http://127.0.0.1:9"
    commit(world, *runs_for(world))
    assert check(world, "--with-ollama") == EXIT_CANNOT_RUN
    assert "cannot reach Ollama" in capsys.readouterr().err


def test_without_with_ollama_no_request_is_made(world: World, fake_ollama: FakeOllama) -> None:
    use_fake_ollama(world, fake_ollama)
    commit(world, *runs_for(world))
    assert check(world) == EXIT_OK
    assert fake_ollama.requests == []  # CI has no Ollama; check never needs one


def test_with_ollama_asks_no_ollama_about_a_generator_that_is_not_one(
    world: World, fake_ollama: FakeOllama
) -> None:
    judge(world.config)["base_url"] = f"{fake_ollama.url}/v1"
    llm(world.config)["base_url"] = "http://127.0.0.1:9"  # an endpoint that is no Ollama
    answers, scores = runs_for(world)
    generator = Generator(ref="components.llms.other", model=None, digest=None)
    answers = answers.model_copy(update={"generator": generator})
    scores = scores.model_copy(
        update={"generator": generator, "answers": answers_identity(answers)}
    )
    commit(world, answers, scores)
    assert check(world, "--with-ollama") == EXIT_OK


def test_a_metric_the_judge_mostly_failed_to_score_fails_check(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    # The story's warning: a judge that misparses quietly produces confident numbers. The
    # one item it scored is 1.0, so the mean alone would pass (S2-5's second review).
    answers, scores = runs_for(world)
    failures = dict(scores.failures, faithfulness=Failures(parse=1))
    commit(world, answers, scores.model_copy(update={"failures": failures}))
    assert check(world) == EXIT_FLOOR_MISSED
    assert "faithfulness: 1 of 1 answerable items have no score (1 parse" in (
        capsys.readouterr().err
    )


def test_unscored_items_within_max_unscored_pass(world: World) -> None:
    _, scores = runs_for(world)
    failures = dict(scores.failures, faithfulness=Failures(parse=1))
    items = [*scores.items] + [
        ScoreItem(id=f"extra-{n}", answerable=True) for n in range(9)
    ]  # 1 of 10 unscored: exactly max_unscored
    floors = GenerationFloors.model_validate(THRESHOLDS["generation"])
    run = scores.model_copy(update={"failures": failures, "items": items})
    assert too_many_unscored(run, floors) == []


def test_a_base_model_never_recorded_is_not_held_against_the_run(world: World) -> None:
    answers, scores = runs_for(world)
    judge_ = scores.judge.model_copy(update={"base_digest": None})
    scores = scores.model_copy(update={"judge": judge_})
    live = {"mistral:latest": MISTRAL_DIGEST, "rag-judge:latest": JUDGE_DIGEST}
    assert digest_drift(answers, scores, live) == []


def test_an_ollama_host_without_a_port_means_ollamas_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box")
    assert ollama_root(None) == "http://gpu-box:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:8080/")
    assert ollama_root(None) == "http://gpu-box:8080"
    assert ollama_root("http://localhost:11434/v1") == "http://localhost:11434"


def test_a_failed_write_leaves_the_last_run_and_no_stray_file(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers, _ = runs_for(world)
    target = world.eval_dir / "runs" / "answers-latest.json"
    write_run(answers, target)
    before = target.read_bytes()

    def interrupted(*args: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("rag_qa.evaluation.gate.os.replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_run(answers.model_copy(update={"limit": 1}), target)
    assert target.read_bytes() == before
    assert sorted(p.name for p in target.parent.iterdir()) == ["answers-latest.json"]


def test_modelfile_directives_drop_comments_but_keep_block_text() -> None:
    text = (
        "# header\nFROM gemma2:9b\n\nPARAMETER num_ctx 8192  \n"
        'SYSTEM """\n# not a comment: part of the system text\nJudge strictly.\n"""\n# end\n'
    )
    assert fp.modelfile_directives(text) == [
        "FROM gemma2:9b",
        "PARAMETER num_ctx 8192",
        'SYSTEM """',
        "# not a comment: part of the system text",
        "Judge strictly.",
        '"""',
    ]
