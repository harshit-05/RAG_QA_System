"""Tier-1 retrieval evaluation (S2-3): the metrics, the floors and ``rag-eval retrieval``.

The arithmetic runs on hand-built documents. The command runs end to end against a tiny
index that the real ingest built from ``sample_corpus`` with the deterministic fake
embedder, so nothing downloads (DEC-11). Its tests also make building an LLM fail: tier 1
must run where there is no Ollama, as in CI's ``eval-retrieval`` job.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import REPO_ROOT, MakeConfig
from langchain_core.documents import Document

from rag_qa.evaluation import cli
from rag_qa.evaluation.cli import EXIT_CANNOT_RUN, EXIT_FLOOR_MISSED, EXIT_OK, main
from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.evaluation.gate import RetrievalFloors, ThresholdsError, below_floors, load_thresholds
from rag_qa.evaluation.retrieval import aggregate, score_item
from rag_qa.settings import ENV_VECTOR_STORE_PATH

CORPUS = Path("/corpus")
PAPER = {"source": "paper.pdf", "pages": ["7"]}


def item(*expected_sources: dict[str, Any]) -> GoldenItem:
    """An answerable record with these expected sources, validated as the loader does."""
    record = {
        "id": "q",
        "question": "?",
        "ground_truth": "a",
        "answerable": True,
        "expected_sources": list(expected_sources),
    }
    return GoldenItem.model_validate_json(json.dumps(record))


def chunk(source: str, label: str | None = None, **metadata: Any) -> Document:
    """A chunk of ``source``, on the page printed as ``label``."""
    if label is not None:
        metadata["page_label"] = label
    return Document(page_content="text", metadata={"source": source, **metadata})


def pdf(*pages: str) -> list[Document]:
    """Chunks of ``/corpus/paper.pdf`` on these pages, in rank order."""
    return [chunk("/corpus/paper.pdf", page) for page in pages]


# ---- the metrics, on hand-built documents ----


def test_a_hit_at_rank_1() -> None:
    score = score_item(item(PAPER), pdf("7", "2", "3"), CORPUS)
    assert (score.hit, score.rank, score.reciprocal_rank) == (True, 1, 1.0)


def test_a_hit_at_rank_3() -> None:
    score = score_item(item(PAPER), pdf("2", "3", "7", "9"), CORPUS)
    assert (score.rank, score.reciprocal_rank) == (3, 1 / 3)
    assert [c.match for c in score.retrieved] == [None, None, "page", None]


def test_no_hit() -> None:
    score = score_item(item(PAPER), pdf("2", "3"), CORPUS)
    assert (score.hit, score.rank, score.reciprocal_rank, score.recall) == (False, None, 0.0, 0.0)


def test_recall_is_the_share_of_expected_pages_covered_each_counted_once() -> None:
    expected = {"source": "paper.pdf", "pages": ["3", "9"]}
    score = score_item(item(expected), pdf("9", "9", "4"), CORPUS)
    assert (score.covered, score.expected, score.recall) == (1, 2, 0.5)


def test_recall_counts_the_pages_of_every_expected_source() -> None:
    paper = {"source": "paper.pdf", "pages": ["3", "9"]}
    other = {"source": "other.pdf", "pages": ["1"]}
    score = score_item(item(paper, other), [*pdf("3"), chunk("/corpus/other.pdf", "1")], CORPUS)
    assert (score.covered, score.expected) == (2, 3)


def test_pages_match_by_their_printed_label() -> None:
    # As citation() prints them: a preface's "iii"; a label over the page index, which
    # differs from it; and the index plus one when a chunk has no label.
    expected = {"source": "paper.pdf", "pages": ["iii", "7"]}
    docs = [
        chunk("/corpus/paper.pdf", "iii"),
        chunk("/corpus/paper.pdf", "7", page=0),
        chunk("/corpus/paper.pdf", page=2),
    ]
    score = score_item(item(expected), docs, CORPUS)
    assert [(c.page, c.match) for c in score.retrieved] == [
        ("iii", "page"),
        ("7", "page"),
        ("3", None),
    ]


def test_the_same_page_of_another_file_is_no_match() -> None:
    assert not score_item(item(PAPER), [chunk("/corpus/other.pdf", "7")], CORPUS).hit


@pytest.mark.parametrize(
    ("source", "expected_source", "hit"),
    [
        pytest.param("/corpus/sub/paper.pdf", "sub/paper.pdf", True, id="absolute, as in v0.2"),
        pytest.param("sub/paper.pdf", "sub/paper.pdf", True, id="relative, as from S2-6"),
        # An index built from a corpus at another path: SourceRef reduces the source to
        # its file name, which matches only a file at the corpus root.
        pytest.param("/elsewhere/paper.pdf", "paper.pdf", True, id="elsewhere, by file name"),
        pytest.param("/elsewhere/sub/paper.pdf", "sub/paper.pdf", False, id="elsewhere, nested"),
    ],
)
def test_a_source_is_matched_relative_to_the_corpus(
    source: str, expected_source: str, hit: bool
) -> None:
    expected = {"source": expected_source, "pages": ["7"]}
    assert score_item(item(expected), [chunk(source, "7")], CORPUS).hit is hit


def test_an_also_page_counts_for_the_hit_and_the_rank_but_not_recall() -> None:
    expected = {"source": "paper.pdf", "pages": ["1"], "also_pages": ["2"]}
    score = score_item(item(expected), pdf("2", "5"), CORPUS)
    assert (score.hit, score.rank, score.recall) == (True, 1, 0.0)
    assert score.retrieved[0].match == "also"


def test_an_expected_page_ranked_below_an_also_page_takes_the_also_pages_rank() -> None:
    expected = {"source": "paper.pdf", "pages": ["1"], "also_pages": ["2"]}
    score = score_item(item(expected), pdf("5", "2", "1"), CORPUS)
    assert (score.rank, score.recall) == (2, 1.0)


def test_a_source_listed_without_pages_matches_any_chunk_of_that_file() -> None:
    docs = [*pdf("1"), chunk("/corpus/notes.txt")]
    score = score_item(item({"source": "notes.txt"}), docs, CORPUS)
    assert (score.rank, score.covered, score.expected) == (2, 1, 1)


def test_the_aggregates_are_the_means() -> None:
    retrieved = (pdf("7"), pdf("1", "2", "7"), pdf("1"))  # rank 1, rank 3, a miss
    means = aggregate([score_item(item(PAPER), docs, CORPUS) for docs in retrieved])
    assert means.hit_rate == pytest.approx(2 / 3)
    assert means.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert means.recall == pytest.approx(2 / 3)


# ---- the floors ----


def write_floors(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_floors_load_and_an_int_is_a_number(tmp_path: Path) -> None:
    path = write_floors(tmp_path, "retrieval: {hit_rate: 0.75, mrr: 0, recall: 1}\n")
    assert load_thresholds(path).retrieval == RetrievalFloors(hit_rate=0.75, mrr=0.0, recall=1.0)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("retrieval: {hit_rate: 0.7, mrr: 0.5, recal: 0.6}\n", "retrieval.recal: Extra inputs"),
        ("retreival: {hit_rate: 0.7, mrr: 0.5, recall: 0.6}\n", "retreival: Extra inputs"),
        ("retrieval: {hit_rate: 0.7, mrr: 0.5}\n", "retrieval.recall: Field required"),
        # A negative floor could never fail: a typo must not switch a gate off.
        ("retrieval: {hit_rate: -0.1, mrr: 0.5, recall: 0.6}\n", "greater than or equal to 0"),
        ("retrieval: {hit_rate: '0.7', mrr: 0.5, recall: 0.6}\n", "valid number"),
        ("retrieval: {hit_rate: true, mrr: 0.5, recall: 0.6}\n", "valid number"),
        ("", "is empty"),
        ("retrieval: [0.7\n", "not valid YAML"),
    ],
)
def test_a_bad_thresholds_file_is_refused(tmp_path: Path, text: str, problem: str) -> None:
    with pytest.raises(ThresholdsError) as exc:
        load_thresholds(write_floors(tmp_path, text))
    assert problem in str(exc.value)


def test_a_missing_thresholds_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ThresholdsError, match="cannot read the thresholds file"):
        load_thresholds(tmp_path / "missing.yaml")


def test_a_metric_exactly_at_its_floor_meets_it() -> None:
    floors = RetrievalFloors(hit_rate=0.75, mrr=0.5, recall=0.6)
    assert below_floors({"hit_rate": 0.75, "mrr": 0.5, "recall": 0.6}, floors) == []
    below = below_floors({"hit_rate": 0.7, "mrr": 0.5, "recall": 0.59}, floors)
    assert below == ["hit_rate", "recall"]


def test_the_committed_floors_load() -> None:
    assert load_thresholds(REPO_ROOT / "eval" / "thresholds.yaml").retrieval


# ---- rag-eval retrieval, end to end on a tiny index ----

# sample_corpus indexes as 7 one-chunk documents, and the retriever returns 5 of them.
# Which 5 depends on the fake embedder, so the test set asks only what does not:
EVERY_FILE = {
    "id": "every-file",  # every chunk matches: rank 1, recall 5 of 7
    "question": "What do the documents say?",
    "ground_truth": "Everything.",
    "answerable": True,
    "expected_sources": [
        {"source": "guide.pdf", "pages": ["i", "ii", "1"]},
        {"source": "notes.txt"},
        {"source": "README.md"},
        {"source": "sub/report.docx"},
        {"source": "sub/deeper/LOUD.TXT"},
    ],
}
NO_SUCH_PAGE = {
    "id": "no-such-page",  # a miss whatever is retrieved
    "question": "What is on page 99?",
    "ground_truth": "Nothing.",
    "answerable": True,
    "expected_sources": [{"source": "guide.pdf", "pages": ["99"]}],
}
UNANSWERABLE = {
    "id": "unanswerable",  # skipped: tier 2's decline rate only
    "question": "What is the capital of Australia?",
    "ground_truth": "I could not find the answer in the provided documents.",
    "answerable": False,
}
# So: hit rate 1/2, MRR 1/2, recall 5/14 (0.357).
MET = "retrieval: {hit_rate: 0.5, mrr: 0.5, recall: 0.35}\n"


def write_golden(path: Path, *records: dict[str, Any] | str) -> Path:
    lines = (r if isinstance(r, str) else json.dumps(r) for r in records)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def evals(fake_rag: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``fake_rag``'s config, with the test set and passing floors in ``eval/`` beside it.

    Building an LLM fails the test: tier 1 must not need Ollama.
    """

    def no_llm(config: Any) -> None:
        raise AssertionError("tier 1 built an LLM; it must run without Ollama")

    monkeypatch.setattr("rag_qa.chain.build_llm", no_llm)
    monkeypatch.setattr("rag_qa.components.build_llm", no_llm)
    folder = fake_rag.parent / "eval"
    folder.mkdir()
    write_golden(folder / "eval_dataset.jsonl", EVERY_FILE, NO_SUCH_PAGE, UNANSWERABLE)
    write_floors(folder, MET)
    return fake_rag


def run(config: Path, *args: str) -> int:
    return main(["retrieval", "--config", str(config), *args])


def test_every_floor_met_exits_0_with_misses_listed_first(evals: Path, capsys: Any) -> None:
    assert run(evals) == EXIT_OK
    out = capsys.readouterr().out
    assert out.index("no-such-page") < out.index("every-file")  # misses first
    assert "5/7" in out  # every-file's recall
    assert "1 of 2" in out  # the hit rate's count
    assert "Every floor in" in out


def test_the_eval_files_default_to_eval_beside_the_config(
    evals: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anchored like paths: run from another directory, the defaults still find them.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert run(evals) == EXIT_OK


def test_a_missed_floor_exits_1_and_names_it(evals: Path, tmp_path: Path, capsys: Any) -> None:
    strict = write_floors(tmp_path, "retrieval: {hit_rate: 0.51, mrr: 0, recall: 0}\n")
    assert run(evals, "--thresholds", str(strict)) == EXIT_FLOOR_MISSED
    captured = capsys.readouterr()
    assert "hit_rate 0.500 < 0.51" in captured.err
    assert "BELOW" in captured.out


def test_json_holds_every_question_in_golden_order(evals: Path, tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    assert run(evals, "--json", str(out)) == EXIT_OK
    report = json.loads(out.read_text())
    assert report["aggregate"] == pytest.approx({"hit_rate": 0.5, "mrr": 0.5, "recall": 5 / 14})
    assert report["below_floor"] == []
    assert report["retrieval"]["retriever_kwargs"] == {"search_kwargs": {"k": 5}}
    every_file, no_such_page = report["items"]
    assert (every_file["id"], every_file["rank"], every_file["covered"]) == ("every-file", 1, 5)
    assert all(chunk["match"] == "page" for chunk in every_file["retrieved"])
    assert (no_such_page["hit"], no_such_page["rank"]) == (False, None)


def test_no_index_exits_2_and_says_to_ingest(
    evals: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "no-index"))
    assert run(evals) == EXIT_CANNOT_RUN
    assert "Run rag-ingest first" in capsys.readouterr().err


def test_an_error_while_retrieving_exits_2_with_its_traceback(
    evals: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # The error CI's offline eval step gets when a model is not in the cache. Exit 1
    # would read as "a floor is missed" (S2-3's second review).
    def offline(config: Any, items: Any) -> None:
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr("rag_qa.evaluation.cli.evaluate_retrieval", offline)
    assert run(evals) == EXIT_CANNOT_RUN
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "retrieval could not run: OSError: We couldn't connect" in err


def test_a_bad_golden_set_exits_2(evals: Path, tmp_path: Path, capsys: Any) -> None:
    bad = write_golden(tmp_path / "bad.jsonl", EVERY_FILE, "{oops")
    assert run(evals, "--dataset", str(bad)) == EXIT_CANNOT_RUN
    assert "bad.jsonl:2: Invalid JSON" in capsys.readouterr().err


def test_a_missing_golden_set_exits_2(evals: Path, tmp_path: Path, capsys: Any) -> None:
    assert run(evals, "--dataset", str(tmp_path / "missing.jsonl")) == EXIT_CANNOT_RUN
    assert "cannot read the golden set" in capsys.readouterr().err


def test_a_golden_set_with_nothing_answerable_exits_2(
    evals: Path, tmp_path: Path, capsys: Any
) -> None:
    only = write_golden(tmp_path / "only.jsonl", UNANSWERABLE)
    assert run(evals, "--dataset", str(only)) == EXIT_CANNOT_RUN
    assert "no answerable records" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("retrieval: {hit_rate: 0.5, mrr: 0.5}\n", "invalid thresholds file"),
        (None, "cannot read the thresholds file"),
    ],
)
def test_a_bad_or_missing_thresholds_file_exits_2(
    evals: Path, tmp_path: Path, capsys: Any, text: str | None, problem: str
) -> None:
    path = tmp_path / "floors" / "thresholds.yaml"
    if text is not None:
        path.parent.mkdir()
        path.write_text(text)
    assert run(evals, "--thresholds", str(path)) == EXIT_CANNOT_RUN
    assert problem in capsys.readouterr().err


def test_an_invalid_config_exits_2(make_config: MakeConfig, capsys: Any) -> None:
    nope = "components.retrievers.nope"
    path = make_config(lambda c: c["pipeline"]["query"].update(retriever=nope))
    assert run(path) == EXIT_CANNOT_RUN
    assert "no entry 'nope'" in capsys.readouterr().err


def test_a_json_file_that_cannot_be_written_exits_2(
    evals: Path, tmp_path: Path, capsys: Any
) -> None:
    assert run(evals, "--json", str(tmp_path / "missing-dir" / "run.json")) == EXIT_CANNOT_RUN
    assert "cannot write" in capsys.readouterr().err


def test_help_does_no_work(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def no_run(*args: Any) -> None:
        raise AssertionError("--help evaluated something")

    monkeypatch.setattr(cli, "evaluate_retrieval", no_run)
    with pytest.raises(SystemExit) as exc:
        main(["retrieval", "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "--thresholds" in out and "Exit status" in out


def test_a_command_is_required(capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == EXIT_CANNOT_RUN  # argparse's usage-error status
    assert "COMMAND" in capsys.readouterr().err
