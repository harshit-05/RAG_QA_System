"""``rag-eval``: the evaluation harness's command line (FR-7, DEC-15).

``rag-eval retrieval`` scores retrieval against the golden set. That is tier 1, the half
of the quality bar CI recomputes on every pull request and every push to ``main``
(:mod:`rag_qa.evaluation.retrieval`). It prints one row per answerable question, misses
first, then the means against their floors. S2-5 adds ``generate``, ``score`` and
``check``, for tier 2.

The eval files live in ``eval/`` beside the config file (ARCHITECTURE.md §2.3), the way
``paths`` are anchored to it: ``eval/eval_dataset.jsonl`` and ``eval/thresholds.yaml``.
So a run works from any directory. A path given on the command line is relative to the
working directory, as usual.

``rag-eval retrieval`` exit codes: **0** every floor is met; **1** a floor is missed;
**2** it could not run: no index, a bad config, dataset or thresholds file, a ``--json``
file it cannot write, a command-line usage error (as ``argparse`` uses), or any other
error while retrieving, such as a model missing offline or an unreadable index (printed
with its traceback). So exit 1 only ever means that retrieval ran and scored below a
floor, which is how CI's ``eval-retrieval`` job is read (S2-3's second review).
"""

import argparse
import json
import string
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_qa.chain import NoIndexError
from rag_qa.config import ConfigError, load_config, resolve_config_path
from rag_qa.evaluation.dataset import ExpectedSource, GoldenItem, GoldenSetError, load_golden
from rag_qa.evaluation.gate import RetrievalFloors, ThresholdsError, below_floors, load_thresholds
from rag_qa.evaluation.retrieval import (
    ItemScore,
    RetrievalScores,
    aggregate,
    evaluate_retrieval,
)
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_CONFIG

EXIT_OK = 0
EXIT_FLOOR_MISSED = 1
EXIT_CANNOT_RUN = 2

EVAL_DIR = "eval"
DATASET_FILE = "eval_dataset.jsonl"
THRESHOLDS_FILE = "thresholds.yaml"

#: How the table marks a chunk that counts: on an expected page, or on an also-page.
MARKS = {"page": "*", "also": "+", None: ""}
LABELS = {"hit_rate": "hit rate", "mrr": "MRR", "recall": "recall"}


def eval_dir(config_file: Path) -> Path:
    """Where the eval files are by default: ``eval/`` beside the config file."""
    return config_file.parent / EVAL_DIR


def _error(message: object) -> None:
    # Flush first: in a pipe, as in a CI log, stdout is block-buffered and stderr is not,
    # so the error would otherwise print above the table it is about.
    sys.stdout.flush()
    print(f"Error: {message}", file=sys.stderr)


def _cannot_run(problem: object) -> int:
    _error(problem)
    return EXIT_CANNOT_RUN


def _shown(path: Path) -> Path:
    """``path`` for display: relative to the working directory when it is under it."""
    return path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path


def _file_keys(items: Sequence[GoldenItem], scores: Sequence[ItemScore]) -> dict[str, str]:
    """A short key for every file the table names (A, B, …), in path order.

    Corpus paths are long (one is 90 characters), so the rows name files by key and a
    legend above the table spells each one out.
    """
    files = {s.source for item in items for s in item.expected_sources}
    files |= {chunk.source for score in scores for chunk in score.retrieved}
    letters = string.ascii_uppercase
    return {
        file: letters[i] if i < len(letters) else f"F{i + 1}"
        for i, file in enumerate(sorted(files))
    }


def _expected_cell(sources: Sequence[ExpectedSource], keys: dict[str, str]) -> str:
    cells = []
    for expected in sources:
        cell = keys[expected.source]
        if expected.pages:
            cell += ":" + ",".join(expected.pages)
        if expected.also_pages:
            cell += " +" + ",".join(expected.also_pages)
        cells.append(cell)
    return "  ".join(cells)


def _retrieved_cell(score: ItemScore, keys: dict[str, str]) -> str:
    return " ".join(
        keys[chunk.source] + ("" if chunk.page is None else f":{chunk.page}") + MARKS[chunk.match]
        for chunk in score.retrieved
    )


def print_items(items: Sequence[GoldenItem], scores: Sequence[ItemScore]) -> None:
    """One row per question, worst first: misses, then by rank, then by recall."""
    keys = _file_keys(items, scores)
    by_id = {item.id: item for item in items}
    print("Files:")
    for file, key in keys.items():
        print(f"  {key}  {file}")
    rows = [
        (
            "hit" if score.hit else "MISS",
            "-" if score.rank is None else str(score.rank),
            f"{score.covered}/{score.expected}",
            score.id,
            _expected_cell(by_id[score.id].expected_sources, keys),
            _retrieved_cell(score, keys),
        )
        for score in sorted(scores, key=lambda s: (s.reciprocal_rank, s.recall))
    ]
    header = ("result", "rank", "recall", "id", "expected", "retrieved, in rank order")
    # Every column but the last is padded to its widest cell; rank and recall align right.
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header) - 1)]
    print()
    for row in [header, *rows]:
        cells = [
            cell.rjust(width) if i in (1, 2) else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(row[:-1], widths, strict=True))
        ]
        print("  ".join([*cells, row[-1]]))
    print("\n* an expected page   + an also-page (counts for the hit and the rank, not recall)")


def print_aggregates(
    means: RetrievalScores, floors: RetrievalFloors, below: list[str], scores: Sequence[ItemScore]
) -> None:
    print(f"\n{'':10} {'mean':>6} {'floor':>6}")
    for name, floor in floors.model_dump().items():
        value = getattr(means, name)
        verdict = "BELOW" if name in below else "ok"
        extra = f"  {sum(s.hit for s in scores)} of {len(scores)}" if name == "hit_rate" else ""
        print(f"{LABELS[name]:10} {value:6.3f} {floor:6.3f}  {verdict:5}{extra}".rstrip())


def report_json(
    config: RagConfig,
    scores: Sequence[ItemScore],
    means: RetrievalScores,
    floors: RetrievalFloors,
    below: list[str],
) -> dict[str, Any]:
    """The run as JSON: what produced it, the means and floors, and every question in the
    golden set's order, so two runs (say, with and without a reranker) compare line by
    line."""
    query = config.pipeline.query
    return {
        "retrieval": {
            "retriever": query.retriever,
            "retriever_kwargs": config.retriever(query.retriever).kwargs(),
            "reranker": query.reranker,
            # With a reranker, the retriever fetches this many in place of its own k (S2-4).
            "reranker_candidates": query.reranker_candidates,
        },
        "aggregate": asdict(means),
        "floors": floors.model_dump(),
        "below_floor": below,
        "items": [
            {
                "id": score.id,
                "hit": score.hit,
                "rank": score.rank,
                "reciprocal_rank": score.reciprocal_rank,
                "recall": score.recall,
                "covered": score.covered,
                "expected": score.expected,
                "retrieved": [asdict(chunk) for chunk in score.retrieved],
            }
            for score in scores
        ],
    }


def run_retrieval(args: argparse.Namespace) -> int:
    """``rag-eval retrieval``. Returns the exit status; see the module docstring."""
    config_file = resolve_config_path(args.config).resolve()
    try:
        config = load_config(config_file)
    except ConfigError as e:
        return _cannot_run(e)
    dataset = Path(args.dataset) if args.dataset else eval_dir(config_file) / DATASET_FILE
    thresholds_file = (
        Path(args.thresholds) if args.thresholds else eval_dir(config_file) / THRESHOLDS_FILE
    )
    try:
        items = load_golden(dataset)
        floors = load_thresholds(thresholds_file).retrieval
    except OSError as e:  # the golden set's own; the thresholds loader wraps its own
        return _cannot_run(f"cannot read the golden set {e.filename}: {e.strerror}")
    except (GoldenSetError, ThresholdsError) as e:
        return _cannot_run(e)
    answerable = [item for item in items if item.answerable]
    if not answerable:
        return _cannot_run(f"{dataset} has no answerable records to score retrieval against")

    query = config.pipeline.query
    print(
        f"Tier 1: retrieval, over the {len(answerable)} answerable questions in "
        f"{_shown(dataset.resolve())}"
    )
    reranking = (
        f"{query.reranker}, over {query.reranker_candidates} candidates in place of k"
        if query.reranker
        else "none"
    )
    print(
        f"Retriever: {query.retriever} {config.retriever(query.retriever).kwargs()}, "
        f"reranker: {reranking}"
    )
    try:
        scores = evaluate_retrieval(config, answerable)
    except NoIndexError as e:
        return _cannot_run(e)
    # Deliberately broad: whatever stops retrieval from running is "could not run",
    # never a floor verdict. Uncaught, it would exit 1, and CI reads exit 1 as "retrieval
    # quality moved". A model missing offline did exactly that (S2-3's second review).
    except Exception as e:  # noqa: BLE001
        sys.stdout.flush()
        traceback.print_exc()
        return _cannot_run(f"retrieval could not run: {type(e).__name__}: {e}")
    means = aggregate(scores)
    below = below_floors(asdict(means), floors)

    print()
    print_items(answerable, scores)
    print_aggregates(means, floors, below, scores)
    if args.json:
        try:
            Path(args.json).write_text(
                json.dumps(report_json(config, scores, means, floors, below), indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            return _cannot_run(f"cannot write {args.json}: {e.strerror}")
    shown = _shown(thresholds_file.resolve())
    if below:
        missed = "; ".join(
            f"{name} {getattr(means, name):.3f} < {getattr(floors, name)}" for name in below
        )
        _error(f"below the floor in {shown}: {missed}")
        return EXIT_FLOOR_MISSED
    print(f"\nEvery floor in {shown} is met.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval",
        description="Evaluate the pipeline against the golden set.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    retrieval = commands.add_parser(
        "retrieval",
        help="score retrieval against the golden set (tier 1; no LLM needed)",
        description="Score retrieval against the golden set: hit rate, MRR and recall over "
        "the answerable questions, checked against the floors (tier 1). Needs the index and "
        "the embedder only.",
        epilog="Exit status: 0 every floor is met; 1 a floor is missed; 2 could not run "
        "(no index; a bad config, dataset or thresholds file; a --json file that cannot be "
        "written; a usage error; or any other error while retrieving, with its traceback).",
    )
    retrieval.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    retrieval.add_argument(
        "--dataset",
        metavar="PATH",
        help=f"golden set (default: {EVAL_DIR}/{DATASET_FILE} beside the config file)",
    )
    retrieval.add_argument(
        "--thresholds",
        metavar="PATH",
        help=f"floors to check (default: {EVAL_DIR}/{THRESHOLDS_FILE} beside the config file)",
    )
    retrieval.add_argument(
        "--json",
        metavar="OUT",
        help="also write the per-question results and the means to this JSON file",
    )
    retrieval.set_defaults(run=run_retrieval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-eval``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work
    status: int = args.run(args)
    return status


if __name__ == "__main__":
    sys.exit(main())
