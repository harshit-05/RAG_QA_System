"""``rag-eval``: the evaluation harness's command line (FR-7, DEC-15).

Two tiers, by what can be computed where:

- ``rag-eval retrieval`` scores retrieval against the golden set: tier 1, which CI
  recomputes on every pull request and every push to ``main``
  (:mod:`rag_qa.evaluation.retrieval`). It prints one row per answerable question, misses
  first, then the means against their floors.
- Tier 2 runs offline, for hours, and is committed (S2-5):
  ``rag-eval generate`` answers every golden question through ``stream_answer``
  (:mod:`.generation`); ``rag-eval score`` scores those answers with RAGAs and a local
  judge (:mod:`.ragas_scoring`, which needs the ``eval`` extra and Ollama); and
  ``rag-eval check`` holds the committed run to its floors and to the checkout
  (:mod:`.gate`), with no models, so CI can run it.

The eval files live in ``eval/`` beside the config file (ARCHITECTURE.md §2.3), the way
``paths`` are anchored to it: the golden set, the floors, the judge's Modelfile and
``runs/``. ``--eval-dir`` points the tier-2 commands at another folder. So a run works
from any directory. A path given on the command line is relative to the working
directory, as usual.

Exit codes, for every command: **0** it ran and passed; **1** it ran and failed: a floor
is missed, or (``check``) the committed run is stale; **2** it could not run: a bad
config, dataset, thresholds or run file, no usable index, Ollama or the judge unusable,
a file it cannot write, a command-line usage error (as ``argparse`` uses), or any other
error while running (printed with its traceback). So exit 1 only ever means a measured
verdict, which is how CI reads it (S2-3's second review). ``generate`` and ``score``
never exit 1: they record, and ``check`` judges.
"""

import argparse
import asyncio
import json
import string
import sys
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_qa.chain import NoIndexError
from rag_qa.config import ConfigError, load_config, resolve_config_path
from rag_qa.evaluation.dataset import ExpectedSource, GoldenItem, GoldenSetError, load_golden
from rag_qa.evaluation.fingerprint import PARTS, FingerprintError, evaluation_of, fingerprint
from rag_qa.evaluation.gate import (
    GENERATION_METRICS,
    AnswerItem,
    OllamaError,
    RetrievalFloors,
    RunFileError,
    ScoresRun,
    ThresholdsError,
    below_floors,
    below_generation_floors,
    digest_drift,
    load_answers,
    load_scores,
    load_thresholds,
    ollama_digests,
    ollama_root,
    stale,
    too_many_unscored,
    uncovered,
    write_run,
)
from rag_qa.evaluation.generation import StaleIndexError, generate
from rag_qa.evaluation.ragas_scoring import ScoreRequest, ScoringError, score
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
RUNS_DIR = "runs"
ANSWERS_FILE = "answers-latest.json"
SCORES_FILE = "generation-latest.json"

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


def _crashed(what: str, error: Exception) -> int:
    """Any other error while running: the traceback, then exit 2, never 1, which CI reads
    as a verdict (S2-3's second review)."""
    sys.stdout.flush()
    traceback.print_exc()
    return _cannot_run(f"{what} could not run: {type(error).__name__}: {error}")


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
        return _crashed("retrieval", e)
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


# ---- tier 2: generate, score, check (S2-5) -----------------------------------------------


class _CannotRun(Exception):
    """A tier-2 command cannot run; the message says why. Exit 2."""


@dataclass(frozen=True)
class _Setup:
    config_file: Path
    config: Any  # RagConfig
    eval_dir: Path
    golden: list[GoldenItem]


def _setup(args: argparse.Namespace) -> _Setup:
    """The config, the eval folder and the golden set every tier-2 command reads."""
    config_file = resolve_config_path(args.config).resolve()
    try:
        config = load_config(config_file)
    except ConfigError as e:
        raise _CannotRun(e) from e
    folder = Path(args.eval_dir) if args.eval_dir else eval_dir(config_file)
    dataset = folder / DATASET_FILE
    try:
        golden = load_golden(dataset)
    except OSError as e:
        raise _CannotRun(f"cannot read the golden set {dataset}: {e.strerror}") from e
    except GoldenSetError as e:
        raise _CannotRun(e) from e
    return _Setup(config_file, config, folder, golden)


def _write(run: Any, path: Path) -> None:
    try:
        write_run(run, path)
    except OSError as e:
        raise _CannotRun(f"cannot write {path}: {e.strerror}") from e


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.1f} s"


def run_generate(args: argparse.Namespace) -> int:
    """``rag-eval generate``. Returns the exit status; see the module docstring."""
    try:
        setup = _setup(args)
    except _CannotRun as e:
        return _cannot_run(e)
    out = Path(args.out) if args.out else setup.eval_dir / RUNS_DIR / ANSWERS_FILE
    total = len(setup.golden) if args.limit is None else min(args.limit, len(setup.golden))
    print(f"Tier 2: generating answers to {total} golden question(s) with "
          f"{setup.config.pipeline.query.llm}, through stream_answer")

    def progress(n: int, of: int, item: AnswerItem) -> None:
        print(f"[{n}/{of}] {item.id}: ttft {_ms(item.ttft_ms)}, total {_ms(item.total_ms)}",
              flush=True)

    try:
        run = asyncio.run(generate(setup.config, setup.golden, limit=args.limit, progress=progress))
    except (FingerprintError, NoIndexError, StaleIndexError) as e:
        return _cannot_run(e)
    # Deliberately broad: whatever stops generation (Ollama down, a model error) is "could
    # not run", never a verdict.
    except Exception as e:  # noqa: BLE001
        return _crashed("generation", e)
    try:
        _write(run, out)
    except _CannotRun as e:
        return _cannot_run(e)
    print(f"\nWrote {len(run.items)} answer(s) to {_shown(out.resolve())}. "
          f"Generator digest: {run.generator.digest or 'not recorded'}")
    return EXIT_OK


def _print_scores(run: ScoresRun) -> None:
    print(f"\n{'':18} {'mean':>6}  failures: parse, truncated, no score; retried")
    for name in GENERATION_METRICS:
        value = run.aggregate.get(name)
        shown = "-" if value is None else f"{value:.3f}"
        failures = run.failures.get(name)
        extra = (
            f"  {failures.parse}, {failures.truncated}, {failures.no_score}; {failures.retried}"
            if failures is not None
            else ""
        )
        print(f"{name:18} {shown:>6}{extra}")
    judge = run.judge
    print(f"\nJudge {judge.model} ({judge.digest[:12]}, on {judge.base}), num_ctx {judge.num_ctx}, "
          f"served {judge.served_context}; largest prompt {judge.max_prompt_tokens} tokens, "
          f"largest answer {judge.max_completion_tokens}")


def run_score(args: argparse.Namespace) -> int:
    """``rag-eval score``. Returns the exit status; see the module docstring."""
    try:
        setup = _setup(args)
        answers_file = (
            Path(args.answers) if args.answers else setup.eval_dir / RUNS_DIR / ANSWERS_FILE
        )
        answers = load_answers(answers_file)
    except (_CannotRun, RunFileError) as e:
        return _cannot_run(e)
    if args.out is None and (args.answers is not None or answers.limit is not None):
        # Only the committed answers, in full, are scored into the committed scores file:
        # a smoke run must never replace a 13-hour baseline.
        return _cannot_run(
            "scoring an answers file other than the committed one, or a --limit one, "
            "needs --out: the committed scores file is for the committed answers"
        )
    out = Path(args.out) if args.out else setup.eval_dir / RUNS_DIR / SCORES_FILE
    print(f"Tier 2: scoring {_shown(answers_file.resolve())} with RAGAs")
    request = ScoreRequest(setup.config, setup.golden, answers, setup.eval_dir, args.limit)
    try:
        run = asyncio.run(score(request, progress=lambda line: print(line, flush=True)))
    except (ScoringError, FingerprintError) as e:
        return _cannot_run(e)
    except Exception as e:  # noqa: BLE001
        return _crashed("scoring", e)
    try:
        _write(run, out)
    except _CannotRun as e:
        return _cannot_run(e)
    _print_scores(run)
    print(f"\nWrote the scores to {_shown(out.resolve())}. rag-eval check holds them to the "
          f"floors.")
    return EXIT_OK


def run_check(args: argparse.Namespace) -> int:
    """``rag-eval check``. Returns the exit status; see the module docstring."""
    try:
        setup = _setup(args)
        runs = setup.eval_dir / RUNS_DIR
        thresholds_file = setup.eval_dir / THRESHOLDS_FILE
        floors = load_thresholds(thresholds_file).generation
        if floors is None:
            raise _CannotRun(
                f"{thresholds_file} has no generation: section, so tier 2 has no floors yet "
                f"(S2-5b sets them from the first baseline)"
            )
        answers = load_answers(runs / ANSWERS_FILE)
        scores = load_scores(runs / SCORES_FILE)
        current = fingerprint(setup.config, setup.golden, setup.eval_dir)
    except (_CannotRun, ThresholdsError, RunFileError, FingerprintError) as e:
        return _cannot_run(e)

    problems = stale(answers, scores, current) + uncovered(setup.golden, answers, scores)
    if args.with_ollama:
        roots = {ollama_root(evaluation_of(setup.config).judge.base_url)}
        if answers.generator.model is not None:  # an Ollama model, with a digest to compare
            llm_spec = setup.config.component(setup.config.pipeline.query.llm).spec()
            roots.add(ollama_root(llm_spec.get("base_url")))
        try:
            live = {name: d for root in sorted(roots) for name, d in ollama_digests(root).items()}
        except OllamaError as e:
            return _cannot_run(f"--with-ollama: {e}")
        problems += digest_drift(answers, scores, live)
    below = below_generation_floors(scores.aggregate, floors)
    problems += too_many_unscored(scores, floors)

    print(f"Tier 2: the committed run, generated {answers.generated_at}, scored {scores.scored_at}")
    print("\nFingerprint (recorded vs this checkout):")
    for part in PARTS:
        same = scores.fingerprint.get(part) == current[part]
        print(f"  {part:11} {'ok' if same else 'CHANGED'}")
    print(f"\n{'':18} {'score':>6} {'floor':>6}")
    for name, floor in floors.metric_floors().items():
        value = scores.aggregate.get(name)
        verdict = "BELOW" if name in below else ("ok" if floor is not None else "not gated")
        print(f"{name:18} {'-' if value is None else f'{value:.3f}':>6} "
              f"{'-' if floor is None else f'{floor:.3f}':>6}  {verdict}")
    for name, f in scores.failures.items():
        if not (f.parse or f.truncated or f.no_score):
            continue
        print(f"  {name}: {f.parse} parse failure(s), {f.truncated} truncated, "
              f"{f.no_score} without a score, of the items scored")

    shown = _shown(thresholds_file.resolve())
    if problems:
        _error("the committed tier-2 run does not stand:\n"
               + "\n".join(f"  - {p}" for p in problems))
    if below:
        missed = "; ".join(f"{name} {scores.aggregate.get(name)} < {getattr(floors, name)}"
                           for name in below)
        _error(f"below the floor in {shown}: {missed}")
    if problems or below:
        return EXIT_FLOOR_MISSED
    print(f"\nThe committed run matches this checkout, and every floor in {shown} is met.")
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

    def tier2(name: str, help_: str, description: str) -> argparse.ArgumentParser:
        command = commands.add_parser(
            name,
            help=help_,
            description=description,
            epilog="Exit status: 0 it ran (and, for check, passed); 1 check only: a floor is "
            "missed or the committed run is stale; 2 could not run, with the reason (and a "
            "traceback for an unexpected error).",
        )
        command.add_argument(
            "--config",
            metavar="PATH",
            help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
        )
        command.add_argument(
            "--eval-dir",
            metavar="DIR",
            help=f"the eval folder: golden set, floors, judge Modelfile, {RUNS_DIR}/ "
            f"(default: {EVAL_DIR}/ beside the config file)",
        )
        return command

    def positive(text: str) -> int:
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
        return value

    generate_ = tier2(
        "generate",
        "answer every golden question through stream_answer (tier 2; needs the index and Ollama)",
        "Answer every golden question the way users get answers, and record the answers, the "
        "chunks each prompt got, the sources and the timings, with the fingerprint of what "
        "produced them. Refuses an index that lags the corpus.",
    )
    score_ = tier2(
        "score",
        "score an answers file with RAGAs and the local judge (tier 2; needs the eval extra)",
        "Score an answers file with RAGAs: faithfulness, answer relevancy, context precision "
        "and context recall, judged by rag-judge through Ollama; plus the decline rate on "
        "the questions the corpus cannot answer. Needs no index and no generator.",
    )
    for command, default in ((generate_, ANSWERS_FILE), (score_, SCORES_FILE)):
        command.add_argument(
            "--limit", type=positive, metavar="N", help="only the first N questions (needs --out)"
        )
        command.add_argument(
            "--out", metavar="PATH", help=f"where to write (default: {RUNS_DIR}/{default})"
        )
    score_.add_argument(
        "--answers", metavar="PATH", help=f"the answers file (default: {RUNS_DIR}/{ANSWERS_FILE})"
    )
    check_ = tier2(
        "check",
        "check the committed tier-2 run against the floors and this checkout (no models)",
        "Check the committed tier-2 run: its fingerprint against this checkout (a stale run "
        "fails, naming the re-run that fixes it), its coverage of the golden set, and its "
        "scores against the generation floors. Needs no models and no Ollama.",
    )
    check_.add_argument(
        "--with-ollama",
        action="store_true",
        help="also compare the recorded model digests with the live ones (local only)",
    )
    generate_.set_defaults(run=run_generate)
    score_.set_defaults(run=run_score)
    check_.set_defaults(run=run_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-eval``). Returns the exit status; see the module docstring."""
    parser = build_parser()
    args = parser.parse_args(argv)  # --help and usage errors exit here, before any work
    if getattr(args, "limit", None) is not None and args.out is None:
        # A partial run must never replace the committed one.
        parser.error(f"{args.command} --limit needs --out: a partial run must not overwrite "
                     f"the committed file")
    try:
        status: int = args.run(args)
    # The backstop: whatever a command did not foresee is "could not run", never exit 1,
    # which CI reads as a verdict. Each command catches what it expects itself.
    except Exception as e:  # noqa: BLE001
        return _crashed(f"rag-eval {args.command}", e)
    return status


if __name__ == "__main__":
    sys.exit(main())
