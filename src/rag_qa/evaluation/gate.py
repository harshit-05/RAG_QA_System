"""The quality bar's floors, tier 2's run files, and what ``rag-eval check`` checks.

DEC-15: each tier's floors are its first measured baseline minus a tolerance, and they
are only ever raised deliberately. ``eval/thresholds.yaml`` has one section per tier::

    retrieval:  {hit_rate: ..., mrr: ..., recall: ...}     # tier 1, rag-eval retrieval (S2-3)
    generation: {faithfulness: ..., answer_relevancy: ..., context_precision: ...,
                 context_recall: ..., decline_rate: ...}   # tier 2, rag-eval check (S2-5)

Tier 2 runs offline and is committed as two files in ``eval/runs/`` (ARCHITECTURE.md
§2.4): ``answers-latest.json`` from ``rag-eval generate`` and ``generation-latest.json``
from ``rag-eval score``. ``rag-eval check`` fails CI when a score is below its floor, or
when the committed run is **stale**: its fingerprint (:mod:`.fingerprint`) no longer
matches the checkout. Each moved part names the re-run that fixes it.

CI's model-free job runs ``check``, so this module stays light: pydantic, YAML, the
fingerprint (``langchain_core`` at most) and the standard library's ``urllib`` for
Ollama. No model stack (``tests/test_architecture.py``).
"""

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.evaluation.fingerprint import GENERATION_PARTS, METRICS, moved, stale_message
from rag_qa.manifest import spec_identity


class ThresholdsError(ValueError):
    """The thresholds file is missing, unreadable or invalid. The message says where."""


class _Strict(BaseModel):
    # strict: a hand-edited floor should be a number, never "0.8" or `true`. Strict mode
    # still accepts an int for a float, so `mrr: 0` loads.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class RetrievalFloors(_Strict):
    """Tier 1's floors. A metric meets its floor when it is at or above it."""

    # ge=0: a negative floor could never fail, and a typo must not switch a gate off. No
    # upper bound: a floor above 1 can never be met, which is how a negative control
    # shows that the gate can fail.
    hit_rate: float = Field(ge=0)
    mrr: float = Field(ge=0)
    recall: float = Field(ge=0)


#: A tier-2 floor: a number, or ``null`` for a metric recorded but not gated. S2-5b leaves
#: out a metric whose two scorings differ by more than 0.1 (DEC-15). The key stays required,
#: so a typo can never switch a gate off.
Floor = Annotated[float, Field(ge=0)] | None

#: Tier 2's metrics, in the order they are printed: the four RAGAs ones, then the decline
#: rate over the unanswerable questions.
GENERATION_METRICS = (*METRICS, "decline_rate")


class GenerationFloors(_Strict):
    """Tier 2's floors (S2-5b sets them). A metric meets its floor at or above it."""

    faithfulness: Floor
    answer_relevancy: Floor
    context_precision: Floor
    context_recall: Floor
    decline_rate: Floor
    #: The largest share of the answerable items a gated RAGAs metric may leave unscored:
    #: a parse failure, a truncated judgement or no number. A mean over the few items a
    #: misparsing judge did score would otherwise pass as confidently as a full one.
    max_unscored: float = Field(ge=0, le=1)

    def metric_floors(self) -> dict[str, float | None]:
        """The floors by metric, without ``max_unscored``."""
        return self.model_dump(exclude={"max_unscored"})


class Thresholds(_Strict):
    """The whole thresholds file. ``generation`` is optional until S2-5b sets it; only
    ``rag-eval check`` needs it."""

    retrieval: RetrievalFloors
    generation: GenerationFloors | None = None


def load_thresholds(path: str | Path) -> Thresholds:
    """The floors in ``path``. Raises :class:`ThresholdsError` naming every problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ThresholdsError(f"cannot read the thresholds file {path}: {e.strerror}") from e
    except yaml.YAMLError as e:
        raise ThresholdsError(f"thresholds file {path} is not valid YAML:\n{e}") from e
    if raw is None:
        raise ThresholdsError(f"thresholds file {path} is empty: it needs a retrieval: section")
    try:
        return Thresholds.model_validate(raw)
    except ValidationError as e:
        problems = []
        for detail in e.errors():
            where = ".".join(str(part) for part in detail["loc"])
            problems.append(f"  {where}: {detail['msg']}" if where else f"  {detail['msg']}")
        # `from None`: the lines above say everything the pydantic dump would.
        raise ThresholdsError(f"invalid thresholds file {path}:\n" + "\n".join(problems)) from None


def below_floors(values: Mapping[str, float], floors: BaseModel) -> list[str]:
    """The names of the metrics in ``values`` that are below their floor, in the floors'
    order. A metric exactly at its floor meets it."""
    return [
        name for name, floor in floors.model_dump().items() if values[name] < floor
    ]


def below_generation_floors(
    aggregate: Mapping[str, float | None], floors: GenerationFloors
) -> list[str]:
    """The gated tier-2 metrics that miss their floor. A metric with no score at all (every
    item failed to score) misses it: a gate must not pass on no evidence."""
    return [
        name
        for name, floor in floors.metric_floors().items()
        if floor is not None and ((value := aggregate.get(name)) is None or value < floor)
    ]


def too_many_unscored(scores: "ScoresRun", floors: GenerationFloors) -> list[str]:
    """The gated RAGAs metrics with more unscored items than ``max_unscored`` allows."""
    answerable = sum(item.answerable for item in scores.items)
    problems = []
    for name, floor in floors.metric_floors().items():
        failures = scores.failures.get(name)
        if floor is None or failures is None or not answerable:
            continue
        unscored = failures.parse + failures.truncated + failures.no_score
        if unscored / answerable > floors.max_unscored:
            problems.append(
                f"{name}: {unscored} of {answerable} answerable items have no score ("
                f"{failures.parse} parse, {failures.truncated} truncated, "
                f"{failures.no_score} no score), above max_unscored {floors.max_unscored}"
            )
    return problems


# ---- the run files (ARCHITECTURE.md §2.4) ------------------------------------------------


class RunFileError(ValueError):
    """A run file is missing, unreadable or not the shape this version writes."""


class _Run(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Generator(_Run):
    """The model that answered. Its digest is recorded, never hashed: CI cannot see it."""

    ref: str  # the config reference, e.g. components.llms.mistral_ollama
    model: str | None  # the Ollama tag, when the llm is ChatOllama
    digest: str | None  # from Ollama's /api/tags when generate ran


class AnswerItem(_Run):
    """One golden question, answered through ``stream_answer``."""

    id: str
    question: str
    answerable: bool
    answer: str
    contexts: list[str]  # the chunks the prompt got, in its [n] order
    sources: list[dict[str, Any]]  # SourceRef fields, as the stream reported them
    ttft_ms: float | None
    total_ms: float


class AnswersRun(_Run):
    """``answers-latest.json``: what ``rag-eval generate`` wrote."""

    format: Literal[1] = 1
    kind: Literal["answers"] = "answers"
    generated_at: str
    limit: int | None  # --limit, which a committed run never has
    fingerprint: dict[str, str]  # the four generation parts
    generator: Generator
    items: list[AnswerItem]


class Failures(_Run):
    """Why some items have no score for one metric, counted; and how often the judge was
    asked to correct an answer that did not parse, so a judge that often misparses shows
    even when its retries succeed."""

    parse: int = 0  # the judge's output never parsed, retries included
    truncated: int = 0  # the judge's output hit max_tokens
    no_score: int = 0  # ragas returned no number (no statements to judge, say)
    retried: int = 0  # parse errors followed by a retry, whether or not the retry parsed


class JudgeRecord(_Run):
    """The judge as it was served, recorded by ``rag-eval score``. Digests are recorded,
    never hashed."""

    model: str
    digest: str
    base: str  # the Modelfile's FROM
    base_digest: str | None
    num_ctx: int
    served_context: int | None  # Ollama's /api/ps once the judge was loaded
    max_prompt_tokens: int
    max_completion_tokens: int
    mode: str
    embedding_model: str


class ScoreItem(_Run):
    """One question, scored. Answerable ones have ``scores`` (``None`` where a metric
    failed, with the reason in ``failures``); unanswerable ones have ``declined``."""

    id: str
    answerable: bool
    scores: dict[str, float | None] = Field(default_factory=dict)
    failures: dict[str, str] = Field(default_factory=dict)
    declined: bool | None = None
    leaked: list[str] = Field(default_factory=list)


class ScoresRun(_Run):
    """``generation-latest.json``: what ``rag-eval score`` wrote."""

    format: Literal[1] = 1
    kind: Literal["generation"] = "generation"
    generated_at: str  # the answers file's
    scored_at: str
    limit: int | None
    #: The sha256 of the answers file scored (:func:`answers_identity`), so ``check`` can
    #: tell that the committed scores are of the committed answers: a re-generate with
    #: unchanged inputs (a re-pulled model, unhashed code) keeps every fingerprint part.
    answers: str
    fingerprint: dict[str, str]  # all six parts
    generator: Generator
    judge: JudgeRecord
    versions: dict[str, str]  # ragas, instructor, openai: recorded; the first two are hashed
    failures: dict[str, Failures]
    aggregate: dict[str, float | None]
    items: list[ScoreItem]


def _load_run[RunT: _Run](model: type[RunT], path: Path, what: str) -> RunT:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise RunFileError(f"cannot read the {what} {path}: {e.strerror}") from e
    try:
        return model.model_validate_json(data)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        more = f" (and {e.error_count() - 1} more)" if e.error_count() > 1 else ""
        raise RunFileError(
            f"{path} is not the {what} this version writes: {where}: {first['msg']}{more}"
        ) from None


def load_answers(path: Path) -> AnswersRun:
    return _load_run(AnswersRun, path, "answers file")


def load_scores(path: Path) -> ScoresRun:
    return _load_run(ScoresRun, path, "scores file")


def write_run(run: BaseModel, path: Path) -> None:
    """Write a run file atomically: a temporary file of its own, ``fsync``, then
    ``os.replace``. An hours-long run must never leave half a file where the last good one
    was, nor a stray temporary file in ``eval/runs/`` when the write fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(run.model_dump_json(indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def answers_identity(answers: AnswersRun) -> str:
    """The sha256 of an answers file's content, as canonical JSON."""
    return spec_identity(answers.model_dump(mode="json"))


def stale(answers: AnswersRun, scores: ScoresRun, current: Mapping[str, str]) -> list[str]:
    """Why the committed run no longer stands for the checkout, one line per reason."""
    problems = []
    if scores.answers != answers_identity(answers):
        disagree = [
            part
            for part in GENERATION_PARTS
            if answers.fingerprint.get(part) != scores.fingerprint.get(part)
        ]
        problems.append(
            "the scores were not computed from the committed answers file"
            + (f" ({', '.join(disagree)} differ)" if disagree else "")
            + ": re-run score on it"
        )
    problems += [stale_message(part) for part in moved(scores.fingerprint, current)]
    return problems


def uncovered(golden: Sequence[GoldenItem], answers: AnswersRun, scores: ScoresRun) -> list[str]:
    """Whether the run covers exactly the golden set: a ``--limit`` run never passes."""
    wanted = [item.id for item in golden]
    problems = []
    runs = {
        "answers": [item.id for item in answers.items],
        "scores": [item.id for item in scores.items],
    }
    for what, run_ids in runs.items():
        missing = [i for i in wanted if i not in run_ids]
        extra = [i for i in run_ids if i not in wanted]
        if missing or extra or len(run_ids) != len(set(run_ids)):
            problems.append(
                f"the {what} file covers {len(run_ids)} of the golden set's {len(wanted)} "
                f"questions"
                + (f"; missing {', '.join(missing[:5])}" if missing else "")
                + (f"; not in the golden set: {', '.join(extra[:5])}" if extra else "")
                + ": re-run generate and score without --limit"
            )
    return problems


# ---- Ollama's native API, through urllib: nothing CI's job lacks -------------------------


class OllamaError(Exception):
    """Ollama could not be asked, or answered something unusable."""


DEFAULT_OLLAMA = "http://localhost:11434"


def ollama_root(url: str | None) -> str:
    """Ollama's native API root from a configured URL: the OpenAI endpoint's ``/v1`` is
    dropped. ``None`` means ``$OLLAMA_HOST``, then the default. An ``$OLLAMA_HOST`` with no
    port means Ollama's own, 11434, as Ollama and its clients read it."""
    from_env = url is None
    url = url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA
    if "://" not in url:
        url = f"http://{url}"
    url = url.rstrip("/").removesuffix("/v1")
    parts = urllib.parse.urlsplit(url)
    if from_env and parts.port is None:
        url = urllib.parse.urlunsplit(parts._replace(netloc=f"{parts.netloc}:11434"))
    return url


def ollama_tag(model: str) -> str:
    """A model name as ``/api/tags`` lists it: ``mistral`` is ``mistral:latest``."""
    return model if ":" in model else f"{model}:latest"


def _ollama(
    root: str, path: str, body: Mapping[str, Any] | None = None, timeout: float = 10
) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        root + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Ollama at {root} answered {path} with HTTP {e.code}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise OllamaError(f"cannot reach Ollama at {root} ({e}). Is it running?") from e


def ollama_digests(root: str) -> dict[str, str]:
    """Every local model's digest, by its full tag (``/api/tags``)."""
    models = _ollama(root, "/api/tags").get("models", [])
    return {m["name"]: m["digest"] for m in models}


def ollama_show(root: str, model: str) -> dict[str, Any] | None:
    """``/api/show`` for ``model``, or ``None`` when Ollama has no such model."""
    try:
        shown: dict[str, Any] = _ollama(root, "/api/show", {"model": model})
    except OllamaError as e:
        if isinstance(e.__cause__, urllib.error.HTTPError) and e.__cause__.code == 404:
            return None
        raise
    return shown


def served_parameters(shown: Mapping[str, Any]) -> dict[str, str]:
    """``/api/show``'s ``parameters``: one ``name value`` per line. Repeated names
    (``stop``) keep their last value, which nothing here reads."""
    parameters = {}
    for line in str(shown.get("parameters", "")).splitlines():
        words = line.split(None, 1)
        if len(words) == 2:
            parameters[words[0]] = words[1].strip().strip('"')
    return parameters


def ollama_context(root: str, model: str) -> int | None:
    """The context a loaded model is served with (``/api/ps``), or ``None`` when it is not
    loaded or this Ollama does not say."""
    for loaded in _ollama(root, "/api/ps").get("models", []):
        if loaded.get("name") == ollama_tag(model):
            context = loaded.get("context_length")
            return context if isinstance(context, int) else None
    return None


def digest_drift(
    answers: AnswersRun, scores: ScoresRun, live: Mapping[str, str]
) -> list[str]:
    """``check --with-ollama``: the recorded digests against the live ones. Re-pulling a
    model behind the same tag changes answers without changing any hashed input."""
    problems = []
    recorded = [
        (answers.generator.model, answers.generator.digest, "generate and score", False),
        (scores.judge.model, scores.judge.digest, "score", False),
        (scores.judge.base, scores.judge.base_digest, "score", True),
    ]
    for model, digest, rerun, is_base in recorded:
        if model is None:
            continue  # not an Ollama model: nothing to compare
        # Flagged, not matched by name: a generator that is the judge's base model (gemma2
        # answering) must still report a digest it never recorded (second reviewer).
        if digest is None and is_base:
            # The base was not listed when the run scored (removed after `ollama create`,
            # or a path). The judge's own digest already pins its layers, and no re-score
            # could record this one, so it is no reason to fail.
            continue
        now = live.get(ollama_tag(model))
        if digest is None:
            problems.append(f"{model}: the run recorded no digest; re-run {rerun} with Ollama up")
        elif now is None:
            problems.append(f"{model} is not in this Ollama (the run used {digest[:12]})")
        elif now != digest:
            problems.append(
                f"{model} changed since the run ({digest[:12]} then, {now[:12]} now): "
                f"re-run {rerun}"
            )
    return problems

