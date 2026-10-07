"""Tier 2's second half: score an answers file with RAGAs and a local judge (``rag-eval score``).

**The only module that imports ragas, openai or instructor**, all from the optional
``eval`` extra and only inside functions, so ``rag-eval check`` and the test suite run
without them (DEC-15). Coverage omits this module: its ragas wiring runs only in a real
score. Its pure parts, the refusals and the decline rate, are tested all the same.

What a run does, in order, refusing with a message (exit 2) before any model loads:

1. the config has an ``evaluation`` section, and every unanswerable golden record's
   ``ground_truth`` contains its ``decline_marker`` (the golden-set loader cannot know
   the config, S2-2's review);
2. the answers file's four generation parts match this checkout, so a Colab run never
   scores stale answers. ``references`` and ``judge`` are computed here;
3. Ollama serves ``rag-judge`` with exactly the Modelfile's ``num_ctx``. Ollama's OpenAI
   endpoint cannot set the context per request, and its 4k default would truncate
   RAGAs' prompts silently (caveat 22);
4. the installed ragas and instructor are the ones the fingerprint names.

Then it scores each answerable item with four metrics (faithfulness, answer relevancy,
context precision, context recall), one judge call at a time, and computes the decline
rate over the unanswerable ones. **A truncated judgement is an error, not a number:**
every judge call's token counts are checked, and a prompt within :data:`CONTEXT_MARGIN`
tokens of ``num_ctx``, or a prompt and answer that fill it, aborts the run.

**Failures are counted by cause** (S2-5's spike). instructor raises the same
``InstructorRetryException`` for a judge whose output never parsed and for a connection
error or a timeout. Only the first is a judge failure, counted per metric, with the item's
score left out. Anything else aborts the run, so a dead Ollama can never read as a column
of parse failures.
"""

import importlib.metadata
import importlib.util
import json
import math
import os
import re
import sys
import types
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.evaluation.fingerprint import (
    GENERATION_PARTS,
    INSTRUCTOR_VERSION,
    JUDGE_MODE,
    METRICS,
    RAGAS_VERSION,
    JudgeSettings,
    evaluation_of,
    fingerprint,
    moved,
    parse_modelfile,
    read_modelfile,
    stale_message,
)
from rag_qa.evaluation.gate import (
    AnswerItem,
    AnswersRun,
    Failures,
    JudgeRecord,
    OllamaError,
    ScoreItem,
    ScoresRun,
    answers_identity,
    ollama_context,
    ollama_digests,
    ollama_root,
    ollama_show,
    ollama_tag,
    served_parameters,
)
from rag_qa.schema import RagConfig

#: The closest any judge prompt may come to ``num_ctx`` (DEC-15).
CONTEXT_MARGIN = 256

#: langchain-community 0.4.2 removed this module, and ragas 0.4.3 imports it at the top of
#: ``ragas/llms/base.py`` (vibrantlabsai/ragas#2741), so ragas does not import at all.
#: ragas uses the class only in an ``isinstance()`` list on its legacy LangChain path,
#: which the judge here (an ``InstructorLLM``) never takes.
REMOVED_MODULE = "langchain_community.chat_models.vertexai"

#: What re-creates the judge, for every refusal about it.
CREATE_HINT = "Run: ollama create {model} -f {modelfile}"


class ScoringError(Exception):
    """Scoring cannot start, or must stop: the message says why and what fixes it."""


# ---- the refusals before anything loads --------------------------------------------------


def check_references(golden: Sequence[GoldenItem], marker: str) -> None:
    """Every unanswerable record's ``ground_truth`` is a refusal the decline rate would
    count, so the golden set and the config agree on what a decline looks like."""
    bad = [
        item.id
        for item in golden
        if not item.answerable and marker.lower() not in item.ground_truth.lower()
    ]
    if bad:
        raise ScoringError(
            f"the ground_truth of unanswerable record(s) {', '.join(bad)} does not contain "
            f"the decline marker {marker!r}. Their ground_truth is the refusal sentence, "
            f"which the marker must be part of"
        )


def check_answers(
    answers: AnswersRun, current: Mapping[str, str], golden: Sequence[GoldenItem]
) -> None:
    """The answers file was generated from this checkout, for this golden set."""
    differ = moved(answers.fingerprint, {part: current[part] for part in GENERATION_PARTS})
    if differ:
        reasons = "; ".join(stale_message(part) for part in differ)
        raise ScoringError(
            f"the answers file was generated from another checkout ({reasons}). Score it "
            f"at the commit it was generated at, or re-run generate here"
        )
    known = {item.id for item in golden}
    unknown = [item.id for item in answers.items if item.id not in known]
    if unknown:
        raise ScoringError(f"the answers file has questions the golden set has not: {unknown}")


def _same(served: str | None, written: str) -> bool:
    """Whether a served parameter is the Modelfile's: ``0`` and ``0.0`` are one value."""
    if served is None:
        return False
    try:
        return float(served) == float(written)
    except ValueError:
        return served == written


def _not_from_the_modelfile(
    root: str, shown: Mapping[str, Any], served: Mapping[str, str], settings: JudgeSettings
) -> list[str]:
    """What the served judge has that its Modelfile and its base do not give it: a
    parameter or a SYSTEM or TEMPLATE text left from an older Modelfile (S2-5's second
    reviewer). A created model serves its base's parameters, SYSTEM and TEMPLATE, with the
    Modelfile's own in their place.

    Needs the base model to know what it gives. When Ollama cannot show it (removed after
    ``ollama create``, or a FROM that is a file), only what the Modelfile sets is checked:
    each PARAMETER by the caller, and the SYSTEM and TEMPLATE here.
    """
    try:
        base = ollama_show(root, settings.base)
    except OllamaError:
        base = None
    differ = []
    if base is not None:
        inherited = served_parameters(base)
        differ += [
            f"{name} {served[name]} (neither the Modelfile nor {settings.base} sets it)"
            for name in sorted(set(served) - set(inherited) - set(settings.parameters))
        ]
        differ += [
            f"{name} {served.get(name, 'unset')} ({settings.base} has {value}, and the "
            f"Modelfile does not set it)"
            for name, value in inherited.items()
            if name not in settings.parameters and not _same(served.get(name), value)
        ]
    for keyword, written in (("SYSTEM", settings.system), ("TEMPLATE", settings.template)):
        key = keyword.lower()
        if written is not None:
            wanted = written
        elif base is not None:
            wanted = str(base.get(key) or "")
        else:
            continue
        if str(shown.get(key) or "").strip() != wanted.strip():
            differ.append(f"a {keyword} that is not the one its Modelfile gives it")
    return differ


def check_judge(
    root: str, model: str, modelfile: str, settings: JudgeSettings
) -> tuple[str, str | None]:
    """The judge is served as its Modelfile says: built on its FROM, with each of its
    parameters, ``num_ctx`` first, and nothing its Modelfile and base do not give it
    (:func:`_not_from_the_modelfile`). Returns its digest and its base model's (recorded,
    never hashed).

    The Modelfile's text is what the run's ``judge`` part hashes. An edited Modelfile that
    was never re-created would otherwise be recorded as the judge while the old one ran.
    """
    hint = CREATE_HINT.format(model=model, modelfile=modelfile)
    shown = ollama_show(root, model)
    if shown is None:
        raise ScoringError(f"Ollama has no model {model!r}. {hint}")
    served = served_parameters(shown)
    if not _same(served.get("num_ctx"), str(settings.num_ctx)):
        context = served.get("num_ctx") or "unset (Ollama defaults to 4k)"
        raise ScoringError(
            f"{model} is served with num_ctx {context}, but its Modelfile says "
            f"{settings.num_ctx}: RAGAs' prompts could be truncated silently (caveat 22). {hint}"
        )
    differ = [
        f"{name} {served.get(name, 'unset')} (the Modelfile says {value})"
        for name, value in settings.parameters.items()
        if not _same(served.get(name), value)
    ]
    differ += _not_from_the_modelfile(root, shown, served, settings)
    parent = str(shown.get("details", {}).get("parent_model") or "")
    if parent and ollama_tag(parent) != ollama_tag(settings.base):
        differ.append(f"built on {parent} (the Modelfile says FROM {settings.base})")
    if differ:
        raise ScoringError(
            f"{model} is not served as its Modelfile says: {'; '.join(differ)}. It was "
            f"probably edited without re-creating the model. {hint}"
        )
    digests = ollama_digests(root)
    digest = digests.get(ollama_tag(model))
    if digest is None:
        raise ScoringError(f"Ollama does not list {model!r} in /api/tags. {hint}")
    return digest, digests.get(ollama_tag(settings.base))


def check_ragas() -> dict[str, str]:
    """The installed versions of the eval extra; ragas and instructor must be the ones the
    fingerprint names. Returns them for the record."""
    try:
        versions = {
            name: importlib.metadata.version(name) for name in ("ragas", "instructor", "openai")
        }
    except importlib.metadata.PackageNotFoundError as e:
        raise ScoringError(
            f"rag-eval score needs the eval extra ({e.name} is not installed): "
            f"uv sync --extra eval"
        ) from None
    for name, pinned, constant in (
        ("ragas", RAGAS_VERSION, "RAGAS_VERSION"),
        ("instructor", INSTRUCTOR_VERSION, "INSTRUCTOR_VERSION"),
    ):
        if versions[name] != pinned:
            raise ScoringError(
                f"{name} {versions[name]} is installed, but this checkout scores with "
                f"{pinned} (rag_qa.evaluation.fingerprint.{constant}, held to uv.lock): "
                f"uv sync --extra eval"
            )
    return versions


# ---- the decline rate: deterministic, no judge --------------------------------------------


def leak_pattern(word: str) -> re.Pattern[str]:
    """``word`` as a whole word, in any case: "COCO" is not in "cocoa", and "H100" is in
    "the H100." Lookarounds rather than ``\\b``, which would never match after a word that
    ends in punctuation, such as "U.S."."""
    return re.compile(rf"(?<!\w){re.escape(word.lower())}(?!\w)")


def declines(answer: str, marker: str, must_not_contain: Sequence[str]) -> tuple[bool, list[str]]:
    """Whether ``answer`` is a decline (DEC-15), and the leak words it contains.

    A decline contains the marker, a phrase compared as a substring in any case, and none
    of ``must_not_contain``. A refusal that names a leak word to explain itself counts as
    a leak: an accepted cost (S2-5's story).
    """
    text = answer.lower()
    leaked = [word for word in must_not_contain if leak_pattern(word).search(text)]
    return marker.lower() in text and not leaked, leaked


def mean(values: Sequence[float | None]) -> float | None:
    """The mean of the scored values, or ``None`` when nothing was scored."""
    scored = [value for value in values if value is not None]
    return fmean(scored) if scored else None


# ---- the judge -------------------------------------------------------------------------


def import_ragas() -> types.SimpleNamespace:
    """ragas and what the judge is built from, imported only when scoring.

    Turns ragas's usage reporting off first: it posts to its maker unless told not to, and
    reads the setting once. Then stands in for :data:`REMOVED_MODULE` if it is missing,
    with a class nothing is an instance of, which is the truth here.
    """
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    if REMOVED_MODULE not in sys.modules and importlib.util.find_spec(REMOVED_MODULE) is None:
        stand_in = types.ModuleType(REMOVED_MODULE)
        stand_in.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[REMOVED_MODULE] = stand_in
    from instructor.core.exceptions import (
        AsyncValidationError,
        IncompleteOutputException,
        InstructorRetryException,
        ResponseParsingError,
    )
    from openai import AsyncOpenAI
    from pydantic import ValidationError
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import collections

    return types.SimpleNamespace(
        AsyncOpenAI=AsyncOpenAI,
        HuggingFaceEmbeddings=HuggingFaceEmbeddings,
        llm_factory=llm_factory,
        collections=collections,
        IncompleteOutputException=IncompleteOutputException,
        InstructorRetryException=InstructorRetryException,
        # What instructor 1.17 retries as a parse failure (its retry module's own list).
        PARSE_ERRORS=(ValidationError, json.JSONDecodeError, AsyncValidationError,
                      ResponseParsingError),
    )


def classify(error: BaseException, ragas: types.SimpleNamespace) -> str | None:
    """``"parse"`` or ``"truncated"`` for a judge failure, ``None`` for anything else.

    instructor wraps whatever ended its last attempt in ``InstructorRetryException``: a
    connection error or a timeout too, even after an earlier attempt failed to parse
    (a timeout on the retry, say). So the exception's *cause*, the final error, decides:
    only an output that did not parse is the judge failing.
    """
    if isinstance(error, ragas.IncompleteOutputException):
        return "truncated"
    if isinstance(error, ragas.InstructorRetryException) and isinstance(
        error.__cause__, ragas.PARSE_ERRORS
    ):
        return "parse"
    return None


@dataclass
class CallLog:
    """Every judge call's token counts, from instructor's ``completion:response`` hook.

    A hook cannot stop the run (instructor turns its exceptions into warnings), so an
    overflow is kept here and the scoring loop raises it after the metric returns.
    """

    num_ctx: int
    metric: str = ""
    calls: int = 0
    max_prompt: int = 0
    max_completion: int = 0
    overflow: str | None = None
    retried: Counter[str] = field(default_factory=Counter)

    def on_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        self.calls += 1
        if not isinstance(prompt, int) or not isinstance(completion, int):
            self.overflow = self.overflow or (
                f"a {self.metric} call reported no token usage, so truncation cannot be ruled out"
            )
            return
        self.max_prompt = max(self.max_prompt, prompt)
        self.max_completion = max(self.max_completion, completion)
        if prompt > self.num_ctx - CONTEXT_MARGIN or prompt + completion >= self.num_ctx:
            self.overflow = self.overflow or (
                f"a {self.metric} call used {prompt} prompt and {completion} answer tokens, "
                f"within {CONTEXT_MARGIN} of num_ctx {self.num_ctx} or past it: Ollama "
                f"truncates or shifts the context silently, so the judgement is not to be "
                f"trusted. Raise num_ctx in the judge's Modelfile"
            )

    def on_parse_error(
        self, error: BaseException, *, is_last_attempt: bool = False, **attempt: Any
    ) -> None:
        """Counts the parse errors instructor then retries; one on a call's last attempt
        is that call's failure, which the scoring loop counts as ``parse``.

        instructor passes ``attempt_number`` and ``max_attempts`` too, and calls a handler
        that cannot take all three with the error alone: hence ``**attempt``.
        """
        if not is_last_attempt:
            self.retried[self.metric] += 1


def metric_inputs(item: AnswerItem, reference: str) -> dict[str, dict[str, Any]]:
    """What each metric is asked, from one answered item and its reference."""
    return {
        "faithfulness": {
            "user_input": item.question,
            "response": item.answer,
            "retrieved_contexts": item.contexts,
        },
        "answer_relevancy": {"user_input": item.question, "response": item.answer},
        "context_precision": {
            "user_input": item.question,
            "reference": reference,
            "retrieved_contexts": item.contexts,
        },
        "context_recall": {
            "user_input": item.question,
            "retrieved_contexts": item.contexts,
            "reference": reference,
        },
    }


def unscorable(inputs: Mapping[str, Any]) -> bool:
    """Whether ragas would refuse these inputs outright: an empty answer or no contexts.
    Counted as no score, never sent to the judge."""
    return any(not inputs[key] for key in ("response", "retrieved_contexts") if key in inputs)


@dataclass(frozen=True)
class ScoreRequest:
    """Everything a score run needs, checked."""

    config: RagConfig
    golden: Sequence[GoldenItem]
    answers: AnswersRun
    eval_dir: Path
    limit: int | None = None


async def score(
    request: ScoreRequest,
    progress: Callable[[str], None] | None = None,
) -> ScoresRun:
    """Score ``request.answers``; see the module docstring. Raises :class:`ScoringError`."""
    config, golden, answers = request.config, request.golden, request.answers
    evaluation = evaluation_of(config)
    judge = evaluation.judge
    say = progress or (lambda _: None)

    check_references(golden, evaluation.decline_marker)
    current = fingerprint(config, golden, request.eval_dir)
    check_answers(answers, current, golden)
    settings = parse_modelfile(read_modelfile(request.eval_dir.resolve() / judge.modelfile))
    root = ollama_root(judge.base_url)
    try:
        modelfile = request.eval_dir.resolve() / judge.modelfile
        if modelfile.is_relative_to(Path.cwd()):  # as the user would type it
            modelfile = modelfile.relative_to(Path.cwd())
        digest, base_digest = check_judge(root, judge.model, str(modelfile), settings)
    except OllamaError as e:
        raise ScoringError(str(e)) from e
    versions = check_ragas()

    ragas = import_ragas()
    client = ragas.AsyncOpenAI(
        base_url=judge.base_url, api_key="ollama", max_retries=0, timeout=judge.timeout_s
    )
    # ragas sends its own temperature, top_p and max_tokens with every call, which override
    # the Modelfile's. So the Modelfile's are sent: what is hashed is what runs.
    top_p = settings.parameters.get("top_p")
    sampling = {
        "temperature": settings.temperature,
        "seed": settings.seed,
        "max_tokens": settings.num_predict,
        "top_p": None if top_p is None else float(top_p),
    }
    llm = ragas.llm_factory(
        judge.model, client=client, **{k: v for k, v in sampling.items() if v is not None}
    )
    mode = str(getattr(llm.client, "mode", "")).rsplit(".", 1)[-1].lower()
    if mode != JUDGE_MODE:
        raise ScoringError(
            f"ragas built the judge in instructor mode {mode!r}, not {JUDGE_MODE!r}: this "
            f"ragas no longer behaves as S2-5's spike measured"
        )
    log = CallLog(num_ctx=settings.num_ctx)
    llm.client.on("completion:response", log.on_response)
    llm.client.on("parse:error", log.on_parse_error)
    embeddings = ragas.HuggingFaceEmbeddings(model=judge.embedding_model, device="cpu")
    metrics = {
        name: getattr(ragas.collections, spec["class"])(
            llm=llm,
            **({"embeddings": embeddings} if spec["class"] == "AnswerRelevancy" else {}),
            **{key: value for key, value in spec.items() if key != "class"},
        )
        for name, spec in METRICS.items()
    }

    references = {item.id: item.ground_truth for item in golden}
    must_not = {item.id: item.must_not_contain for item in golden}
    items = answers.items if request.limit is None else answers.items[: request.limit]
    counts: dict[str, Counter[str]] = {name: Counter() for name in METRICS}
    served_context = None
    polled = False
    scored: list[ScoreItem] = []
    for n, item in enumerate(items, 1):
        if not item.answerable:
            declined, leaked = declines(item.answer, evaluation.decline_marker, must_not[item.id])
            scored.append(
                ScoreItem(id=item.id, answerable=False, declined=declined, leaked=leaked)
            )
            say(f"[{n}/{len(items)}] {item.id}: {'declined' if declined else 'did not decline'}"
                + (f", leaked {leaked}" if leaked else ""))
            continue
        values: dict[str, float | None] = {}
        failures: dict[str, str] = {}
        for name, inputs in metric_inputs(item, references[item.id]).items():
            log.metric = name
            if unscorable(inputs):
                values[name], failures[name] = None, "no_score"
                counts[name]["no_score"] += 1
                continue
            try:
                value = (await metrics[name].ascore(**inputs)).value
            except Exception as e:
                kind = classify(e, ragas)
                if kind is None:
                    raise ScoringError(
                        f"the judge call for {name} on {item.id} failed: {type(e).__name__}: "
                        f"{e}. Nothing was written"
                    ) from e
                values[name], failures[name] = None, kind
                counts[name][kind] += 1
            else:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    values[name], failures[name] = None, "no_score"
                    counts[name]["no_score"] += 1
                else:
                    values[name] = float(value)
            if log.overflow:
                raise ScoringError(f"{log.overflow}. Nothing was written")
            if not polled and log.calls:  # once, while the judge is surely loaded
                polled = True
                try:
                    served_context = ollama_context(root, judge.model)
                except OllamaError as e:
                    raise ScoringError(str(e)) from e
                if served_context is None:
                    say(f"Note: Ollama's /api/ps does not report {judge.model}'s context; "
                        f"its num_ctx was checked through /api/show only")
                elif served_context != settings.num_ctx:
                    raise ScoringError(
                        f"Ollama serves {judge.model} with a {served_context}-token context, "
                        f"not its Modelfile's {settings.num_ctx}. Nothing was written"
                    )
        scored.append(ScoreItem(id=item.id, answerable=True, scores=values, failures=failures))
        shown = ", ".join(
            f"{name} {'-' if value is None else f'{value:.3f}'}" for name, value in values.items()
        )
        say(f"[{n}/{len(items)}] {item.id}: {shown}")

    aggregate: dict[str, float | None] = {
        name: mean([i.scores.get(name) for i in scored if i.answerable]) for name in METRICS
    }
    unanswerable = [i for i in scored if not i.answerable]
    aggregate["decline_rate"] = (
        sum(bool(i.declined) for i in unanswerable) / len(unanswerable) if unanswerable else None
    )
    return ScoresRun(
        generated_at=answers.generated_at,
        scored_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        limit=request.limit if request.limit is not None else answers.limit,
        answers=answers_identity(answers),
        fingerprint={
            **{part: answers.fingerprint[part] for part in GENERATION_PARTS},
            "references": current["references"],
            "judge": current["judge"],
        },
        generator=answers.generator,
        judge=JudgeRecord(
            model=judge.model,
            digest=digest,
            base=settings.base,
            base_digest=base_digest,
            num_ctx=settings.num_ctx,
            served_context=served_context,
            max_prompt_tokens=log.max_prompt,
            max_completion_tokens=log.max_completion,
            mode=mode,
            embedding_model=judge.embedding_model,
        ),
        versions=versions,
        failures={
            name: Failures(
                parse=counts[name]["parse"],
                truncated=counts[name]["truncated"],
                no_score=counts[name]["no_score"],
                retried=log.retried[name],
            )
            for name in METRICS
        },
        aggregate=aggregate,
        items=scored,
    )
