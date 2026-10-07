"""The tier-2 fingerprint: what a committed run was computed from (DEC-15, ARCHITECTURE.md §2.1).

Tier 2 runs offline, for hours, and is committed. ``rag-eval check`` then fails CI when the
committed run no longer matches the repository. So a run records six hashes, one per kind
of input, and the part that moved names the re-run that fixes it:

=============  ===============================================================  ===============
part           hashes                                                           when it moves
=============  ===============================================================  ===============
``query``      the llm, retriever and reranker specs, ``reranker_candidates``,  generate, score
               the prompt template, and the prompt rendered over two fixed
               documents (the *probe*)
``ingestion``  the embedder identity, the splitter, the loader map and          generate, score
               :data:`~rag_qa.manifest.CHUNKING_VERSION`
``corpus``     the sha256 of every file ``rag-ingest`` would index              generate, score
``questions``  each golden record's id, question, answerable flag and           generate, score
               ``must_not_contain``
``references`` each golden record's ``ground_truth``                            score
``judge``      the judge model, its Modelfile (comments aside), the answer-     score
               relevancy embedder, the metric set, the decline marker, and the
               ragas and instructor versions
=============  ===============================================================  ===============

Each is canonical JSON (sorted keys) hashed with sha256, by
:func:`rag_qa.manifest.spec_identity`. Specs are hashed by content, never by their
reference name, so renaming a config entry moves nothing. Left out because they are
machine-specific: paths, ``base_url`` and device keys. Left out because tier 2 never reads
them: the golden set's ``notes`` and ``expected_sources``. Model digests are recorded beside
the fingerprint, never hashed: CI has no Ollama to recompute them.

The ingestion identities and the corpus walk are :mod:`rag_qa.manifest`'s and
:mod:`rag_qa.ingest`'s own, reused unchanged, so a run and the index agree on what "the
same embedder" and "the corpus" mean. Code is not hashed, except through the probe, which
moves ``query`` when :func:`rag_qa.chain.format_docs` or ``citation()`` changes
(ARCHITECTURE.md §2.1, "Honest limit").

``rag-eval check`` imports this in CI's model-free job, so it imports ``langchain_core``
and ``rag_qa.chain`` at most: never ragas, openai or the embedder's stack
(``tests/test_architecture.py``).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document

from rag_qa import chain
from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.ingest import CorpusScan, IngestReport, scan_corpus
from rag_qa.manifest import CHUNKING_VERSION, embedder_identity, spec_identity
from rag_qa.schema import Evaluation, RagConfig

PARTS: Final = ("query", "ingestion", "corpus", "questions", "references", "judge")
#: What ``rag-eval generate`` records, and ``rag-eval score`` copies and checks.
GENERATION_PARTS: Final = ("query", "ingestion", "corpus", "questions")

#: The re-run that brings a run up to date once a part has moved.
RERUN: Final[Mapping[str, str]] = {
    "query": "generate and score",
    # generate refuses an index that lags the corpus or config, so rag-ingest comes first.
    "ingestion": "rag-ingest, then generate and score",
    "corpus": "rag-ingest, then generate and score",
    "questions": "generate and score",
    "references": "score",
    "judge": "score",
}
#: What a part covers, in a stale message.
COVERS: Final[Mapping[str, str]] = {
    "query": "the llm, retriever, reranker, prompt, or how the prompt is assembled",
    "ingestion": "the embedder, splitter, loader map, or chunking version",
    "corpus": "a file rag-ingest indexes",
    "questions": "a golden record's id, question, answerable flag or must_not_contain",
    "references": "a golden record's ground_truth",
    "judge": "the judge model, its Modelfile, the embedder, the metrics, the decline "
    "marker, or the ragas or instructor version",
}

#: The ragas that scores. CI has no ragas to ask, so the version is pinned here, and a test
#: holds it to ``uv.lock``: a lock bump fails CI until this follows, and then ``judge``
#: moves. ``rag-eval score`` refuses an installed ragas that differs.
RAGAS_VERSION: Final = "0.4.3"
#: The instructor that parses the judge's answers, pinned and held to ``uv.lock`` as ragas
#: is: in JSON mode it adds its own system message to every judge call, and writes the
#: re-ask after a parse failure, so an upgrade changes the judge's prompts (S2-5's second
#: reviewer). ``rag-eval score`` refuses an installed instructor that differs.
INSTRUCTOR_VERSION: Final = "1.17.0"
#: How the judge's answers are parsed: ragas patches an OpenAI client with instructor's
#: JSON mode (S2-5's spike). ``rag-eval score`` checks that the judge it built uses it.
JUDGE_MODE: Final = "json"
#: The metric set, by the names the scores use, with every setting that changes a score.
#: Answer relevancy asks one question, not ragas's default three: under greedy decoding
#: the three were identical in S2-5's spike, so the mean of three equals one.
METRICS: Final[Mapping[str, Mapping[str, Any]]] = {
    "faithfulness": {"class": "Faithfulness"},
    "answer_relevancy": {"class": "AnswerRelevancy", "strictness": 1},
    "context_precision": {"class": "ContextPrecision"},  # with a reference
    "context_recall": {"class": "ContextRecall"},
}

#: Keys of a query component that choose where or how it runs, never what it answers, so
#: editing one never forces a re-run: the server and its HTTP clients (timeouts included),
#: the device, how long Ollama keeps the model loaded, and the start-up check that it
#: exists. ``num_thread`` is deliberately not one: a different thread count can change the
#: floating-point sums, and so, rarely, an answer (STATUS.md backlog, S2-5b).
RUNTIME_KEYS: Final = (
    "base_url",
    "device",
    "client_kwargs",
    "async_client_kwargs",
    "sync_client_kwargs",
    "keep_alive",
    "validate_model_on_init",
)

#: What the prompt is rendered over for the probe: one chunk with a page, one without, so
#: a change to how either kind is cited moves ``query``.
PROBE_DOCS: Final = (
    Document(
        page_content="The first probe passage.",
        metadata={"source": "probe/first.pdf", "page": 6, "page_label": "7"},
    ),
    Document(page_content="The second probe passage.", metadata={"source": "probe/second.docx"}),
)
PROBE_QUESTION: Final = "What do the probe passages say?"


class FingerprintError(Exception):
    """A part cannot be computed from the checkout. The message says what to fix."""


def stale_message(part: str) -> str:
    """What a moved part means, and the re-run that fixes it."""
    return f"{part} changed: re-run {RERUN[part]} ({COVERS[part]})"


def moved(recorded: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """The parts of ``current`` that ``recorded`` does not match, in :data:`PARTS` order."""
    return [part for part in PARTS if part in current and recorded.get(part) != current[part]]


def _without_runtime(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A query component's spec minus :data:`RUNTIME_KEYS` and ``model_kwargs.device``."""
    kept = {key: value for key, value in spec.items() if key not in RUNTIME_KEYS}
    model_kwargs = kept.pop("model_kwargs", None)
    if isinstance(model_kwargs, Mapping):
        model_kwargs = {key: value for key, value in model_kwargs.items() if key != "device"}
    if model_kwargs:
        kept["model_kwargs"] = model_kwargs
    return kept


def query_part(config: RagConfig) -> str:
    query = config.pipeline.query
    # Through the chain module's own names, so the probe follows the code that answers.
    messages = chain.build_prompt(config).format_messages(
        context=chain.format_docs(PROBE_DOCS), question=PROBE_QUESTION
    )
    reranker = None if query.reranker is None else config.component(query.reranker).spec()
    return spec_identity(
        {
            "llm": _without_runtime(config.component(query.llm).spec()),
            "retriever": config.retriever(query.retriever).kwargs(),
            "reranker": None if reranker is None else _without_runtime(reranker),
            "reranker_candidates": query.reranker_candidates,
            "prompt": {"system": query.prompt.system, "human": query.prompt.human},
            "probe": [[message.type, message.content] for message in messages],
        }
    )


def ingestion_part(config: RagConfig) -> str:
    ingestion = config.pipeline.ingestion
    return spec_identity(
        {
            "embedder": embedder_identity(config.component(ingestion.embedder).spec()),
            "splitter": spec_identity(config.component(ingestion.splitter).spec()),
            "loaders": {
                extension: spec_identity(config.component(ref).spec())
                for extension, ref in ingestion.loaders.items()
            },
            # The code that makes chunks, versioned (S2-6's third review): a bump changes
            # the chunks, so the contexts, so the answers.
            "chunking": CHUNKING_VERSION,
        }
    )


def corpus_scan(config: RagConfig) -> CorpusScan:
    """The corpus exactly as ``rag-ingest`` walks it: the same function, so the same
    pruning, extension map, symlink rule and file-name check.

    Raises :class:`FingerprintError` when the corpus is missing, or when anything in it
    would fail ``rag-ingest`` too (an unreadable file or folder, a name that is not UTF-8).
    """
    report = IngestReport()
    try:
        scan = scan_corpus(config, report)
    except FileNotFoundError as e:
        raise FingerprintError(str(e)) from e
    if report.failed:
        failed = "; ".join(f"{name}: {error}" for name, error in report.failed)
        raise FingerprintError(
            f"rag-ingest cannot read part of the corpus either, so it has no fingerprint. "
            f"Fix these first: {failed}"
        )
    return scan


def corpus_part(scan: CorpusScan) -> str:
    return spec_identity({file.name: file.state.sha256 for file in scan.files})


def questions_part(golden: Iterable[GoldenItem]) -> str:
    # By id: the file's order changes no answer, so it must not force a re-run.
    records = sorted(
        (
            {
                "id": item.id,
                "question": item.question,
                "answerable": item.answerable,
                "must_not_contain": list(item.must_not_contain),
            }
            for item in golden
        ),
        key=lambda record: str(record["id"]),
    )
    return spec_identity({"questions": records})


def references_part(golden: Iterable[GoldenItem]) -> str:
    return spec_identity({item.id: item.ground_truth for item in golden})


def evaluation_of(config: RagConfig) -> Evaluation:
    """The config's ``evaluation`` section, which scoring and checking need."""
    if config.evaluation is None:
        raise FingerprintError(
            "the config has no 'evaluation' section, which rag-eval score and check need: "
            "the decline marker and the judge (config.yaml shows it)"
        )
    return config.evaluation


def modelfile_path(config: RagConfig, eval_dir: Path) -> Path:
    return eval_dir / evaluation_of(config).judge.modelfile


def read_modelfile(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise FingerprintError(f"cannot read the judge's Modelfile {path}: {e}") from e


def modelfile_directives(text: str) -> list[str]:
    """A Modelfile's text without its comment and blank lines: what Ollama builds from.

    Hashed in place of the raw text, so editing a comment never forces a re-score, while
    a FROM, PARAMETER, SYSTEM or TEMPLATE edit still does. A line inside a ``\"\"\"``
    block (a SYSTEM or TEMPLATE text) is kept whatever it holds, ``#`` included.
    """
    kept = []
    in_block = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not in_block and (not stripped or stripped.startswith("#")):
            continue
        kept.append(line)
        if stripped.count('"""') % 2:
            in_block = not in_block
    return kept


def judge_part(evaluation: Evaluation, modelfile_text: str) -> str:
    judge = evaluation.judge
    return spec_identity(
        {
            "model": judge.model,
            "modelfile": modelfile_directives(modelfile_text),
            "embedding_model": judge.embedding_model,
            "metrics": METRICS,
            "mode": JUDGE_MODE,
            # It defines the decline rate, which score computes.
            "decline_marker": evaluation.decline_marker,
            "ragas": RAGAS_VERSION,
            "instructor": INSTRUCTOR_VERSION,
        }
    )


def fingerprint(
    config: RagConfig,
    golden: Sequence[GoldenItem],
    eval_dir: Path | None = None,
    parts: Sequence[str] = PARTS,
) -> dict[str, str]:
    """The ``parts`` asked for, computed from the checkout. ``judge`` needs ``eval_dir``,
    where the Modelfile is. Raises :class:`FingerprintError`."""
    computed: dict[str, str] = {}
    for part in parts:
        if part == "query":
            computed[part] = query_part(config)
        elif part == "ingestion":
            computed[part] = ingestion_part(config)
        elif part == "corpus":
            computed[part] = corpus_part(corpus_scan(config))
        elif part == "questions":
            computed[part] = questions_part(golden)
        elif part == "references":
            computed[part] = references_part(golden)
        elif part == "judge":
            if eval_dir is None:
                raise ValueError("the judge part needs the eval folder")
            text = read_modelfile(modelfile_path(config, eval_dir))
            computed[part] = judge_part(evaluation_of(config), text)
        else:
            raise ValueError(f"unknown fingerprint part {part!r}")
    return computed


@dataclass(frozen=True)
class JudgeSettings:
    """What ``rag-eval score`` needs from the judge's Modelfile."""

    base: str  # FROM: the model the judge is built on, whose digest is recorded too
    num_ctx: int
    num_predict: int | None
    temperature: float | None
    seed: int | None
    #: Every PARAMETER, as written: ``score`` checks that the served judge has each one,
    #: so an edited Modelfile that was never re-created cannot pass for the one it hashes.
    #: A repeated name (``stop``) keeps its last value.
    parameters: Mapping[str, str] = field(default_factory=dict)
    #: The SYSTEM and TEMPLATE texts, when the Modelfile sets them: ``score`` checks the
    #: served judge has them too, as it does each PARAMETER (S2-5's second reviewer).
    system: str | None = None
    template: str | None = None


def modelfile_instructions(text: str) -> list[tuple[str, str]]:
    """A Modelfile's instructions as ``(KEYWORD, argument)``, in order.

    Keywords are upper-cased, as Ollama reads them case-insensitively; ``#`` and blank
    lines are skipped. An argument in ``\"\"\"`` may span lines, and comes back without its
    quotes; a single-line ``"argument"`` loses its quotes too.
    """
    lines = text.splitlines()
    instructions = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        keyword, *rest = line.split(None, 1)
        argument = rest[0].strip() if rest else ""
        if argument.startswith('"""'):
            body = argument[3:]
            while '"""' not in body and i < len(lines):
                body += "\n" + lines[i]
                i += 1
            argument = body.split('"""', 1)[0]
        elif len(argument) >= 2 and argument[0] == argument[-1] == '"':
            argument = argument[1:-1]
        instructions.append((keyword.upper(), argument))
    return instructions


def parse_modelfile(text: str) -> JudgeSettings:
    """The FROM, PARAMETER, SYSTEM and TEMPLATE instructions of an Ollama Modelfile, as
    :class:`JudgeSettings` (:func:`modelfile_instructions`).

    Raises :class:`FingerprintError` when there is no FROM or no ``num_ctx``: the context
    the judge is served with is what ``score`` checks, and a Modelfile without one would get
    Ollama's 4k default (caveat 22).
    """
    base = None
    parameters: dict[str, str] = {}
    texts: dict[str, str] = {}
    for keyword, argument in modelfile_instructions(text):
        if keyword == "FROM" and argument:
            base = argument.split()[0]
        elif keyword == "PARAMETER":
            words = argument.split(None, 1)
            if len(words) == 2:
                parameters[words[0].lower()] = words[1].strip().strip('"')
        elif keyword in ("SYSTEM", "TEMPLATE"):
            texts[keyword] = argument
    if base is None:
        raise FingerprintError("the judge's Modelfile has no FROM line")
    if "num_ctx" not in parameters:
        raise FingerprintError(
            "the judge's Modelfile sets no num_ctx, so Ollama would serve it with its 4k "
            "default and truncate RAGAs' prompts silently. Add PARAMETER num_ctx 8192"
        )

    def number(name: str, kind: type[int] | type[float]) -> Any:
        value = parameters.get(name)
        if value is None:
            return None
        try:
            return kind(value)
        except ValueError:
            raise FingerprintError(
                f"the judge's Modelfile sets {name} to {value!r}, which is not a number"
            ) from None

    return JudgeSettings(
        base=base,
        num_ctx=number("num_ctx", int),
        num_predict=number("num_predict", int),
        temperature=number("temperature", float),
        seed=number("seed", int),
        parameters=parameters,
        system=texts.get("SYSTEM"),
        template=texts.get("TEMPLATE"),
    )
