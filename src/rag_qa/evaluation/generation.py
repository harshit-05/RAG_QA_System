"""Tier 2's first half: answer every golden question as users get it (``rag-eval generate``).

Each question goes through :func:`rag_qa.answering.stream_answer`, the stream the CLI
and the API render (DEC-14), so tier 2 measures exactly what users get. The run records,
per question, the answer, the chunks the prompt got (``contexts``, which RAGAs judges
against), the sources as the stream reported them, and the time to the first token and
to the end. It also records the four generation parts of the fingerprint
(:mod:`.fingerprint`) and the generator's Ollama digest, which is recorded, never hashed.
``rag-eval score`` then needs only this file, so scoring can run on Colab (DEC-15).

**The index must match the corpus and config the run records.** The fingerprint is
computed from the checkout, but the answers come from the index. An index not rebuilt
since a corpus edit would answer from old chunks under a fingerprint that claims the new
ones, and ``check`` would pass it (maintainer, 2026-10-06). So the run refuses an index
whose manifest disagrees with the same corpus scan the fingerprint hashes.
"""

import dataclasses
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_qa.answering import Done, Sources, Token, stream_answer
from rag_qa.chain import QueryPipeline, build_query_pipeline, check_index
from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.evaluation.fingerprint import GENERATION_PARTS, corpus_part, corpus_scan, fingerprint
from rag_qa.evaluation.gate import (
    AnswerItem,
    AnswersRun,
    Generator,
    OllamaError,
    ollama_digests,
    ollama_root,
    ollama_tag,
)
from rag_qa.ingest import CorpusScan
from rag_qa.manifest import CHUNKING_VERSION, diff, spec_identity
from rag_qa.schema import RagConfig

#: The ``_target_`` whose model is an Ollama tag, so has a digest to record.
OLLAMA_CHAT = "langchain_ollama.ChatOllama"


class StaleIndexError(Exception):
    """The index is not what the corpus and config would build now: re-run ``rag-ingest``."""


def check_index_current(config: RagConfig, scan: CorpusScan) -> None:
    """Refuse an index that lags the corpus or the config (module docstring).

    Raises :class:`~rag_qa.chain.NoIndexError` or its subclass first, as every front end
    does; then :class:`StaleIndexError` when the manifest's documents, splitter or
    chunking version differ from what ``rag-ingest`` would build now. The embedder's
    identity is :func:`~rag_qa.chain.check_index`'s own check. Reads files only.
    """
    _, manifest = check_index(config)
    changes = diff(manifest, {file.name: file.state for file in scan.files})
    problems = []
    counts = [
        f"{len(names)} {what}"
        for names, what in (
            (changes.added, "added"),
            (changes.changed, "changed"),
            (changes.removed, "removed"),
        )
        if names
    ]
    if counts:
        named = [*changes.added, *changes.changed, *changes.removed][:3]
        problems.append(f"documents {', '.join(counts)} (e.g. {', '.join(named)})")
    splitter = config.pipeline.ingestion.splitter
    if manifest.splitter.identity != spec_identity(config.component(splitter).spec()):
        problems.append("the splitter is not the one that built it")
    if manifest.chunking != CHUNKING_VERSION:
        problems.append(f"it was chunked by version {manifest.chunking}, not {CHUNKING_VERSION}")
    if problems:
        raise StaleIndexError(
            f"the index is not up to date with the corpus and config: {'; '.join(problems)}. "
            f"Run rag-ingest first, so the answers come from the corpus this run records."
        )


def generator_record(config: RagConfig) -> Generator:
    """The model that answers, with its Ollama digest when it is an Ollama model.

    The digest is ``None`` when Ollama cannot say; ``check --with-ollama`` then reports
    that the run recorded none, rather than this run failing for want of a record.
    """
    ref = config.pipeline.query.llm
    spec = config.component(ref).spec()
    model = spec.get("model")
    if spec["_target_"] != OLLAMA_CHAT or not isinstance(model, str):
        return Generator(ref=ref, model=None, digest=None)
    try:
        digest = ollama_digests(ollama_root(spec.get("base_url"))).get(ollama_tag(model))
    except OllamaError:
        digest = None
    return Generator(ref=ref, model=model, digest=digest)


def _recording(pipeline: QueryPipeline, captured: list[Document]) -> QueryPipeline:
    """``pipeline`` whose retrieval also keeps what it returned in ``captured``: exactly
    the chunks the prompt is rendered over, in the order it numbers them."""
    inner = pipeline.retrieve
    if inner is None:  # build_query_pipeline(require_index=True) never leaves it unset
        raise ValueError("the pipeline has no retrieval half")

    async def retrieve(question: str) -> list[Document]:
        docs = await inner.ainvoke(question)
        captured[:] = docs
        return docs

    return dataclasses.replace(pipeline, retrieve=RunnableLambda(retrieve))


async def answer(pipeline: QueryPipeline, item: GoldenItem) -> AnswerItem:
    """One golden question, answered through :func:`~rag_qa.answering.stream_answer`."""
    captured: list[Document] = []
    recording = _recording(pipeline, captured)
    text: list[str] = []
    sources: list[dict[str, object]] = []
    done = None
    async for event in stream_answer(recording, item.question):
        if isinstance(event, Sources):
            sources = [dataclasses.asdict(source) for source in event.sources]
        elif isinstance(event, Token):
            text.append(event.text)
        elif isinstance(event, Done):
            done = event
    if done is None:  # stream_answer always ends with Done unless it raised
        raise RuntimeError("the answer stream ended without its Done event")
    return AnswerItem(
        id=item.id,
        question=item.question,
        answerable=item.answerable,
        answer="".join(text),
        contexts=[doc.page_content for doc in captured],
        sources=sources,
        ttft_ms=done.ttft_ms,
        total_ms=done.total_ms,
    )


def prepare(config: RagConfig, golden: Sequence[GoldenItem]) -> dict[str, str]:
    """Everything a run can refuse before a model loads: the four generation parts, and
    an index that lags them. Raises ``FingerprintError``, ``NoIndexError`` (or its
    subclass) or :class:`StaleIndexError`."""
    scan = corpus_scan(config)
    check_index_current(config, scan)
    parts = fingerprint(config, golden, parts=("query", "ingestion", "questions"))
    parts["corpus"] = corpus_part(scan)  # the scan the index was just checked against
    return {part: parts[part] for part in GENERATION_PARTS}


async def generate(
    config: RagConfig,
    golden: Sequence[GoldenItem],
    *,
    limit: int | None = None,
    progress: Callable[[int, int, AnswerItem], None] | None = None,
) -> AnswersRun:
    """Answer the golden set's first ``limit`` questions (all, by default).

    The fingerprint covers the whole golden set either way; ``check`` refuses a run that
    does not cover every question. ``progress`` is called after each answer, with its
    1-based number and the total.
    """
    parts = prepare(config, golden)
    generator = generator_record(config)
    items = list(golden if limit is None else golden[:limit])
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    pipeline = build_query_pipeline(config)
    answers = []
    try:
        for n, item in enumerate(items, 1):
            answers.append(await answer(pipeline, item))
            if progress is not None:
                progress(n, len(items), answers[-1])
    finally:
        await pipeline.aclose()
    return AnswersRun(
        generated_at=generated_at,
        limit=limit,
        fingerprint=parts,
        generator=generator,
        items=answers,
    )
