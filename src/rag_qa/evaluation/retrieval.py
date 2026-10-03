"""Tier 1 of the quality bar: retrieval scored against the golden set (FR-7, DEC-15).

Every answerable golden question goes through the retrieval half that
:func:`rag_qa.chain.build_retrieve` builds. That is exactly what the prompt would
receive, in the order the prompt numbers it. Each retrieved chunk is matched to the
record's ``expected_sources`` by its corpus-relative ``source`` and its printed page
label, both as :func:`rag_qa.answering.source_refs` reports them. So tier 1 scores the
sources a user is shown, before S2-6 makes ``source`` relative in the index and after.

Three numbers for each question, then their means over the questions:

- **hit**: a chunk is on an expected page, or on one of its ``also_pages``;
- **reciprocal rank**: ``1 / rank`` of the first such chunk, or 0 for a miss;
- **recall**: the share of the expected ``pages`` the chunks cover. ``also_pages`` are
  outside its denominator. They repeat the whole answer, while recall measures the pages
  the ground truth was written from (S2-2).

A source listed without pages (a format that has none) matches any chunk of that file,
and counts as one page towards recall.

Deterministic, and it needs only the embedder and the index: no LLM, so no Ollama. That
is what lets CI recompute it on every push (ARCHITECTURE.md §2.5, ``eval-retrieval``).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal

from langchain_core.documents import Document

from rag_qa.answering import source_refs
from rag_qa.chain import NoIndexError, build_retrieve
from rag_qa.components import build_embedder
from rag_qa.evaluation.dataset import GoldenItem
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import store_exists

#: Why a chunk counts: it is on an expected page, or on an also-page.
Match = Literal["page", "also"]


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk, as tier 1 matched it."""

    source: str  # corpus-relative, as SourceRef.source reports it
    page: str | None  # the printed page label; None for formats without pages
    match: Match | None  # None: neither an expected page nor an also-page


@dataclass(frozen=True)
class ItemScore:
    """One golden question's retrieval, scored."""

    id: str
    retrieved: tuple[RetrievedChunk, ...]  # in rank order
    covered: int  # expected pages the chunks cover
    expected: int  # expected pages: recall's denominator, never 0

    @property
    def rank(self) -> int | None:
        """The 1-based rank of the first chunk on an expected page or an also-page."""
        return next((n for n, chunk in enumerate(self.retrieved, 1) if chunk.match), None)

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.rank is None else 1 / self.rank

    @property
    def recall(self) -> float:
        return self.covered / self.expected


@dataclass(frozen=True)
class RetrievalScores:
    """The means over the scored questions, which the floors are checked against."""

    hit_rate: float
    mrr: float
    recall: float


def score_item(item: GoldenItem, docs: Sequence[Document], corpus_root: Path) -> ItemScore:
    """Score one question's retrieved ``docs``, given in rank order."""
    # (source, page) pairs. A page of None stands for a whole file listed without pages.
    pages: set[tuple[str, str | None]] = set()
    also: set[tuple[str, str]] = set()
    for expected in item.expected_sources:
        if expected.pages:
            pages.update((expected.source, page) for page in expected.pages)
            also.update((expected.source, page) for page in expected.also_pages)
        else:
            pages.add((expected.source, None))

    retrieved = []
    covered: set[tuple[str, str | None]] = set()
    for ref in source_refs(list(docs), corpus_root):
        # Its own page, or its whole file when that file is listed without pages.
        unit = next((u for u in ((ref.source, ref.page), (ref.source, None)) if u in pages), None)
        match: Match | None
        if unit is not None:
            covered.add(unit)
            match = "page"
        elif ref.page is not None and (ref.source, ref.page) in also:
            match = "also"
        else:
            match = None
        retrieved.append(RetrievedChunk(source=ref.source, page=ref.page, match=match))
    return ItemScore(
        id=item.id, retrieved=tuple(retrieved), covered=len(covered), expected=len(pages)
    )


def aggregate(scores: Sequence[ItemScore]) -> RetrievalScores:
    """The means of hit, reciprocal rank and recall. ``scores`` must not be empty."""
    return RetrievalScores(
        hit_rate=fmean(score.hit for score in scores),
        mrr=fmean(score.reciprocal_rank for score in scores),
        recall=fmean(score.recall for score in scores),
    )


def evaluate_retrieval(config: RagConfig, items: Iterable[GoldenItem]) -> list[ItemScore]:
    """Retrieve for every answerable item and score it, in the golden set's order.

    Unanswerable items are skipped: they have no expected sources, and count only towards
    tier 2's decline rate. Raises :class:`~rag_qa.chain.NoIndexError` before any model is
    loaded when there is no index.
    """
    store = config.paths.vector_store
    if not store_exists(store):
        raise NoIndexError(
            f"no index at '{store}'. Run rag-ingest first, with the same "
            f"RAG_VECTOR_STORE_PATH when the index is a scratch one."
        )
    # The same embedder the index was built with: build_embedder is the one place that
    # choice is made. No LLM is built.
    retrieve = build_retrieve(config, build_embedder(config))
    return [
        score_item(item, retrieve.invoke(item.question), config.paths.data)
        for item in items
        if item.answerable
    ]
