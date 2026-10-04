"""Our own cross-encoder reranker (FR-4, DEC-16): DEC-5 step 2 of 3.

The retriever ranks chunks by embedding similarity. The question and each chunk are
embedded apart, which is fast but coarse. A cross-encoder reads the question and a chunk
together and scores how well the chunk answers it. That is slower per chunk, so it only
rescores the retriever's candidates (``pipeline.query.reranker_candidates``), and keeps the
best ``top_n`` for the prompt.

It calls ``sentence_transformers.CrossEncoder`` directly. The disabled entry it replaces
needed three classes from two maintenance-mode packages for that one model call:
``langchain_classic``'s ``ContextualCompressionRetriever`` and ``CrossEncoderReranker``,
and ``langchain_community``'s ``HuggingFaceCrossEncoder``.

Its settings are closed, and none of them reaches the model's loading options, so a config
cannot pass ``trust_remote_code`` through a reranker (the trust boundary in
:mod:`rag_qa.registry`).
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseDocumentCompressor):
    """Rescore retrieved chunks against the question, and keep the best ``top_n``.

    The model is loaded once, at construction: a download on first use (about 90 MB for
    ms-marco-MiniLM-L-6-v2), then from the HF cache, offline included. ``rerank_score`` is
    the model's own score: for the ms-marco models a logit, where higher is more relevant
    and there is no fixed range. ``min_score`` is in the same units.
    """

    # BaseDocumentCompressor keeps pydantic's default, which ignores unknown keys, so a
    # misspelled `topn: 3` in the config would be dropped silently (langchain-core 1.6.3).
    model_config = ConfigDict(extra="forbid")

    model_name: str
    top_n: int = Field(default=5, ge=1)
    device: str = "cpu"
    #: Drop chunks that score below this. Off by default: the Sources-relevance knob,
    #: tuned once tier 2 exists (S2-5).
    min_score: float | None = None

    _model: CrossEncoder = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        # The name positionally, the device by keyword: CrossEncoder's current signature.
        # It still accepts the older spellings (model_name=, a positional device), but says
        # so only to a logger, never as a warning the deprecation gate would see.
        self._model = CrossEncoder(self.model_name, device=self.device)
        if self._model.num_labels != 1:
            raise ValueError(
                f"{self.model_name!r} gives {self._model.num_labels} scores per pair; a "
                f"reranker needs a cross-encoder with one relevance score (num_labels=1)"
            )

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Callbacks | None = None
    ) -> list[Document]:
        """The best ``top_n`` of ``documents`` that score at least ``min_score``, best first.

        Returns copies with ``rerank_score`` in their metadata, and never modifies the
        documents given: FAISS hands back the very objects its docstore holds, so writing
        into them would change the loaded index.
        """
        if not documents:
            return []
        scores = self._model.predict(
            [(query, doc.page_content) for doc in documents], show_progress_bar=False
        )
        # Python floats, not numpy's: the score travels on to SourceRef and to JSON.
        # sorted() is stable with reverse=True too, so equal scores keep the retriever's order.
        ranked = sorted(
            zip(map(float, scores), documents, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
        kept = [pair for pair in ranked if self.min_score is None or pair[0] >= self.min_score]
        return [
            doc.model_copy(update={"metadata": {**doc.metadata, "rerank_score": score}})
            for score, doc in kept[: self.top_n]
        ]
