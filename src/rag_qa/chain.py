"""Query pipeline construction: the parts every front end answers with (ARCHITECTURE.md §2.4).

:func:`build_query_pipeline` returns a :class:`QueryPipeline`, the two halves of an
answer kept apart:

- ``retrieve``: question in, chunks out. Awaited, never streamed.
- ``prompt`` and ``llm``: :func:`rag_qa.answering.stream_answer` renders the prompt,
  then streams the model itself, never ``prompt | llm | StrOutputParser()``. That is
  what makes a cancel close the stream to the model every time (DEC-14).

:func:`build_rag_chain` composes the same parts into the Phase 0 chain, for the
``invoke`` contract of §0.4, relied on by the evaluation harness:

    chain.invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}

Imports come only from ``langchain_core`` plus our own modules (DEC-1 rule 1). None
of the legacy chain helpers: the ``RetrievalQA`` this replaces now lives only in the
maintenance-mode "classic" package, which application code never imports.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from typing import Any

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from rag_qa.components import aclose_llm, build_embedder, build_llm, build_reranker
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import open_store, store_exists


class NoIndexError(Exception):
    """There is no index to retrieve from: ``rag-ingest`` has not built one yet."""


@dataclass(frozen=True)
class QueryPipeline:
    """The parts of an answer, built once and shared by every question.

    ``retrieve`` is ``None`` only when the pipeline was built with
    ``require_index=False`` and there was no index (the API before its first ingest,
    S2-7); :func:`rag_qa.answering.stream_answer` then raises :class:`NoIndexError`.
    """

    retrieve: Runnable[str, list[Document]] | None
    prompt: ChatPromptTemplate  # rendered by stream_answer, which then streams llm itself
    llm: BaseChatModel  # streamed directly; also kept so aclose() can close its clients
    corpus_root: Path  # sources are reported relative to it

    async def aclose(self) -> None:
        """Close the model's HTTP clients. Call it on the loop that used them."""
        await aclose_llm(self.llm)


def citation(doc: Document) -> str:
    """Human-readable source for one chunk: ``file.pdf, p. 3``.

    Omits the page for formats that have none (docx, txt); see :func:`page_label`.
    """
    name = Path(doc.metadata.get("source", "unknown source")).name
    page = page_label(doc)
    return f"{name}, p. {page}" if page is not None else name


def page_label(doc: Document) -> str | None:
    """The page a chunk came from, as printed: the loader's ``page_label`` when present,
    else the 0-indexed ``page`` plus one, else ``None``."""
    metadata = doc.metadata
    page = metadata.get("page_label")
    if page is None and "page" in metadata:
        page = metadata["page"] + 1
    return None if page is None else str(page)


def format_docs(docs: Sequence[Document]) -> str:
    """Render retrieved chunks for the prompt, numbered so the model can cite them.

    The CLI prints sources with the same ``[n]`` numbering, so a citation in the
    answer maps directly to a line in the source list.
    """
    return "\n\n".join(
        f"[{i}] ({citation(doc)})\n{doc.page_content}" for i, doc in enumerate(docs, 1)
    )


def build_retrieve(config: RagConfig, embeddings: Embeddings) -> Runnable[str, list[Document]]:
    """The retrieval half: the configured retriever over the index, then the reranker if any.

    Needs no LLM, so tier-1 evaluation runs it without Ollama, and the API can swap it
    after an ingest. ``embeddings`` must be :func:`rag_qa.components.build_embedder`'s,
    the model the index was built with.

    With a reranker, the retriever fetches ``pipeline.query.reranker_candidates`` chunks in
    place of its own ``k``, and the reranker cuts them to its ``top_n`` (DEC-16). One switch
    sets both, so the candidates never reach the prompt unreranked: 20 chunks would overflow
    the model's context window silently (caveat 7).
    """
    query = config.pipeline.query
    reranker = build_reranker(config)
    kwargs = config.retriever(query.retriever).kwargs()  # a fresh copy, so safe to change
    if reranker is not None:
        kwargs["search_kwargs"]["k"] = query.reranker_candidates
    retriever: Runnable[str, list[Document]] = open_store(
        embeddings, config.paths.vector_store
    ).as_retriever(**kwargs)
    return retriever if reranker is None else _rerank_after(retriever, reranker)


def _rerank_after(
    retriever: Runnable[str, list[Document]], reranker: BaseDocumentCompressor
) -> Runnable[str, list[Document]]:
    """``retriever``, then ``reranker`` over the chunks it found, as one Runnable.

    One function rather than a sequence, because the reranker needs the question as well
    as the chunks. Awaited, it runs in an executor thread, retrieval included (DEC-14).
    """

    def retrieve_and_rerank(question: str) -> list[Document]:
        return list(reranker.compress_documents(retriever.invoke(question), question))

    return RunnableLambda(retrieve_and_rerank)


def build_prompt(config: RagConfig) -> ChatPromptTemplate:
    """The chat prompt: ``{"context": str, "question": str}`` in, messages out.

    Instructions and retrieved content stay in different chat roles (OWASP LLM01).
    The one prompt both :func:`build_query_pipeline` and :func:`build_rag_chain` use.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", config.pipeline.query.prompt.system),
            ("human", config.pipeline.query.prompt.human),
        ]
    )


def build_query_pipeline(config: RagConfig, *, require_index: bool = True) -> QueryPipeline:
    """Build both halves of an answer from a loaded config.

    With ``require_index`` (the CLI and evaluation), a missing index raises
    :class:`NoIndexError` before any model is built. Without it (the API), a missing
    index leaves ``retrieve`` as ``None`` and the LLM half is still built; with an
    index, both halves are built either way.

    Cannot mutate ``config``: it is frozen, and every component dict used here is a
    fresh copy from ``spec()`` / ``kwargs()``. Deliberately silent (no progress
    prints); front ends print their own progress.
    """
    has_index = store_exists(config.paths.vector_store)
    if require_index and not has_index:
        raise NoIndexError(
            f"no index at '{config.paths.vector_store}'. Run rag-ingest first to build it "
            f"from the corpus."
        )
    llm = build_llm(config)
    # The same embedder ingestion used — build_embedder is the one place that choice
    # is made (see rag_qa.components).
    retrieve = build_retrieve(config, build_embedder(config)) if has_index else None
    return QueryPipeline(
        retrieve=retrieve,
        prompt=build_prompt(config),
        llm=llm,
        corpus_root=config.paths.data,
    )


def build_rag_chain(config: RagConfig) -> Runnable[dict[str, str], dict[str, Any]]:
    """The §0.4 chain, composed from the same parts as :func:`build_query_pipeline`.

    **For** ``invoke`` **only, never for cancellable streaming** (DEC-14). Its
    ``RunnablePassthrough.assign`` steps run under ``RunnableParallel``, which never
    cancels its step tasks: a cancelled consumer returns while the generation runs on.
    Even its inner ``prompt | llm | StrOutputParser()`` would not do: a sequence runs each
    chunk in its own task, and a cancel at one of those boundaries leaves the model's
    stream open. Front ends stream :func:`rag_qa.answering.stream_answer` instead.

    Streaming it still yields ``question``, then the whole ``context`` in one chunk,
    then ``answer`` token by token.
    """
    llm = build_llm(config)
    retrieve = build_retrieve(config, build_embedder(config))
    to_prompt_inputs: RunnableLambda[dict[str, Any], dict[str, str]] = RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )
    return (
        RunnablePassthrough.assign(context=itemgetter("question") | retrieve)
        | RunnablePassthrough.assign(
            answer=to_prompt_inputs | build_prompt(config) | llm | StrOutputParser()
        )
    )
