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
from rag_qa.manifest import Manifest, embedder_identity
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import NO_INDEX, in_the_way, live_index, open_store


class NoIndexError(Exception):
    """There is no index to retrieve from: ``rag-ingest`` has not built one yet."""


class IncompatibleIndexError(NoIndexError):
    """There is an index, but not one this config can use: re-run ``rag-ingest``.

    It was built by v0.2, with no manifest to say which embedder made it, or by a different
    embedder, whose vectors cannot be compared with the config's (DEC-17). A subclass of
    :class:`NoIndexError`, so every front end that refuses a missing index refuses this one
    the same way: ``rag-query`` and ``rag-eval`` exit 2, and the API counts it as no index
    (DEC-18).
    """


def check_index(config: RagConfig) -> tuple[Path, Manifest]:
    """The live index generation and its manifest, once the manifest shows the config can
    use it (DEC-17).

    Reads small files only and loads no model, so a front end refuses an unusable index
    before it builds the embedder, the reranker or the LLM. Raises :class:`NoIndexError`
    when there is no index, and :class:`IncompatibleIndexError` when there is one this
    config cannot use: something ``rag-ingest`` will not write over
    (:func:`rag_qa.vectorstore.in_the_way`), then the rule
    :func:`rag_qa.vectorstore.live_index` shares with ``rag-ingest``, then the embedder's
    identity. The dimension, the identity's backstop,
    needs the embedder, so :func:`build_retrieve` checks it.
    """
    store = config.paths.vector_store
    blocked = in_the_way(store)
    if blocked is not None:
        # Not "re-run rag-ingest": it refuses to write over whatever this is (second review).
        raise IncompatibleIndexError(
            f"'{store}' {blocked}, so there is no index to use. Point paths.vector_store "
            f"or RAG_VECTOR_STORE_PATH at the index rag-ingest wrote, or move this aside "
            f"and run rag-ingest."
        )
    found = live_index(store)
    if found == NO_INDEX:
        raise NoIndexError(
            f"no index at '{store}'. Run rag-ingest first to build it from the corpus "
            f"(with the same RAG_VECTOR_STORE_PATH, for a scratch index)."
        )
    if isinstance(found, str):
        raise IncompatibleIndexError(
            f"cannot use the index at '{store}': {found}. Re-run rag-ingest to rebuild it."
        )
    generation, manifest = found
    ref = config.pipeline.ingestion.embedder
    if manifest.embedder.identity != embedder_identity(config.component(ref).spec()):
        then = f"{ref} as it was then" if manifest.embedder.ref == ref else manifest.embedder.ref
        raise IncompatibleIndexError(
            f"the index at '{store}' was built with a different embedder ({then}) than the "
            f"config's ({ref}), and their vectors cannot be compared. Re-run rag-ingest "
            f"--rebuild to rebuild it with {ref}."
        )
    return generation, manifest


def _check_dimension(embeddings: Embeddings, manifest: Manifest, store: Path) -> None:
    """The identity's backstop (DEC-17): the configured embedder must make vectors of the
    size the index holds. One probe embedding."""
    size = len(embeddings.embed_query("dimension check"))
    if size != manifest.embedder.dimension:
        raise IncompatibleIndexError(
            f"the index at '{store}' holds {manifest.embedder.dimension}-dimension vectors, "
            f"but the configured embedder makes {size}-dimension ones. Re-run rag-ingest "
            f"--rebuild."
        )


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

    Raises :class:`NoIndexError` or :class:`IncompatibleIndexError` before the reranker
    loads (:func:`check_index`, then the dimension). ``embeddings`` is built by then, so
    callers that build it run :func:`check_index` first, as :func:`build_query_pipeline`
    and tier 1 do. It opens the generation it checked, not the symlink, so a flip in
    between cannot pair one generation's manifest with another's index.
    """
    store = config.paths.vector_store
    generation, manifest = check_index(config)
    _check_dimension(embeddings, manifest, store)
    try:
        index = open_store(embeddings, generation)
    # Deliberately broad, and re-raised: whatever stops a generation opening (a corrupt
    # pickle, FAISS's own errors) has one remedy, rebuilding it, and deserves a refusal
    # rather than a traceback. rag-ingest rebuilds such an index on its own next run.
    except Exception as e:
        raise IncompatibleIndexError(
            f"cannot open the index at '{store}' ({type(e).__name__}: {e}). Re-run "
            f"rag-ingest to rebuild it."
        ) from e
    query = config.pipeline.query
    reranker = build_reranker(config)
    kwargs = config.retriever(query.retriever).kwargs()  # a fresh copy, so safe to change
    if reranker is not None:
        kwargs["search_kwargs"]["k"] = query.reranker_candidates
    retriever: Runnable[str, list[Document]] = index.as_retriever(**kwargs)
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
    :class:`NoIndexError`, and one the config cannot use :class:`IncompatibleIndexError`.
    The manifest's refusals come before any model is built. The dimension and an index
    that will not open need the embedder, but are still refused before the reranker and
    the LLM: retrieval is built first, so a refusal never waits on Ollama. Without
    ``require_index`` (the API), either leaves ``retrieve`` as ``None`` and the LLM half
    is still built; with a usable index, both halves are built either way.

    Cannot mutate ``config``: it is frozen, and every component dict used here is a
    fresh copy from ``spec()`` / ``kwargs()``. Deliberately silent (no progress
    prints); front ends print their own progress.
    """
    retrieve = None
    try:
        check_index(config)  # files only, so nothing has loaded when it refuses
        # The same embedder ingestion used — build_embedder is the one place that
        # choice is made (see rag_qa.components).
        retrieve = build_retrieve(config, build_embedder(config))
    except NoIndexError:  # IncompatibleIndexError too: the API counts it as no index
        if require_index:
            raise
    llm = build_llm(config)
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
    then ``answer`` token by token. Refuses an unusable index as
    :func:`build_query_pipeline` does, before the embedder and the LLM.
    """
    check_index(config)  # before the embedder: build_retrieve's own check comes after it
    retrieve = build_retrieve(config, build_embedder(config))
    llm = build_llm(config)
    to_prompt_inputs: RunnableLambda[dict[str, Any], dict[str, str]] = RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )
    return (
        RunnablePassthrough.assign(context=itemgetter("question") | retrieve)
        | RunnablePassthrough.assign(
            answer=to_prompt_inputs | build_prompt(config) | llm | StrOutputParser()
        )
    )
