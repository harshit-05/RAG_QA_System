"""Query pipeline construction: the parts every front end answers with (ARCHITECTURE.md §2.4).

:func:`build_query_pipeline` returns a :class:`QueryPipeline`, the two halves of an
answer kept apart:

- ``retrieve``: a :class:`RetrievalQuery` in, the question and the documents it may search
  (S3-5), chunks out. Awaited, never streamed.
- ``prompt`` and ``llm``: :func:`rag_qa.answering.stream_answer` renders the prompt,
  then streams the model itself, never ``prompt | llm | StrOutputParser()``. That is
  what makes a cancel close the stream to the model every time (DEC-14).

:func:`build_rag_chain` composes the same parts into the Phase 0 chain, for the
``invoke`` contract of §0.4, relied on by the evaluation harness:

    chain.invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}

**Reading a Qdrant index (DEC-19).** A retrieval half binds to the generation it resolved,
never to the alias, so a switch mid-question cannot mix two generations. A generation is
deleted two switches later, so a reader that lives longer than that resolves the alias
again: :func:`build_retrieve`'s, which ``rag-query`` and ``rag-eval`` use, before every
question, and the API before every request (:mod:`rag_qa.api.state`). A search whose
generation has gone resolves it once more.

Imports come only from ``langchain_core`` plus our own modules (DEC-1 rule 1). None
of the legacy chain helpers: the ``RetrievalQA`` this replaces now lives only in the
maintenance-mode "classic" package, which application code never imports.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Any

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from rag_qa import resilience
from rag_qa.components import (
    FALLBACK_SERVICE,
    aclose_llm,
    build_embedder,
    build_fallback_llm,
    build_llm,
    build_reranker,
    build_sparse_encoder,
    build_store,
    configured_sparse,
    service_of,
)
from rag_qa.manifest import Manifest, embedder_identity, same_sparse
from rag_qa.resilience import CircuitBreaker
from rag_qa.schema import RagConfig
from rag_qa.sparse import Bm25Encoder
from rag_qa.vectorstore import (
    NO_INDEX,
    GenerationGoneError,
    QdrantStore,
    in_the_way,
    live_index,
    newest_complete,
    retriever,
)

#: Where README says how to point the alias at a named generation, which the "no index"
#: message cites when a complete generation has lost its alias (DEC-19).
POINT_THE_ALIAS = 'README, "Pointing the alias at a generation"'


class NoIndexError(Exception):
    """There is no index to retrieve from: ``rag-ingest`` has not built one yet."""


class IncompatibleIndexError(NoIndexError):
    """There is an index, but not one this config can use: re-run ``rag-ingest``.

    It was built by a different embedder, whose vectors cannot be compared with the
    config's (DEC-17), or its live generation cannot be used, or something in Qdrant takes
    the alias's name. A subclass of :class:`NoIndexError`, so every front end that refuses
    a missing index refuses this one the same way: ``rag-query`` and ``rag-eval`` exit 2,
    and the API counts it as no index (DEC-18).
    """


def check_index(config: RagConfig, *, store: QdrantStore | None = None) -> tuple[str, Manifest]:
    """The live generation and its manifest, once the manifest shows the config can use it.

    Reads the alias and one generation's metadata, and loads no model, so a front end
    refuses an unusable index before it builds the embedder, the reranker or the LLM.
    Raises :class:`NoIndexError` when there is no index, and :class:`IncompatibleIndexError`
    when there is one this config cannot use: something ``rag-ingest`` will not switch
    away from (:func:`rag_qa.vectorstore.in_the_way`), then the rule
    :func:`rag_qa.vectorstore.live_index` shares with ``rag-ingest``, then the embedder's
    identity. The dimension, the identity's backstop, needs the embedder, so
    :func:`open_retrieval` checks it. Raises the store's errors when Qdrant cannot be used.
    """
    store = store if store is not None else build_store(config)
    blocked = in_the_way(store)
    if blocked is not None:
        # Not "re-run rag-ingest": it refuses to touch whatever this is (S2-6's review).
        raise IncompatibleIndexError(
            f"{store.where} {blocked}, so there is no index to use. Set "
            f"RAG_QDRANT_COLLECTION to the alias rag-ingest publishes, or remove this from "
            f"Qdrant and run rag-ingest."
        )
    found = live_index(store)
    if found == NO_INDEX:
        raise NoIndexError(_no_index(store))
    if isinstance(found, str):
        raise IncompatibleIndexError(
            f"cannot use the index in {store.where}: {found}. Re-run rag-ingest to rebuild it."
        )
    generation, manifest = found
    ref = config.pipeline.ingestion.embedder
    if manifest.embedder.identity != embedder_identity(config.component(ref).spec()):
        then = f"{ref} as it was then" if manifest.embedder.ref == ref else manifest.embedder.ref
        raise IncompatibleIndexError(
            f"the index in {store.where} was built with a different embedder ({then}) than "
            f"the config's ({ref}), and their vectors cannot be compared. Re-run rag-ingest "
            f"--rebuild to rebuild it with {ref}."
        )
    if config.retriever(config.pipeline.query.retriever).is_hybrid:
        # A hybrid retriever searches the index's sparse vectors with queries this config's
        # encoder makes: they must be that encoder's (DEC-20). A dense one ignores them.
        wanted = configured_sparse(config)
        if not same_sparse(manifest.sparse, wanted):
            had = (
                "no sparse vectors"
                if manifest.sparse is None
                else f"sparse vectors from another encoder ({manifest.sparse.ref} as it was then)"
            )
            raise IncompatibleIndexError(
                f"the index in {store.where} has {had}, but the config's retriever "
                f"{config.pipeline.query.retriever} is hybrid and searches those of "
                f"{config.pipeline.ingestion.sparse_encoder}. Re-run rag-ingest to rebuild it."
            )
    return generation, manifest


def _no_index(store: QdrantStore) -> str:
    """Why there is no index, and what to do: ingest, or point the alias back at the
    newest complete generation when one survived its live collection (DEC-19)."""
    complete = newest_complete(store)
    if complete is not None:
        return (
            f"no index is published as {store.where}, but its generation {complete} is "
            f"complete: the live collection may have been deleted, which deletes the alias "
            f"with it. Point the alias at {complete} to serve it again ({POINT_THE_ALIAS}), "
            f"or run rag-ingest to publish a new one."
        )
    return (
        f"no index in {store.where}. Run rag-ingest first to build it from the corpus "
        f"(with the same RAG_QDRANT_COLLECTION, for a scratch index)."
    )


def live_generation(store: QdrantStore) -> str | None:
    """The generation the alias names now, or ``None`` when there is no alias.

    One small request (``GET /aliases``), cheap enough for the API to make before every
    request, to notice that an ingest run outside it switched a new generation in.
    """
    return store.alias_target()


def _check_dimension(embeddings: Embeddings, manifest: Manifest, store: QdrantStore) -> None:
    """The identity's backstop (DEC-17): the configured embedder must make vectors of the
    size the index holds. One probe embedding."""
    size = len(embeddings.embed_query("dimension check"))
    if size != manifest.embedder.dimension:
        raise IncompatibleIndexError(
            f"the index in {store.where} holds {manifest.embedder.dimension}-dimension "
            f"vectors, but the configured embedder makes {size}-dimension ones. Re-run "
            f"rag-ingest --rebuild."
        )


@dataclass(frozen=True)
class RetrievalQuery:
    """What the retrieval half is asked: a question, and the corpus documents it may search
    (S3-5, FR-3).

    ``sources`` are corpus-relative paths, as the manifest and ``GET /v1/documents`` name
    them; empty means the whole corpus. The front ends check them against the served
    manifest before asking, and tier 1 and tier 2 pass none.
    """

    question: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPipeline:
    """The parts of an answer, built once and shared by every question.

    ``retrieve`` is ``None`` only in the API while it has no usable index (S2-7);
    :func:`rag_qa.answering.stream_answer` then raises :class:`NoIndexError`. The API
    swaps the retrieval half with ``dataclasses.replace``, so an answer in flight keeps the
    one it started with; the breakers are carried over with the rest.

    **The models** (DEC-21, S3-6): ``llm`` answers, and ``fallback_llm``, the hosted
    fallback, only when ``llm`` cannot. Each has a breaker shared by every answer, and a
    reference, which :class:`~rag_qa.answering.Done` names for the one that answered.
    """

    retrieve: Runnable[RetrievalQuery, list[Document]] | None
    prompt: ChatPromptTemplate  # rendered by stream_answer, which then streams llm itself
    llm: BaseChatModel  # streamed directly; also kept so aclose() can close its clients
    corpus_root: Path  # sources are reported relative to it
    llm_ref: str = "llm"  # its config reference, as Done.model names it
    fallback_llm: BaseChatModel | None = None
    fallback_ref: str = "fallback"
    llm_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("the language model")
    )
    fallback_breaker: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker(FALLBACK_SERVICE)
    )

    async def aclose(self) -> None:
        """Close both models' HTTP clients. Call it on the loop that used them."""
        try:
            await aclose_llm(self.llm)
        finally:
            if self.fallback_llm is not None:
                await aclose_llm(self.fallback_llm)


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


@dataclass(frozen=True)
class Retrieval:
    """A retrieval half, with the index generation it opened and that generation's
    manifest: what the API serves, lists and reports (DEC-18)."""

    retrieve: Runnable[RetrievalQuery, list[Document]]
    generation: str  # the generation it is bound to, as live_generation() names it
    manifest: Manifest
    reranker: BaseDocumentCompressor | None = None  # the one it reranks with, for a reopen


def _checked(config: RagConfig, store: QdrantStore, embeddings: Embeddings) -> tuple[str, Manifest]:
    """The live generation and its manifest, once the config can use them, the dimension
    included: everything a refusal needs, before any reranker loads."""
    generation, manifest = check_index(config, store=store)
    _check_dimension(embeddings, manifest, store)
    return generation, manifest


def _bound(
    config: RagConfig,
    store: QdrantStore,
    generation: str,
    embeddings: Embeddings,
    reranker: BaseDocumentCompressor | None,
    sparse: Bm25Encoder | None,
) -> Runnable[RetrievalQuery, list[Document]]:
    """A retrieval half bound to ``generation``: the retriever, then the reranker if any.

    The question goes to both; the sources only to the retriever, which searches nothing
    else (:func:`rag_qa.vectorstore.source_filter`). One function rather than a sequence,
    because the reranker needs the question as well as the chunks. Awaited, it runs in an
    executor thread, retrieval included (DEC-14).
    """
    query = config.pipeline.query
    spec = config.retriever(query.retriever)
    # With a reranker, the retriever fetches its candidates in place of its own k (DEC-16);
    # the schema requires reranker_candidates whenever a reranker is configured.
    k = spec.k()
    if reranker is not None and query.reranker_candidates is not None:
        k = query.reranker_candidates
    found = retriever(
        store, generation, embeddings, spec.kwargs(), sparse=sparse if spec.is_hybrid else None, k=k
    )

    def retrieve(asked: RetrievalQuery) -> list[Document]:
        docs = found.invoke(asked.question, sources=asked.sources)
        if reranker is None:
            return docs
        return list(reranker.compress_documents(docs, asked.question))

    return RunnableLambda(retrieve)


def open_retrieval(
    config: RagConfig,
    embeddings: Embeddings,
    *,
    reranker: BaseDocumentCompressor | None = None,
    store: QdrantStore | None = None,
) -> Retrieval:
    """The retrieval half: the configured retriever over the index, then the reranker if any.

    Needs no LLM, so tier-1 evaluation runs it without Ollama, and the API can swap it
    after an ingest. ``embeddings`` must be :func:`rag_qa.components.build_embedder`'s,
    the model the index was built with.

    With a reranker, the retriever fetches ``pipeline.query.reranker_candidates`` chunks in
    place of its own ``k``, and the reranker cuts them to its ``top_n`` (DEC-16). One switch
    sets both, so the candidates never reach the prompt unreranked: 20 chunks would overflow
    the model's context window silently (caveat 7).

    ``reranker`` is the config's own, built once by a caller that opens the index more
    than once: the API reopens it after every ingest, and a cross-encoder takes seconds and
    about 90 MB to load (S2-4). Left out, it is built here from the config.

    Raises :class:`NoIndexError` or :class:`IncompatibleIndexError` before the reranker
    loads (:func:`check_index`, then the dimension). ``embeddings`` is built by then, so
    callers that build it run :func:`check_index` first, as :func:`build_query_pipeline`
    and tier 1 do.

    **Bound to the generation it checked**, never to the alias, so a switch in between
    cannot pair one generation's manifest with another's points (DEC-19). When a search
    finds that generation gone, deleted two switches later, the half resolves the alias
    once more, checks the generation it names as above, and searches that; it never
    retries the search it lost. ``generation`` and ``manifest`` stay what was opened.
    """
    store = store if store is not None else build_store(config)
    generation, manifest = _checked(config, store, embeddings)
    if reranker is None:
        reranker = build_reranker(config)
    sparse = build_sparse_encoder(config)  # no model: a few settings, cheap on every open
    bound = _bound(config, store, generation, embeddings, reranker, sparse)

    def retrieve(asked: RetrievalQuery) -> list[Document]:
        # One budget for every attempt this retrieval makes, retries included: the API
        # holds its retrieval lock throughout (DEC-21, design review 2).
        with resilience.deadline(store.deadline_s):
            try:
                return bound.invoke(asked)
            except GenerationGoneError:
                again, _ = _checked(config, store, embeddings)
                return _bound(config, store, again, embeddings, reranker, sparse).invoke(asked)

    return Retrieval(
        retrieve=RunnableLambda(retrieve),
        generation=generation,
        manifest=manifest,
        reranker=reranker,
    )


def build_retrieve(
    config: RagConfig,
    embeddings: Embeddings,
    *,
    reranker: BaseDocumentCompressor | None = None,
    store: QdrantStore | None = None,
    on_new_generation: Callable[[Manifest], None] | None = None,
) -> Runnable[RetrievalQuery, list[Document]]:
    """A retrieval half that resolves the alias again before every question (DEC-19).

    For a reader that may outlive two ingests: ``rag-query``'s session and ``rag-eval``.
    Each question costs one ``GET /aliases``. When the alias has moved, the new generation
    is opened as :func:`open_retrieval` opens one: its manifest read once, the embedder
    and the dimension checked, the reranker reused; and ``on_new_generation`` hears its
    manifest first, and may refuse it by raising. Otherwise the question searches the
    generation already open. Raises what :func:`open_retrieval` raises, at the start and
    whenever the alias has moved.
    """
    store = store if store is not None else build_store(config)
    opened = open_retrieval(config, embeddings, reranker=reranker, store=store)
    reranker = opened.reranker  # built once, by the first open, and reused on every reopen

    def retrieve(asked: RetrievalQuery) -> list[Document]:
        nonlocal opened
        with resilience.deadline(store.deadline_s):  # the alias read counts against it too
            if store.alias_target() != opened.generation:
                moved = open_retrieval(config, embeddings, reranker=reranker, store=store)
                if on_new_generation is not None:
                    on_new_generation(moved.manifest)
                opened = moved
            return opened.retrieve.invoke(asked)

    return RunnableLambda(retrieve)


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


def build_query_pipeline(
    config: RagConfig,
    *,
    store: QdrantStore | None = None,
    on_new_generation: Callable[[Manifest], None] | None = None,
    fallback: bool = False,
) -> QueryPipeline:
    """Build both halves of an answer from a loaded config (the CLI and evaluation).

    A missing index raises :class:`NoIndexError`, and one the config cannot use
    :class:`IncompatibleIndexError`. The manifest's refusals come before any model is
    built. The dimension needs the embedder, but is still refused before the reranker and
    the LLM: retrieval is built first, so a refusal never waits on Ollama. The retrieval
    half resolves the alias before every question (:func:`build_retrieve`, which
    ``on_new_generation`` is passed to). The API, which must start without an index,
    composes its pipeline from :func:`open_retrieval` and :func:`assemble` (S2-7).

    With ``fallback``, the hosted fallback is built too when ``pipeline.query.fallback_llm``
    names one (DEC-21), and the model without asking Ollama anything, since Ollama down is
    the fallback's case. ``rag-query`` asks for it; tier 2 never does, so it measures the
    model it names, and nothing it sends leaves the host.

    Cannot mutate ``config``: it is frozen, and every component dict used here is a
    fresh copy from ``spec()`` / ``kwargs()``. Deliberately silent (no progress
    prints); front ends print their own progress.
    """
    store = store if store is not None else build_store(config)
    check_index(config, store=store)  # metadata only, so nothing has loaded when it refuses
    # The same embedder ingestion used — build_embedder is the one place that choice is
    # made (see rag_qa.components).
    retrieve = build_retrieve(
        config, build_embedder(config), store=store, on_new_generation=on_new_generation
    )
    spare = build_fallback_llm(config) if fallback else None
    llm = build_llm(config) if spare is None else build_llm(config, validate=False)
    return assemble(config, retrieve, llm, spare)


def assemble(
    config: RagConfig,
    retrieve: Runnable[RetrievalQuery, list[Document]] | None,
    llm: BaseChatModel,
    fallback_llm: BaseChatModel | None,
) -> QueryPipeline:
    """A pipeline over ``retrieve`` and the models built for ``config``: its prompt and its
    corpus, each model's reference, and a breaker for each, named after its service
    (DEC-21). The one composition :func:`build_query_pipeline` and the API share."""
    query = config.pipeline.query
    return QueryPipeline(
        retrieve=retrieve,
        prompt=build_prompt(config),
        llm=llm,
        corpus_root=config.paths.data,
        llm_ref=query.llm,
        fallback_llm=fallback_llm,
        fallback_ref=query.fallback_llm or "fallback",
        llm_breaker=CircuitBreaker(service_of(config)),
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
    store = build_store(config)
    check_index(config, store=store)  # before the embedder: build_retrieve's comes after it
    retrieve = build_retrieve(config, build_embedder(config), store=store)
    llm = build_llm(config)
    to_prompt_inputs: RunnableLambda[dict[str, Any], dict[str, str]] = RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )
    asked: RunnableLambda[str, RetrievalQuery] = RunnableLambda(
        lambda question: RetrievalQuery(question)
    )
    return RunnablePassthrough.assign(
        context=itemgetter("question") | asked | retrieve
    ) | RunnablePassthrough.assign(
        answer=to_prompt_inputs | build_prompt(config) | llm | StrOutputParser()
    )
