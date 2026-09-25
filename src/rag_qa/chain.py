"""Query chain construction: the LCEL composition fixed by ARCHITECTURE.md §0.4.

Contract, relied on by the CLI now and by the Phase 2 API and evaluation harness:

    chain.invoke({"question": q}) -> {"question": str, "context": list[Document], "answer": str}

Streaming (``chain.stream`` / ``chain.astream``) yields ``question``, then the whole
``context`` in one chunk, then ``answer`` token by token. That order is what lets a
streaming client show citations before the first answer token arrives.

Imports come only from ``langchain_core`` plus our own modules (DEC-1 rule 1). None
of the legacy chain helpers: the ``RetrievalQA`` this replaces now lives only in the
maintenance-mode "classic" package, which application code never imports.
"""

from operator import itemgetter
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag_qa.registry import build_object
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import open_store


def citation(doc):
    """Human-readable source for one chunk: ``file.pdf, p. 3``.

    Uses the loader's ``page_label`` (the printed page number) when present, falls
    back to the 0-indexed ``page`` plus one, and omits the page for formats that
    have none (docx, txt).
    """
    metadata = doc.metadata
    name = Path(metadata.get("source", "unknown source")).name
    page = metadata.get("page_label")
    if page is None and "page" in metadata:
        page = metadata["page"] + 1
    return f"{name}, p. {page}" if page is not None else name


def format_docs(docs):
    """Render retrieved chunks for the prompt, numbered so the model can cite them.

    The CLI prints sources with the same ``[n]`` numbering, so a citation in the
    answer maps directly to a line in the source list.
    """
    return "\n\n".join(
        f"[{i}] ({citation(doc)})\n{doc.page_content}" for i, doc in enumerate(docs, 1)
    )


def build_rag_chain(config: RagConfig):
    """Build the RAG chain from a loaded config (see :func:`rag_qa.config.load_config`).

    Cannot mutate ``config``: it is frozen, and every component dict used here is a
    fresh copy from ``spec()`` / ``kwargs()``. (The old implementation wrote a live
    retriever object into the config dict on the reranker path, so a reused config
    stopped being inert.) Deliberately silent (no progress prints) because the Phase 2
    API calls it too; front ends print their own progress.
    """
    query = config.pipeline.query

    llm = build_object(config.component(query.llm).spec())
    # The query-time embedder is always the ingestion embedder: an index and the
    # queries against it must share an embedding model.
    embeddings = build_object(config.component(config.pipeline.ingestion.embedder).spec())

    retriever = open_store(embeddings, config.paths.vector_store).as_retriever(
        **config.retriever(query.retriever).kwargs()
    )
    if query.reranker is not None:  # disabled until Phase 2 (FR-4)
        reranker_spec = {**config.component(query.reranker).spec(), "base_retriever": retriever}
        retriever = build_object(reranker_spec)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query.prompt.system),
            ("human", query.prompt.human),
        ]
    )
    to_prompt_inputs = RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )

    return (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever)
        | RunnablePassthrough.assign(answer=to_prompt_inputs | prompt | llm | StrOutputParser())
    )
