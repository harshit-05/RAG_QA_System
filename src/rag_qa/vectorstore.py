"""Vector store persistence: the only place the store implementation appears.

This is the Phase 3 swap seam (ARCHITECTURE.md §0.5, DEC-3). Callers only ever use
``.as_retriever()`` and ``.add_documents()`` on what these two functions return, so
migrating FAISS to Qdrant/pgvector means rewriting these two bodies and nothing else.

Safety invariant (ISS-16): ``allow_dangerous_deserialization=True`` unpickles the
index, which is only safe because the index directory is produced exclusively by
``rag-ingest`` on this host. Never point this at an index from an untrusted source.
"""

from langchain_community.vectorstores import FAISS


def create_store(chunks, embeddings, config):
    """Build a fresh index from chunks and persist it to disk."""
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config["paths"]["vector_store"])
    return db


def open_store(embeddings, config):
    """Open the persisted index built by a previous ingestion run."""
    return FAISS.load_local(
        config["paths"]["vector_store"],
        embeddings,
        allow_dangerous_deserialization=True,
    )
