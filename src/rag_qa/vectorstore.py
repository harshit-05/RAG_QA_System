"""Vector store persistence: the only place the store implementation appears.

This is the Phase 3 swap seam (ARCHITECTURE.md §0.5, DEC-3). Callers only ever use
``.as_retriever()`` and ``.add_documents()`` on what these two functions return, so
migrating FAISS to Qdrant/pgvector means rewriting these two bodies and nothing else.

Safety invariant (ISS-16): ``allow_dangerous_deserialization=True`` unpickles the
index, which is only safe because the index directory is produced exclusively by
``rag-ingest`` on this host. Never point this at an index from an untrusted source.

Both functions take the store's location, not the whole config: the smallest contract
that works, so a Phase 3 backend needing a host and collection instead of a directory
changes the argument here and at two call sites, not a config shape everyone reads.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS


def create_store(chunks, embeddings, path: Path) -> FAISS:
    """Build a fresh index from chunks and persist it to ``path``."""
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(path))
    return db


def open_store(embeddings, path: Path) -> FAISS:
    """Open the persisted index a previous ingestion run wrote to ``path``."""
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
