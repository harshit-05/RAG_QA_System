"""Building the pipeline's components from a validated config.

The shared assembly layer (ARCHITECTURE.md §1.2): :mod:`rag_qa.chain` and
:mod:`rag_qa.ingest` both build from here instead of each resolving references
themselves. Dependency direction: ``components → {registry, schema}``; it imports
neither ``chain`` nor ``ingest``, so ADR-009's rule (``ingest`` never imports
``chain``) is untouched.

The invariant that motivates the module is :func:`build_embedder`. An index and the
queries against it must use the same embedding model, or retrieval silently compares
incomparable vectors — wrong answers, not a crash. Before S1-3 that held because two
modules happened to contain the same line; now it holds because there is one
function, and it always reads ``pipeline.ingestion.embedder``.
"""

import inspect
from pathlib import Path
from typing import Any

from langchain_core.documents import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from rag_qa.registry import build_object
from rag_qa.schema import RagConfig


def build_embedder(config: RagConfig) -> Embeddings:
    """The embedder for both ingestion and querying: always ``pipeline.ingestion.embedder``."""
    return build_object(config.component(config.pipeline.ingestion.embedder).spec())


def build_splitter(config: RagConfig) -> Any:
    """The ingestion text splitter (anything with ``split_documents``)."""
    return build_object(config.component(config.pipeline.ingestion.splitter).spec())


def build_llm(config: RagConfig) -> BaseChatModel:
    """The query-time chat model."""
    return build_object(config.component(config.pipeline.query.llm).spec())


def build_reranker(config: RagConfig) -> BaseDocumentCompressor | None:
    """The query-time reranker, or ``None`` when ``pipeline.query.reranker`` is unset.

    Building one loads its model (DEC-16). CI's eval-retrieval job calls this to warm its
    HF cache, so the cache holds exactly the files the offline eval step will open.
    """
    ref = config.pipeline.query.reranker
    return None if ref is None else build_object(config.component(ref).spec())


async def aclose_llm(llm: BaseChatModel) -> None:
    """Close the chat model's HTTP clients; a no-op for a model without them.

    ``ChatOllama`` builds an async and a sync ``ollama`` client at construction (the
    sync one already holds a connection when ``validate_model_on_init`` is set) and has
    no public close, so this reaches the private ``_async_client`` / ``_client``
    (ARCHITECTURE.md §2.1, DEC-14). Kept narrow: only those two names, and only their
    ``close``, awaited when it is a coroutine (``AsyncClient.close``) and called plainly
    otherwise (``Client.close``). Run it on the loop that used the async client.
    """
    for name in ("_async_client", "_client"):
        close = getattr(getattr(llm, name, None), "close", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result


def build_loader(config: RagConfig, path: Path) -> Any | None:
    """The loader for one file, or ``None`` when no loader is mapped to its suffix.

    Matching is on the lowercased suffix, so ``REPORT.PDF`` uses the ``.pdf`` loader.
    This is the only way a loader is built — through ``build_object`` like every
    other component, so the ``_target_`` allowlist (S1-2) covers it — replacing the
    old direct ``import_from_string`` call in ``ingest.py``.
    """
    ref = config.pipeline.ingestion.loaders.get(path.suffix.lower())
    if ref is None:
        return None
    return build_object({**config.component(ref).spec(), "file_path": str(path)})
