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

:func:`preflight` is the start-up check of the language model's server that ``rag-serve``
and ``rag-query`` share (S2-7, S2-8). It lives here, not in the ``api`` package, because
the CLI never imports that package (ARCHITECTURE.md §2.2). :func:`build_store` is the
vector store's one build path, and every command checks the store before anything else
(DEC-19).
"""

import asyncio
import inspect
from pathlib import Path
from typing import Any

from langchain_core.documents import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from rag_qa.config import ConfigError
from rag_qa.registry import build_object
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_QDRANT_PREFIX, describe_settings_error
from rag_qa.vectorstore import QdrantStore

#: What to try when an Ollama model cannot be reached, shown under the error.
OLLAMA_HINT = "Is Ollama running? Try: systemctl status ollama"


def build_embedder(config: RagConfig) -> Embeddings:
    """The embedder for both ingestion and querying: always ``pipeline.ingestion.embedder``."""
    return build_object(config.component(config.pipeline.ingestion.embedder).spec())


def build_store(config: RagConfig) -> QdrantStore:
    """The vector store both halves share: ``pipeline.vector_store`` (DEC-19).

    Sends no request; :meth:`~rag_qa.vectorstore.QdrantStore.check` does. Raises
    :class:`~rag_qa.config.ConfigError` when the entry cannot make a store: its kwargs, a
    ``RAG_QDRANT_*`` setting, or a ``_target_`` that is not one.
    """
    ref = config.pipeline.vector_store
    try:
        store = build_object(config.component(ref).spec())
    except ValidationError as e:  # the RAG_QDRANT_* settings, read by the store itself
        raise ConfigError(
            f"Invalid environment:\n{describe_settings_error(e, ENV_QDRANT_PREFIX)}"
        ) from None
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Cannot build the vector store {ref}: {e}") from e
    if not isinstance(store, QdrantStore):
        raise ConfigError(
            f"{ref} builds a {type(store).__name__}, not a store: its _target_ must be "
            f"rag_qa.vectorstore.QdrantStore"
        )
    return store


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


async def llm_reachable(llm: BaseChatModel, timeout_s: float = 2.0) -> bool | None:
    """Whether the chat model's server answers within ``timeout_s``: the API's readiness
    check and ``rag-serve``'s start-up check (DEC-18).

    For ``ChatOllama`` that is a list of its models (``GET /api/tags``), sent through the
    model's own async client, so the host, its parsing and the client's settings are the
    ones answers use. Reached through the private ``_async_client``, narrowly, as
    :func:`aclose_llm` does. ``None`` for a model with no such client, which has no server
    to ask.
    """
    list_models = getattr(getattr(llm, "_async_client", None), "list", None)
    if list_models is None:
        return None
    try:
        await asyncio.wait_for(list_models(), timeout_s)
    # Deliberately broad: refused, timed out, a proxy's error page, an unexpected reply.
    # Each means the same thing here, that answers would fail.
    except Exception:  # noqa: BLE001
        return False
    return True


def _reason(error: Exception) -> str:
    """What a failed model build says, without pydantic's wrapping around it."""
    if isinstance(error, ValidationError):
        return "; ".join(
            str(detail["msg"]).removeprefix("Value error, ") for detail in error.errors()
        )
    return f"{type(error).__name__}: {error}"


async def _reachable_then_close(llm: BaseChatModel) -> bool | None:
    try:
        return await llm_reachable(llm)
    finally:
        await aclose_llm(llm)


def preflight(config: RagConfig) -> str | None:
    """Why the language model's server cannot be used, or ``None`` when it answers.

    It builds the model, which with ``validate_model_on_init`` asks Ollama for its models,
    then asks once more with a time limit (the readiness check's), then closes the
    model's clients. The caller builds its own afterwards. It runs its own event loop, so
    call it before the caller's loop starts. A Ctrl-C is not caught here.
    """
    ref = config.pipeline.query.llm
    target = config.component(ref).spec()["_target_"]
    hint = f"\n{OLLAMA_HINT}" if str(target).startswith("langchain_ollama.") else ""
    try:
        llm = build_llm(config)
    # Deliberately broad: whatever stops the model being built stops the start, and it
    # deserves a reason rather than a traceback.
    except Exception as e:  # noqa: BLE001
        return f"cannot build the language model {ref}: {_reason(e)}{hint}"
    if asyncio.run(_reachable_then_close(llm)) is False:
        return f"the language model {ref} did not answer within 2 s{hint}"
    return None


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
