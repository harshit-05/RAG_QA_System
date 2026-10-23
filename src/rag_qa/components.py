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

:func:`preflight` is the start-up check of the language model's server that ``rag-serve``,
``rag-query`` and ``rag-eval generate`` share (S2-7, S2-8, S3-6). It lives here, not in the
``api`` package, because the CLI never imports that package (ARCHITECTURE.md §2.2).
:func:`build_store` is the vector store's one build path, and every command checks the store
before anything else (DEC-19).

**Timeouts and the hosted fallback** (DEC-21, S3-6). :func:`build_llm` and
:func:`build_fallback_llm` turn a timeout spelled as its phases into its HTTP client's own
``Timeout``, which YAML cannot spell. The fallback gets its key from
``RAG_FALLBACK_API_KEY`` and an HTTP client of its own; ``langchain_openai`` and ``openai``
come from the ``fallback`` extra, and are imported only to build it.
"""

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from rag_qa import resilience
from rag_qa.config import ConfigError
from rag_qa.manifest import SparseRecord, sparse_record
from rag_qa.registry import build_object
from rag_qa.schema import CLIENT_KWARGS, RagConfig
from rag_qa.settings import (
    ENV_FALLBACK_API_KEY,
    ENV_QDRANT_PREFIX,
    FallbackSettings,
    describe_settings_error,
)
from rag_qa.sparse import Bm25Encoder
from rag_qa.vectorstore import QdrantStore

#: What to try when an Ollama model cannot be reached, shown under the error.
OLLAMA_HINT = "Is Ollama running? Try: systemctl status ollama"
#: How to install what the hosted fallback is built on (DEC-21).
FALLBACK_EXTRA_HINT: Final = "install it with `uv sync --extra fallback`"
#: How messages name the hosted fallback's service, its breaker's included.
FALLBACK_SERVICE: Final = "the hosted fallback"
#: The ``_target_`` prefix of a model whose HTTP client is httpx2's, through openai.
OPENAI_PREFIX: Final = "langchain_openai."


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


def build_sparse_encoder(config: RagConfig) -> Bm25Encoder | None:
    """The sparse encoder for both ingestion and hybrid queries: always
    ``pipeline.ingestion.sparse_encoder``, as :func:`build_embedder` is always the
    ingestion embedder (DEC-20). ``None`` when it is unset.

    Raises :class:`~rag_qa.config.ConfigError` when the entry cannot make one: its
    settings, or a ``_target_`` that is not one.
    """
    ref = config.pipeline.ingestion.sparse_encoder
    if ref is None:
        return None
    try:
        encoder = build_object(config.component(ref).spec())
    except (TypeError, ValueError) as e:
        raise ConfigError(f"Cannot build the sparse encoder {ref}: {e}") from e
    if not isinstance(encoder, Bm25Encoder):
        raise ConfigError(
            f"{ref} builds a {type(encoder).__name__}, not a sparse encoder: its _target_ "
            f"must be rag_qa.sparse.Bm25Encoder"
        )
    return encoder


def configured_sparse(config: RagConfig) -> SparseRecord | None:
    """The sparse record an index built with ``config`` has: the ingestion sparse encoder's
    identity and :data:`~rag_qa.sparse.SPARSE_VERSION`, or ``None`` when none is set.
    ``rag-ingest`` writes it, and the query side and ``rag-eval generate`` compare with it
    (:func:`rag_qa.manifest.same_sparse`)."""
    ref = config.pipeline.ingestion.sparse_encoder
    return None if ref is None else sparse_record(ref, config.component(ref).spec())


def build_splitter(config: RagConfig, extension: str | None = None) -> Any:
    """An ingestion text splitter (anything with ``split_documents``): the one for files
    with this lowercased suffix (``pipeline.ingestion.splitters``, S3-4), else the default
    ``pipeline.ingestion.splitter``, which is also what no ``extension`` gives."""
    ingestion = config.pipeline.ingestion
    ref = ingestion.splitter if extension is None else ingestion.splitter_for(extension)
    return build_object(config.component(ref).spec())


def build_llm(config: RagConfig, *, validate: bool = True) -> BaseChatModel:
    """The query-time chat model, ``pipeline.query.llm``.

    ``validate=False`` builds it without asking its server anything: its
    ``validate_model_on_init``, a runtime key that tier 2 does not hash, is turned off. A
    pipeline with a hosted fallback builds it so, since Ollama may be down then, which is
    the fallback's case (DEC-21).
    """
    spec = config.component(config.pipeline.query.llm).spec()
    if not validate and "validate_model_on_init" in spec:
        spec["validate_model_on_init"] = False
    return _build_model(spec)


def service_of(config: RagConfig) -> str:
    """How messages, and its breaker, name the service behind ``pipeline.query.llm``:
    Ollama for an Ollama model, else "the language model"."""
    target = str(config.component(config.pipeline.query.llm).spec()["_target_"])
    return "Ollama" if target.startswith("langchain_ollama.") else "the language model"


def raised_by_fallback(error: BaseException) -> bool:
    """Whether ``error`` came from the hosted fallback: marked so by the answer that was
    reading its stream (:func:`rag_qa.resilience.mark_fallback`), which is how its stream's
    own HTTP errors are told from Ollama's; or openai's own errors, and their subclasses in
    langchain-openai, by module name, so nothing here imports openai."""
    return resilience.is_marked_fallback(error) or any(
        kind.__module__.split(".")[0] in ("openai", "langchain_openai")
        for kind in type(error).__mro__
    )


def _openai() -> Any:
    """The ``openai`` package, from the ``fallback`` extra (or ``eval``'s), imported only to
    build a model on it: the tier-2 gate never loads it (``tests/test_architecture.py``)."""
    try:
        import openai
    except ImportError as e:
        raise ConfigError(f"the openai package is not installed: {FALLBACK_EXTRA_HINT}") from e
    return openai


def _with_timeouts(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with each timeout spelled as its phases made into its HTTP client's
    ``Timeout`` (S3-6): httpx's in ChatOllama's ``*client_kwargs``, openai's (httpx2's) in
    a ``langchain_openai`` model's own ``timeout``. YAML cannot spell either; the schema
    checked the phases at load. A number is left as it is, for every phase."""
    built = dict(spec)
    for key in CLIENT_KWARGS:
        holder = built.get(key)
        if isinstance(holder, dict) and isinstance(holder.get("timeout"), dict):
            built[key] = {**holder, "timeout": httpx.Timeout(**holder["timeout"])}
    phases = built.get("timeout")
    if isinstance(phases, dict):
        on_openai = str(built["_target_"]).startswith(OPENAI_PREFIX)
        built["timeout"] = (_openai().Timeout if on_openai else httpx.Timeout)(**phases)
    return built


def _build_model(spec: dict[str, Any]) -> BaseChatModel:
    """A chat model from its spec, with its timeouts made (:func:`_with_timeouts`).

    A ``langchain_openai`` model gets an async HTTP client of its own. langchain-openai
    1.6.2 otherwise shares one, cached per base URL and timeout, among all its instances
    (``_cached_async_httpx_client``), so closing one model's would close every other's;
    owning it lets :func:`aclose_llm` close it (S3-6, caveat 38).
    """
    built = _with_timeouts(spec)
    if str(built["_target_"]).startswith(OPENAI_PREFIX):
        built["http_async_client"] = _openai().DefaultAsyncHttpxClient()
    return build_object(built)


def build_fallback_llm(config: RagConfig) -> BaseChatModel | None:
    """The hosted fallback, ``pipeline.query.fallback_llm``, or ``None`` when none is set
    (DEC-21). The schema has checked its entry (D8 of S3-6's plan).

    Its key is ``RAG_FALLBACK_API_KEY``'s, passed by this code: left to its own default,
    ``ChatOpenAI`` would read ``OPENAI_API_KEY``, or a LangSmith gateway key, and send it to
    the entry's ``base_url``. Raises :class:`~rag_qa.config.ConfigError` when the key is
    not set, naming the variable and never a value, or when the ``fallback`` extra is not
    installed. Building it sends nothing: no question leaves the host until it answers.
    """
    ref = config.pipeline.query.fallback_llm
    if ref is None:
        return None
    key = FallbackSettings().api_key
    if key is None:
        raise ConfigError(
            f"the hosted fallback {ref} needs its API key in {ENV_FALLBACK_API_KEY}, which is "
            f"not set. Keys come from the environment only: set it in this shell, or in .env "
            f"and load that with `set -a; . ./.env; set +a`"
        )
    try:
        return _build_model({**config.component(ref).spec(), "api_key": key})
    except ImportError as e:  # the allowlist's import of langchain_openai
        raise ConfigError(
            f"cannot build the hosted fallback {ref}: {e}. It needs the fallback extra: "
            f"{FALLBACK_EXTRA_HINT}"
        ) from e


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
    (ARCHITECTURE.md §2.1, DEC-14). ``ChatOpenAI``'s async client is its public
    ``root_async_client``, whose HTTP client :func:`build_fallback_llm` gave it, so closing it
    closes no other model's (S3-6); its sync one is the library's shared default, never used
    to answer, and is left alone. Kept narrow: only those names, and only their ``close``,
    awaited when it is a coroutine (``AsyncClient.close``) and called plainly otherwise
    (``Client.close``). Run it on the loop that used the async client.
    """
    for name in ("_async_client", "_client", "root_async_client"):
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


async def _reachable_then_close(llm: BaseChatModel, *also: BaseChatModel) -> bool | None:
    try:
        return await llm_reachable(llm)
    finally:
        for model in (llm, *also):
            await aclose_llm(model)


@dataclass(frozen=True)
class Preflight:
    """What the start-up check of the language model found.

    ``problem``: why the command cannot start (exit 2). ``warning``: what is wrong, but
    covered: Ollama is not answering, and the hosted fallback will until it does (DEC-21).
    """

    problem: str | None = None
    warning: str | None = None


def preflight(config: RagConfig, *, fallback: bool = True) -> Preflight:
    """The start-up check of the language model's server (S2-7, S2-8, S3-6).

    It builds the model, which with ``validate_model_on_init`` asks Ollama for its models,
    then asks once more with a time limit (the readiness check's), then closes the
    model's clients. The caller builds its own afterwards. It runs its own event loop, so
    call it before the caller's loop starts. A Ctrl-C is not caught here.

    **With a hosted fallback** (``fallback``, and ``pipeline.query.fallback_llm`` set), Ollama
    down is not a refusal but a :attr:`Preflight.warning`: the fallback answers until it is
    back (DEC-21, design review 1). The model is then built without asking Ollama anything,
    as the pipeline builds it. The fallback is built too, which needs its key and its extra,
    and closed: it is asked nothing, so nothing leaves the host here. ``rag-eval generate``
    passes ``fallback=False``, since tier 2 never uses one.
    """
    ref = config.pipeline.query.llm
    target = config.component(ref).spec()["_target_"]
    hint = f"\n{OLLAMA_HINT}" if str(target).startswith("langchain_ollama.") else ""
    spare_ref = config.pipeline.query.fallback_llm if fallback else None
    spare = None
    if spare_ref is not None:
        try:
            spare = build_fallback_llm(config)
        except ConfigError as e:
            return Preflight(problem=str(e))
        # Deliberately broad, as for the model below.
        except Exception as e:  # noqa: BLE001
            return Preflight(problem=f"cannot build the hosted fallback {spare_ref}: {_reason(e)}")
    also = () if spare is None else (spare,)
    try:
        llm = build_llm(config) if spare is None else build_llm(config, validate=False)
    # Deliberately broad: whatever stops the model being built stops the start, and it
    # deserves a reason rather than a traceback.
    except Exception as e:  # noqa: BLE001
        if also:
            asyncio.run(_close(*also))
        return Preflight(problem=f"cannot build the language model {ref}: {_reason(e)}{hint}")
    if asyncio.run(_reachable_then_close(llm, *also)) is not False:
        return Preflight()
    down = f"the language model {ref} did not answer within 2 s"
    if spare_ref is None:
        return Preflight(problem=f"{down}{hint}")
    where = config.component(spare_ref).spec()["base_url"]
    return Preflight(
        warning=(
            f"{down}. Until it does, the hosted fallback {spare_ref} answers, and each "
            f"question and its retrieved chunks are sent to {where}.{hint}"
        )
    )


async def _close(*models: BaseChatModel) -> None:
    for model in models:
        await aclose_llm(model)


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
