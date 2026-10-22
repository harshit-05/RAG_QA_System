"""The vector store: the only module that talks to Qdrant (ARCHITECTURE.md §3.1, DEC-19).

The Phase 3 seam (§0.5). Nothing else imports ``qdrant_client`` (``tests/test_architecture.py``),
so every error the client raises is translated here: :class:`StoreUnavailableError` when
Qdrant cannot be reached or refuses us, :class:`GenerationGoneError` when the generation a
search names has been deleted, :class:`StoreError` for any other answer. The other modules
catch those.

**Generations behind an alias.** A store's ``collection`` names an alias. Each version of
the index is a collection of its own, a *generation*, named
``<collection>--gen-<UTC stamp with microseconds>-<8 hex>``, so names sort in the order
they were made. :func:`write_generation` builds the next one in full, alongside the live
one, which is only ever read:

1. create it: one dense vector, ``dense``, by cosine; a payload index on
   ``metadata.source``; the manifest in its metadata, beside ``complete: false``;
2. copy the live generation's points into it, without the deleted ones;
3. upsert the new points, waiting for each batch, then check the count;
4. mark it ``complete: true``;
5. switch the alias to it in one ``update_collection_aliases`` call, which deletes and
   creates the alias together. Qdrant applies alias actions atomically, so a reader finds
   the old generation or the new one, never neither and never half of one.

A crash before step 5 publishes nothing. :func:`recover`, run under the writer lock at the
start of every ingestion and after every switch, keeps the alias's target and the newest
generation older than it, for readers still on it, and deletes the rest.

**Readers bind to a generation**, never to the alias: :func:`retriever` searches the one
its caller resolved, so a switch between the check and the search cannot mix two. The
previous generation survives one more switch; a search whose generation has gone raises
:class:`GenerationGoneError`, and the caller resolves the alias again
(:mod:`rag_qa.chain`).

**Every cut is ours.** Search is exact (``SearchParams(exact=True)``), and where points tie
at a limit, the server keeps a set that depends on the collection's layout, which changes
with the insertion order and across restarts (design review 2). So :func:`search` asks for
more than it needs, widens the request while the point at the cut ties the last one
fetched, then orders by score and point ID itself, and cuts.

**Hybrid retrieval, fused by us** (DEC-20, S3-5). A generation written with a sparse encoder
holds a second named vector, ``bm25``, whose IDF Qdrant applies (:mod:`rag_qa.sparse`). A
hybrid retriever runs two searches, dense and sparse, each cut to ``prefetch`` by the rule
above, and :func:`fuse` merges them by reciprocal rank fusion with an explicit ``rrf_k``,
ordered by score, then point ID. Qdrant's own fusion is not used: it breaks no ties, and
each of its prefetches would cut its own list by the collection's layout.

**A ``sources`` filter** limits every search to the named corpus documents: a ``MatchAny``
on ``metadata.source``, which each generation indexes.

**The manifest lives in the generation's metadata**, under one key and written whole, so
no stale field can survive a merge (Qdrant merges top-level metadata keys, and replaces a
nested object whole). It is read once per generation, never on a hot path:
``GET /collections/{name}`` returns all of it, and a request is limited to 32 MiB.
"""

import json
import math
import re
import secrets
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, SecretStr
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_qa.manifest import Manifest, ManifestError, manifest_payload, parse_manifest
from rag_qa.schema import SEARCH_TYPES, collection_problem
from rag_qa.settings import StoreSettings
from rag_qa.sparse import Bm25Encoder, SparseVector

#: What to try when Qdrant cannot be reached, shown under the error.
QDRANT_HINT: Final = "Is Qdrant running? docker compose up -d qdrant"
#: What to check when something answers at the store's URL, but not as Qdrant does.
URL_HINT: Final = (
    "Check RAG_QDRANT_URL, or the url of the config's vector store: it names Qdrant's "
    "REST API, such as http://localhost:6333"
)

#: What :func:`live_index` says when there is no index at all, as against one that cannot
#: be used: the query side tells "run rag-ingest first" from "re-run rag-ingest" by it.
NO_INDEX: Final = "there is no index yet"

#: A generation's dense vector, the embedder's.
VECTOR: Final = "dense"
#: Its sparse vector, the BM25 encoder's (DEC-20): only in a generation whose manifest has a
#: sparse record.
SPARSE: Final = "bm25"
#: The payload key a ``sources`` filter matches, which every generation indexes.
SOURCE_KEY: Final = "metadata.source"
#: The metadata keys: the manifest, written whole, and whether the generation is complete.
MANIFEST_KEY: Final = "manifest"
COMPLETE_KEY: Final = "complete"
#: Between the alias and the stamp in a generation's name. A collection name has no hyphen
#: (:data:`rag_qa.schema.COLLECTION_NAME`), so a generation's name parses one way only.
SEPARATOR: Final = "--gen-"
#: Generation stamps sort in the order they were made. Microseconds, because a second
#: cannot order several runs made in one.
_STAMP: Final = "%Y%m%dT%H%M%S%fZ"
#: Points per upsert or scroll request: far below the 32 MiB request limit at 384
#: dimensions and a chunk of text each.
BATCH: Final = 256
#: The oldest server with collection metadata, where the manifest lives (Qdrant 1.16).
MIN_SERVER: Final = (1, 16)
#: How many points beyond the cut a search asks for first (:func:`search`).
CUT_MARGIN: Final = 10


class StoreError(Exception):
    """Qdrant answered, but not as the store needs. The message says what it said."""


class StoreUnavailableError(StoreError):
    """Qdrant cannot be reached, did not answer in time, or refused the request. Every
    command stops on it with exit 2 and :data:`QDRANT_HINT`, having published nothing."""


class GenerationGoneError(StoreError):
    """The generation a search named has been deleted: two switches pruned it, or someone
    did by hand. Its reader resolves the alias again; the search is never retried as is."""


def connect(url: str, api_key: SecretStr | None, timeout_s: float) -> QdrantClient:
    """A client for the server at ``url``. Builds no connection and sends no request.

    The tests replace this function, so ingestion and the query side share one in-memory
    client. ``check_compatibility`` is off: it asks the server for its version in a thread
    of its own at construction, and warns when Qdrant is down, before any command could
    say so properly. :meth:`QdrantStore.check` asks instead.
    """
    return QdrantClient(
        url=url,
        api_key=None if api_key is None else api_key.get_secret_value(),
        timeout=math.ceil(timeout_s),
        check_compatibility=False,
    )


class QdrantStore:
    """The index's home: a Qdrant server, and the alias that names the live generation.

    The ``_target_`` of ``components.vector_stores.qdrant`` (``rag_qa.components.build_store``).
    ``RAG_QDRANT_URL`` and ``RAG_QDRANT_COLLECTION`` override ``url`` and ``collection``,
    and the API key comes from ``RAG_QDRANT_API_KEY`` only, never from a config file (§9).
    Building one sends no request; :meth:`check` does. ``client`` is for tests.
    """

    def __init__(
        self,
        url: str,
        collection: str,
        timeout_s: float = 10.0,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        settings = StoreSettings()
        self.url = settings.url or url
        self.collection = settings.collection or collection
        problem = collection_problem(self.collection)
        if problem is not None:
            raise ValueError(problem)
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be a positive number of seconds, not {timeout_s}")
        self.timeout_s = timeout_s
        self.client = (
            client if client is not None else connect(self.url, settings.api_key, timeout_s)
        )

    def __repr__(self) -> str:  # never the key
        return f"QdrantStore(url={self.url!r}, collection={self.collection!r})"

    @property
    def where(self) -> str:
        """The index, named for a message: ``the Qdrant collection 'rag_qa' at <url>``."""
        return f"the Qdrant collection {self.collection!r} at {self.url}"

    def check(self) -> None:
        """Ask the server for its version: every command's first step (DEC-19).

        Raises :class:`StoreUnavailableError` when it cannot be reached or refuses us, when
        what answers is not Qdrant's API (a wrong URL, :data:`URL_HINT`), or when it is
        older than :data:`MIN_SERVER`, which has no collection metadata.
        """
        with _as_qdrant(self):
            version = self.client.info().version
        parts = tuple(int(part) for part in re.findall(r"\d+", version)[:2])
        if parts < MIN_SERVER:
            raise StoreUnavailableError(
                f"Qdrant at {self.url} is version {version}, but the index needs "
                f"{'.'.join(map(str, MIN_SERVER))} or newer (collection metadata)"
            )
        # The other half of "can we use it": a key Qdrant refuses shows here, not mid-run.
        with _as_qdrant(self):
            self.client.get_aliases()

    def alias_target(self) -> str | None:
        """The collection the alias names now, or ``None`` when there is no alias.

        One small request (``GET /aliases``), cheap enough for the API to make on every
        request and ``rag-query`` before every question (DEC-19).
        """
        with _translated(self):
            aliases = self.client.get_aliases().aliases
        return next((a.collection_name for a in aliases if a.alias_name == self.collection), None)

    def close(self) -> None:
        """Close the client's connections."""
        self.client.close()


@contextmanager
def _translated(store: QdrantStore, generation: str | None = None) -> Iterator[None]:
    """Turn the client's errors into this module's, worded for the person running the command.

    A 404 becomes :class:`GenerationGoneError` when ``generation`` names what was asked for.
    """
    try:
        yield
    except ResponseHandlingException as e:  # refused, reset, timed out: no answer at all
        raise StoreUnavailableError(
            f"cannot use Qdrant at {store.url} ({type(e.source).__name__}: {e.source}). "
            f"{QDRANT_HINT}"
        ) from e
    except UnexpectedResponse as e:
        if e.status_code in (401, 403):
            raise StoreUnavailableError(
                f"Qdrant at {store.url} refused the request ({e.status_code}): check "
                f"RAG_QDRANT_API_KEY"
            ) from e
        if e.status_code == 404 and generation is not None:
            raise GenerationGoneError(f"the index generation {generation} is gone") from e
        raise StoreError(
            f"Qdrant at {store.url} answered {e.status_code}: {_server_error(e)}"
        ) from e


@contextmanager
def _as_qdrant(store: QdrantStore) -> Iterator[None]:
    """:func:`_translated`, for :meth:`QdrantStore.check`, where any answer but Qdrant's
    means a store that cannot be used. A wrong URL can reach another server: a path Qdrant
    does not serve (a 404), or another service on the port (a body that is not Qdrant's
    JSON). Every command must stop on it with exit 2 and a hint, never a traceback (S3-2's
    second review)."""
    try:
        with _translated(store):
            yield
    except StoreUnavailableError as e:
        source = getattr(e.__cause__, "source", None)
        if isinstance(source, ValueError):  # JSON of another shape: the client wraps it
            raise _not_qdrant(store, source) from e
        raise
    except StoreError as e:  # an HTTP answer Qdrant does not give to these requests
        raise StoreUnavailableError(f"{e}. {URL_HINT}") from e
    except ValueError as e:  # a body that is not JSON at all: the client does not wrap it
        raise _not_qdrant(store, e) from e


def _not_qdrant(store: QdrantStore, error: ValueError) -> StoreUnavailableError:
    """The refusal of an answer that is not Qdrant's, with the error's first line only:
    pydantic's runs to many."""
    shown = str(error).splitlines()[0] if str(error) else ""
    return StoreUnavailableError(
        f"{store.url} does not answer as Qdrant does ({type(error).__name__}: {shown}). {URL_HINT}"
    )


def _server_error(error: UnexpectedResponse) -> str:
    """The server's own error text, from its JSON body, or the HTTP reason."""
    try:
        return str(json.loads(error.content)["status"]["error"])
    except (ValueError, KeyError, TypeError):
        return error.reason_phrase or "no reason given"


# ---- names --------------------------------------------------------------------------------


def generation_pattern(store: QdrantStore) -> re.Pattern[str]:
    """The names of ``store``'s generations; group 1 is the stamp."""
    return re.compile(re.escape(store.collection + SEPARATOR) + r"(\d{8}T\d{12}Z)-[0-9a-f]{8}")


def _next_stamp(store: QdrantStore, live: str | None) -> str:
    """A stamp for a new generation, sorting after the live one's even if the clock has
    stepped back: recovery keeps "the newest generation older than the live one" by name,
    so a new generation that sorted first would cost the previous one its place."""
    stamp = datetime.now(UTC).strftime(_STAMP)
    found = generation_pattern(store).fullmatch(live) if live else None
    if found is not None and found.group(1) >= stamp:
        after = datetime.strptime(found.group(1), _STAMP).replace(tzinfo=UTC)
        stamp = (after + timedelta(microseconds=1)).strftime(_STAMP)
    return stamp


def _collections(store: QdrantStore) -> set[str]:
    """Every real collection on the server. Aliases are not listed: ``get_collections``
    omits them, where ``collection_exists`` resolves an alias to its target."""
    with _translated(store):
        return {c.name for c in store.client.get_collections().collections}


def _generations(store: QdrantStore) -> list[str]:
    """``store``'s generations, oldest first."""
    pattern = generation_pattern(store)
    return sorted(name for name in _collections(store) if pattern.fullmatch(name))


# ---- what is published ---------------------------------------------------------------------


def in_the_way(store: QdrantStore) -> str | None:
    """What stops ``store``'s alias being managed here, phrased to follow
    :attr:`QdrantStore.where`; ``None`` when nothing is in the way.

    A real collection named like the alias, or an alias naming anything but one of its
    generations: someone's own data, which ``rag-ingest`` must never switch away from or
    delete (S2-6's second review). The server would refuse the clash anyway, a collection
    named like an alias and an alias like a collection, but only once a run had embedded.
    """
    if store.collection in _collections(store):
        return "is a collection of its own, not the alias rag-ingest switches between generations"
    target = store.alias_target()
    if target is not None and not generation_pattern(store).fullmatch(target):
        return f"is an alias of {target!r}, which is not one of its generations"
    return None


class _Gone:
    """A generation that disappeared between two requests."""


_GONE: Final = _Gone()


def _metadata(store: QdrantStore, generation: str) -> dict[str, Any] | _Gone:
    """A generation's metadata: one ``GET /collections/{name}``, which returns all of it."""
    try:
        with _translated(store, generation):
            info = store.client.get_collection(generation)
    except GenerationGoneError:
        return _GONE
    except ValueError:  # local mode's "Collection … not found"
        if generation not in _collections(store):
            return _GONE
        raise
    return dict(info.config.metadata or {})


def _complete(generation: str, metadata: Mapping[str, Any]) -> Manifest | str:
    """The generation's manifest when it is complete, else why it is not.

    Complete means marked so, after its count was checked, and with a manifest that names
    the generation itself: a copy restored under another name is not complete until its
    manifest is rewritten (DEC-25).
    """
    if metadata.get(COMPLETE_KEY) is not True:
        return "it is incomplete: the run that built it stopped before publishing it"
    try:
        manifest = parse_manifest(metadata.get(MANIFEST_KEY))
    except ManifestError as e:
        return f"its manifest cannot be used: {e}"
    if manifest.generation != generation:
        return f"its manifest describes {manifest.generation!r}, not this collection"
    return manifest


def live_index(store: QdrantStore) -> tuple[str, Manifest] | str:
    """The live generation and its manifest, or why there is none to use.

    The one rule for "a usable index" that ``rag-ingest`` (to update it) and the query side
    (to answer from it) share. It reads the alias and the target's metadata, and loads no
    model. Whether the embedder matches the config is the callers' to check.
    """
    blocked = in_the_way(store)
    if blocked is not None:
        return f"{store.where} {blocked}"
    for _ in range(2):
        target = store.alias_target()
        if target is None:
            return NO_INDEX
        metadata = _metadata(store, target)
        if isinstance(metadata, _Gone):
            # Switched and pruned between the two requests: look once more.
            continue
        found = _complete(target, metadata)
        if isinstance(found, str):
            return f"its live generation {target} cannot be used: {found}"
        return target, found
    return "the alias moved twice while it was read; try again"


def newest_complete(store: QdrantStore) -> str | None:
    """The newest complete generation, whatever the alias says: the one to point the alias
    back at when its live collection was deleted and took the alias with it (DEC-19)."""
    for name in reversed(_generations(store)):
        metadata = _metadata(store, name)
        if not isinstance(metadata, _Gone) and isinstance(_complete(name, metadata), Manifest):
            return name
    return None


# ---- writing a generation ------------------------------------------------------------------


@dataclass(frozen=True)
class OpenGeneration:
    """The live generation, as an update starts from it: its name and its point IDs."""

    name: str
    ids: frozenset[str]


def _scroll(store: QdrantStore, generation: str, *, everything: bool) -> Iterator[list[Any]]:
    """A generation's points, a page at a time, in point ID order."""
    offset = None
    while True:
        with _translated(store, generation):
            page, offset = store.client.scroll(
                generation,
                limit=BATCH,
                offset=offset,
                with_payload=everything,
                with_vectors=everything,
            )
        yield page
        if offset is None:
            return


def open_generation(store: QdrantStore, generation: str) -> OpenGeneration:
    """The point IDs ``generation`` holds, for an update to check against its manifest.

    Raises :class:`GenerationGoneError` when it has gone, and the store's other errors.
    """
    ids = frozenset(
        str(point.id) for page in _scroll(store, generation, everything=False) for point in page
    )
    return OpenGeneration(name=generation, ids=ids)


@dataclass(frozen=True)
class Embedded:
    """Chunks ready to add, as parallel lists: their IDs, texts, metadata and vectors, and
    their sparse vectors when the manifest has a sparse encoder (``None`` otherwise)."""

    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    vectors: list[list[float]]
    sparse: list[SparseVector] | None = None


@dataclass(frozen=True)
class Published:
    """What :func:`write_generation` published."""

    generation: str  # the new generation's collection, which its manifest names


def write_generation(
    store: QdrantStore,
    manifest: Manifest,
    *,
    base: OpenGeneration | None,
    delete: Sequence[str],
    add: Embedded,
) -> Published:
    """Build the next generation and publish it with one alias switch (the module docstring).

    With ``base`` (the live generation), its points are copied, all but ``delete``, then
    ``add`` is upserted: so a document whose loader changed gets its old chunk IDs back with
    new content. Without it, the generation holds ``add`` alone, which must not be empty
    (a full rebuild). The vectors' size is the manifest's embedder dimension. With a sparse
    record in the manifest, the generation has a sparse vector as well, with Qdrant's IDF,
    and ``add`` must carry one per chunk; without one, it must carry none. Copied points
    keep the vectors they have, so ``base`` must have the same vector configuration:
    ``rag-ingest`` rebuilds in full whenever the sparse record changes.

    Nothing is published until the last request, the switch. An exception before it
    leaves an incomplete generation for :func:`recover`, which the next run calls under
    the writer lock. A switch request that fails is read back: when the alias names the
    new generation, the server applied it and only its answer was lost, so the generation
    is published and returned. Pruning after the switch is the caller's job, through
    :func:`recover`.
    """
    if base is None and not add.ids:
        raise ValueError("a new index needs at least one chunk")
    hybrid = manifest.sparse is not None
    if hybrid != (add.sparse is not None):
        raise ValueError(
            "the chunks' sparse vectors must come with a sparse record in the manifest, and "
            "only with one"
        )
    live = store.alias_target()
    name = f"{store.collection}{SEPARATOR}{_next_stamp(store, live)}-{secrets.token_hex(4)}"
    written = manifest.model_copy(update={"generation": name})
    expected = sum(record.chunks for record in written.documents.values())
    client = store.client
    with _translated(store):
        client.create_collection(
            name,
            vectors_config={
                VECTOR: models.VectorParams(
                    size=written.embedder.dimension, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config=(
                {SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
                if hybrid
                else None
            ),
            metadata={MANIFEST_KEY: manifest_payload(written), COMPLETE_KEY: False},
        )
        client.create_payload_index(
            name, "metadata.source", models.PayloadSchemaType.KEYWORD, wait=True
        )
    if base is not None:
        dropped = set(delete)
        for page in _scroll(store, base.name, everything=True):
            kept = [
                models.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in page
                if str(point.id) not in dropped
            ]
            if kept:
                with _translated(store):
                    client.upsert(name, points=kept, wait=True)
    sparse = add.sparse if add.sparse is not None else [None] * len(add.ids)
    for start in range(0, len(add.ids), BATCH):
        batch = slice(start, start + BATCH)
        points = [
            models.PointStruct(
                id=chunk,
                vector=_vectors(vector, lexical),
                payload={"page_content": text, "metadata": metadata},
            )
            for chunk, text, metadata, vector, lexical in zip(
                add.ids[batch],
                add.texts[batch],
                add.metadatas[batch],
                add.vectors[batch],
                sparse[batch],
                strict=True,
            )
        ]
        with _translated(store):
            client.upsert(name, points=points, wait=True)
    with _translated(store):
        count = client.count(name, exact=True).count
    if count != expected:
        raise StoreError(
            f"the new generation {name} holds {count} chunks, but its manifest lists "
            f"{expected}; nothing was published"
        )
    with _translated(store):
        client.update_collection(name, metadata={COMPLETE_KEY: True})
    try:
        with _translated(store):
            _switch(store, name)
    except StoreError:
        if not _switched_to(store, name):
            raise
    return Published(generation=name)


def _sparse(vector: SparseVector) -> models.SparseVector:
    """Our sparse vector as Qdrant's."""
    return models.SparseVector(indices=list(vector.indices), values=list(vector.values))


def _vectors(dense: list[float], sparse: SparseVector | None) -> dict[str, Any]:
    """A new point's named vectors: the dense one, and the sparse one when there is one."""
    if sparse is None:
        return {VECTOR: dense}
    return {VECTOR: dense, SPARSE: _sparse(sparse)}


def _switched_to(store: QdrantStore, generation: str) -> bool:
    """Whether the alias names ``generation``, read back after its switch request failed.

    The server may have applied the switch and lost only the answer (a read timeout, a
    reset connection). Then the run published, and must say so rather than "published
    nothing" with exit 2 (S3-2's second review). A read that fails too leaves the
    failure as it was: the next run's :func:`recover` keeps whichever generation is live.
    """
    try:
        return store.alias_target() == generation
    except StoreError:
        return False


def _switch(store: QdrantStore, generation: str) -> None:
    """Point the alias at ``generation``: delete and create in one request, which Qdrant
    applies atomically. The delete is sent on the first publish too, with no alias to
    delete; the server accepts that (checked on 1.19.2)."""
    store.client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=store.collection)
            ),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=generation, alias_name=store.collection
                )
            ),
        ]
    )


def recover(store: QdrantStore) -> list[str]:
    """Delete what earlier runs left among ``store``'s generations; return the names deleted.

    - **The alias names a generation:** keep it and the newest generation older than it,
      the previous one, which a reader may still be searching. Every other goes, newer ones
      included: a generation newer than the live one is a run that stopped before its
      switch, so it is incomplete by definition.
    - **The alias names anything else:** nothing here can be told live; touch nothing.
    - **No alias:** delete only the incomplete generations (design review 2). Qdrant deletes
      an alias with its target, so a lost live collection leaves no alias, and the
      previous generation, by then the last usable copy, must survive it.

    Call it under the writer lock: at the start of a run, and after a switch to prune.
    """
    generations = _generations(store)
    target = store.alias_target()
    if target is not None:
        if target not in generations:
            return []
        older = [name for name in generations if name < target]
        keep = {target, *older[-1:]}
    else:
        keep = set()
        for name in generations:
            metadata = _metadata(store, name)
            if not isinstance(metadata, _Gone) and isinstance(_complete(name, metadata), Manifest):
                keep.add(name)
    deleted = []
    for name in generations:
        if name not in keep:
            with _translated(store):
                store.client.delete_collection(name)  # one already gone counts as deleted
            deleted.append(name)
    return deleted


# ---- reading -------------------------------------------------------------------------------


def source_filter(sources: Sequence[str]) -> models.Filter | None:
    """The filter that keeps the chunks of ``sources``, corpus-relative paths, or ``None``
    for no filter when there are none."""
    if not sources:
        return None
    return models.Filter(
        must=[models.FieldCondition(key=SOURCE_KEY, match=models.MatchAny(any=list(sources)))]
    )


def search(
    store: QdrantStore,
    generation: str,
    query: list[float] | SparseVector,
    k: int,
    *,
    sources: Sequence[str] = (),
) -> list[Any]:
    """The ``k`` points that score highest for ``query`` in ``generation``, by exact search,
    cut by our own rule: score, then point ID (the module docstring).

    ``query`` is an embedding, searched against the dense vectors by cosine, or a
    :class:`~rag_qa.sparse.SparseVector`, against the sparse ones, which Qdrant scores with
    its IDF; a sparse search finds only points that share a term with it. With ``sources``,
    only those documents' chunks are searched (:func:`source_filter`).

    The request asks for :data:`CUT_MARGIN` more than ``k``, and doubles its limit while
    the last point it got ties the point at the cut: then more tied points may lie beyond
    the limit, and the server would have chosen among them by layout. Once the last point
    scores below the cut, every point tied with the cut is in hand.
    """
    using, sent = (SPARSE, _sparse(query)) if isinstance(query, SparseVector) else (VECTOR, query)
    limit = k + CUT_MARGIN
    while True:
        try:
            with _translated(store, generation):
                points = store.client.query_points(
                    generation,
                    query=sent,
                    using=using,
                    query_filter=source_filter(sources),
                    limit=limit,
                    with_payload=True,
                    search_params=models.SearchParams(exact=True),
                ).points
        except ValueError as e:  # local mode's "Collection … not found"
            if generation not in _collections(store):
                raise GenerationGoneError(f"the index generation {generation} is gone") from e
            raise
        if len(points) < limit or points[-1].score < points[k - 1].score:
            break
        limit *= 2
    return sorted(points, key=lambda point: (-point.score, str(point.id)))[:k]


def fuse(
    ranked: Sequence[Sequence[Any]],
    *,
    rrf_k: int,
    limit: int,
    weights: Sequence[float] | None = None,
) -> list[Any]:
    """Reciprocal rank fusion of ``ranked`` lists of points, each best first (DEC-20).

    A point scores ``weight / (rrf_k + rank)``, summed over the lists it is in, with
    one-based ranks; every list weighs 1 unless ``weights`` says otherwise. The fused
    points are ordered by that score, ties by point ID, and cut to ``limit``. Ties are the
    rule, not the exception: a point at rank r of one list scores the same whichever list
    that is. Each point keeps its payload from the first list that holds it; its ``score``
    is left as that list's, since the fused score is ours alone.
    """
    if weights is None:
        weights = [1.0] * len(ranked)
    if len(weights) != len(ranked):
        raise ValueError(f"{len(ranked)} lists to fuse, but {len(weights)} weights")
    scores: dict[str, float] = {}
    points: dict[str, Any] = {}
    for weight, found in zip(weights, ranked, strict=True):
        for rank, point in enumerate(found, 1):
            key = str(point.id)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            points.setdefault(key, point)
    order = sorted(scores, key=lambda key: (-scores[key], key))
    return [points[key] for key in order[:limit]]


class QdrantRetriever(BaseRetriever):
    """Retrieval from one generation: dense, as FAISS's retriever returned it, or hybrid.

    The same ``page_content``, metadata and ``id`` for each chunk, and nothing else: no
    score or collection keys in the metadata, which the prompt and the sources read.
    Synchronous only, so awaiting it runs this same code in an executor, as every answer
    and tier 1 do (``ainvoke``).

    **Hybrid** (``sparse`` set): the dense and sparse searches each keep ``prefetch``
    points, and :func:`fuse` merges them with ``rrf_k``, down to ``k``. A question with no
    tokens, all stopwords, has no sparse search, and the dense list is fused alone.

    **Sources:** ``invoke(question, sources=[...])`` limits every search to those corpus
    documents. LangChain hands a retriever's extra keyword arguments to
    :meth:`_get_relevant_documents` once its signature names them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: QdrantStore
    generation: str
    embeddings: Embeddings
    k: int
    sparse: Bm25Encoder | None = None
    prefetch: int | None = None
    rrf_k: int | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        sources: Sequence[str] = (),
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        if self.sparse is None or self.prefetch is None or self.rrf_k is None:
            found = search(self.store, self.generation, vector, self.k, sources=sources)
        else:
            lists = [search(self.store, self.generation, vector, self.prefetch, sources=sources)]
            lexical = self.sparse.encode_query(query)
            if lexical.indices:
                lists.append(
                    search(self.store, self.generation, lexical, self.prefetch, sources=sources)
                )
            found = fuse(lists, rrf_k=self.rrf_k, limit=self.k)
        docs = []
        for point in found:
            payload = point.payload or {}
            docs.append(
                Document(
                    id=str(point.id),
                    page_content=payload["page_content"],
                    metadata=payload["metadata"],
                )
            )
        return docs


def retriever(
    store: QdrantStore,
    generation: str,
    embeddings: Embeddings,
    spec: Mapping[str, Any],
    *,
    sparse: Bm25Encoder | None = None,
    k: int,
) -> BaseRetriever:
    """A retriever over ``generation``, bound to it, returning ``k`` chunks.

    ``spec`` is the config's retriever settings (``RetrieverSpec.kwargs()``); the schema
    has refused any search but similarity already. ``k`` is the caller's: the spec's own,
    or the reranker's candidates (DEC-16). A hybrid spec needs ``sparse``, the encoder that
    wrote the generation's sparse vectors, which encodes the queries too.
    """
    search_type = spec.get("search_type")
    if search_type is not None and search_type not in SEARCH_TYPES:
        raise ValueError(f"search_type {search_type!r} is not supported by the store")
    if spec.get("mode", "dense") != "hybrid":
        return QdrantRetriever(store=store, generation=generation, embeddings=embeddings, k=k)
    if sparse is None:
        raise ValueError("a hybrid retriever needs the sparse encoder that wrote the index")
    return QdrantRetriever(
        store=store,
        generation=generation,
        embeddings=embeddings,
        k=k,
        sparse=sparse,
        prefetch=spec["prefetch"],
        rrf_k=spec["rrf_k"],
    )
