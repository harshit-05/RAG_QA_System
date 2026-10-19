"""Vector store persistence: the only place the store implementation appears.

This is the Phase 3 swap seam (ARCHITECTURE.md §0.5, DEC-3). Nothing outside this module
touches FAISS, and its functions take the store's location, never the config. Both sides
ask :func:`live_index` whether there is an index to use. The query side then calls
:func:`open_store`; ingestion calls :func:`open_generation`, :func:`write_generation` and
:func:`recover`. Migrating to Qdrant means rewriting these bodies.

**Generations and a symlink flip (DEC-17).** ``paths.vector_store`` (``vectorstore/db_faiss``)
is a symlink to a generation folder beside it, ``db_faiss.gen-<UTC stamp>-<8 hex>``, which
holds ``index.faiss``, ``index.pkl`` and ``manifest.json``. FAISS writes its two files
without any atomicity, so no run ever writes into the live generation. It builds a new
one, ``fsync``\\ s it, then replaces the symlink in one ``rename(2)``: a reader finds the old
generation or the new one, never neither and never half of one. The previous generation
survives one more flip, for readers still opening it; :func:`recover` removes the rest.

Safety invariant (ISS-16): ``allow_dangerous_deserialization=True`` unpickles the index.
That is only safe because every generation is produced by ``rag-ingest`` on this host.
Never point this at an index from an untrusted source.
"""

import os
import re
import secrets
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rag_qa.manifest import MANIFEST_FILE, Manifest, ManifestError, load_manifest, save_manifest

#: The two files FAISS writes into a generation, beside the manifest.
INDEX_FILES = ("index.faiss", "index.pkl")

#: What :func:`live_index` says when there is no index at all, as against one that cannot
#: be used: the query side tells "run rag-ingest first" from "re-run rag-ingest" by it.
NO_INDEX = "there is no index yet"

#: Generation names start with this stamp, so they sort in the order they were made.
#: Microseconds, because a second cannot order several runs made in one.
_STAMP = "%Y%m%dT%H%M%S.%fZ"


def _stamp() -> str:
    return datetime.now(UTC).strftime(_STAMP)


def _generation_name(store: Path) -> re.Pattern[str]:
    return re.compile(re.escape(store.name) + r"\.gen-(\d{8}T\d{6}\.\d{6}Z)-[0-9a-f]{8}")


def _live_name(store: Path) -> str | None:
    """The name of the generation folder beside ``store`` that its symlink points to,
    however the link is written (bare, ``./name``, absolute, a trailing slash).

    ``None`` when ``store`` is not a symlink, or points anywhere but at one of its
    generations beside it: then nothing beside it can be told live or not.
    """
    if not store.is_symlink():
        return None
    target = store.parent / os.readlink(store)  # an absolute target replaces the parent
    if not _generation_name(store).fullmatch(target.name):
        return None
    if target.parent.resolve() != store.parent.resolve():
        return None
    return target.name


def _next_stamp(store: Path) -> str:
    """A stamp for a new generation, sorting after the live one's even if the clock has
    stepped back: recovery keeps "the newest generation older than the live one" by name,
    so a new generation that sorted first would cost the previous one its place."""
    stamp = _stamp()
    live_name = _live_name(store)
    live = _generation_name(store).fullmatch(live_name) if live_name else None
    if live is not None and live.group(1) >= stamp:
        after = datetime.strptime(live.group(1), _STAMP).replace(tzinfo=UTC)
        stamp = (after + timedelta(microseconds=1)).strftime(_STAMP)
    return stamp


def _temporary_link_name(store: Path) -> re.Pattern[str]:
    return re.compile(re.escape(store.name) + r"\.tmp-[0-9a-f]{8}")


def generation_path(store: Path) -> Path | None:
    """The live generation: the folder beside ``store`` that it links to, whether or not
    that folder still exists.

    ``None`` when ``store`` is not a symlink to one of its generations: no index yet, a
    v0.2 directory, or something else in the way (:func:`in_the_way`).
    """
    name = _live_name(store)
    return None if name is None else store.parent / name


def is_legacy_store(store: Path) -> bool:
    """Whether ``store`` is an index written by v0.2: a real folder holding FAISS's two
    files and nothing else. Any other folder is not one, and is never renamed aside."""
    if not store.is_dir() or store.is_symlink():
        return False
    try:
        return {path.name for path in store.iterdir()} == set(INDEX_FILES)
    except OSError:
        return False


def in_the_way(store: Path) -> str | None:
    """What sits at the store path that is neither nothing, nor an index this code wrote or
    can migrate, phrased to follow the path; ``None`` when nothing is in the way.

    ``rag-ingest`` refuses to touch any of these: a file, a folder that is not a v0.2
    index (a mis-set path can name the corpus itself), or a link to anything but one of
    the index's generations beside it (someone's own link to another disk, say).
    """
    if store.is_symlink():
        if _live_name(store) is None:
            return f"is a link to {os.readlink(store)!r}, not to one of the index's generations"
        return None
    if store.is_dir():
        if is_legacy_store(store):
            return None
        if all((store / name).is_file() for name in (*INDEX_FILES, MANIFEST_FILE)):
            # One generation named directly, or a copy of one (`cp -rL`): an index, but the
            # path must name the link to it, which the next flip moves (second review).
            return (
                "is an index generation folder, not the link that rag-ingest points at the "
                "live generation"
            )
        return "is a folder that is not an index (v0.2's held only index.faiss and index.pkl)"
    if store.exists():
        return "is a file, not an index"
    return None


def live_index(store: Path) -> tuple[Path, Manifest] | str:
    """The live generation and its manifest, or why there is none to use.

    The one rule for "a usable index" that ``rag-ingest`` (to update it) and the query
    side (to answer from it) share, so that the two never disagree. It reads files only,
    loading no model and no index: the index files must be there, and the manifest must
    load. Whether the index opens, and whether its embedder matches the config, are the
    callers' to check. Lives here because what "an index exists" means is store-specific.
    """
    blocked = in_the_way(store)
    if blocked is not None:
        return f"the index path {blocked}"
    if is_legacy_store(store):
        return "the index was built by v0.2 and has no manifest"
    generation = generation_path(store)
    if generation is None:
        return NO_INDEX
    missing = [name for name in INDEX_FILES if not (generation / name).is_file()]
    if missing:
        return f"the index is missing {' and '.join(missing)}"
    try:
        manifest = load_manifest(generation / MANIFEST_FILE)
    except ManifestError as e:
        return f"its manifest cannot be used: {e}"
    if manifest is None:
        return "the index has no manifest"
    return generation, manifest


def store_exists(store: Path) -> bool:
    """Whether a usable published index is at ``store`` (:func:`live_index`). A v0.2
    directory is not one: it has no manifest, so nothing shows which embedder built it."""
    return not isinstance(live_index(store), str)


def open_store(embeddings: Embeddings, path: Path) -> VectorStore:
    """Open the persisted index a previous ingestion run wrote to ``path``: a generation
    folder, or the symlink to one."""
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


class _NoEmbedder(Embeddings):
    """Stands in where FAISS requires an embedder and nothing is embedded.

    Writing a generation adds vectors embedded already and deletes by ID, and saving
    writes no embedder, so it never embeds. With this in place of the configured embedder,
    a run that only deletes never loads the model.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("nothing is embedded while an index generation is written")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("an index generation being written is never queried")


@dataclass(frozen=True)
class OpenGeneration:
    """A generation loaded into memory for an update, with the chunk IDs it holds.

    Only :func:`write_generation` changes it, and only in memory: the files it came from
    are never written to.
    """

    path: Path
    ids: frozenset[str]
    db: FAISS = field(repr=False)


def open_generation(generation: Path) -> OpenGeneration:
    """Load a generation for an update, once: its IDs are checked, then the same object is
    updated and saved as the next generation.

    Raises whatever opening it raises (a missing file, a corrupt pickle, FAISS's own
    errors), and ``ValueError`` when the FAISS index and its docstore disagree on how many
    chunks there are.
    """
    db = FAISS.load_local(str(generation), _NoEmbedder(), allow_dangerous_deserialization=True)
    mapped = db.index_to_docstore_id
    ids = frozenset(mapped.values())
    if not db.index.ntotal == len(mapped) == len(ids):
        raise ValueError(
            f"its FAISS index holds {db.index.ntotal} vectors, mapped to {len(ids)} chunk IDs"
        )
    return OpenGeneration(path=generation, ids=ids, db=db)


@dataclass(frozen=True)
class Embedded:
    """Chunks ready to add, as parallel lists: their IDs, texts, metadata and vectors."""

    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    vectors: list[list[float]]


@dataclass(frozen=True)
class Published:
    """What :func:`write_generation` published."""

    generation: str  # the new generation's folder name, which its manifest records
    legacy: str | None = None  # where a v0.2 directory at the store path was moved, if any


def write_generation(
    store: Path,
    manifest: Manifest,
    *,
    base: OpenGeneration | None,
    delete: Sequence[str],
    add: Embedded,
) -> Published:
    """Build the next generation of the index and publish it with one atomic flip (DEC-17).

    With ``base`` (the live generation, in memory), ``delete`` is applied to it, then
    ``add``. Delete comes first, because a document whose loader changed gets the same
    chunk IDs back. Without ``base``, a fresh index is built from ``add``, which must not
    be empty (a full rebuild). Either way, the live generation on disk is only ever read.
    That matters: FAISS's own add is not atomic, and puts the vectors in before its
    docstore rejects a duplicate ID.

    Then, in order:

    1. a fresh folder, with the index and ``manifest`` written into it (the manifest's
       ``generation`` set to the folder's name);
    2. every file in it, then the folder itself, ``fsync``\\ ed;
    3. a v0.2 directory at ``store`` moved aside to ``<name>.v02-<stamp>``: the one moment
       with no index at ``store``, once, on the first run after the upgrade;
    4. a temporary symlink to the new folder, ``os.replace``\\ d over ``store``, and the
       parent folder ``fsync``\\ ed.

    An exception before step 4 publishes nothing. A half-built folder is left for
    :func:`recover`, which the next run calls under the writer lock. Pruning old
    generations after a flip is the caller's job, also through :func:`recover`.
    """
    pairs = list(zip(add.texts, add.vectors, strict=True))
    if base is None:
        if not pairs:
            raise ValueError("a new index needs at least one chunk")
        db = FAISS.from_embeddings(pairs, _NoEmbedder(), metadatas=add.metadatas, ids=add.ids)
    else:
        db = base.db
        if delete:
            db.delete(list(delete))
        if pairs:
            db.add_embeddings(pairs, metadatas=add.metadatas, ids=add.ids)

    name = f"{store.name}.gen-{_next_stamp(store)}-{secrets.token_hex(4)}"
    folder = store.parent / name
    folder.mkdir()
    db.save_local(str(folder))
    save_manifest(manifest.model_copy(update={"generation": name}), folder / MANIFEST_FILE)
    for path in folder.iterdir():
        _fsync(path)
    _fsync(folder, directory=True)
    # The new folder's own entry in its parent, durable before any link can name it: an
    # fsync after the flip alone would let a crash keep the link and lose the folder.
    _fsync(store.parent, directory=True)

    legacy = None
    if is_legacy_store(store):
        legacy = f"{store.name}.v02-{_stamp()}"
        os.rename(store, store.with_name(legacy))
        _fsync(store.parent, directory=True)
    _flip(store, name)
    return Published(generation=name, legacy=legacy)


def _flip(store: Path, generation: str) -> None:
    """Point ``store`` at ``generation`` in one ``rename(2)``, then make that durable."""
    link = store.with_name(f"{store.name}.tmp-{secrets.token_hex(4)}")
    # Relative, so the folder holding the store can be moved or renamed as a whole.
    os.symlink(generation, link)
    os.replace(link, store)
    _fsync(store.parent, directory=True)


def _fsync(path: Path, *, directory: bool = False) -> None:
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if directory else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def recover(store: Path) -> list[str]:
    """Delete what earlier runs left beside ``store``, and return the names deleted.

    Keeps the generation ``store`` links to, and the newest generation older than it: the
    previous one, which a reader may still be opening. Deletes every other generation
    folder, and every temporary link. An unreferenced generation newer than the live one
    is a crash before the flip, so it is incomplete by definition and goes too. With no
    symlink, nothing is published, and every generation goes. With a symlink pointing
    anywhere else, nothing can be told live, and no generation is touched. Names outside
    those two patterns, such as a ``.v02-`` directory or the lock file, are never touched.

    The link is read however it is written (:func:`_live_name`): a hand-made rollback such
    as ``ln -sfn "$PWD/vectorstore/db_faiss.gen-…" vectorstore/db_faiss`` must not read as
    "no generation is live".

    Call it under the writer lock: at the start of a run, and after a flip to prune.
    """
    parent = store.parent
    if not parent.is_dir():
        return []
    pattern = _generation_name(store)
    generations = sorted(
        path.name
        for path in parent.iterdir()
        if pattern.fullmatch(path.name) and path.is_dir() and not path.is_symlink()
    )
    target = _live_name(store)
    if store.is_symlink() and target is None:
        generations = []  # a link elsewhere: nothing here can be told live or not
    keep = set()
    if target is not None:
        keep.add(target)
        # The names start with a fixed-width UTC stamp, so string order is creation order.
        older = [name for name in generations if name < target]
        if older:
            keep.add(older[-1])
    deleted = []
    for name in generations:
        if name not in keep:
            shutil.rmtree(parent / name)
            deleted.append(name)
    temporary = _temporary_link_name(store)
    for path in sorted(parent.iterdir()):
        if temporary.fullmatch(path.name) and path.is_symlink():
            path.unlink()
            deleted.append(path.name)
    return deleted
