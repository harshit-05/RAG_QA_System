"""The ingestion manifest: what each index generation holds (DEC-17, DEC-19, ARCHITECTURE.md §3.4).

Every generation carries its manifest in its own collection's metadata, so a query replica
needs no shared disk (DEC-19). It records which embedder and splitter built the index, how
the store searches it, and, for each document, the sha256 of the file's bytes, its loader's
and splitter's identities and how many chunks it has. ``rag-ingest`` diffs the corpus
against it to embed only what changed. The query side compares its embedder identity with
the config's before comparing any vectors.

**Version 2** (S3-2). A document's chunk IDs are derived (:meth:`Manifest.chunk_ids`), not
listed, which keeps a document near 200 bytes. The whole shape of §3.4 is written now,
``sparse`` and the per-document ``splitter`` included: the records are strict, so a field
added by a later story would make every manifest written before it unreadable. Version 1
is not read: it only ever lived beside a FAISS index.

Pure Python, like :mod:`rag_qa.schema`: pydantic and the standard library, no LangChain
(``tests/test_architecture.py``). S2-5's fingerprint reuses :func:`spec_identity` and
:func:`embedder_identity` unchanged, so a committed tier-2 run and the index agree on what
"the same embedder" means.
"""

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_VERSION: Final = 2

#: How this code turns a file into chunks. The identities see the config only, never the
#: code, so **bump it whenever the same file and config would make other chunks**: a change
#: to ``rag_qa.loaders``, to the metadata ``rag_qa.ingest`` adds, or a pypdf, docx2txt or
#: text-splitters upgrade that changes their output. Then every index rebuilds in full,
#: rather than keeping the old chunks of every unchanged file for good.
#: ``tests/test_ingest_incremental.py`` pins the sample corpus's chunks to it.
CHUNKING_VERSION: Final = 1

#: The namespace of every chunk ID. Changing it would change every ID, so it never changes.
CHUNK_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/harshit-05/RAG_QA_System/chunk"
)

#: Embedder spec keys that choose where or how visibly the model runs, never which vectors
#: it makes: left out of the identity, so ``minilm_cuda`` on Colab and ``minilm_cpu`` here
#: are the same embedder (DEC-17). ``multi_process`` is deliberately not one of them (S2-6):
#: langchain-huggingface 1.2.2's multi-process path encodes without ``encode_kwargs``, so
#: switching it can change the vectors.
EMBEDDER_RUNTIME_KEYS = ("device", "show_progress", "cache_folder")
#: The same, inside the embedder's ``model_kwargs``.
EMBEDDER_RUNTIME_MODEL_KWARGS = ("device",)

#: Per embedder class, keys whose value equals what leaving them out gives: dropped, so
#: spelling a default out never reads as another embedder (S2-6's second review). Only for
#: classes listed here, with defaults checked against the installed library
#: (``tests/test_manifest.py``): a default is a fact about one class in one version, so an
#: unlisted class keeps every key it is given, which costs at worst a needless rebuild.
EMBEDDER_DEFAULTS: Final[Mapping[str, Mapping[str, Any]]] = {
    "langchain_huggingface.HuggingFaceEmbeddings": {
        "encode_kwargs": {},
        "query_encode_kwargs": {},
        "multi_process": False,
    },
}


class ManifestError(Exception):
    """A manifest is there but cannot be used: unreadable, not JSON, of an unknown version,
    or not the shape this version writes."""


def file_sha256(path: Path) -> str:
    """The sha256 of the file's bytes, read in blocks rather than whole."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def spec_identity(spec: Mapping[str, Any]) -> str:
    """The sha256 of a component spec as canonical JSON: keys sorted, no spaces, UTF-8.

    The spec's content, never its reference name, so renaming a config entry changes
    nothing.
    """
    canonical = json.dumps(
        spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedder_identity(spec: Mapping[str, Any]) -> str:
    """:func:`spec_identity` of an embedder spec, minus the keys that cannot change its vectors.

    Those are :data:`EMBEDDER_RUNTIME_KEYS` and ``model_kwargs.device``. Everything else
    stays in: ``_target_``, ``model_name``, ``encode_kwargs``, ``query_encode_kwargs``,
    ``multi_process`` and any other ``model_kwargs``. One wrong exclusion would let two
    embedders that differ pass for one, and queries would compare incomparable vectors.

    A ``model_kwargs`` left empty goes too, since ``{}`` is what leaving it out means:
    ``model_kwargs: {device: cpu}`` and no ``model_kwargs`` are the same embedder. So does
    any key set to its default for a class in :data:`EMBEDDER_DEFAULTS`.
    """
    target = spec.get("_target_")
    defaults = EMBEDDER_DEFAULTS.get(target, {}) if isinstance(target, str) else {}
    kept = {
        key: value
        for key, value in spec.items()
        if key not in EMBEDDER_RUNTIME_KEYS and not (key in defaults and value == defaults[key])
    }
    model_kwargs = kept.pop("model_kwargs", None)
    if isinstance(model_kwargs, Mapping):
        model_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key not in EMBEDDER_RUNTIME_MODEL_KWARGS
        }
    if model_kwargs:  # anything left, including a malformed non-mapping, stays in
        kept["model_kwargs"] = model_kwargs
    return spec_identity(kept)


def chunk_id(relative_path: str, sha256: str, index: int) -> str:
    """The ID of a document's ``index``-th chunk: ``uuid5`` over ``path:sha256:index``.

    Deterministic, so an unchanged document keeps its IDs, and a manifest need only count
    them. The path is in the key because two files with identical bytes would otherwise
    share IDs, and deleting one document would delete the other's chunks. UUIDs are what
    Qdrant accepts as point IDs (DEC-3).
    """
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{relative_path}:{sha256}:{index}"))


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class EmbedderRecord(_Record):
    ref: str  # the config reference, for messages; the identity is what is compared
    identity: str
    dimension: int  # the backstop behind the identity


class SplitterRecord(_Record):
    ref: str
    identity: str


class StoreRecord(_Record):
    """How the store searches the generation (DEC-19): by cosine, always exactly."""

    distance: Literal["cosine"]
    exact: Literal[True]


#: The only store record this version writes or reads.
STORE: Final = StoreRecord(distance="cosine", exact=True)


class DocumentRecord(_Record):
    """One corpus file. :attr:`Manifest.documents` keys it by its corpus-relative path."""

    sha256: str
    loader: str
    loader_identity: str
    #: The identity of the splitter that chunked it: the default one, until S3-4 gives each
    #: extension its own.
    splitter: str
    chunks: int = Field(ge=0)
    ingested_at: str


class Manifest(_Record):
    """What one index generation holds, in the shape of ARCHITECTURE.md §3.4."""

    version: Literal[2] = MANIFEST_VERSION
    generation: str  # the collection it describes: what the alias names when it is live
    chunking: int  # the CHUNKING_VERSION that made its chunks
    embedder: EmbedderRecord
    splitter: SplitterRecord  # the default splitter
    #: The sparse encoder's record, null until S3-11b sets one (DEC-20). A later story
    #: widens the type; a null stays readable.
    sparse: None
    store: StoreRecord
    documents: dict[str, DocumentRecord]

    def chunk_ids_of(self, name: str) -> list[str]:
        """The IDs of document ``name``'s chunks, in order, derived as ingestion made them."""
        record = self.documents[name]
        return [chunk_id(name, record.sha256, i) for i in range(record.chunks)]

    def chunk_ids(self) -> set[str]:
        """Every chunk ID the manifest's documents hold."""
        return {cid for name in self.documents for cid in self.chunk_ids_of(name)}


def parse_manifest(data: object) -> Manifest:
    """The manifest in ``data``, a generation's metadata entry as JSON reads it.

    Raises :class:`ManifestError` when it cannot be used: absent, not an object, of
    another version, or not this version's shape. The messages name no collection, so
    callers can say where it came from.
    """
    if data is None:
        raise ManifestError("there is none")
    if not isinstance(data, dict):
        raise ManifestError(f"it is not a JSON object but a {type(data).__name__}")
    version = data.get("version")
    if version != MANIFEST_VERSION or isinstance(version, bool):
        raise ManifestError(
            f"unknown manifest version {version!r}; this version reads {MANIFEST_VERSION}"
        )
    try:
        return Manifest.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise ManifestError(
            f"it is not a version-{MANIFEST_VERSION} manifest: {where}: {first['msg']}"
            + (f" (and {e.error_count() - 1} more)" if e.error_count() > 1 else "")
        ) from e


def manifest_payload(manifest: Manifest) -> dict[str, Any]:
    """``manifest`` as JSON, whole: what a generation's metadata holds under its key."""
    return manifest.model_dump(mode="json")


@dataclass(frozen=True)
class FileState:
    """What the corpus walk found for one file: its bytes' sha256 and the loader it maps to."""

    sha256: str
    loader: str  # the loader's config reference, e.g. components.loaders.pdf
    loader_identity: str  # spec_identity of that loader's spec


@dataclass(frozen=True)
class Changes:
    """How the corpus differs from a manifest, as sorted corpus-relative paths."""

    added: tuple[str, ...]
    changed: tuple[str, ...]
    removed: tuple[str, ...]
    unchanged: tuple[str, ...]


def diff(manifest: Manifest | None, files: Mapping[str, FileState]) -> Changes:
    """Sort the corpus's ``files`` against ``manifest`` (DEC-17).

    A document has changed when its bytes **or** its loader identity differ. Without the
    second, an edit to the extension map or to a loader's spec would leave the old
    loader's chunks in the index for good. With no manifest, every file is added.
    """
    known = manifest.documents if manifest is not None else {}
    added, changed, unchanged = [], [], []
    for name, state in files.items():
        record = known.get(name)
        if record is None:
            added.append(name)
        elif (record.sha256, record.loader_identity) != (state.sha256, state.loader_identity):
            changed.append(name)
        else:
            unchanged.append(name)
    removed = [name for name in known if name not in files]
    return Changes(
        added=tuple(sorted(added)),
        changed=tuple(sorted(changed)),
        removed=tuple(sorted(removed)),
        unchanged=tuple(sorted(unchanged)),
    )
