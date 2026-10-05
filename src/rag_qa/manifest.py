"""The ingestion manifest: what each index generation holds (DEC-17, ARCHITECTURE.md §2.4).

``manifest.json`` sits inside every generation, beside the index it describes. It records
which embedder and splitter built the index and, for each document, the sha256 of the
file's bytes, its loader's identity and the IDs of its chunks. ``rag-ingest`` diffs the
corpus against it to embed only what changed. The query side compares its embedder
identity with the config's before comparing any vectors.

Pure Python, like :mod:`rag_qa.schema`: pydantic and the standard library, no LangChain
(``tests/test_architecture.py``). S2-5's fingerprint reuses :func:`spec_identity` and
:func:`embedder_identity` unchanged, so a committed tier-2 run and the index agree on what
"the same embedder" means.
"""

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION: Final = 1

#: The namespace of every chunk ID. Changing it would change every ID, so it never changes.
CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/harshit-05/RAG_QA_System/chunk")

#: Embedder spec keys that choose where or how visibly the model runs, never which vectors
#: it makes: left out of the identity, so ``minilm_cuda`` on Colab and ``minilm_cpu`` here
#: are the same embedder (DEC-17). ``multi_process`` is deliberately not one of them (S2-6):
#: langchain-huggingface 1.2.2's multi-process path encodes without ``encode_kwargs``, so
#: switching it can change the vectors.
EMBEDDER_RUNTIME_KEYS = ("device", "show_progress", "cache_folder")
#: The same, inside the embedder's ``model_kwargs``.
EMBEDDER_RUNTIME_MODEL_KWARGS = ("device",)


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
    ``model_kwargs: {device: cpu}`` and no ``model_kwargs`` are the same embedder.
    """
    kept = {key: value for key, value in spec.items() if key not in EMBEDDER_RUNTIME_KEYS}
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

    Deterministic, so an unchanged document keeps its IDs. The path is in the key because
    two files with identical bytes would otherwise share IDs: FAISS rejects duplicates, and
    deleting one document would delete the other's chunks. UUIDs are also what Qdrant
    accepts as point IDs (DEC-3).
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


class DocumentRecord(_Record):
    """One corpus file. :attr:`Manifest.documents` keys it by its corpus-relative path."""

    sha256: str
    loader: str
    loader_identity: str
    chunk_ids: list[str]
    ingested_at: str


class Manifest(_Record):
    """What one index generation holds, in the shape of ARCHITECTURE.md §2.4."""

    version: Literal[1] = MANIFEST_VERSION
    generation: str  # the folder it sits in: what the symlink points to when it is live
    embedder: EmbedderRecord
    splitter: SplitterRecord
    documents: dict[str, DocumentRecord]

    def chunk_ids(self) -> set[str]:
        """Every chunk ID the manifest lists, across its documents."""
        return {cid for record in self.documents.values() for cid in record.chunk_ids}


def load_manifest(path: Path) -> Manifest | None:
    """The manifest at ``path``, or ``None`` when there is none (a v0.2 index, say).

    Raises :class:`ManifestError` when one is there but cannot be used. The messages name
    no path, so callers can log them as they are.
    """
    try:
        data = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except OSError as e:
        raise ManifestError(f"cannot read it ({type(e).__name__}: {e.strerror})") from e
    except ValueError as e:  # JSONDecodeError, and UnicodeDecodeError
        raise ManifestError(f"it is not valid JSON ({e})") from e
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


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Write ``manifest`` to ``path`` atomically: a temporary file, ``fsync``, ``os.replace``.

    A reader finds the old file or the new one, never half of one. Making the rename
    itself durable takes an ``fsync`` of the folder, which the caller does once every file
    is in (:func:`rag_qa.vectorstore.write_generation`).
    """
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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
