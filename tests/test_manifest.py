"""The ingestion manifest (S2-6, DEC-17): pure functions, with no index and no model.

The identity tests are the story's review note made executable: every key the embedder
identity leaves out is checked to change nothing, and every key it keeps to change it.
One wrong exclusion would let two embedders that differ pass for one, and the query-side
guard would then let incomparable vectors through.
"""

import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_CONFIG

from rag_qa.config import load_config
from rag_qa.manifest import (
    CHUNK_NAMESPACE,
    EMBEDDER_DEFAULTS,
    Changes,
    DocumentRecord,
    EmbedderRecord,
    FileState,
    Manifest,
    ManifestError,
    SplitterRecord,
    chunk_id,
    diff,
    embedder_identity,
    file_sha256,
    load_manifest,
    save_manifest,
    spec_identity,
)

# --- hashing and identities --------------------------------------------------------------


def test_file_sha256_reads_a_large_file_in_blocks_and_matches_hashlib(tmp_path: Path) -> None:
    data = os.urandom(3 * (1 << 18) + 7)  # several of hashlib.file_digest's 256 KiB blocks
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_spec_identity_reads_content_not_key_order() -> None:
    spec = {"_target_": "x.Y", "size": 8, "nested": {"b": 1, "a": 2}}
    reordered = {"nested": {"a": 2, "b": 1}, "size": 8, "_target_": "x.Y"}
    assert spec_identity(spec) == spec_identity(reordered)
    assert spec_identity(spec) != spec_identity({**spec, "size": 9})


MINILM: dict[str, Any] = {
    "_target_": "langchain_huggingface.HuggingFaceEmbeddings",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_kwargs": {"device": "cpu"},
}


@pytest.mark.parametrize(
    ("key", "value"),
    [("device", "cuda"), ("show_progress", True), ("cache_folder", "/elsewhere/models")],
)
def test_keys_that_cannot_change_the_vectors_leave_the_identity_alone(key: str, value: Any) -> None:
    assert embedder_identity({**MINILM, key: value}) == embedder_identity(MINILM)


def test_the_device_in_model_kwargs_leaves_the_identity_alone() -> None:
    on_gpu = {**MINILM, "model_kwargs": {"device": "cuda"}}
    assert embedder_identity(on_gpu) == embedder_identity(MINILM)


@pytest.mark.parametrize("left", [{}, None], ids=["empty", "null"])
def test_model_kwargs_holding_only_a_device_is_the_same_as_none(left: Any) -> None:
    # What is left once the device goes is the default (first review): no model_kwargs.
    without = {key: value for key, value in MINILM.items() if key != "model_kwargs"}
    assert embedder_identity(without) == embedder_identity(MINILM)
    assert embedder_identity({**without, "model_kwargs": left}) == embedder_identity(MINILM)


@pytest.mark.parametrize(
    ("key", "value"),
    [("encode_kwargs", {}), ("query_encode_kwargs", {}), ("multi_process", False)],
)
def test_a_default_spelled_out_is_the_same_embedder(key: str, value: Any) -> None:
    # Before the second review each of these read as another embedder: a needless rebuild,
    # a refused query, and a stale tier-2 fingerprint (S2-5).
    assert embedder_identity({**MINILM, key: value}) == embedder_identity(MINILM)


def test_a_default_counts_only_for_a_class_whose_defaults_are_listed() -> None:
    # In HuggingFaceBgeEmbeddings, encode_kwargs defaults to normalising, and {} turns that
    # off: an unlisted class keeps every key it is given.
    bge = {**MINILM, "_target_": "langchain_community.embeddings.HuggingFaceBgeEmbeddings"}
    assert embedder_identity({**bge, "encode_kwargs": {}}) != embedder_identity(bge)


@pytest.mark.parametrize("target", sorted(EMBEDDER_DEFAULTS))
def test_the_listed_defaults_are_the_installed_library_s(target: str) -> None:
    # A library upgrade that changes a default would otherwise equate two embedders.
    module, _, name = target.rpartition(".")
    fields = getattr(importlib.import_module(module), name).model_fields
    for key, value in EMBEDDER_DEFAULTS[target].items():
        assert fields[key].get_default(call_default_factory=True) == value, key


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("_target_", "langchain_huggingface.HuggingFaceEndpointEmbeddings"),
        ("model_name", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
        ("encode_kwargs", {"normalize_embeddings": True}),
        ("query_encode_kwargs", {"prompt": "query: "}),
        # Not a device key: langchain-huggingface 1.2.2's multi-process path encodes
        # without encode_kwargs, so switching it can change the vectors (S2-6).
        ("multi_process", True),
    ],
)
def test_every_other_key_changes_the_identity(key: str, value: Any) -> None:
    assert embedder_identity({**MINILM, key: value}) != embedder_identity(MINILM)


def test_other_model_kwargs_change_the_identity() -> None:
    pinned = {**MINILM, "model_kwargs": {"device": "cpu", "revision": "0123abc"}}
    assert embedder_identity(pinned) != embedder_identity(MINILM)


def test_the_real_configs_cpu_and_cuda_entries_are_one_embedder() -> None:
    # The DEC-12 offload path: an index embedded on Colab with a _cuda entry serves the
    # _cpu entry here. A different model is a different embedder.
    config = load_config(REAL_CONFIG)

    def identity(name: str) -> str:
        return embedder_identity(config.component(f"components.embedders.{name}").spec())

    assert identity("minilm_cuda") == identity("minilm_cpu")
    assert identity("multilingual_mpnet_cuda") == identity("multilingual_mpnet_cpu")
    assert identity("multilingual_mpnet_cpu") != identity("minilm_cpu")


def test_embedder_identity_leaves_the_spec_untouched() -> None:
    spec = {**MINILM, "model_kwargs": {"device": "cpu"}}
    embedder_identity(spec)
    assert spec["model_kwargs"] == {"device": "cpu"}


# --- chunk IDs ---------------------------------------------------------------------------


def test_chunk_ids_are_deterministic_uuid5s_keyed_by_path_hash_and_index() -> None:
    sha = "f" * 64
    first = chunk_id("a/notes.txt", sha, 0)
    assert first == chunk_id("a/notes.txt", sha, 0)
    assert uuid.UUID(first).version == 5
    others = {
        chunk_id("b/notes.txt", sha, 0),  # the same bytes elsewhere (DEC-17)
        chunk_id("a/notes.txt", "e" * 64, 0),
        chunk_id("a/notes.txt", sha, 1),
    }
    assert first not in others
    assert len(others) == 3


def test_chunk_ids_never_drift() -> None:
    # A changed namespace or key format would change every ID in every existing index,
    # and each update would then delete and re-add every unchanged chunk.
    assert str(CHUNK_NAMESPACE) == "016cda6d-006a-50af-ab8e-e46eebadb043"
    assert chunk_id("2412.14140v2.pdf", "0" * 64, 0) == "40ec7f75-8c19-5032-8e4f-5f2db21e3782"


# --- the manifest file -------------------------------------------------------------------


def record(sha256: str = "a" * 64, identity: str = "loader-1", *ids: str) -> DocumentRecord:
    return DocumentRecord(
        sha256=sha256,
        loader="components.loaders.txt",
        loader_identity=identity,
        chunk_ids=list(ids) or [chunk_id("x", sha256, 0)],
        ingested_at="2026-10-05T12:00:00Z",
    )


def manifest(**documents: DocumentRecord) -> Manifest:
    return Manifest(
        generation="db_faiss.gen-20261005T120000.000000Z-0a1b2c3d",
        embedder=EmbedderRecord(ref="components.embedders.fake", identity="e", dimension=8),
        splitter=SplitterRecord(ref="components.splitters.english_recursive", identity="s"),
        documents=documents,
    )


def test_a_manifest_round_trips_in_the_documented_shape(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = manifest(**{"guide.pdf": record()})
    save_manifest(original, path)

    data = json.loads(path.read_text())  # ARCHITECTURE.md §2.4, key for key
    assert sorted(data) == [
        "chunking", "documents", "embedder", "generation", "splitter", "version",
    ]
    assert data["version"] == 1
    assert data["chunking"] == 1
    assert sorted(data["embedder"]) == ["dimension", "identity", "ref"]
    assert sorted(data["splitter"]) == ["identity", "ref"]
    assert sorted(data["documents"]["guide.pdf"]) == [
        "chunk_ids", "ingested_at", "loader", "loader_identity", "sha256",
    ]
    assert load_manifest(path) == original
    assert not path.with_name("manifest.json.tmp").exists()


def test_a_save_interrupted_before_its_rename_leaves_the_old_manifest_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    old = manifest(**{"a.txt": record()})
    save_manifest(old, path)

    def crash(src: Any, dst: Any) -> None:
        raise OSError("crashed before the rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="crashed"):
        save_manifest(manifest(**{"b.txt": record()}), path)
    assert load_manifest(path) == old


def test_a_manifest_from_before_the_chunking_version_reads_as_version_1(tmp_path: Path) -> None:
    # The first S2-6 indexes have no "chunking" key, and version 1 made them: they stay
    # usable without a rebuild.
    path = tmp_path / "manifest.json"
    save_manifest(manifest(**{"a.txt": record()}), path)
    data = json.loads(path.read_text())
    del data["chunking"]
    path.write_text(json.dumps(data))
    loaded = load_manifest(path)
    assert loaded is not None and loaded.chunking == 1


def test_no_manifest_file_is_none_not_an_error(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "manifest.json") is None


def test_a_manifest_that_cannot_be_read_is_refused_without_its_path(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").mkdir()
    with pytest.raises(ManifestError, match=r"cannot read it \(IsADirectoryError") as exc:
        load_manifest(tmp_path / "manifest.json")
    assert str(tmp_path) not in str(exc.value)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("{oops", "not valid JSON"),
        (b"\xff\xfe\xfa", "not valid JSON"),
        ("[1, 2]", "not a JSON object but a list"),
        ('{"version": 2}', "unknown manifest version 2"),
        ('{"version": true}', "unknown manifest version True"),
        ('{"generation": "g"}', "unknown manifest version None"),
        ('{"version": 1, "generation": "g"}', "not a version-1 manifest: embedder: Field required"),
    ],
)
def test_an_unusable_manifest_is_refused_with_the_reason(
    tmp_path: Path, text: str | bytes, problem: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    with pytest.raises(ManifestError, match=problem):
        load_manifest(path)


def test_a_mistyped_field_is_refused_rather_than_coerced(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    save_manifest(manifest(**{"a.txt": record()}), path)
    data = json.loads(path.read_text())
    data["embedder"]["dimension"] = "384"
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match=r"embedder\.dimension"):
        load_manifest(path)


# --- the diff ----------------------------------------------------------------------------


def state(sha256: str, identity: str = "loader-1") -> FileState:
    return FileState(sha256=sha256, loader="components.loaders.txt", loader_identity=identity)


def test_diff_sorts_the_corpus_into_added_changed_removed_and_unchanged() -> None:
    old = manifest(
        **{
            "same.txt": record("1" * 64),
            "edited.txt": record("1" * 64),
            "moved.md": record("1" * 64),
            "gone.txt": record("1" * 64),
        }
    )
    files = {
        "same.txt": state("1" * 64),
        "edited.txt": state("2" * 64),  # new bytes
        "moved.md": state("1" * 64, identity="loader-2"),  # same bytes, another loader
        "new.txt": state("3" * 64),
    }
    assert diff(old, files) == Changes(
        added=("new.txt",),
        changed=("edited.txt", "moved.md"),
        removed=("gone.txt",),
        unchanged=("same.txt",),
    )


def test_with_no_manifest_every_file_is_added() -> None:
    files = {"b.txt": state("1" * 64), "a.txt": state("2" * 64)}
    assert diff(None, files) == Changes(
        added=("a.txt", "b.txt"), changed=(), removed=(), unchanged=()
    )
