"""The ingestion manifest (S2-6, DEC-17, DEC-19): pure functions, with no index and no model.

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
    STORE,
    Changes,
    DocumentRecord,
    EmbedderRecord,
    FileState,
    Manifest,
    ManifestError,
    SparseRecord,
    SplitterRecord,
    chunk_id,
    diff,
    embedder_identity,
    file_sha256,
    manifest_payload,
    parse_manifest,
    same_sparse,
    sparse_record,
    spec_identity,
)
from rag_qa.sparse import SPARSE_VERSION

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
    # An unlisted class keeps every key it is given: its defaults are unchecked, so {} may
    # or may not be one. (Here it is, in langchain-community 0.4.2; S2-9's review.)
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


# --- the manifest, version 2 (DEC-19) -----------------------------------------------------


def record(sha256: str = "a" * 64, identity: str = "loader-1", chunks: int = 1) -> DocumentRecord:
    return DocumentRecord(
        sha256=sha256,
        loader="components.loaders.txt",
        loader_identity=identity,
        splitter="s",
        chunks=chunks,
        ingested_at="2026-10-05T12:00:00Z",
    )


def manifest(**documents: DocumentRecord) -> Manifest:
    return Manifest(
        generation="rag_qa--gen-20261005T120000000000Z-0a1b2c3d",
        chunking=1,
        embedder=EmbedderRecord(ref="components.embedders.fake", identity="e", dimension=8),
        splitter=SplitterRecord(ref="components.splitters.english_recursive", identity="s"),
        sparse=None,
        store=STORE,
        documents=documents,
    )


def test_a_manifest_round_trips_in_the_documented_shape() -> None:
    original = manifest(**{"guide.pdf": record(chunks=3)})
    data = json.loads(json.dumps(manifest_payload(original)))  # as the metadata holds it
    assert sorted(data) == [  # ARCHITECTURE.md §3.4, key for key
        "chunking",
        "documents",
        "embedder",
        "generation",
        "sparse",
        "splitter",
        "store",
        "version",
    ]
    assert data["version"] == 2
    assert data["sparse"] is None
    assert data["store"] == {"distance": "cosine", "exact": True}
    assert sorted(data["embedder"]) == ["dimension", "identity", "ref"]
    assert sorted(data["splitter"]) == ["identity", "ref"]
    assert sorted(data["documents"]["guide.pdf"]) == [
        "chunks",
        "ingested_at",
        "loader",
        "loader_identity",
        "sha256",
        "splitter",
    ]
    assert parse_manifest(data) == original


def test_the_architecture_s_example_manifest_parses() -> None:
    # The shape §3.4 documents, so the doc and the code cannot drift apart.
    example = {
        "version": 2,
        "generation": "rag_qa--gen-20261020T120000123456Z-1a2b3c4d",
        "chunking": 1,
        "embedder": {"ref": "components.embedders.minilm_cpu", "identity": "x", "dimension": 384},
        "splitter": {"ref": "components.splitters.english_recursive", "identity": "y"},
        "sparse": None,
        "store": {"distance": "cosine", "exact": True},
        "documents": {
            "2412.14140v2.pdf": {
                "sha256": "z",
                "loader": "components.loaders.pdf",
                "loader_identity": "w",
                "splitter": "y",
                "chunks": 412,
                "ingested_at": "2026-10-20T12:00:00Z",
            }
        },
    }
    assert parse_manifest(example).documents["2412.14140v2.pdf"].chunks == 412


def test_a_sparse_record_round_trips_and_a_null_still_parses() -> None:
    # S3-5 widened the type: a generation with sparse vectors records their encoder, and
    # every manifest written since S3-2, with null, stays readable.
    spec = {"_target_": "rag_qa.sparse.Bm25Encoder", "k1": 1.2, "b": 0.75, "avg_len": 100.76}
    sparse = sparse_record("components.sparse_encoders.bm25", spec)
    assert sparse == SparseRecord(
        ref="components.sparse_encoders.bm25", identity=spec_identity(spec), version=SPARSE_VERSION
    )
    hybrid = manifest(**{"a.txt": record()}).model_copy(update={"sparse": sparse})
    data = json.loads(json.dumps(manifest_payload(hybrid)))
    assert sorted(data["sparse"]) == ["identity", "ref", "version"]
    assert parse_manifest(data) == hybrid
    assert parse_manifest(manifest_payload(manifest(**{"a.txt": record()}))).sparse is None


def test_the_same_sparse_vectors_are_told_by_identity_and_version_never_the_name() -> None:
    a = SparseRecord(ref="components.sparse_encoders.bm25", identity="i", version=1)
    assert same_sparse(None, None)
    assert not same_sparse(a, None)
    assert not same_sparse(None, a)
    assert same_sparse(a, a.model_copy(update={"ref": "components.sparse_encoders.renamed"}))
    assert not same_sparse(a, a.model_copy(update={"identity": "j"}))
    assert not same_sparse(a, a.model_copy(update={"version": 2}))


def test_chunk_ids_are_derived_from_each_document_s_count() -> None:
    # Version 2 counts chunks rather than listing their IDs (DEC-19): chunk_id makes them.
    m = manifest(**{"a.txt": record("1" * 64, chunks=2), "b.txt": record("2" * 64, chunks=0)})
    assert m.chunk_ids_of("a.txt") == [chunk_id("a.txt", "1" * 64, i) for i in range(2)]
    assert m.chunk_ids_of("b.txt") == []
    assert m.chunk_ids() == set(m.chunk_ids_of("a.txt"))


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (None, "there is none"),
        ([1, 2], "not a JSON object but a list"),
        ({"version": 1, "generation": "g"}, "unknown manifest version 1; this version reads 2"),
        ({"version": True}, "unknown manifest version True"),
        ({"generation": "g"}, "unknown manifest version None"),
        ({"version": 2, "generation": "g"}, "not a version-2 manifest: chunking: Field required"),
    ],
)
def test_an_unusable_manifest_is_refused_with_the_reason(data: Any, problem: str) -> None:
    with pytest.raises(ManifestError, match=problem):
        parse_manifest(data)


@pytest.mark.parametrize(
    ("edit", "where"),
    [
        (lambda d: d["embedder"].update(dimension="384"), r"embedder\.dimension"),
        (lambda d: d["store"].update(distance="dot"), r"store\.distance"),
        (lambda d: d["store"].update(exact=False), r"store\.exact"),
        (lambda d: d.update(sparse={"ref": "x"}), "sparse"),
        (lambda d: d.update(sparse={"ref": "x", "identity": "i", "version": "1"}), "sparse"),
        (lambda d: d.update(sparse={"ref": "x", "identity": "i", "version": 1, "k1": 1}), "sparse"),
        (lambda d: d.pop("sparse"), "sparse: Field required"),
        (lambda d: d["documents"]["a.txt"].update(chunks=-1), r"documents\.a\.txt\.chunks"),
        (lambda d: d["documents"]["a.txt"].update(chunk_ids=[]), "Extra inputs"),
        (lambda d: d.update(extra=1), "extra: Extra inputs"),
    ],
    ids=[
        "a mistyped dimension",
        "another distance",
        "an approximate search",
        "a sparse record missing its identity and version",
        "a sparse version as text",
        "a sparse record with a field it does not have",
        "no sparse key",
        "a negative count",
        "version 1's chunk_ids",
        "a field this version does not write",
    ],
)
def test_a_manifest_off_the_strict_shape_is_refused_not_coerced(edit: Any, where: str) -> None:
    data = manifest_payload(manifest(**{"a.txt": record()}))
    edit(data)
    with pytest.raises(ManifestError, match=where):
        parse_manifest(data)


# --- the diff ----------------------------------------------------------------------------


def state(sha256: str, identity: str = "loader-1", splitter: str = "s") -> FileState:
    return FileState(
        sha256=sha256, loader="components.loaders.txt", loader_identity=identity, splitter=splitter
    )


def test_diff_sorts_the_corpus_into_added_changed_removed_and_unchanged() -> None:
    old = manifest(
        **{
            "same.txt": record("1" * 64),
            "edited.txt": record("1" * 64),
            "moved.md": record("1" * 64),
            "resplit.md": record("1" * 64),
            "gone.txt": record("1" * 64),
        }
    )
    files = {
        "same.txt": state("1" * 64),
        "edited.txt": state("2" * 64),  # new bytes
        "moved.md": state("1" * 64, identity="loader-2"),  # same bytes, another loader
        "resplit.md": state("1" * 64, splitter="s2"),  # same bytes, another splitter (S3-4)
        "new.txt": state("3" * 64),
    }
    assert diff(old, files) == Changes(
        added=("new.txt",),
        changed=("edited.txt", "moved.md", "resplit.md"),
        removed=("gone.txt",),
        unchanged=("same.txt",),
    )


def test_with_no_manifest_every_file_is_added() -> None:
    files = {"b.txt": state("1" * 64), "a.txt": state("2" * 64)}
    assert diff(None, files) == Changes(
        added=("a.txt", "b.txt"), changed=(), removed=(), unchanged=()
    )
