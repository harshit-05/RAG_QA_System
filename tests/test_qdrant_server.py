"""The store against a real Qdrant server: what local mode cannot show (DEC-19, §3.5).

Local mode is a reimplementation in Python. It applies a failing alias batch halfway, keeps
a ``null`` metadata key, hands vectors back at another precision, and decides ties by its
own stable sort. So the alias switch, recovery, the metadata round trip, the copy and the
cut rule are shown here too, on the server.

Skipped unless ``RAG_TEST_QDRANT_URL`` names a server (conftest): CI's ``eval-retrieval``
job runs them against its service container, with ``pytest -m qdrant_server``. Each test
works under a collection name of its own, and removes everything under it afterwards, so it
never touches an index on the same server.
"""

import secrets
from collections.abc import Iterator

import pytest
from conftest import QDRANT_TEST_URL
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from test_vectorstore import (
    ENCODER,
    HYBRID,
    QUERY_1,
    SPARSE_RECORD,
    FixedEmbeddings,
    chunks_of,
    doc,
    manifest,
    publish,
    publish_hybrid,
    sparse_tied_collection,
    texts_of,
    tied_collection,
)

from rag_qa import vectorstore
from rag_qa.manifest import chunk_id, manifest_payload
from rag_qa.vectorstore import (
    GenerationGoneError,
    QdrantStore,
    live_index,
    newest_complete,
    open_generation,
    recover,
    retriever,
    search,
    write_generation,
)

pytestmark = pytest.mark.qdrant_server


@pytest.fixture
def server() -> Iterator[QdrantClient]:
    assert QDRANT_TEST_URL is not None  # the marker's skip guarantees it
    client = QdrantClient(url=QDRANT_TEST_URL, check_compatibility=False, timeout=30)
    yield client
    client.close()


@pytest.fixture
def collection(server: QdrantClient) -> Iterator[str]:
    """A collection name of this test's own; everything under it is removed afterwards."""
    name = f"t_{secrets.token_hex(6)}"
    yield name
    for alias in server.get_aliases().aliases:
        if alias.alias_name.startswith(name):
            server.update_collection_aliases(
                change_aliases_operations=[
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=alias.alias_name)
                    )
                ]
            )
    for found in server.get_collections().collections:
        if found.name.startswith(name):
            server.delete_collection(found.name)


def store_of(server: QdrantClient, collection: str) -> QdrantStore:
    assert QDRANT_TEST_URL is not None
    return QdrantStore(url=QDRANT_TEST_URL, collection=collection, client=server)


def generations_of(server: QdrantClient, collection: str) -> list[str]:
    prefix = f"{collection}--gen-"
    return sorted(c.name for c in server.get_collections().collections if c.name.startswith(prefix))


def test_the_store_checks_the_server(server: QdrantClient, collection: str) -> None:
    store_of(server, collection).check()


def test_the_alias_switch_and_recovery(server: QdrantClient, collection: str) -> None:
    # The first publish sends the alias's delete too, with no alias to delete: the server
    # accepts it (design review 2). Each later switch is one request; recovery after each
    # leaves the live generation and the previous one.
    store = store_of(server, collection)
    names = []
    for _ in range(4):
        names.append(publish(store, **{"a.txt": 2}))
        recover(store)
    assert store.alias_target() == names[-1]
    assert generations_of(server, collection) == names[-2:]


def test_a_failed_alias_batch_leaves_the_alias_as_it_was(
    server: QdrantClient, collection: str
) -> None:
    # What local mode gets wrong: it applies the delete, then fails the create. The server
    # applies alias actions atomically, which is what the switch relies on.
    store = store_of(server, collection)
    live = publish(store, **{"a.txt": 1})
    with pytest.raises(UnexpectedResponse):
        server.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=collection)),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=f"{collection}_missing", alias_name=collection
                    )
                ),
            ]
        )
    assert store.alias_target() == live


def test_with_no_alias_recovery_keeps_every_complete_generation(
    server: QdrantClient, collection: str
) -> None:
    store = store_of(server, collection)
    previous = publish(store, **{"a.txt": 1})
    live = publish(store, **{"a.txt": 1})
    incomplete = f"{collection}--gen-29991231T235959999999Z-0badf00d"
    server.create_collection(incomplete, vectors_config={}, metadata={"complete": False})
    server.delete_collection(live)
    assert store.alias_target() is None  # the server deleted the alias with its target
    assert recover(store) == [incomplete]
    assert generations_of(server, collection) == [previous]
    assert newest_complete(store) == previous


def test_the_manifest_round_trips_whole_through_collection_metadata(
    server: QdrantClient, collection: str
) -> None:
    store = store_of(server, collection)
    written = manifest(**{"a.txt": doc("a.txt", 3), "sub/b.pdf": doc("sub/b.pdf", 2)})
    add = chunks_of("a.txt", 3)
    add = type(add)(
        ids=add.ids + chunks_of("sub/b.pdf", 2).ids,
        texts=add.texts + chunks_of("sub/b.pdf", 2).texts,
        metadatas=add.metadatas + chunks_of("sub/b.pdf", 2).metadatas,
        vectors=add.vectors + chunks_of("sub/b.pdf", 2).vectors,
    )
    generation = write_generation(store, written, base=None, delete=[], add=add).generation
    found = live_index(store)
    assert isinstance(found, tuple)
    assert found[1] == written.model_copy(update={"generation": generation})
    # Marking it complete merged a top-level key; the manifest beside it came back whole.
    held = server.get_collection(generation).config.metadata or {}
    assert held["complete"] is True
    assert held["manifest"] == manifest_payload(found[1])
    # The payload index on metadata.source exists on the server (local mode ignores it).
    assert "metadata.source" in (server.get_collection(generation).payload_schema or {})


def test_a_copy_is_bit_for_bit(server: QdrantClient, collection: str) -> None:
    # An update copies the live generation's points, vectors included: a copy that moved
    # their last bits would let ranks drift from a fresh ingest over successive updates.
    store = store_of(server, collection)
    live = publish(store, **{"a.txt": 50})
    copied = write_generation(
        store,
        manifest(**{"a.txt": doc("a.txt", 50)}),
        base=open_generation(store, live),
        delete=[],
        add=type(chunks_of("a.txt", 0))(ids=[], texts=[], metadatas=[], vectors=[]),
    ).generation

    def vectors(generation: str) -> dict[str, list[float]]:
        found, _ = server.scroll(generation, limit=100, with_vectors=True)
        return {str(p.id): p.vector["dense"] for p in found}  # type: ignore[index]

    assert vectors(copied) == vectors(live)


def test_tied_points_give_one_cut_from_collections_built_in_different_orders(
    server: QdrantClient, collection: str
) -> None:
    # Design review 2: with points tied at a limit, the server kept another set for each
    # insertion order. A tie test must compare builds in different orders: within one
    # collection the server always agreed with itself.
    store = store_of(server, collection)
    orders = [list(range(30)), list(reversed(range(30))), [(i * 7) % 30 for i in range(30)]]
    results = []
    for n, order in enumerate(orders):
        name = f"{collection}_ties_{n}"
        tied_collection(server, name, order)
        results.append([str(p.id) for p in search(store, name, [1.0, 0.0], 10)])
    assert results[0] == results[1] == results[2]


def test_a_search_of_a_deleted_generation_is_gone(server: QdrantClient, collection: str) -> None:
    # The server answers 404, never 500, and the reader resolves the alias again.
    store = store_of(server, collection)
    old = publish(store, **{"a.txt": 1})
    publish(store, **{"a.txt": 1})
    server.delete_collection(old)
    with pytest.raises(GenerationGoneError, match=old):
        search(store, old, [1.0, 1.0, 0.5], 1)


# ---- hybrid retrieval and the sources filter (S3-5, DEC-20) --------------------------------


def test_a_hybrid_query_and_a_filtered_query_on_the_server(
    server: QdrantClient, collection: str
) -> None:
    # The server's IDF and filter, through our two searches and our fusion: the same answer
    # as local mode gives (test_vectorstore.py).
    store = store_of(server, collection)
    texts = ["alpha one", "beta two", "gamma three", "delta four", "zebra five"]
    generation = publish_hybrid(store, **{"a.txt": texts, "b.txt": ["zebra b", "beta b"]})
    params = server.get_collection(generation).config.params
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
    found = retriever(store, generation, FixedEmbeddings(), HYBRID, sparse=ENCODER, k=2)
    assert [d.page_content for d in found.invoke("zebra", sources=("a.txt",))] == [
        "zebra five",
        "alpha one",
    ]
    only_b = found.invoke("zebra", sources=("b.txt",))
    assert [d.page_content for d in only_b] == ["zebra b", "beta b"]
    dense = retriever(store, generation, FixedEmbeddings(), {"search_kwargs": {"k": 5}}, k=5)
    assert {d.metadata["source"] for d in dense.invoke("q", sources=("b.txt",))} == {"b.txt"}
    assert found.invoke("zebra", sources=("gone.txt",)) == []


def test_a_fused_tie_across_the_two_lists_is_cut_by_point_id(
    server: QdrantClient, collection: str
) -> None:
    # With prefetch 2, the dense list holds the first two chunks and the sparse list the
    # fifth alone: the first and the fifth both score 1 / (rrf_k + 1). A cut of 1 falls in
    # that tie, and the point ID decides it, whatever the insertion order.
    store = store_of(server, collection)
    texts = ["alpha one", "beta two", "gamma three", "delta four", "zebra five"]
    spec = {**HYBRID, "prefetch": 2}
    results = []
    for order in (texts, list(reversed(texts))):
        parts = texts_of("a.txt", texts)
        ranked = sorted(range(len(texts)), key=lambda i: order.index(texts[i]))
        add = type(parts)(
            ids=[parts.ids[i] for i in ranked],
            texts=[parts.texts[i] for i in ranked],
            metadatas=[parts.metadatas[i] for i in ranked],
            vectors=[parts.vectors[i] for i in ranked],
            sparse=[parts.sparse[i] for i in ranked] if parts.sparse else None,
        )
        records = {"a.txt": doc("a.txt", len(texts))}
        hybrid = manifest(**records).model_copy(update={"sparse": SPARSE_RECORD})
        generation = write_generation(store, hybrid, base=None, delete=[], add=add).generation
        fused = retriever(store, generation, FixedEmbeddings(), spec, sparse=ENCODER, k=1)
        both = retriever(store, generation, FixedEmbeddings(), spec, sparse=ENCODER, k=2)
        results.append(
            ([d.id for d in fused.invoke("zebra")], [d.id for d in both.invoke("zebra")])
        )
    tied = sorted([chunk_id("a.txt", "a" * 64, 0), chunk_id("a.txt", "a" * 64, 4)])
    assert results[0] == results[1] == ([tied[0]], tied)


def test_a_tie_at_each_branch_s_own_cut_gives_one_answer_from_builds_in_different_orders(
    server: QdrantClient, collection: str
) -> None:
    # Design review 2: each branch's cut through tied points is ours. Thirty points tie in
    # both branches; two collections built in different orders must give the same sparse
    # cut, the same dense cut, and so the same fused list (fuse is what the hybrid
    # retriever runs over the two). Within one collection the server always agrees with
    # itself, so only different builds prove anything.
    store = store_of(server, collection)
    results = []
    for n, order in enumerate([list(range(30)), [(i * 7) % 30 for i in range(30)]]):
        name = f"{collection}_sparse_ties_{n}"
        ids = sparse_tied_collection(server, name, order)
        sparse = search(store, name, QUERY_1, 8)
        dense = search(store, name, [1.0, 1.0], 8)
        fused = vectorstore.fuse([dense, sparse], rrf_k=60, limit=8)
        results.append([[str(p.id) for p in found] for found in (sparse, dense, fused)])
    assert results[0] == results[1]
    sparse_cut, dense_cut, _ = results[0]
    assert sparse_cut[5:] == sorted(ids)[:3]  # the 5 that score higher, then ties by ID
    assert dense_cut == sorted([*ids, *sparse_cut[:5]])[:8]  # every dense vector ties
