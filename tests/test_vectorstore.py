"""The Qdrant store (S3-2, DEC-19), hermetically, on the in-memory client.

Generations, the alias switch, recovery and the cut rule, each against the module's own
functions. What only the server can show (its atomic alias batch, its 404, ties decided by
a collection's layout) is in ``test_qdrant_server.py``, under the ``qdrant_server`` marker.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import COLLECTION, generations, live_generation, metadata
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_qa import vectorstore
from rag_qa.manifest import (
    STORE,
    DocumentRecord,
    EmbedderRecord,
    Manifest,
    SplitterRecord,
    chunk_id,
)
from rag_qa.vectorstore import (
    NO_INDEX,
    QDRANT_HINT,
    Embedded,
    GenerationGoneError,
    QdrantStore,
    StoreError,
    StoreUnavailableError,
    in_the_way,
    live_index,
    newest_complete,
    open_generation,
    recover,
    retriever,
    search,
    write_generation,
)

DIMENSION = 3


def make_store(client: QdrantClient, collection: str = COLLECTION) -> QdrantStore:
    return QdrantStore(url="http://qdrant.test:6333", collection=collection, client=client)


def doc(name: str, chunks: int, sha256: str = "a" * 64) -> DocumentRecord:
    return DocumentRecord(
        sha256=sha256,
        loader="components.loaders.txt",
        loader_identity="l",
        splitter="s",
        chunks=chunks,
        ingested_at="2026-10-17T00:00:00Z",
    )


def manifest(**documents: DocumentRecord) -> Manifest:
    return Manifest(
        generation="",
        chunking=1,
        embedder=EmbedderRecord(ref="components.embedders.fake", identity="e", dimension=DIMENSION),
        splitter=SplitterRecord(ref="components.splitters.split", identity="s"),
        sparse=None,
        store=STORE,
        documents=documents,
    )


def chunks_of(name: str, chunks: int, sha256: str = "a" * 64, tag: str = "") -> Embedded:
    """``name``'s chunks as ingestion makes them: derived IDs, text, metadata, a vector each."""
    ids = [chunk_id(name, sha256, i) for i in range(chunks)]
    return Embedded(
        ids=ids,
        texts=[f"{name} chunk {i}{tag}" for i in range(chunks)],
        metadatas=[{"source": name, "page": i, "page_label": str(i + 1)} for i in range(chunks)],
        vectors=[[1.0, float(i + 1), 0.5] for i in range(chunks)],
    )


def merged(*parts: Embedded) -> Embedded:
    return Embedded(
        ids=[i for p in parts for i in p.ids],
        texts=[t for p in parts for t in p.texts],
        metadatas=[m for p in parts for m in p.metadatas],
        vectors=[v for p in parts for v in p.vectors],
    )


def publish(store: QdrantStore, **documents: int) -> str:
    """A full rebuild holding ``documents`` (name → chunk count); returns its generation."""
    add = merged(*(chunks_of(name, n) for name, n in documents.items()))
    records = {name: doc(name, n) for name, n in documents.items()}
    return write_generation(store, manifest(**records), base=None, delete=[], add=add).generation


def point_ids(client: QdrantClient, generation: str) -> set[str]:
    points, _ = client.scroll(generation, limit=10_000)
    return {str(p.id) for p in points}


# ---- names ---------------------------------------------------------------------------------


def test_a_generation_is_named_after_its_alias_and_sorts_by_its_stamp(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    first = publish(store, **{"a.txt": 1})
    second = publish(store, **{"a.txt": 1})
    assert vectorstore.generation_pattern(store).fullmatch(first)
    assert first.startswith("rag_qa--gen-") and first < second


def test_a_new_generation_sorts_after_the_live_one_even_if_the_clock_stepped_back(
    qdrant: QdrantClient,
) -> None:
    # Recovery keeps "the newest generation older than the live one" by name, so a name
    # that sorted first would cost the previous generation its place (S2-6's first review).
    store = make_store(qdrant)
    future = f"{COLLECTION}--gen-29991231T235959999999Z-ffffffff"
    stamp = vectorstore._next_stamp(store, future)
    assert stamp == "30000101T000000000000Z"  # one microsecond after the live one
    assert f"{COLLECTION}--gen-{stamp}-00000000" > future
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    assert vectorstore._next_stamp(store, None) >= now


def test_a_collection_name_with_a_hyphen_is_refused(qdrant: QdrantClient) -> None:
    # A hyphen would let one alias's generations be read as another's (rag--gen-…).
    with pytest.raises(ValueError, match="1 to 48 lowercase letters"):
        make_store(qdrant, collection="rag-qa")


# ---- writing and publishing ----------------------------------------------------------------


def test_the_first_publish_switches_the_alias_to_a_complete_generation(
    qdrant: QdrantClient,
) -> None:
    store = make_store(qdrant)
    assert store.alias_target() is None
    generation = publish(store, **{"a.txt": 2, "b.txt": 1})
    assert live_generation(qdrant) == generation
    held = metadata(qdrant, generation)
    assert held["complete"] is True
    assert held["manifest"]["generation"] == generation  # the manifest names itself
    assert qdrant.count(generation, exact=True).count == 3
    found = live_index(store)
    assert isinstance(found, tuple) and found[0] == generation
    assert found[1].documents["a.txt"].chunks == 2


def test_an_update_copies_the_live_generation_without_the_deleted_points(
    qdrant: QdrantClient,
) -> None:
    store = make_store(qdrant)
    live = publish(store, **{"keep.txt": 2, "gone.txt": 1, "edit.txt": 1})
    before = point_ids(qdrant, live)
    live_manifest = live_index(store)
    assert isinstance(live_manifest, tuple)
    old = live_manifest[1]
    # edit.txt changes its loader but not its bytes: the same chunk IDs come back, new text.
    replaced = chunks_of("edit.txt", 1, tag=" (relabelled)")
    added = chunks_of("new.txt", 2)
    documents = {
        "keep.txt": old.documents["keep.txt"],
        "edit.txt": doc("edit.txt", 1),
        "new.txt": doc("new.txt", 2),
    }
    delete = [*old.chunk_ids_of("gone.txt"), *old.chunk_ids_of("edit.txt")]
    published = write_generation(
        store,
        manifest(**documents),
        base=open_generation(store, live),
        delete=delete,
        add=merged(replaced, added),
    ).generation

    assert live_generation(qdrant) == published
    assert point_ids(qdrant, published) == {
        *old.chunk_ids_of("keep.txt"),
        *replaced.ids,
        *added.ids,
    }
    # The live generation was only ever read.
    assert point_ids(qdrant, live) == before
    assert metadata(qdrant, live)["manifest"]["documents"].keys() == {
        "keep.txt",
        "gone.txt",
        "edit.txt",
    }
    texts = {
        str(p.id): (p.payload or {})["page_content"]
        for p in qdrant.scroll(published, limit=100, with_payload=True)[0]
    }
    assert texts[replaced.ids[0]] == "edit.txt chunk 0 (relabelled)"


def test_a_copy_keeps_each_point_s_vector_and_payload(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    live = publish(store, **{"a.txt": 3})
    copied = write_generation(
        store,
        manifest(**{"a.txt": doc("a.txt", 3)}),
        base=open_generation(store, live),
        delete=[],
        add=Embedded(ids=[], texts=[], metadatas=[], vectors=[]),
    ).generation

    def points(generation: str) -> dict[str, Any]:
        found, _ = qdrant.scroll(generation, limit=100, with_payload=True, with_vectors=True)
        return {str(p.id): (p.vector, p.payload) for p in found}

    before, after = points(live), points(copied)
    assert after.keys() == before.keys()
    for pid, (vector, payload) in after.items():
        assert payload == before[pid][1]
        # Local mode hands back float64 once and float32 after a copy; the server copies
        # bit for bit, which test_qdrant_server.py checks.
        assert vector["dense"] == pytest.approx(before[pid][0]["dense"], rel=1e-6)


def test_a_new_index_needs_a_chunk(qdrant: QdrantClient) -> None:
    empty = Embedded(ids=[], texts=[], metadatas=[], vectors=[])
    with pytest.raises(ValueError, match="at least one chunk"):
        write_generation(make_store(qdrant), manifest(), base=None, delete=[], add=empty)
    assert generations(qdrant) == []


def test_a_count_that_disagrees_with_the_manifest_publishes_nothing(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    live = publish(store, **{"a.txt": 1})
    with pytest.raises(StoreError, match="holds 1 chunks, but its manifest lists 5"):
        write_generation(
            store,
            manifest(**{"b.txt": doc("b.txt", 5)}),
            base=None,
            delete=[],
            add=chunks_of("b.txt", 1),
        )
    assert live_generation(qdrant) == live
    orphan = max(generations(qdrant))
    assert orphan != live and metadata(qdrant, orphan)["complete"] is False
    assert recover(store) == [orphan]


@pytest.mark.parametrize(
    "step",
    ["create_collection", "upsert", "count", "update_collection", "update_collection_aliases"],
)
def test_a_crash_at_any_step_leaves_the_live_generation_and_one_recovery_cleans_up(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    # The hermetic half of story step 11: the moment between the new collection and the
    # switch lasts a second or two, too short to hit by hand. An exception at each request
    # of write_generation stands in for a crash there: nothing is published, the live
    # generation is whole, and the next run's recovery removes what was left.
    store = make_store(qdrant)
    previous = publish(store, **{"a.txt": 1})
    live = publish(store, **{"a.txt": 2})
    real = getattr(qdrant, step)

    def crash(*args: Any, **kwargs: Any) -> Any:
        if step == "create_collection":
            real(*args, **kwargs)  # the collection exists; the process dies right after
        raise RuntimeError(f"killed during {step}")

    monkeypatch.setattr(qdrant, step, crash)
    with pytest.raises(RuntimeError, match="killed"):
        publish(store, **{"a.txt": 3})
    monkeypatch.setattr(qdrant, step, real)

    assert live_generation(qdrant) == live
    assert qdrant.count(live, exact=True).count == 2
    left = set(generations(qdrant)) - {previous, live}
    assert len(left) == 1  # the run's own, incomplete
    assert recover(store) == sorted(left)
    assert generations(qdrant) == [previous, live]


# ---- recovery --------------------------------------------------------------------------------


def test_recovery_after_each_switch_leaves_the_live_and_the_previous_generation(
    qdrant: QdrantClient,
) -> None:
    # write_generation prunes nothing itself; ingestion calls recover() after each switch.
    store = make_store(qdrant)
    names = []
    for _ in range(4):
        names.append(publish(store, **{"a.txt": 1}))
        recover(store)
    assert generations(qdrant) == names[-2:]
    assert live_generation(qdrant) == names[-1]


def test_recovery_deletes_older_generations_and_newer_orphans(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    oldest, previous, live = (publish(store, **{"a.txt": 1}) for _ in range(3))
    orphan = f"{COLLECTION}--gen-29991231T235959999999Z-0badf00d"  # a run killed pre-switch
    qdrant.create_collection(orphan, vectors_config={})
    assert recover(store) == sorted([oldest, orphan])
    assert generations(qdrant) == [previous, live]


def test_with_no_alias_recovery_keeps_every_complete_generation(qdrant: QdrantClient) -> None:
    # Design review 2: deleting the alias's target deletes the alias with it. "Keep the
    # target and the newest older one" would then delete every generation, the previous
    # one included, which is by then the last usable copy.
    store = make_store(qdrant)
    older, previous, live = (publish(store, **{"a.txt": 1}) for _ in range(3))
    incomplete = f"{COLLECTION}--gen-29991231T235959999999Z-0badf00d"
    qdrant.create_collection(incomplete, vectors_config={}, metadata={"complete": False})
    qdrant.delete_collection(live)  # by hand, or a drill
    assert store.alias_target() is None
    assert recover(store) == [incomplete]
    assert generations(qdrant) == [older, previous]
    assert newest_complete(store) == previous


def test_a_generation_whose_manifest_names_another_collection_is_not_complete(
    qdrant: QdrantClient,
) -> None:
    # A copy restored under a new name carries the old manifest (DEC-25): not complete
    # until its manifest is rewritten, so a no-alias recovery removes it.
    store = make_store(qdrant)
    real = publish(store, **{"a.txt": 1})
    copy = f"{COLLECTION}--gen-29991231T235959999999Z-c0b1ed00"
    qdrant.create_collection(copy, vectors_config={}, metadata=metadata(qdrant, real))
    qdrant.delete_collection(real)
    assert recover(store) == [copy]


def test_with_an_alias_elsewhere_recovery_touches_nothing(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    first, second = (publish(store, **{"a.txt": 1}) for _ in range(2))
    qdrant.create_collection("someone_elses", vectors_config={})
    qdrant.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=COLLECTION)),
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name="someone_elses", alias_name=COLLECTION
                )
            ),
        ]
    )
    assert recover(store) == []
    assert generations(qdrant) == [first, second]


def test_recovery_leaves_other_collections_and_other_aliases_generations_alone(
    qdrant: QdrantClient,
) -> None:
    store, scratch = make_store(qdrant), make_store(qdrant, "scratch")
    ours = publish(store, **{"a.txt": 1})
    theirs = [publish(scratch, **{"a.txt": 1}) for _ in range(3)]
    for name in ("rag_qa_other", f"{COLLECTION}--gen-not-a-stamp", "rag_qa--gen-x"):
        qdrant.create_collection(name, vectors_config={})
    assert recover(store) == []
    assert generations(qdrant, "scratch") == theirs
    assert {c.name for c in qdrant.get_collections().collections} >= {ours, "rag_qa_other"}


def test_recovery_with_nothing_there_does_nothing(qdrant: QdrantClient) -> None:
    assert recover(make_store(qdrant)) == []


# ---- what is in the way, and what is published ----------------------------------------------


def test_nothing_is_in_the_way_of_a_new_or_a_published_alias(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    assert in_the_way(store) is None
    publish(store, **{"a.txt": 1})
    assert in_the_way(store) is None


def test_a_real_collection_named_like_the_alias_is_in_the_way(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    qdrant.create_collection(COLLECTION, vectors_config={})
    blocked = in_the_way(store)
    assert blocked is not None and "is a collection of its own" in blocked
    assert live_index(store) == f"{store.where} {blocked}"


def test_an_alias_of_anything_but_a_generation_is_in_the_way(qdrant: QdrantClient) -> None:
    qdrant.create_collection("hand_made", vectors_config={})
    qdrant.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name="hand_made", alias_name=COLLECTION)
            )
        ]
    )
    blocked = in_the_way(make_store(qdrant))
    assert blocked == "is an alias of 'hand_made', which is not one of its generations"


def test_live_index_says_why_there_is_no_index_to_use(qdrant: QdrantClient) -> None:
    # The one rule rag-ingest and the query side share.
    store = make_store(qdrant)
    assert live_index(store) == NO_INDEX
    live = publish(store, **{"a.txt": 1})
    qdrant.update_collection(live, metadata={"manifest": {"version": 1}})
    found = live_index(store)
    assert isinstance(found, str) and "unknown manifest version 1" in found
    qdrant.update_collection(live, metadata={"complete": False})
    found = live_index(store)
    assert isinstance(found, str) and f"its live generation {live} cannot be used" in found
    assert "incomplete" in found


def test_a_generation_gone_between_the_alias_and_its_metadata_is_looked_up_once_more(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two switches between the two requests prune the generation the alias first named.
    store = make_store(qdrant)
    first = publish(store, **{"a.txt": 1})
    second = publish(store, **{"a.txt": 1})
    names = iter([first, second])
    monkeypatch.setattr(store, "alias_target", lambda: next(names))
    qdrant.delete_collection(first)  # the target read first is gone by its metadata read
    found = live_index(store)
    assert isinstance(found, tuple) and found[0] == second


def test_an_alias_that_keeps_moving_is_reported_not_looped_on(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(qdrant)
    gone = f"{COLLECTION}--gen-20990101T000000000000Z-deadbeef"  # named right, never made
    monkeypatch.setattr(store, "alias_target", lambda: gone)
    assert live_index(store) == "the alias moved twice while it was read; try again"


def test_a_server_error_without_a_json_body_gives_its_reason() -> None:
    error = UnexpectedResponse(
        status_code=502, reason_phrase="Bad Gateway", content=b"<html>", headers={}
    )  # type: ignore[arg-type]
    store = QdrantStore(url="http://q", collection="rag_qa", client=Refusing(error))  # type: ignore[arg-type]
    with pytest.raises(StoreError, match="answered 502: Bad Gateway"):
        store.alias_target()


def test_newest_complete_is_none_with_no_generation(qdrant: QdrantClient) -> None:
    assert newest_complete(make_store(qdrant)) is None


# ---- reading: the cut rule and the documents ------------------------------------------------


def tied_collection(client: QdrantClient, name: str, order: list[int]) -> list[str]:
    """``name`` holding 30 points that tie exactly for any query, inserted in ``order``,
    and 5 nearer ones; returns the tied points' IDs."""
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    ids = [chunk_id("tie.txt", "0" * 64, i) for i in range(30)]
    near = [chunk_id("near.txt", "0" * 64, i) for i in range(5)]
    points = [
        models.PointStruct(id=ids[i], vector={"dense": [1.0, 1.0]}, payload={"i": i}) for i in order
    ]
    points += [
        models.PointStruct(id=pid, vector={"dense": [1.0, 0.01 * n]}, payload={})
        for n, pid in enumerate(near)
    ]
    client.upsert(name, points=points, wait=True)
    return ids


@pytest.mark.parametrize("k", [6, 10, 20])
def test_points_tied_at_the_cut_give_the_same_result_whatever_the_insertion_order(
    qdrant: QdrantClient, k: int
) -> None:
    # Design review 2: the server kept a different set of tied points per insertion order.
    # Our cut decides ties by point ID, so every order gives the same k.
    store = make_store(qdrant)
    orders = [list(range(30)), list(reversed(range(30))), [(i * 7) % 30 for i in range(30)]]
    results = []
    for n, order in enumerate(orders):
        ids = tied_collection(qdrant, f"ties_{n}", order)
        found = search(store, f"ties_{n}", [1.0, 0.0], k)
        results.append([str(p.id) for p in found])
    assert results[0] == results[1] == results[2]
    tied = sorted(ids)
    assert results[0][5:] == tied[: k - 5]  # the 5 nearer ones first, then ties by ID


def test_the_search_widens_its_request_while_the_last_point_ties_the_cut(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(qdrant)
    tied_collection(qdrant, "ties", list(range(30)))
    limits: list[int] = []
    real = qdrant.query_points

    def spy(*args: Any, **kwargs: Any) -> Any:
        limits.append(kwargs["limit"])
        return real(*args, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", spy)
    search(store, "ties", [1.0, 0.0], 8)
    # 8 + 10 = 18 fetched; the 18th ties the 8th, so 36, which is more than the 35 points.
    assert limits == [18, 36]
    limits.clear()
    search(store, "ties", [1.0, 0.0], 3)  # the cut falls among the 5 nearer: no tie
    assert limits == [13]


class FixedEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0, 0.5]


def test_the_retriever_returns_documents_as_faiss_did(qdrant: QdrantClient) -> None:
    # The same page_content, metadata and id: no score or collection keys, which the
    # prompt and the sources read (DEC-19).
    store = make_store(qdrant)
    generation = publish(store, **{"a.txt": 3})
    found = retriever(
        store, generation, FixedEmbeddings(), {"search_kwargs": {"k": 2}}, k=2
    ).invoke("q")
    assert [d.metadata for d in found] == [
        {"source": "a.txt", "page": 0, "page_label": "1"},
        {"source": "a.txt", "page": 1, "page_label": "2"},
    ]
    assert [d.page_content for d in found] == ["a.txt chunk 0", "a.txt chunk 1"]
    assert [d.id for d in found] == [chunk_id("a.txt", "a" * 64, i) for i in range(2)]


def test_a_search_of_a_deleted_generation_says_it_is_gone(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    old = publish(store, **{"a.txt": 1})
    publish(store, **{"a.txt": 1})
    qdrant.delete_collection(old)
    with pytest.raises(GenerationGoneError, match=old):
        search(store, old, [1.0, 1.0, 0.5], 1)


def test_the_retriever_refuses_a_search_it_does_not_run(qdrant: QdrantClient) -> None:
    with pytest.raises(ValueError, match="'mmr' is not supported"):
        retriever(make_store(qdrant), "g", FixedEmbeddings(), {"search_type": "mmr"}, k=1)


def test_open_generation_lists_its_point_ids(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    generation = publish(store, **{"a.txt": 300})  # more than one scroll page
    assert open_generation(store, generation).ids == {
        chunk_id("a.txt", "a" * 64, i) for i in range(300)
    }


# ---- the store, its settings and its errors -------------------------------------------------


def test_the_environment_overrides_the_url_and_the_collection_and_holds_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def connect(url: str, api_key: SecretStr | None, timeout_s: float) -> QdrantClient:
        seen.update(url=url, key=api_key, timeout=timeout_s)
        return QdrantClient(":memory:")

    monkeypatch.setattr(vectorstore, "connect", connect)
    monkeypatch.setenv("RAG_QDRANT_URL", "http://elsewhere:6333")
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "scratch_1")
    monkeypatch.setenv("RAG_QDRANT_API_KEY", "s3cret-key")
    store = QdrantStore(url="http://localhost:6333", collection="rag_qa", timeout_s=4)
    assert (store.url, store.collection) == ("http://elsewhere:6333", "scratch_1")
    assert seen["url"] == "http://elsewhere:6333" and seen["timeout"] == 4
    assert seen["key"].get_secret_value() == "s3cret-key"
    assert "s3cret" not in repr(store) and "s3cret" not in store.where


def test_a_non_positive_timeout_is_refused(qdrant: QdrantClient) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        QdrantStore(url="http://x", collection="rag_qa", timeout_s=0, client=qdrant)


class Refusing:
    """A client whose every call raises ``error``: the server's answers, without one."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            raise self.error

        return call


def refused_with(status: int, error: str = "boom") -> UnexpectedResponse:
    body = json.dumps({"status": {"error": error}}).encode()
    return UnexpectedResponse(status_code=status, reason_phrase="x", content=body, headers={})  # type: ignore[arg-type]


def test_an_unreachable_server_is_unavailable_with_the_hint() -> None:
    refused = ResponseHandlingException(ConnectionRefusedError(111, "Connection refused"))
    store = QdrantStore(url="http://127.0.0.1:6333", collection="rag_qa", client=Refusing(refused))  # type: ignore[arg-type]
    with pytest.raises(StoreUnavailableError) as exc:
        store.check()
    assert "cannot use Qdrant at http://127.0.0.1:6333" in str(exc.value)
    assert QDRANT_HINT in str(exc.value)


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_is_unavailable_and_says_which_setting(status: int) -> None:
    store = QdrantStore(url="http://q", collection="rag_qa", client=Refusing(refused_with(status)))  # type: ignore[arg-type]
    with pytest.raises(
        StoreUnavailableError, match=f"refused the request \\({status}\\): check RAG_QDRANT_API_KEY"
    ):
        store.alias_target()


def test_any_other_answer_is_a_store_error_with_the_server_s_words() -> None:
    store = QdrantStore(
        url="http://q", collection="rag_qa", client=Refusing(refused_with(500, "disk full"))
    )  # type: ignore[arg-type]
    with pytest.raises(StoreError, match="answered 500: disk full") as exc:
        store.alias_target()
    assert not isinstance(exc.value, StoreUnavailableError)


def test_a_404_on_a_search_is_gone() -> None:
    store = QdrantStore(url="http://q", collection="rag_qa", client=Refusing(refused_with(404)))  # type: ignore[arg-type]
    with pytest.raises(GenerationGoneError, match="rag_qa--gen-x is gone"):
        search(store, "rag_qa--gen-x", [1.0], 1)


def test_check_refuses_a_server_too_old_for_collection_metadata(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        qdrant, "info", lambda: models.VersionInfo(title="qdrant", version="1.15.4")
    )
    with pytest.raises(StoreUnavailableError, match="version 1.15.4, but the index needs 1.16"):
        make_store(qdrant).check()


def test_check_passes_a_current_server(qdrant: QdrantClient) -> None:
    make_store(qdrant).check()  # local mode reports the client's version, 1.19.1
