"""The Qdrant store (S3-2, DEC-19), hermetically, on the in-memory client.

Generations, the alias switch, recovery and the cut rule, each against the module's own
functions. What only the server can show (its atomic alias batch, its 404, ties decided by
a collection's layout) is in ``test_qdrant_server.py``, under the ``qdrant_server`` marker.

The retries, the breaker and the deadline (S3-6) run the real client over a fake transport,
:class:`FakeQdrant`, and a hung server is a real socket that never answers.
"""

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from conftest import COLLECTION, generations, live_generation, metadata
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag_qa import resilience, vectorstore
from rag_qa.api.state import serialized
from rag_qa.manifest import (
    STORE,
    DocumentRecord,
    EmbedderRecord,
    Manifest,
    SparseRecord,
    SplitterRecord,
    chunk_id,
)
from rag_qa.resilience import QDRANT_ATTEMPTS, CircuitBreaker
from rag_qa.sparse import SPARSE_VERSION, Bm25Encoder, SparseVector
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


LOST = ResponseHandlingException(TimeoutError("timed out"))  # the client's read timeout


def test_a_switch_the_server_applied_but_whose_answer_was_lost_is_published(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S3-2's second review: the alias moved, so the run published, and must not raise
    # StoreUnavailableError, which every command reports as "published nothing", exit 2.
    store = make_store(qdrant)
    live = publish(store, **{"a.txt": 1})
    real = qdrant.update_collection_aliases

    def applied_then_lost(*args: Any, **kwargs: Any) -> Any:
        real(*args, **kwargs)
        raise LOST

    monkeypatch.setattr(qdrant, "update_collection_aliases", applied_then_lost)
    published = publish(store, **{"a.txt": 2})
    assert published != live
    assert live_generation(qdrant) == published


@pytest.mark.parametrize("read_back", ["answers", "fails too"])
def test_a_switch_that_failed_and_was_not_applied_still_fails(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch, read_back: str
) -> None:
    store = make_store(qdrant)
    live = publish(store, **{"a.txt": 1})
    real_get_aliases = qdrant.get_aliases
    switches = []

    def lost(*args: Any, **kwargs: Any) -> Any:
        switches.append(1)
        if read_back == "fails too":  # from here on: write_generation read the alias first
            monkeypatch.setattr(qdrant, "get_aliases", Refusing(LOST).call)
        raise LOST

    monkeypatch.setattr(qdrant, "update_collection_aliases", lost)
    with pytest.raises(StoreUnavailableError, match=r"did not answer in time \(TimeoutError\)"):
        publish(store, **{"a.txt": 2})
    # It failed at the switch, not before, and the switch was retried as every request is
    # (S3-6): a repeat of an applied switch is a no-op.
    assert switches == [1] * QDRANT_ATTEMPTS
    monkeypatch.setattr(qdrant, "get_aliases", real_get_aliases)
    assert live_generation(qdrant) == live


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


def test_every_search_asks_the_server_for_an_exact_scan(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DEC-19 and the manifest's store.exact promise exact search, but local mode ignores
    # the flag and a small collection is scanned in full whatever it says. Only the request
    # shows it: past the server's indexing threshold, exact=False would be approximate
    # search, and nothing else would notice (S3-2 review).
    store = make_store(qdrant)
    tied_collection(qdrant, "ties", list(range(30)))
    asked: list[Any] = []
    real = qdrant.query_points

    def spy(*args: Any, **kwargs: Any) -> Any:
        asked.append(kwargs.get("search_params"))
        return real(*args, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", spy)
    search(store, "ties", [1.0, 0.0], 8)  # widens: two requests
    search(store, "ties", [1.0, 0.0], 3)
    assert len(asked) == 3
    assert all(params is not None and params.exact is True for params in asked)


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


# ---- hybrid retrieval and the sources filter (S3-5, DEC-20) -------------------------------

ENCODER = Bm25Encoder(k1=1.2, b=0.75, avg_len=3.0)
SPARSE_RECORD = SparseRecord(
    ref="components.sparse_encoders.bm25", identity="b", version=SPARSE_VERSION
)
#: A hybrid retriever's settings, as RetrieverSpec.kwargs() dumps them.
HYBRID = {"mode": "hybrid", "search_kwargs": {"k": 2}, "prefetch": 10, "rrf_k": 60}


def texts_of(name: str, texts: list[str]) -> Embedded:
    """``name``'s chunks with these texts, ranked by the dense query of
    :class:`FixedEmbeddings` in their order, and BM25 vectors of their texts."""
    plain = chunks_of(name, len(texts))
    return Embedded(
        ids=plain.ids,
        texts=texts,
        metadatas=plain.metadatas,
        vectors=plain.vectors,
        sparse=ENCODER.encode_documents(texts),
    )


def publish_hybrid(store: QdrantStore, **documents: list[str]) -> str:
    """A full rebuild with sparse vectors, of ``documents`` (name → chunk texts)."""
    parts = [texts_of(name, texts) for name, texts in documents.items()]
    add = Embedded(
        ids=[i for p in parts for i in p.ids],
        texts=[t for p in parts for t in p.texts],
        metadatas=[m for p in parts for m in p.metadatas],
        vectors=[v for p in parts for v in p.vectors],
        sparse=[s for p in parts for s in p.sparse or []],
    )
    records = {name: doc(name, len(texts)) for name, texts in documents.items()}
    hybrid = manifest(**records).model_copy(update={"sparse": SPARSE_RECORD})
    return write_generation(store, hybrid, base=None, delete=[], add=add).generation


def point(pid: str, payload: str = "") -> Any:
    return SimpleNamespace(id=pid, payload={"from": payload}, score=0.0)


def test_rrf_sums_one_over_rrf_k_plus_the_one_based_rank_over_the_lists() -> None:
    dense = [point("a"), point("b"), point("c")]
    sparse = [point("c"), point("d")]
    fused = vectorstore.fuse([dense, sparse], rrf_k=60, limit=10)
    scores = {"a": 1 / 61, "b": 1 / 62, "c": 1 / 63 + 1 / 61, "d": 1 / 62}
    assert [p.id for p in fused] == sorted(scores, key=lambda pid: (-scores[pid], pid))
    assert [p.id for p in fused] == ["c", "a", "b", "d"]  # b and d tie: by point ID


def test_a_fused_tie_is_ordered_by_point_id_and_cut_where_it_falls() -> None:
    # Ties are the rule in RRF: rank r of either list scores the same (DEC-20).
    dense = [point("x"), point("m")]
    sparse = [point("y"), point("b")]
    assert [p.id for p in vectorstore.fuse([dense, sparse], rrf_k=60, limit=3)] == ["x", "y", "b"]
    assert [p.id for p in vectorstore.fuse([sparse, dense], rrf_k=60, limit=3)] == ["x", "y", "b"]


def test_rrf_k_and_the_weights_are_the_caller_s() -> None:
    dense = [point("a"), point("b")]
    sparse = [point("b"), point("a")]
    # Equal weights: a tie, by ID. A heavier sparse list puts its own first first.
    assert [p.id for p in vectorstore.fuse([dense, sparse], rrf_k=1, limit=2)] == ["a", "b"]
    weighted = vectorstore.fuse([dense, sparse], rrf_k=1, limit=2, weights=[1.0, 2.0])
    assert [p.id for p in weighted] == ["b", "a"]
    with pytest.raises(ValueError, match="2 lists to fuse, but 1 weights"):
        vectorstore.fuse([dense, sparse], rrf_k=1, limit=2, weights=[1.0])


def test_a_fused_point_keeps_the_payload_of_the_first_list_that_holds_it() -> None:
    fused = vectorstore.fuse([[point("a", "dense")], [point("a", "sparse")]], rrf_k=60, limit=1)
    assert fused[0].payload == {"from": "dense"}
    assert vectorstore.fuse([[], []], rrf_k=60, limit=5) == []


def test_a_hybrid_generation_holds_both_vectors_with_qdrant_s_idf(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    generation = publish_hybrid(store, **{"a.txt": ["glider small", "zebra stripes"]})
    params = qdrant.get_collection(generation).config.params
    assert params.sparse_vectors is not None
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
    found, _ = qdrant.scroll(generation, limit=10, with_vectors=True)
    by_id = {str(p.id): p.vector for p in found}
    first = by_id[chunk_id("a.txt", "a" * 64, 0)]
    expected = ENCODER.encode_document("glider small")
    assert first["bm25"].indices == list(expected.indices)
    assert first["bm25"].values == pytest.approx(list(expected.values))
    assert metadata(qdrant, generation)["manifest"]["sparse"] == SPARSE_RECORD.model_dump()


def test_a_generation_without_a_sparse_record_has_no_sparse_vector(qdrant: QdrantClient) -> None:
    generation = publish(make_store(qdrant), **{"a.txt": 1})
    assert not qdrant.get_collection(generation).config.params.sparse_vectors


def test_sparse_vectors_come_with_a_sparse_record_and_only_with_one(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    with_sparse = texts_of("a.txt", ["glider"])
    without = chunks_of("a.txt", 1)
    hybrid = manifest(**{"a.txt": doc("a.txt", 1)}).model_copy(update={"sparse": SPARSE_RECORD})
    for record, add in ((manifest(**{"a.txt": doc("a.txt", 1)}), with_sparse), (hybrid, without)):
        with pytest.raises(ValueError, match="sparse vectors must come with a sparse record"):
            write_generation(store, record, base=None, delete=[], add=add)
    assert generations(qdrant) == []  # refused before anything was created


def test_an_update_of_a_hybrid_index_copies_the_sparse_vectors_too(qdrant: QdrantClient) -> None:
    store = make_store(qdrant)
    live = publish_hybrid(store, **{"a.txt": ["glider small", "zebra stripes"]})
    added = texts_of("b.txt", ["kestrel island"])
    hybrid = manifest(**{"a.txt": doc("a.txt", 2), "b.txt": doc("b.txt", 1)}).model_copy(
        update={"sparse": SPARSE_RECORD}
    )
    copied = write_generation(
        store, hybrid, base=open_generation(store, live), delete=[], add=added
    ).generation

    def sparse_of(generation: str) -> dict[str, Any]:
        found, _ = qdrant.scroll(generation, limit=10, with_vectors=True)
        return {str(p.id): p.vector["bm25"] for p in found}

    before, after = sparse_of(live), sparse_of(copied)
    assert after.keys() == {*before, *added.ids}
    for pid, vector in before.items():
        assert after[pid].indices == vector.indices
        assert after[pid].values == pytest.approx(vector.values)


def test_a_hybrid_retriever_finds_the_word_the_embedding_ranks_low(qdrant: QdrantClient) -> None:
    # FixedEmbeddings ranks the chunks in order, so dense k=2 keeps the first two. "zebra"
    # is only in the fifth: the sparse search ranks it first, and the fusion lifts it.
    store = make_store(qdrant)
    texts = ["alpha one", "beta two", "gamma three", "delta four", "zebra five"]
    generation = publish_hybrid(store, **{"a.txt": texts})
    dense = retriever(store, generation, FixedEmbeddings(), {"search_kwargs": {"k": 2}}, k=2)
    assert [d.page_content for d in dense.invoke("zebra")] == ["alpha one", "beta two"]
    hybrid = retriever(store, generation, FixedEmbeddings(), HYBRID, sparse=ENCODER, k=2)
    # zebra: 1/61 (sparse) + 1/65 (dense rank 5); alpha: 1/61 (dense rank 1).
    found = hybrid.invoke("zebra")
    assert [d.page_content for d in found] == ["zebra five", "alpha one"]
    assert found[0].metadata == {"source": "a.txt", "page": 4, "page_label": "5"}  # as dense


def test_a_hybrid_retriever_needs_the_encoder(qdrant: QdrantClient) -> None:
    with pytest.raises(ValueError, match="needs the sparse encoder"):
        retriever(make_store(qdrant), "g", FixedEmbeddings(), HYBRID, k=2)


def test_a_question_of_stopwords_alone_searches_dense_only(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty sparse query finds nothing, so it is not sent (S3-5's plan, D7).
    store = make_store(qdrant)
    generation = publish_hybrid(store, **{"a.txt": ["alpha one", "beta two", "gamma three"]})
    used: list[str] = []
    real = qdrant.query_points

    def spy(*args: Any, **kwargs: Any) -> Any:
        used.append(kwargs["using"])
        return real(*args, **kwargs)

    monkeypatch.setattr(qdrant, "query_points", spy)
    hybrid = retriever(store, generation, FixedEmbeddings(), HYBRID, sparse=ENCODER, k=2)
    assert [d.page_content for d in hybrid.invoke("what is it?")] == ["alpha one", "beta two"]
    assert used == ["dense"]
    used.clear()
    hybrid.invoke("gamma")
    assert used == ["dense", "bm25"]


@pytest.mark.parametrize("hybrid", [False, True], ids=["dense", "hybrid"])
def test_sources_limit_every_search_to_those_documents(qdrant: QdrantClient, hybrid: bool) -> None:
    store = make_store(qdrant)
    generation = publish_hybrid(
        store,
        **{"a.txt": ["alpha one", "zebra a"], "b.txt": ["beta two", "zebra b"], "c.txt": ["c"]},
    )
    spec = HYBRID if hybrid else {"search_kwargs": {"k": 2}}
    found = retriever(store, generation, FixedEmbeddings(), spec, sparse=ENCODER, k=2)
    # Unfiltered, each document's first chunk ties on the dense vector, so the two kept
    # come from two documents.
    assert len({d.metadata["source"] for d in found.invoke("zebra")}) == 2
    only_b = found.invoke("zebra", sources=("b.txt",))
    assert [d.metadata["source"] for d in only_b] == ["b.txt", "b.txt"]
    both = found.invoke("zebra", sources=("b.txt", "c.txt"))
    assert {d.metadata["source"] for d in both} <= {"b.txt", "c.txt"}
    assert found.invoke("zebra", sources=("gone.txt",)) == []


def test_the_filter_is_a_match_any_on_the_indexed_source() -> None:
    assert vectorstore.source_filter(()) is None
    flt = vectorstore.source_filter(["a.pdf", "sub/b.md"])
    assert flt == models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.source", match=models.MatchAny(any=["a.pdf", "sub/b.md"])
            )
        ]
    )


def sparse_tied_collection(client: QdrantClient, name: str, order: list[int]) -> list[str]:
    """``name`` with 30 points whose sparse vectors tie exactly for any query on index 1,
    inserted in ``order``, and 5 that score higher; returns the tied points' IDs."""
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=2, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    ids = [chunk_id("tie.txt", "0" * 64, i) for i in range(30)]
    near = [chunk_id("near.txt", "0" * 64, i) for i in range(5)]

    def vector(value: float) -> dict[str, Any]:
        return {"dense": [1.0, 1.0], "bm25": models.SparseVector(indices=[1], values=[value])}

    points = [models.PointStruct(id=ids[i], vector=vector(1.0), payload={}) for i in order]
    points += [
        models.PointStruct(id=pid, vector=vector(2.0 + n), payload={}) for n, pid in enumerate(near)
    ]
    client.upsert(name, points=points, wait=True)
    return ids


QUERY_1 = SparseVector(indices=(1,), values=(1.0,))


def test_a_sparse_search_is_cut_by_our_rule_too(
    qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cut rule holds in each branch (DEC-20): ties at the cut widen the request, and
    # the tied points are kept by point ID, whatever order they were inserted in.
    store = make_store(qdrant)
    limits: list[int] = []
    real = qdrant.query_points

    def spy(*args: Any, **kwargs: Any) -> Any:
        limits.append(kwargs["limit"])
        assert kwargs["using"] == "bm25" and kwargs["search_params"].exact is True
        return real(*args, **kwargs)

    results = []
    for n, order in enumerate([list(range(30)), list(reversed(range(30)))]):
        ids = sparse_tied_collection(qdrant, f"sparse_ties_{n}", order)
        monkeypatch.setattr(qdrant, "query_points", spy)
        results.append([str(p.id) for p in search(store, f"sparse_ties_{n}", QUERY_1, 8)])
        monkeypatch.setattr(qdrant, "query_points", real)
    assert limits == [18, 36, 18, 36]
    assert results[0] == results[1]
    assert results[0][5:] == sorted(ids)[:3]


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


@contextmanager
def answering(status: int, content_type: str, body: bytes) -> Iterator[str]:
    """A loopback HTTP server answering every GET with ``body``: what a wrong store URL
    reaches in place of Qdrant. Yields its URL."""

    class Answer(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:  # no request lines in the test output
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Answer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("status", "content_type", "body", "said"),
    [
        (404, "text/plain", b"Not Found", "answered 404: Not Found"),
        (200, "text/plain", b"Ollama is running", "does not answer as Qdrant does"),
        (200, "application/json", b'{"status": "ok"}', "does not answer as Qdrant does"),
    ],
    ids=["a-path-qdrant-does-not-serve", "another-service-on-the-port", "another-json-api"],
)
def test_check_refuses_whatever_answers_in_place_of_qdrant(
    status: int, content_type: str, body: bytes, said: str
) -> None:
    # S3-2's second review: a wrong RAG_QDRANT_URL that reaches another HTTP server gave a
    # traceback and exit 1 from rag-ingest, rag-query and rag-serve (a 404, a body that
    # is not JSON), or "Is Qdrant running?" while it ran (JSON of another shape, which the
    # client wraps). The real client, so the test sees what the library raises for each.
    # Each must be a store that cannot be used, exit 2, with the URL to check.
    with answering(status, content_type, body) as url:
        client = QdrantClient(url=url, timeout=5, check_compatibility=False)
        try:
            with pytest.raises(StoreUnavailableError) as exc:
                QdrantStore(url=url, collection="rag_qa", client=client).check()
        finally:
            client.close()
    assert said in str(exc.value)
    assert vectorstore.URL_HINT in str(exc.value)
    assert QDRANT_HINT not in str(exc.value)  # something answered: Qdrant may well run


# ---- retries, the breaker and the deadline (S3-6, NFR-6, DEC-21) ---------------------------


class FakeQdrant:
    """Qdrant's REST API, as far as these tests ask it, behind ``httpx.MockTransport``:
    qdrant-client builds its REST client on httpx and hands it extra arguments, so the real
    client, its parsing and its errors run. ``script`` holds what the next requests to a
    ``"METHOD /path"`` get instead of an answer: an exception to raise, or a status code.
    A ``(status, applied)`` step answers ``status`` after doing the work, a lost answer."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.collections: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.script: dict[str, list[Any]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        steps = self.script.get(key)
        if steps:
            step = steps.pop(0)
            if isinstance(step, tuple):  # applied, then the answer is lost
                self.answer(request)
                step = step[0]
            if isinstance(step, BaseException):
                raise step
            return httpx.Response(step, json={"status": {"error": "scripted"}, "time": 0})
        return self.answer(request)

    def answer(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        name = path.removeprefix("/collections/").split("/")[0]

        def ok(result: Any) -> httpx.Response:
            return httpx.Response(200, json={"result": result, "status": "ok", "time": 0})

        if (method, path) == ("GET", "/"):
            return httpx.Response(200, json={"title": "qdrant", "version": "1.19.2"})
        if (method, path) == ("GET", "/aliases"):
            aliases = [{"alias_name": a, "collection_name": c} for a, c in self.aliases.items()]
            return ok({"aliases": aliases})
        if (method, path) == ("GET", "/collections"):
            return ok({"collections": [{"name": c} for c in sorted(self.collections)]})
        if method == "PUT" and path == f"/collections/{name}":
            if name in self.collections:
                return httpx.Response(409, json={"status": {"error": "exists"}, "time": 0})
            self.collections.add(name)
            return ok(True)
        if method == "DELETE" and path == f"/collections/{name}":
            found = name in self.collections
            self.collections.discard(name)
            return ok(found)
        if method == "POST" and path.endswith("/points/query"):
            if name not in self.collections:
                return httpx.Response(404, json={"status": {"error": "Not found"}, "time": 0})
            return ok({"points": []})
        return httpx.Response(500, json={"status": {"error": f"no fake for {method} {path}"}})

    def sent(self, key: str) -> int:
        return sum(f"{r.method} {r.url.path}" == key for r in self.requests)


def fake_store(fake: FakeQdrant, **settings: Any) -> QdrantStore:
    client = QdrantClient(
        url="http://qdrant.test:6333",
        timeout=10,
        check_compatibility=False,
        transport=httpx.MockTransport(fake),
    )
    return QdrantStore(
        url="http://qdrant.test:6333", collection=COLLECTION, client=client, **settings
    )


REFUSED = httpx.ConnectError("All connection attempts failed")
SLOW = httpx.ReadTimeout("timed out")


def test_a_refused_connection_is_tried_again_and_then_answers() -> None:
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [REFUSED, REFUSED]
    store = fake_store(fake)
    assert store.alias_target() is None
    assert fake.sent("GET /aliases") == QDRANT_ATTEMPTS
    assert store.breaker.state == "closed"


@pytest.mark.parametrize("status", [500, 502, 503, 429])
def test_an_answer_that_it_cannot_serve_now_is_tried_again(status: int) -> None:
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [status]
    assert fake_store(fake).alias_target() is None
    assert fake.sent("GET /aliases") == 2


def test_a_refusal_is_not_tried_again_and_says_it_answered() -> None:
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [401, 401, 401]
    store = fake_store(fake)
    with pytest.raises(StoreUnavailableError, match=r"refused the request \(401\)"):
        store.alias_target()
    assert fake.sent("GET /aliases") == 1
    assert store.breaker._failed == 0  # a refusal is an answer: nothing counted against it


def test_a_search_of_a_generation_that_is_gone_is_never_repeated() -> None:
    fake = FakeQdrant()
    store = fake_store(fake)
    gone = "rag_qa--gen-20261022T000000000000Z-00000000"
    with pytest.raises(GenerationGoneError):
        search(store, gone, [0.1, 0.2, 0.3], 5)
    assert fake.sent(f"POST /collections/{gone}/points/query") == 1


def test_every_attempt_spent_says_qdrant_did_not_answer_in_time() -> None:
    # S3-3's backlog line: a read timeout with Qdrant up. The retries absorb one; when every
    # attempt times out, "Is Qdrant running?" would point the wrong way.
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [SLOW] * QDRANT_ATTEMPTS
    with pytest.raises(
        StoreUnavailableError, match=r"did not answer in time \(ReadTimeout\)"
    ) as exc:
        fake_store(fake).alias_target()
    assert QDRANT_HINT not in str(exc.value)


def test_three_failed_attempts_open_the_breaker_and_then_no_request_is_sent() -> None:
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [REFUSED] * QDRANT_ATTEMPTS
    store = fake_store(fake)
    with pytest.raises(StoreUnavailableError, match="cannot use Qdrant"):
        store.alias_target()
    assert store.breaker.state == "open"
    with pytest.raises(StoreUnavailableError) as exc:
        store.alias_target()
    assert fake.sent("GET /aliases") == QDRANT_ATTEMPTS  # refused before any request
    assert str(exc.value).startswith("Qdrant is not answering: 3 attempts in a row failed")
    assert QDRANT_HINT in str(exc.value)


def test_after_the_wait_one_probe_goes_through_and_its_answer_closes_the_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [1000.0]
    fake = FakeQdrant()
    fake.script["GET /aliases"] = [REFUSED] * QDRANT_ATTEMPTS
    store = fake_store(fake)
    store.breaker = CircuitBreaker("Qdrant", clock=lambda: now[0])
    with pytest.raises(StoreUnavailableError):
        store.alias_target()
    now[0] += 30
    assert store.alias_target() is None  # the probe
    assert store.breaker.state == "closed"
    assert fake.sent("GET /aliases") == QDRANT_ATTEMPTS + 1


def test_a_create_whose_answer_was_lost_is_not_sent_again() -> None:
    # The server created it, and the answer timed out: repeated, it would answer 409
    # (checked on 1.19.2). A retry that finds our uniquely named collection counts as done.
    fake = FakeQdrant()
    name = "rag_qa--gen-20261022T000000000000Z-00000000"
    fake.script[f"PUT /collections/{name}"] = [(SLOW,)]
    vectorstore._create(
        fake_store(fake), name, manifest(**{"a.txt": doc("a.txt", 1)}), hybrid=False
    )
    assert fake.sent(f"PUT /collections/{name}") == 1
    assert name in fake.collections


def test_a_delete_whose_answer_was_lost_is_done_when_repeated() -> None:
    fake = FakeQdrant()
    live, older, oldest = (f"rag_qa--gen-2026102{i}T000000000000Z-0000000{i}" for i in (3, 2, 1))
    fake.collections = {live, older, oldest}
    fake.aliases = {COLLECTION: live}
    fake.script[f"DELETE /collections/{oldest}"] = [(SLOW,)]
    assert recover(fake_store(fake)) == [oldest]
    assert fake.sent(f"DELETE /collections/{oldest}") == 2  # the repeat found nothing: done
    assert fake.collections == {live, older}


def test_inside_a_deadline_a_search_asks_for_what_is_left_of_it() -> None:
    # qdrant-client sends a request's timeout to the server and bounds the HTTP request by
    # it too (api_client.py): so the deadline cuts a hung search short.
    fake = FakeQdrant()
    generation = "rag_qa--gen-20261022T000000000000Z-00000000"
    fake.collections = {generation}
    store = fake_store(fake)
    search(store, generation, [0.1, 0.2, 0.3], 5)
    assert "timeout" not in fake.requests[-1].url.params  # no deadline: the client's own
    with resilience.deadline(3.4):
        search(store, generation, [0.1, 0.2, 0.3], 5)
    request = fake.requests[-1]
    assert request.url.params["timeout"] == "4"
    assert request.extensions["timeout"]["read"] == 4


def test_no_request_starts_once_the_deadline_has_passed() -> None:
    fake = FakeQdrant()
    store = fake_store(fake)
    with resilience.deadline(-1), pytest.raises(StoreUnavailableError, match="within 15 s"):
        store.alias_target()
    assert fake.requests == []


@contextmanager
def silent_server() -> Iterator[str]:
    """A hung Qdrant: a loopback server that takes every connection and never answers."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.1)
    taken: list[socket.socket] = []
    stop = threading.Event()

    def take() -> None:
        while not stop.is_set():
            try:
                taken.append(listener.accept()[0])
            except TimeoutError:
                continue

    thread = threading.Thread(target=take, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        stop.set()
        thread.join()
        for connection in taken:
            connection.close()
        listener.close()


def test_a_hung_qdrant_opens_the_breaker_within_one_retrievals_deadline_and_frees_the_lock() -> (
    None
):
    # Design review 2: retrieval holds the API's retrieval lock throughout, so a hung Qdrant
    # must cost one bounded budget, not a timeout per attempt per request. Real sockets and
    # the real client's timeout, at 1 s per attempt.
    with silent_server() as url:
        client = QdrantClient(url=url, timeout=1, check_compatibility=False)
        store = QdrantStore(
            url=url, collection=COLLECTION, timeout_s=1, deadline_s=5, client=client
        )
        lock = threading.Lock()

        def retrieval(asked: Any) -> list[Any]:
            with resilience.deadline(store.deadline_s):
                return search(store, "rag_qa--gen-20261022T000000000000Z-00000000", [0.1] * 3, 5)

        retrieve = serialized(RunnableLambda(retrieval), lock)
        started = time.monotonic()
        try:
            with pytest.raises(StoreUnavailableError, match="did not answer in time"):
                asyncio.run(retrieve.ainvoke("q"))
        finally:
            client.close()
        elapsed = time.monotonic() - started
    assert store.breaker.state == "open"
    assert elapsed < store.deadline_s
    assert lock.acquire(blocking=False)  # the next retrieval is not left waiting
