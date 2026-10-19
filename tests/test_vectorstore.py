"""The index's layout on disk (S2-6, DEC-17): generations, the symlink, and recovery.

Folders and links are made by hand here: recovery reads names, never an index. The
crash and failure paths that run through the real ingest are in
``test_ingest_incremental.py``.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_qa.manifest import EmbedderRecord, Manifest, SplitterRecord, save_manifest
from rag_qa.vectorstore import (
    NO_INDEX,
    Embedded,
    generation_path,
    in_the_way,
    is_legacy_store,
    live_index,
    open_generation,
    recover,
    store_exists,
    write_generation,
)

EMPTY_MANIFEST = Manifest(
    generation="",
    embedder=EmbedderRecord(ref="e", identity="e", dimension=4),
    splitter=SplitterRecord(ref="s", identity="s"),
    documents={},
)


def one_chunk(text: str = "alpha") -> Embedded:
    return Embedded(
        ids=[f"id-{text}"],
        texts=[text],
        metadatas=[{}],
        vectors=DeterministicFakeEmbedding(size=4).embed_documents([text]),
    )


def gen(n: int, store: str = "db_faiss") -> str:
    """The name of the ``n``-th generation of ``store``: a later ``n`` is a newer one."""
    return f"{store}.gen-20261005T12000{n}.000000Z-{n:08x}"


def folders(parent: Path, *names: str) -> None:
    for name in names:
        (parent / name).mkdir()


def test_recovery_keeps_the_live_and_the_previous_generation_only(tmp_path: Path) -> None:
    store = tmp_path / "db_faiss"
    folders(tmp_path, gen(1), gen(2), gen(3), gen(4))  # older, previous, live, crash orphan
    store.symlink_to(gen(3))
    (tmp_path / "db_faiss.tmp-0badc0de").symlink_to(gen(4))  # a flip that never renamed
    folders(tmp_path, "db_faiss.v02-20261005T120000.000000Z", "unrelated")
    (tmp_path / "db_faiss.lock").touch()

    assert sorted(recover(store)) == sorted([gen(1), gen(4), "db_faiss.tmp-0badc0de"])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [
            "db_faiss", gen(2), gen(3),
            "db_faiss.v02-20261005T120000.000000Z", "unrelated", "db_faiss.lock",
        ]
    )


@pytest.mark.parametrize(
    "written",
    [
        pytest.param(lambda folder, name: str(folder / name), id="absolute"),
        pytest.param(lambda folder, name: f"./{name}", id="dot-slash"),
        pytest.param(lambda folder, name: f"{name}/", id="trailing slash"),
    ],
)
def test_recovery_reads_the_link_however_it_is_written(tmp_path: Path, written: Any) -> None:
    # A hand-made rollback (ln -sfn "$PWD/vectorstore/db_faiss.gen-…") once read as "no
    # generation is live", and recovery deleted the live and previous ones (second review).
    store = tmp_path / "db_faiss"
    folders(tmp_path, gen(1), gen(2), gen(3))
    store.symlink_to(written(tmp_path, gen(3)))
    assert recover(store) == [gen(1)]
    assert generation_path(store) == tmp_path / gen(3)


def test_a_link_to_anywhere_else_leaves_every_generation_alone(tmp_path: Path) -> None:
    store = tmp_path / "db_faiss"
    folders(tmp_path, gen(1), gen(2), "elsewhere")
    store.symlink_to("elsewhere")
    assert recover(store) == []
    assert generation_path(store) is None
    assert in_the_way(store) == "is a link to 'elsewhere', not to one of the index's generations"


def test_only_a_folder_holding_exactly_faiss_s_two_files_counts_as_v02(tmp_path: Path) -> None:
    # Any other folder at the store path was renamed aside as "the v0.2 index": with a
    # mis-set path, the corpus itself (second review).
    store = tmp_path / "db_faiss"
    store.mkdir()
    assert not is_legacy_store(store)  # empty
    assert in_the_way(store) is not None
    (store / "index.faiss").touch()
    (store / "index.pkl").touch()
    assert is_legacy_store(store)
    assert in_the_way(store) is None
    (store / "notes.txt").touch()
    assert not is_legacy_store(store)
    assert "is a folder that is not an index" in str(in_the_way(store))


def test_the_new_folder_is_durable_in_its_parent_before_the_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # POSIX makes a mkdir durable only through an fsync of the parent; done after the
    # rename alone, a crash could keep the link and lose the folder (second review).
    from rag_qa import vectorstore

    events: list[str] = []
    fsync, replace = vectorstore._fsync, os.replace

    def recording_fsync(path: Path, *, directory: bool = False) -> None:
        events.append(f"fsync {path.name}")
        fsync(path, directory=directory)

    def recording_replace(src: Any, dst: Any) -> None:
        events.append(f"replace {Path(src).name}")
        replace(src, dst)

    monkeypatch.setattr(vectorstore, "_fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    write_generation(tmp_path / "db_faiss", EMPTY_MANIFEST, base=None, delete=[], add=one_chunk())
    flip = next(i for i, event in enumerate(events) if event.startswith("replace db_faiss.tmp-"))
    assert f"fsync {tmp_path.name}" in events[:flip]


def test_with_no_symlink_nothing_is_published_and_every_generation_goes(tmp_path: Path) -> None:
    # A first run that crashed before its flip, or a v0.2 directory still in place.
    store = tmp_path / "db_faiss"
    folders(tmp_path, gen(1), gen(2))
    store.mkdir()
    assert sorted(recover(store)) == [gen(1), gen(2)]
    assert store.is_dir()


def test_recovery_leaves_alone_whatever_only_resembles_a_generation(tmp_path: Path) -> None:
    store = tmp_path / "db_faiss"
    (tmp_path / gen(1)).write_text("a file, not a generation folder")
    folders(tmp_path, gen(2, store="db_other"))  # another store's generation
    (tmp_path / gen(3)).symlink_to(gen(2, store="db_other"))  # a link, not a folder
    folders(tmp_path, "db_faiss.gen-yesterday")  # not our stamp format
    assert recover(store) == []


def test_recovery_with_no_folder_yet_does_nothing(tmp_path: Path) -> None:
    assert recover(tmp_path / "missing" / "db_faiss") == []


def test_live_index_says_why_there_is_no_index_to_use(tmp_path: Path) -> None:
    # The one rule rag-ingest and the query side share (the first review's finding: two
    # copies of it had already drifted apart).
    store = tmp_path / "db_faiss"
    assert live_index(store) == NO_INDEX
    assert generation_path(store) is None

    store.mkdir()  # a v0.2 index: a real folder, no manifest
    (store / "index.faiss").touch()
    (store / "index.pkl").touch()
    assert is_legacy_store(store)
    assert live_index(store) == "the index was built by v0.2 and has no manifest"

    store.rename(tmp_path / gen(1))
    store.symlink_to(gen(1))
    assert not is_legacy_store(store)
    assert generation_path(store) == tmp_path / gen(1)
    assert live_index(store) == "the index has no manifest"
    (tmp_path / gen(1) / "manifest.json").write_text("{oops")
    assert str(live_index(store)).startswith("its manifest cannot be used: it is not valid JSON")
    save_manifest(EMPTY_MANIFEST, tmp_path / gen(1) / "manifest.json")
    assert live_index(store) == (tmp_path / gen(1), EMPTY_MANIFEST)
    assert store_exists(store)
    (tmp_path / gen(1) / "index.faiss").unlink()
    assert live_index(store) == "the index is missing index.faiss"
    assert not store_exists(store)


def test_a_new_generation_sorts_after_the_live_one_even_if_the_clock_stepped_back(
    tmp_path: Path,
) -> None:
    # Recovery keeps "the newest generation older than the live one" by name, so a name
    # that sorted first would cost the previous generation its place (first review).
    store = tmp_path / "db_faiss"
    future = "db_faiss.gen-20991231T235959.999999Z-00000001"
    (tmp_path / future).mkdir()
    store.symlink_to(future)

    published = write_generation(store, EMPTY_MANIFEST, base=None, delete=[], add=one_chunk())
    assert published.generation.startswith("db_faiss.gen-21000101T000000.000000Z-")
    assert published.generation > future
    assert recover(store) == []  # the clock's "newer" generation is the previous one, kept


def test_open_generation_refuses_an_index_its_docstore_does_not_match(tmp_path: Path) -> None:
    # What a generation saved after a failed add would hold: FAISS appends the vector,
    # then its docstore rejects the duplicate ID. write_generation never saves one; if
    # anything ever did, an update must rebuild rather than trust it.
    embeddings = DeterministicFakeEmbedding(size=4)
    texts = ["alpha", "beta"]
    db = FAISS.from_embeddings(
        list(zip(texts, embeddings.embed_documents(texts), strict=True)), embeddings, ids=["a", "b"]
    )
    with pytest.raises(ValueError, match="already exist"):
        db.add_embeddings([("gamma", embeddings.embed_query("gamma"))], ids=["a"])
    db.save_local(str(tmp_path / "broken"))
    with pytest.raises(ValueError, match="holds 3 vectors, mapped to 2 chunk IDs"):
        open_generation(tmp_path / "broken")


def test_writing_a_generation_never_embeds(tmp_path: Path) -> None:
    # The vectors arrive embedded already, so no model is needed to write one: a run that
    # only deletes never loads the embedder (first review).
    store = tmp_path / "db_faiss"
    first = write_generation(store, EMPTY_MANIFEST, base=None, delete=[], add=one_chunk())
    live = open_generation(tmp_path / first.generation)
    assert live.ids == {"id-alpha"}
    write_generation(store, EMPTY_MANIFEST, base=live, delete=["id-alpha"], add=one_chunk("beta"))
    assert open_generation(generation_path(store) or store).ids == {"id-beta"}


def test_a_new_index_needs_a_chunk(tmp_path: Path) -> None:
    empty = Embedded(ids=[], texts=[], metadatas=[], vectors=[])
    with pytest.raises(ValueError, match="at least one chunk"):
        write_generation(tmp_path / "db_faiss", EMPTY_MANIFEST, base=None, delete=[], add=empty)
    assert os.listdir(tmp_path) == []  # nothing written
