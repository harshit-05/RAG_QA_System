"""Incremental ingestion (S2-6, DEC-17), end to end through the real ``ingest``.

Everything is real: the walk, the loaders, the splitter, the FAISS index, the
generations, the flip and the lock. The exception is the embedder, which is the
deterministic fake, so nothing downloads. Both RAG paths always point at scratch
(caveat 9). ``sample_corpus`` indexes as 5 documents and 7 chunks: the PDF's 3 pages and
one chunk for each of the other 4 files.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import MakeConfig, edit_manifest, live_generation, use_fakes
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_qa.chain import build_query_pipeline, check_index, citation
from rag_qa.config import load_config
from rag_qa.ingest import (
    DIMENSION_PROBE,
    EXIT_CANNOT_START,
    EXIT_OK,
    EXIT_RUN_FAILED,
    IngestReport,
    discover_files,
    ingest,
    main,
)
from rag_qa.loaders import TextLoader
from rag_qa.manifest import CHUNKING_VERSION, file_sha256
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ENV_VECTOR_STORE_PATH
from rag_qa.vectorstore import open_generation, open_store, store_exists

Edit = Callable[[dict[str, Any]], None]
DOCUMENTS = {"README.md", "guide.pdf", "notes.txt", "sub/deeper/LOUD.TXT", "sub/report.docx"}


@dataclass
class Scratch:
    """``sample_corpus`` and a scratch index, with the fakes, plus ways to look inside."""

    corpus: Path
    store: Path
    make_config: MakeConfig

    def config_path(self, edit: Edit | None = None) -> Path:
        def edits(c: dict[str, Any]) -> None:
            use_fakes(c)
            if edit is not None:
                edit(c)

        return self.make_config(edits)

    def config(self, edit: Edit | None = None) -> RagConfig:
        return load_config(self.config_path(edit))

    def ingest(self, edit: Edit | None = None, *, rebuild: bool = False) -> IngestReport:
        return ingest(self.config(edit), rebuild=rebuild)

    def main(self, *args: str) -> int:
        return main(["--config", str(self.config_path()), *args])

    def live(self) -> Path:
        return live_generation(self.store)

    def documents(self) -> dict[str, Any]:
        return json.loads((self.live() / "manifest.json").read_text())["documents"]

    def ids(self) -> set[str]:
        """The chunk IDs the live index holds."""
        return set(open_generation(self.live()).ids)

    def listed(self) -> set[str]:
        """The chunk IDs the live manifest lists."""
        return {cid for record in self.documents().values() for cid in record["chunk_ids"]}

    def generations(self) -> list[str]:
        prefix = f"{self.store.name}.gen-"
        return sorted(p.name for p in self.store.parent.iterdir() if p.name.startswith(prefix))


@pytest.fixture
def scratch(
    sample_corpus: Path, make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Scratch:
    store = tmp_path / "index"
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(store))
    return Scratch(corpus=sample_corpus, store=store, make_config=make_config)


@pytest.fixture
def embedded(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """How many chunks each ``embed_documents`` call got: the cost an update avoids. An
    update's one-text dimension probe is not a chunk, and is left out."""
    calls: list[int] = []
    original = DeterministicFakeEmbedding.embed_documents

    def counting(self: DeterministicFakeEmbedding, texts: list[str]) -> list[list[float]]:
        if texts != [DIMENSION_PROBE]:
            calls.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", counting)
    return calls


def contents(folder: Path) -> dict[str, bytes]:
    """Every file in a generation, byte for byte."""
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


@pytest.fixture
def not_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads any file or folder, so chmod 000 proves nothing")


# --- what is embedded ----------------------------------------------------------------------


def test_the_first_run_is_full_and_a_second_embeds_nothing(
    scratch: Scratch, embedded: list[int]
) -> None:
    first = scratch.ingest()
    assert first.rebuild == "there is no index yet"
    assert (first.added, first.chunks) == (5, 7)
    assert embedded == [7]  # one call over every chunk, as v0.2 embedded them

    second = scratch.ingest()
    assert embedded == [7]  # nothing more
    assert (second.unchanged, second.chunks, second.rebuild, second.published) == (5, 0, None, None)
    assert scratch.generations() == [first.published]  # nothing changed, nothing published


def test_a_run_with_nothing_to_do_says_so_and_exits_0(scratch: Scratch, capsys: Any) -> None:
    assert scratch.main() == EXIT_OK
    capsys.readouterr()
    assert scratch.main() == EXIT_OK
    out = capsys.readouterr().out
    assert "Nothing changed: the index is up to date." in out
    assert "5 unchanged" in out
    assert "Nothing changed; the index at" in out


def test_an_add_a_change_and_a_delete_touch_only_those_documents(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    before = scratch.documents()
    (scratch.corpus / "new.txt").write_text("A new note.\n", encoding="utf-8")
    (scratch.corpus / "notes.txt").write_text("Edited notes.\n", encoding="utf-8")
    (scratch.corpus / "README.md").unlink()
    embedded.clear()

    report = scratch.ingest()
    assert embedded == [2]  # new.txt and notes.txt, one chunk each
    assert (report.added, report.changed, report.removed, report.unchanged) == (1, 1, 1, 3)
    after = scratch.documents()
    assert set(after) == (DOCUMENTS - {"README.md"}) | {"new.txt"}
    for name in ("guide.pdf", "sub/report.docx", "sub/deeper/LOUD.TXT"):
        assert after[name] == before[name]  # untouched: the same IDs and ingested_at
    assert after["notes.txt"]["chunk_ids"] != before["notes.txt"]["chunk_ids"]
    assert scratch.ids() == scratch.listed()  # the index holds exactly what is listed


def test_moving_md_to_another_loader_re_embeds_the_md_files_only(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    before = scratch.documents()["README.md"]
    embedded.clear()

    def markdown_loader(c: dict[str, Any]) -> None:
        loader = {"_target_": "rag_qa.loaders.TextLoader", "encoding": "utf-8"}
        c["components"]["loaders"]["md"] = loader
        c["pipeline"]["ingestion"]["loaders"][".md"] = "components.loaders.md"

    report = scratch.ingest(markdown_loader)
    assert embedded == [1]
    assert (report.changed, report.unchanged) == (1, 4)
    after = scratch.documents()["README.md"]
    assert after["loader"] == "components.loaders.md"
    assert after["loader_identity"] != before["loader_identity"]
    # The same path, bytes and index give the same IDs back: deleted before they are re-added.
    assert after["chunk_ids"] == before["chunk_ids"]
    assert scratch.ids() == scratch.listed()


def test_two_files_with_identical_bytes_both_index_and_deleting_one_keeps_the_other(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "twins"
    for folder in ("a", "b"):
        (corpus / folder).mkdir(parents=True)
        (corpus / folder / "same.txt").write_text("Identical bytes.\n", encoding="utf-8")
    monkeypatch.setenv(ENV_DATA_PATH, str(corpus))
    scratch.ingest()
    documents = scratch.documents()
    assert documents["a/same.txt"]["sha256"] == documents["b/same.txt"]["sha256"]
    assert documents["a/same.txt"]["chunk_ids"] != documents["b/same.txt"]["chunk_ids"]

    (corpus / "a" / "same.txt").unlink()
    report = scratch.ingest()
    assert report.removed == 1
    assert scratch.ids() == set(documents["b/same.txt"]["chunk_ids"])


# --- failures keep their chunks ------------------------------------------------------------


def test_a_file_that_becomes_corrupt_keeps_its_chunks_and_the_run_exits_1(
    scratch: Scratch, capsys: Any
) -> None:
    scratch.ingest()
    before, ids = scratch.documents()["guide.pdf"], scratch.ids()
    (scratch.corpus / "guide.pdf").write_bytes(b"corrupt")

    assert scratch.main() == EXIT_RUN_FAILED
    out, err = capsys.readouterr()
    assert re.search(r"guide\.pdf: \w+Error: ", out)
    assert "1 failed and kept their previous chunks" in out
    assert "the failed documents kept their previous chunks" in err
    assert scratch.documents()["guide.pdf"] == before  # its old entry, old sha256 included
    assert scratch.ids() == ids


def test_a_failed_change_beside_other_changes_still_keeps_its_chunks(scratch: Scratch) -> None:
    scratch.ingest()
    before = scratch.documents()["guide.pdf"]
    (scratch.corpus / "guide.pdf").write_bytes(b"corrupt")
    (scratch.corpus / "new.txt").write_text("Published beside the failure.\n", encoding="utf-8")

    report = scratch.ingest()
    assert report.published is not None  # new.txt went in...
    assert (report.added, report.kept) == (1, 1)
    assert scratch.documents()["guide.pdf"] == before  # ...and guide.pdf kept its chunks
    assert set(before["chunk_ids"]) <= scratch.ids()


def test_a_folder_that_cannot_be_listed_keeps_its_documents_chunks(
    scratch: Scratch, capsys: Any, not_root: None
) -> None:
    scratch.ingest()
    before = scratch.documents()
    (scratch.corpus / "new.txt").write_text("Added while sub/ is locked.\n", encoding="utf-8")
    locked = scratch.corpus / "sub"
    locked.chmod(0)
    try:
        assert scratch.main() == EXIT_RUN_FAILED
    finally:
        locked.chmod(0o755)
    out, err = capsys.readouterr()
    assert "Cannot list sub/" in out
    assert "2 failed and kept their previous chunks" in out
    after = scratch.documents()
    for name in ("sub/report.docx", "sub/deeper/LOUD.TXT"):
        assert after[name] == before[name]  # not "removed" just because they were unseen
    assert "new.txt" in after
    assert "The index was updated" in err


def test_an_indexed_file_that_turns_unreadable_keeps_its_chunks_and_nothing_is_published(
    scratch: Scratch, capsys: Any, not_root: None
) -> None:
    first = scratch.ingest()
    notes = scratch.corpus / "notes.txt"
    notes.chmod(0)
    try:
        assert scratch.main() == EXIT_RUN_FAILED
    finally:
        notes.chmod(0o644)
    out, err = capsys.readouterr()
    assert "Nothing changed apart from the documents that could not be read." in out
    assert "Nothing else changed; the failed documents kept their previous chunks" in err
    assert scratch.generations() == [first.published]


def test_a_corpus_with_no_text_at_all_publishes_nothing_and_exits_1(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    corpus = tmp_path / "blank"
    corpus.mkdir()
    (corpus / "empty.txt").write_text("", encoding="utf-8")  # loads, but splits into nothing
    monkeypatch.setenv(ENV_DATA_PATH, str(corpus))
    assert scratch.main() == EXIT_RUN_FAILED
    out, err = capsys.readouterr()
    assert "no document had any text" in err
    # Not "Rebuilt the index in full", nor "1 added": nothing was (second review).
    assert "A full rebuild was due (there is no index yet), but published nothing." in out
    assert "Documents: 0 added" in out
    assert not scratch.store.exists()


def test_a_prune_that_fails_after_the_flip_does_not_fail_the_run(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    from rag_qa import ingest as module

    calls: list[Path] = []

    def recover_then_fail(store: Path) -> list[str]:
        calls.append(store)
        if len(calls) == 1:
            return []  # recovery at the start
        raise PermissionError(13, "Permission denied", str(store.parent / "old-generation"))

    monkeypatch.setattr(module, "recover", recover_then_fail)
    caplog.set_level(logging.INFO, logger="rag_qa.ingest")
    report = scratch.ingest()
    assert report.published is not None
    assert scratch.live().name == report.published  # the flip stands
    assert "Could not remove an old generation" in caplog.text
    assert str(tmp_path) not in caplog.text


@pytest.mark.parametrize("where", ["while hashing", "while loading"])
def test_a_failure_leaves_no_absolute_path_in_the_report_or_the_log(
    scratch: Scratch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    not_root: None,
    where: str,
) -> None:
    # S2-7 returns both over HTTP (DEC-18), and an OSError's text names its path.
    secret = scratch.corpus / "secret.txt"
    secret.write_text("x\n", encoding="utf-8")
    if where == "while hashing":
        secret.chmod(0)
    else:

        def vanished(self: TextLoader) -> list[Any]:
            raise FileNotFoundError(2, "No such file or directory", self.file_path)

        monkeypatch.setattr(TextLoader, "load", vanished)
    caplog.set_level(logging.INFO, logger="rag_qa.ingest")
    try:
        report = scratch.ingest()
    finally:
        secret.chmod(0o644)

    failures = dict(report.failed)
    assert "secret.txt" in failures
    assert "'secret.txt'" in failures["secret.txt"]  # named relatively
    assert "secret.txt" in caplog.text
    assert str(tmp_path) not in caplog.text
    assert not any(str(tmp_path) in error for _, error in report.failed)


# --- full rebuilds -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "rebuild", "reason"),
    [
        pytest.param(None, True, "--rebuild was given", id="--rebuild"),
        pytest.param(
            lambda c: c["components"]["embedders"]["fake"].update(size=16),
            False,
            "the embedder is not the one that built the index",
            id="embedder",
        ),
        pytest.param(
            lambda c: c["components"]["splitters"]["english_recursive"].update(chunk_size=500),
            False,
            "the splitter is not the one that built the index",
            id="splitter",
        ),
    ],
)
def test_the_flag_or_a_new_embedder_or_splitter_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], edit: Edit | None, rebuild: bool, reason: str
) -> None:
    scratch.ingest()
    embedded.clear()
    report = scratch.ingest(edit, rebuild=rebuild)
    assert report.rebuild is not None and report.rebuild.startswith(reason)
    assert embedded == [7]
    assert (report.added, report.unchanged) == (5, 0)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        pytest.param(
            lambda folder: (folder / "manifest.json").unlink(),
            "the index has no manifest",
            id="no manifest",
        ),
        pytest.param(
            lambda folder: (folder / "manifest.json").write_text("{oops"),
            "its manifest cannot be used: it is not valid JSON",
            id="not JSON",
        ),
        pytest.param(
            lambda folder: (folder / "manifest.json").write_text(json.dumps({"version": 99})),
            "its manifest cannot be used: unknown manifest version 99",
            id="unknown version",
        ),
        pytest.param(
            lambda folder: (folder / "index.faiss").unlink(),
            "the index is missing index.faiss",
            id="missing index file",
        ),
        pytest.param(
            lambda folder: (folder / "index.pkl").write_bytes(b"not a pickle"),
            "the index cannot be opened (UnpicklingError",
            id="corrupt pickle",
        ),
        pytest.param(
            lambda folder: edit_manifest(
                folder.parent / "index",
                lambda m: m["documents"]["README.md"]["chunk_ids"].append(
                    m["documents"]["notes.txt"]["chunk_ids"][0]
                ),
            ),
            "the manifest lists 1 chunk ID(s) more than once",
            id="an ID listed twice",
        ),
    ],
)
def test_a_damaged_index_rebuilds_in_full_even_when_the_corpus_is_unchanged(
    scratch: Scratch, embedded: list[int], damage: Callable[[Path], Any], reason: str
) -> None:
    # With nothing changed, a run that skipped these checks would call the index up to
    # date while rag-query refused it: the two would point at each other (first review).
    scratch.ingest()
    damage(scratch.live())
    embedded.clear()
    report = scratch.ingest()
    assert report.rebuild is not None and report.rebuild.startswith(reason)
    assert embedded == [7]
    assert scratch.ids() == scratch.listed()


def test_a_new_chunking_version_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The code that makes chunks changed, the config did not: without this, every
    # unchanged file would keep its old chunks for good (second review).
    scratch.ingest()
    monkeypatch.setattr("rag_qa.ingest.CHUNKING_VERSION", 2)
    embedded.clear()
    report = scratch.ingest()
    assert report.rebuild == (
        "this version makes chunks differently (chunking version 1 then, 2 now)"
    )
    assert embedded == [7]
    assert json.loads((scratch.live() / "manifest.json").read_text())["chunking"] == 2


def test_an_index_from_before_the_chunking_version_is_kept_not_rebuilt(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    edit_manifest(scratch.store, lambda m: m.pop("chunking"))
    embedded.clear()
    report = scratch.ingest()
    assert (report.rebuild, report.published) == (None, None)
    assert embedded == []


def test_a_manifest_that_lists_an_id_its_index_lacks_rebuilds_in_full(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    edit_manifest(
        scratch.store,
        lambda m: m["documents"]["notes.txt"]["chunk_ids"].append(str(uuid.uuid4())),
    )
    (scratch.corpus / "new.txt").write_text("A change, so the run opens the index.\n")
    embedded.clear()

    report = scratch.ingest()
    assert report.rebuild == (
        "the manifest and its index disagree: 1 chunk(s) listed but missing, 0 held but not listed"
    )
    assert embedded == [8]  # everything, once: checked before anything was embedded
    assert scratch.ids() == scratch.listed()


@pytest.mark.parametrize(
    ("damage", "error"),
    [
        pytest.param(b"not a pickle", "UnpicklingError", id="corrupt pickle"),
        pytest.param(b"", "EOFError", id="empty pickle"),
    ],
)
def test_an_index_that_cannot_be_opened_rebuilds_in_full_and_says_why(
    scratch: Scratch, embedded: list[int], tmp_path: Path, damage: bytes, error: str
) -> None:
    scratch.ingest()
    (scratch.live() / "index.pkl").write_bytes(damage)
    (scratch.corpus / "new.txt").write_text("A change, so the run opens the index.\n")
    embedded.clear()

    report = scratch.ingest()
    assert report.rebuild is not None
    assert report.rebuild.startswith(f"the index cannot be opened ({error}")
    assert str(tmp_path) not in report.rebuild
    assert embedded == [8]


def test_a_run_that_only_deletes_never_loads_the_embedding_model(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Writing a generation embeds nothing (first review): the model is seconds and hundreds
    # of MB to load, for a run that has nothing to embed.
    scratch.ingest()
    (scratch.corpus / "notes.txt").unlink()

    def no_model(config: Any) -> None:
        raise AssertionError("the embedding model was loaded for a run that only deletes")

    monkeypatch.setattr("rag_qa.ingest.build_embedder", no_model)
    report = scratch.ingest()
    assert (report.removed, report.published is not None) == (1, True)
    assert "notes.txt" not in scratch.documents()
    assert scratch.ids() == scratch.listed()


# --- write safety ----------------------------------------------------------------------------


def test_a_crash_before_the_flip_publishes_nothing_and_the_next_run_clears_it(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.ingest()
    live = scratch.live()
    before = contents(live)
    (scratch.corpus / "new.txt").write_text("Added just before the crash.\n", encoding="utf-8")
    replace = os.replace

    def crash_at_the_flip(src: Any, dst: Any) -> None:
        if ".tmp-" in os.fspath(src):  # the flip's rename of the temporary link
            raise OSError("simulated crash before the flip")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", crash_at_the_flip)
    with pytest.raises(OSError, match="simulated crash"):
        scratch.ingest()
    monkeypatch.setattr(os, "replace", replace)

    assert scratch.live() == live
    assert contents(live) == before
    (orphan,) = [name for name in scratch.generations() if name != live.name]
    assert list(scratch.store.parent.glob("index.tmp-*"))
    # Queries still read the old generation, which never had new.txt: checked on what the
    # query side resolves, and on every chunk the store holds, not on a top-5 that might
    # leave new.txt out anyway (second review).
    config = scratch.config()
    assert check_index(config)[0] == live
    assert build_query_pipeline(config).retrieve is not None
    every = open_store(DeterministicFakeEmbedding(size=8), scratch.store).similarity_search(
        "Added just before the crash.", k=20
    )
    assert len(every) == 7
    assert {doc.metadata["source"] for doc in every} == DOCUMENTS

    report = scratch.ingest()  # recovery first: the orphan and the link go
    assert orphan not in scratch.generations()
    assert not list(scratch.store.parent.glob("index.tmp-*"))
    assert report.added == 1
    assert scratch.generations() == sorted([live.name, report.published or ""])


def test_a_failure_in_the_apply_phase_publishes_nothing(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.ingest()
    live = scratch.live()
    before = contents(live)
    taken = scratch.documents()["notes.txt"]["chunk_ids"][0]
    (scratch.corpus / "new.txt").write_text("Its ID will collide.\n", encoding="utf-8")
    # FAISS adds the vector, then its docstore rejects the ID: the in-memory index is now
    # inconsistent, which is why nothing is ever applied to a live generation.
    monkeypatch.setattr("rag_qa.ingest.chunk_id", lambda *args: taken)

    with pytest.raises(ValueError, match="already exist"):
        scratch.ingest()
    assert scratch.live() == live
    assert contents(live) == before
    assert scratch.generations() == [live.name]  # not even a folder was written


def test_a_v02_directory_is_rebuilt_once_and_kept_beside_the_new_index(scratch: Scratch) -> None:
    scratch.store.mkdir()
    (scratch.store / "index.faiss").write_bytes(b"a v0.2 index")
    (scratch.store / "index.pkl").write_bytes(b"its docstore")

    report = scratch.ingest()
    assert report.rebuild == "the index was built by v0.2 and has no manifest"
    assert scratch.store.is_symlink()
    assert store_exists(scratch.store)
    (kept,) = scratch.store.parent.glob("index.v02-*")
    assert (kept / "index.faiss").read_bytes() == b"a v0.2 index"


def test_only_the_live_and_the_previous_generation_survive(scratch: Scratch) -> None:
    names = [scratch.ingest(rebuild=True).published for _ in range(3)]
    assert scratch.generations() == names[1:]  # the names sort in the order they were made
    assert scratch.live().name == names[2]


def test_a_file_where_the_index_goes_exits_2_and_is_left_alone(
    scratch: Scratch, capsys: Any
) -> None:
    scratch.store.write_text("someone's file", encoding="utf-8")
    assert scratch.main() == EXIT_CANNOT_START
    assert "is a file, not an index" in capsys.readouterr().err
    assert scratch.store.read_text(encoding="utf-8") == "someone's file"


def test_a_generation_named_as_the_index_exits_2_and_is_left_alone(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    scratch.ingest()
    live = scratch.live()
    before = contents(live)
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(live))
    assert scratch.main() == EXIT_CANNOT_START
    assert "is an index generation folder" in capsys.readouterr().err
    assert contents(live) == before


def test_an_empty_corpus_publishes_nothing_and_leaves_the_index(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # v0.2's rule, kept: an empty corpus (most often a mis-set RAG_DATA_PATH) is taken for a
    # mistake, not for an instruction to empty the index. A corpus with files is believed.
    scratch.ingest()
    live = scratch.live()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(ENV_DATA_PATH, str(empty))
    assert scratch.main() == EXIT_RUN_FAILED
    assert "No documents were loaded" in capsys.readouterr().err
    assert scratch.live() == live


# --- one writer ------------------------------------------------------------------------------


def test_a_second_ingestion_exits_2_while_another_process_holds_the_lock(
    scratch: Scratch, capsys: Any
) -> None:
    lock = scratch.store.with_name("index.lock")
    hold = (
        "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
        "fcntl.flock(f.fileno(), fcntl.LOCK_EX); print('locked', flush=True); time.sleep(60)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", hold, str(lock)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        assert scratch.main() == EXIT_CANNOT_START
    finally:
        holder.kill()
        holder.wait()
    assert "another ingestion is running" in capsys.readouterr().err
    assert not scratch.store.exists()  # it never started


def test_a_second_ingestion_exits_2_from_another_thread_of_the_same_process(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # The API's case (S2-7): its ingest job runs in a thread of the server's process, so a
    # lock owned by the process, as POSIX lockf's are, would not stop it.
    started, release = threading.Event(), threading.Event()
    load = TextLoader.load

    def held_up(self: TextLoader) -> list[Any]:
        started.set()
        release.wait(30)
        return load(self)

    monkeypatch.setattr(TextLoader, "load", held_up)
    config = scratch.config()
    reports: list[IngestReport] = []
    worker = threading.Thread(target=lambda: reports.append(ingest(config)))
    worker.start()
    try:
        assert started.wait(30)
        assert scratch.main() == EXIT_CANNOT_START
    finally:
        release.set()
        worker.join(30)
    assert "another ingestion is running" in capsys.readouterr().err
    (report,) = reports
    assert report.published is not None  # the first run was unaffected


# --- metadata and the walk -------------------------------------------------------------------


def test_chunks_carry_the_section_7_3_metadata_with_a_corpus_relative_source(
    scratch: Scratch,
) -> None:
    scratch.ingest()
    docs = open_store(DeterministicFakeEmbedding(size=8), scratch.live()).similarity_search(
        "text", k=7
    )
    assert {doc.metadata["source"] for doc in docs} == DOCUMENTS
    listed = scratch.listed()
    common = {"source", "loader", "source_sha256", "ingested_at"}
    pages = {"page", "page_label", "total_pages"}
    for doc in docs:
        source = doc.metadata["source"]
        assert set(doc.metadata) == (common | pages if source == "guide.pdf" else common)
        assert doc.metadata["source_sha256"] == file_sha256(scratch.corpus / source)
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc.metadata["ingested_at"])
        assert doc.id in listed
    # citation() reads only the file name, so a relative source prints the same citations.
    assert sorted(citation(doc) for doc in docs) == [
        "LOUD.TXT", "README.md", "guide.pdf, p. 1", "guide.pdf, p. i", "guide.pdf, p. ii",
        "notes.txt", "report.docx",
    ]


#: ``sample_corpus``'s chunks under each CHUNKING_VERSION, as :func:`chunks_digest` gives.
SAMPLE_CORPUS_CHUNKS = {1: "556f8396d1dba0f5a1f48ff6a8075f78fa862cb5f84bc6c53117b7770b58162b"}


def chunks_digest(generation: Path) -> str:
    """The sha256 of a generation's chunks in index order: text and metadata, less the
    file's hash and the run's time, which are not how chunks are made."""
    db = open_generation(generation).db
    chunks = []
    for position in range(len(db.index_to_docstore_id)):
        doc = db.docstore.search(db.index_to_docstore_id[position])
        assert not isinstance(doc, str)
        metadata = {
            key: value
            for key, value in doc.metadata.items()
            if key not in ("source_sha256", "ingested_at")
        }
        chunks.append([doc.page_content, metadata])
    return hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()


def test_the_sample_corpus_chunks_as_the_chunking_version_says(scratch: Scratch) -> None:
    # The tripwire for CHUNKING_VERSION (second review). If the chunks changed, whether
    # by a change to the loaders or to ingest's metadata, or by a library upgrade, bump
    # CHUNKING_VERSION and pin this digest under the new version: every index then
    # rebuilds once, instead of keeping old chunks for every unchanged file.
    scratch.ingest()
    digest = chunks_digest(scratch.live())
    assert SAMPLE_CORPUS_CHUNKS.get(CHUNKING_VERSION) == digest, (
        f"the sample corpus no longer chunks as CHUNKING_VERSION {CHUNKING_VERSION} did: "
        f"bump it in rag_qa/manifest.py, and pin {digest} under the new version"
    )


def test_the_walk_never_enters_an_ignored_folder(
    sample_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sample_corpus / ".git" / "objects").mkdir(parents=True)
    (sample_corpus / ".git" / "objects" / "pack").write_text("x")
    visited: list[Path] = []
    walk = Path.walk

    def recording(self: Path, *args: Any, **kwargs: Any) -> Any:
        for entry in walk(self, *args, **kwargs):
            visited.append(entry[0].relative_to(sample_corpus))
            yield entry

    monkeypatch.setattr(Path, "walk", recording)
    listing = discover_files(sample_corpus)
    assert sorted(p.as_posix() for p in visited) == [".", "sub", "sub/deeper"]
    # .git and .ipynb_checkpoints once each, unread; the hidden .draft.md; Word's lock file.
    assert listing.ignored == 4


# --- what may sit at the index's path (S2-6's second review) --------------------------------


def test_a_folder_that_is_not_an_index_is_refused_not_renamed_aside(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # Any real folder used to count as "the v0.2 index", and was renamed aside: with
    # RAG_VECTOR_STORE_PATH mis-set to the corpus, the corpus itself.
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(scratch.corpus))
    assert scratch.main() == EXIT_CANNOT_START
    assert "is a folder that is not an index" in capsys.readouterr().err
    assert (scratch.corpus / "guide.pdf").is_file()
    assert not list(scratch.corpus.parent.glob("corpus.v02-*"))


def test_a_link_someone_else_made_at_the_index_path_is_refused_and_left_alone(
    scratch: Scratch, tmp_path: Path, capsys: Any
) -> None:
    # A v0.2 index kept on a bigger disk, linked in by hand: the flip used to replace the
    # link, and write the new index onto the small disk.
    elsewhere = tmp_path / "big-disk" / "db_faiss"
    elsewhere.mkdir(parents=True)
    scratch.store.symlink_to(elsewhere)
    assert scratch.main() == EXIT_CANNOT_START
    assert "not to one of the index's generations" in capsys.readouterr().err
    assert scratch.store.resolve() == elsewhere


# --- more of the second review's cases --------------------------------------------------------


def test_a_file_name_that_is_not_utf8_fails_on_its_own(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It used to abort the whole run in chunk_id, so nothing was indexed (NFR-7).
    (scratch.corpus / os.fsdecode(b"caf\xe9.txt")).write_text("x\n")  # the byte 0xE9: Latin-1
    report = scratch.ingest()
    assert report.published is not None
    assert report.added == 5  # every other document
    (failure,) = report.failed
    assert failure == (
        "caf\\udce9.txt",
        "UnicodeEncodeError: the file name is not valid UTF-8; rename the file",
    )


def test_vectors_of_another_size_under_the_same_spec_rebuild_in_full(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The identity cannot see the weights: a model replaced under the same name can make
    # vectors of another size, which FAISS refused with a bare assert, every run.
    scratch.ingest()
    original = DeterministicFakeEmbedding.embed_documents

    def wider(self: DeterministicFakeEmbedding, texts: list[str]) -> list[list[float]]:
        return [vector * 2 for vector in original(self, texts)]

    monkeypatch.setattr(DeterministicFakeEmbedding, "embed_documents", wider)
    (scratch.corpus / "new.txt").write_text("Embedded by the replaced model.\n")
    report = scratch.ingest()
    assert report.rebuild == (
        "the embedder now makes 16-dimension vectors, but the index holds 8-dimension ones"
    )
    assert report.published is not None
    assert open_generation(scratch.live()).db.index.d == 16


def test_a_folder_named_like_a_word_lock_file_is_still_walked(sample_corpus: Path) -> None:
    # The "~$" rule is for Word's lock files; pruning folders by it dropped their contents.
    (sample_corpus / "~$archive").mkdir()
    (sample_corpus / "~$archive" / "kept.txt").write_text("Still a document.\n")
    files = {p.relative_to(sample_corpus).as_posix() for p in discover_files(sample_corpus).files}
    assert "~$archive/kept.txt" in files


def test_a_new_unreadable_file_is_not_said_to_have_kept_anything(
    scratch: Scratch, capsys: Any, not_root: None
) -> None:
    scratch.ingest()
    locked = scratch.corpus / "locked.txt"
    locked.write_text("never read\n")
    locked.chmod(0)
    try:
        assert scratch.main() == EXIT_RUN_FAILED
    finally:
        locked.chmod(0o644)
    out, err = capsys.readouterr()
    assert "Nothing changed apart from the documents that could not be read." in out
    assert "up to date" not in out.split("Nothing changed apart")[0]
    assert "kept their previous chunks, if they had any" in err


def test_renaming_a_loader_entry_rewrites_the_manifest_without_embedding(
    scratch: Scratch, embedded: list[int]
) -> None:
    # The same spec under a new name: no vector changes, but the manifest (and S2-7's
    # /v1/documents) must not keep naming a loader that no longer exists.
    scratch.ingest()
    embedded.clear()

    def renamed(c: dict[str, Any]) -> None:
        loaders = c["components"]["loaders"]
        loaders["text"] = loaders.pop("txt")
        for extension in (".txt", ".md"):
            c["pipeline"]["ingestion"]["loaders"][extension] = "components.loaders.text"

    report = scratch.ingest(renamed)
    assert embedded == []
    assert report.published is not None
    documents = scratch.documents()
    assert documents["notes.txt"]["loader"] == "components.loaders.text"
    assert documents["guide.pdf"]["loader"] == "components.loaders.pdf"
    assert scratch.ids() == scratch.listed()


def test_no_absolute_path_survives_in_a_described_error(scratch: Scratch) -> None:
    # Paths outside the corpus and the index's folder used to pass through: a moved
    # pickled class names its module's file, and FAISS names the file it read.
    from rag_qa.ingest import _describe

    config = scratch.config()
    moved = AttributeError(
        "Can't get attribute 'X' on <module 'langchain_core.documents.base' from "
        "'/home/someone/project/.venv/lib/python3.12/site-packages/langchain_core/documents/base.py'>"
    )
    described = _describe(moved, config)
    assert "/home/someone" not in described
    assert "from 'base.py'>" in described
    read = f"{scratch.store.parent}/index.gen-1/index.faiss"
    faiss = RuntimeError(f"could not open {read} for reading")
    assert _describe(faiss, config) == (
        "RuntimeError: could not open index.gen-1/index.faiss for reading"
    )
    elsewhere = OSError(13, "Permission denied", "/var/data/secret.txt")  # a PermissionError
    assert _describe(elsewhere, config) == (
        "PermissionError: [Errno 13] Permission denied: 'secret.txt'"
    )
