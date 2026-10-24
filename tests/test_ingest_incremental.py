"""Incremental ingestion (S2-6, DEC-17, DEC-19), end to end through the real ``ingest``.

Everything is real: the walk, the loaders, the splitter, the generations in Qdrant, the
alias switch, recovery and the lock. The exceptions are the embedder, which is the
deterministic fake, so nothing downloads, and Qdrant, which is the test's own in-memory
client (conftest's ``qdrant``). ``sample_corpus`` indexes as 5 documents and 7 chunks: the
PDF's 3 pages and one chunk for each of the other 4 files.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    MakeConfig,
    edit_manifest,
    generations,
    live_generation,
    metadata,
    published,
    use_fakes,
)
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from qdrant_client import QdrantClient, models

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
    scan_corpus,
)
from rag_qa.loaders import TextLoader
from rag_qa.manifest import (
    CHUNKING_VERSION,
    Manifest,
    file_sha256,
    parse_manifest,
    spec_identity,
)
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_DATA_PATH, ENV_VECTOR_STORE_PATH
from rag_qa.sparse import SPARSE_VERSION, Bm25Encoder
from rag_qa.vectorstore import StoreError, StoreUnavailableError

Edit = Callable[[dict[str, Any]], None]
DOCUMENTS = {"README.md", "guide.pdf", "notes.txt", "sub/deeper/LOUD.TXT", "sub/report.docx"}


@dataclass
class Scratch:
    """``sample_corpus`` and the test's Qdrant, with the fakes, plus ways to look inside."""

    corpus: Path
    client: QdrantClient
    state: Path
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

    def live(self) -> str:
        return live_generation(self.client)

    def manifest(self) -> Manifest:
        return parse_manifest(metadata(self.client, self.live())["manifest"])

    def documents(self) -> dict[str, Any]:
        return metadata(self.client, self.live())["manifest"]["documents"]

    def ids(self, generation: str | None = None) -> set[str]:
        """The chunk IDs a generation holds, the live one by default."""
        points, _ = self.client.scroll(generation or self.live(), limit=10_000)
        return {str(point.id) for point in points}

    def chunks(self, generation: str | None = None) -> list[Document]:
        """A generation's chunks as the retriever returns them, in point ID order."""
        points, _ = self.client.scroll(generation or self.live(), limit=10_000, with_payload=True)
        return [
            Document(
                id=str(p.id), page_content=p.payload["page_content"], metadata=p.payload["metadata"]
            )
            for p in points
            if p.payload is not None
        ]

    def listed(self) -> set[str]:
        """The chunk IDs the live manifest's documents derive to."""
        return self.manifest().chunk_ids()

    def generations(self) -> list[str]:
        return generations(self.client)


@pytest.fixture
def scratch(
    sample_corpus: Path,
    make_config: MakeConfig,
    qdrant: QdrantClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Scratch:
    monkeypatch.setenv(ENV_DATA_PATH, str(sample_corpus))
    return Scratch(
        corpus=sample_corpus, client=qdrant, state=tmp_path / "var", make_config=make_config
    )


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
    out, err = capsys.readouterr()
    assert "Nothing changed: the index is up to date." in err  # the log, on stderr (S3-7)
    assert "5 unchanged" in out
    assert "Nothing changed; the Qdrant collection 'rag_qa' at" in out


def test_an_add_a_change_and_a_delete_touch_only_those_documents(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    before, before_ids = scratch.documents(), scratch.manifest().chunk_ids_of("notes.txt")
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
    assert scratch.manifest().chunk_ids_of("notes.txt") != before_ids  # new bytes, new IDs
    assert scratch.ids() == scratch.listed()  # the generation holds exactly what is counted


def test_moving_md_to_another_loader_re_embeds_the_md_files_only(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    before = scratch.documents()["README.md"]
    before_ids = scratch.manifest().chunk_ids_of("README.md")
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
    # The same path, bytes and index give the same IDs back: left out of the copy, then
    # upserted with the new loader's chunks.
    assert scratch.manifest().chunk_ids_of("README.md") == before_ids
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
    documents, manifest = scratch.documents(), scratch.manifest()
    assert documents["a/same.txt"]["sha256"] == documents["b/same.txt"]["sha256"]
    assert manifest.chunk_ids_of("a/same.txt") != manifest.chunk_ids_of("b/same.txt")

    (corpus / "a" / "same.txt").unlink()
    report = scratch.ingest()
    assert report.removed == 1
    assert scratch.ids() == set(manifest.chunk_ids_of("b/same.txt"))


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
    before_ids = scratch.manifest().chunk_ids_of("guide.pdf")
    (scratch.corpus / "guide.pdf").write_bytes(b"corrupt")
    (scratch.corpus / "new.txt").write_text("Published beside the failure.\n", encoding="utf-8")

    report = scratch.ingest()
    assert report.published is not None  # new.txt went in...
    assert (report.added, report.kept) == (1, 1)
    assert scratch.documents()["guide.pdf"] == before  # ...and guide.pdf kept its chunks
    assert set(before_ids) <= scratch.ids()


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
    assert "Cannot list sub/" in err  # the log, on stderr (S3-7)
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
    err = capsys.readouterr().err
    # The first line is the log's, on stderr since S3-7; the second the outcome's.
    assert "Nothing changed apart from the documents that could not be read." in err
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
    assert "Documents:" not in out  # every count is zero, so the line is left out (S2-8)
    assert not published(scratch.client) and scratch.generations() == []


def test_a_prune_that_fails_after_the_flip_does_not_fail_the_run(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    from rag_qa import ingest as module

    calls: list[Any] = []

    def recover_then_fail(store: Any) -> list[str]:
        calls.append(store)
        if len(calls) == 1:
            return []  # recovery at the start
        raise StoreError("Qdrant at http://qdrant.test answered 500: the disk went away")

    monkeypatch.setattr(module, "recover", recover_then_fail)
    caplog.set_level(logging.INFO, logger="rag_qa.ingest")
    report = scratch.ingest()
    assert report.published is not None
    assert scratch.live() == report.published  # the switch stands
    assert "Could not remove an old generation: Qdrant at" in caplog.text
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
    ],
)
def test_the_flag_or_a_new_embedder_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], edit: Edit | None, rebuild: bool, reason: str
) -> None:
    # A new splitter did too, until S3-4: it now re-embeds the documents it chunks (below).
    scratch.ingest()
    embedded.clear()
    report = scratch.ingest(edit, rebuild=rebuild)
    assert report.rebuild is not None and report.rebuild.startswith(reason)
    assert embedded == [7]
    assert (report.added, report.unchanged) == (5, 0)


# --- a splitter per extension (S3-4) -------------------------------------------------------

MARKDOWN = "components.splitters.markdown"


def markdown_for_md(c: dict[str, Any]) -> None:
    c["pipeline"]["ingestion"]["splitters"] = {".md": MARKDOWN}


def resized(splitter: str, chunk_size: int, then: Edit | None = None) -> Edit:
    """Config edit: ``then``, and the splitter entry ``splitter`` cut at ``chunk_size``."""

    def edit(c: dict[str, Any]) -> None:
        if then is not None:
            then(c)
        c["components"]["splitters"][splitter]["chunk_size"] = chunk_size

    return edit


def identity_of(config: RagConfig, ref: str) -> str:
    return spec_identity(config.component(ref).spec())


def test_each_document_records_the_splitter_that_chunked_it(scratch: Scratch) -> None:
    config = scratch.config(markdown_for_md)
    ingest(config)
    default = identity_of(config, config.pipeline.ingestion.splitter)
    markdown = identity_of(config, MARKDOWN)
    assert markdown != default
    assert {name: record["splitter"] for name, record in scratch.documents().items()} == {
        name: markdown if name == "README.md" else default for name in DOCUMENTS
    }
    # The top level names the default, and decides nothing.
    assert scratch.manifest().splitter.identity == default


def test_a_md_file_is_chunked_by_its_own_splitter(scratch: Scratch) -> None:
    paragraph = " ".join(f"word{i:04d}" for i in range(60))  # 539 characters
    text = f"# First\n{paragraph}\n## Second\n{paragraph}\n"
    (scratch.corpus / "README.md").write_text(text, encoding="utf-8")
    scratch.ingest(markdown_for_md)
    readme = [c.page_content for c in scratch.chunks() if c.metadata["source"] == "README.md"]
    # Cut at the heading. The default splitter keeps "## Second" at the end of the first
    # chunk too (tests/test_components.py), and its 150-character overlap starts the second
    # with it as well: so the first chunk's end is what tells the two apart.
    assert sorted(chunk.split("\n")[0] for chunk in readme) == ["# First", "## Second"]
    assert not any(chunk.endswith("## Second") for chunk in readme)


def test_changing_the_md_splitter_re_embeds_the_md_files_only(
    scratch: Scratch, embedded: list[int]
) -> None:
    # The exit's evidence for SRS §7.2 (ARCHITECTURE.md's gap-closure table).
    scratch.ingest(markdown_for_md)
    before = scratch.documents()
    embedded.clear()

    report = scratch.ingest(resized("markdown", 500, then=markdown_for_md))
    assert report.rebuild is None
    assert (report.changed, report.unchanged) == (1, 4)
    assert embedded == [1]  # README.md's one chunk
    after = scratch.documents()
    assert after["README.md"]["splitter"] != before["README.md"]["splitter"]
    for name in DOCUMENTS - {"README.md"}:
        assert after[name] == before[name]  # untouched: the same splitter and ingested_at
    assert scratch.ids() == scratch.listed()


def test_a_new_default_splitter_re_embeds_the_documents_it_chunks_and_rebuilds_nothing(
    scratch: Scratch, embedded: list[int]
) -> None:
    # Where DEC-17 rebuilt in full, each document's own splitter decides (S3-4).
    scratch.ingest(markdown_for_md)
    before = scratch.documents()["README.md"]
    embedded.clear()

    edit = resized("english_recursive", 500, then=markdown_for_md)
    report = scratch.ingest(edit)
    assert report.rebuild is None
    assert (report.changed, report.unchanged) == (4, 1)
    assert embedded == [6]  # the PDF's 3 pages and one chunk each for the other 3
    assert scratch.documents()["README.md"] == before
    # The top level follows the default, at the publish.
    assert scratch.manifest().splitter.identity == identity_of(
        scratch.config(edit), "components.splitters.english_recursive"
    )
    assert scratch.ids() == scratch.listed()


def test_with_no_map_a_new_default_splitter_re_embeds_every_document(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    embedded.clear()
    report = scratch.ingest(resized("english_recursive", 500))
    assert (report.rebuild, report.changed, report.unchanged) == (None, 5, 0)
    assert embedded == [7]


def test_an_override_with_the_default_s_spec_re_embeds_nothing(
    scratch: Scratch, embedded: list[int]
) -> None:
    # The identity is the spec, never the entry's name (ADR-026): the chunks are the same.
    scratch.ingest()
    embedded.clear()

    def a_copy_for_md(c: dict[str, Any]) -> None:
        splitters = c["components"]["splitters"]
        splitters["copy"] = dict(splitters["english_recursive"])
        c["pipeline"]["ingestion"]["splitters"] = {".md": "components.splitters.copy"}

    report = scratch.ingest(a_copy_for_md)
    assert (report.unchanged, report.published) == (5, None)
    assert embedded == []


def test_dropping_an_override_re_embeds_its_files_with_the_default(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest(markdown_for_md)
    embedded.clear()
    report = scratch.ingest()
    assert (report.changed, report.unchanged) == (1, 4)
    assert embedded == [1]
    config = scratch.config()
    default = identity_of(config, config.pipeline.ingestion.splitter)
    assert scratch.documents()["README.md"]["splitter"] == default


# --- a maximum file size (S3-4) ------------------------------------------------------------


def max_file_mb(limit: float | None) -> Edit:
    return lambda c: c["pipeline"]["ingestion"].update(max_file_mb=limit)


def add_big_txt(scratch: Scratch) -> None:
    """5,000 bytes: above 0.004 MB, which every sample corpus file is below (1,327 at most)."""
    (scratch.corpus / "big.txt").write_text("x" * 4999 + "\n", encoding="utf-8")


def test_a_file_above_max_file_mb_is_skipped_reported_and_never_read(
    scratch: Scratch, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_big_txt(scratch)
    hashed: list[str] = []

    def recording(path: Path) -> str:
        hashed.append(path.name)
        return file_sha256(path)

    monkeypatch.setattr("rag_qa.ingest.file_sha256", recording)
    path = scratch.config_path(max_file_mb(0.004))
    assert main(["--config", str(path)]) == EXIT_OK  # skipped, as an unmapped file is
    out, err = capsys.readouterr()
    assert "  - Skipped big.txt (0.005 MB, above the 0.004 MB limit)" in err  # the log (S3-7)
    assert "Skipped 1 file(s) above the 0.004 MB limit (max_file_mb)." in out
    assert "big.txt" not in hashed  # a limit is there so that the file is never read
    assert set(scratch.documents()) == DOCUMENTS

    report = scratch.ingest(max_file_mb(0.004))
    assert (report.too_large, report.max_file_mb) == ([("big.txt", 5000)], 0.004)
    assert report.found == len(DOCUMENTS)  # mapped to a loader, but not admitted


@pytest.mark.parametrize(("limit", "admitted"), [(0.000249, True), (0.000248, False)])
def test_a_file_at_the_limit_is_admitted_and_one_byte_more_is_not(
    scratch: Scratch, limit: float, admitted: bool
) -> None:
    # 0.000249 MB is 248.99999999999997 bytes as a float product, which would turn the
    # 249-byte file away: the limit is compared exactly, as written.
    (scratch.corpus / "edge.txt").write_text("x" * 248 + "\n", encoding="utf-8")
    report = scratch.ingest(max_file_mb(limit))
    assert ("edge.txt" in scratch.documents()) is admitted
    assert ("edge.txt" in dict(report.too_large)) is not admitted


def test_a_file_indexed_before_a_lowered_limit_is_removed_like_a_deleted_one(
    scratch: Scratch, embedded: list[int]
) -> None:
    add_big_txt(scratch)
    scratch.ingest(max_file_mb(None))  # no limit
    assert "big.txt" in scratch.documents()
    embedded.clear()

    report = scratch.ingest(max_file_mb(0.004))
    assert (report.removed, report.unchanged) == (1, 5)
    assert embedded == []  # its chunks only go
    assert "big.txt" not in scratch.documents()
    assert scratch.ids() == scratch.listed()

    report = scratch.ingest(max_file_mb(None))  # the limit lifted: it comes back
    assert (report.added, report.unchanged) == (1, 5)


def drop_a_point(scratch: Scratch) -> None:
    (victim,) = scratch.manifest().chunk_ids_of("notes.txt")
    scratch.client.delete(scratch.live(), points_selector=models.PointIdsList(points=[victim]))


def add_a_stray_point(scratch: Scratch) -> None:
    stray = models.PointStruct(
        id="00000000-0000-5000-8000-000000000000",
        vector={"dense": [0.1] * 8},
        payload={"page_content": "x", "metadata": {}},
    )
    scratch.client.upsert(scratch.live(), points=[stray], wait=True)


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        pytest.param(
            lambda s: s.client.update_collection(s.live(), metadata={"manifest": None}),
            "its manifest cannot be used: there is none",
            id="no manifest",
        ),
        pytest.param(
            lambda s: edit_manifest(s.client, lambda m: m.update(version=99)),
            "its manifest cannot be used: unknown manifest version 99",
            id="unknown version",
        ),
        pytest.param(
            lambda s: edit_manifest(s.client, lambda m: m.update(generation="rag_qa--gen-x")),
            "its manifest describes 'rag_qa--gen-x', not this collection",
            id="a manifest naming another collection",
        ),
        pytest.param(
            lambda s: s.client.update_collection(s.live(), metadata={"complete": False}),
            "it is incomplete",
            id="marked incomplete",
        ),
        pytest.param(
            drop_a_point,
            "the manifest and its generation disagree: 1 chunk(s) counted but missing, "
            "0 held but not counted",
            id="a point missing",
        ),
        pytest.param(
            add_a_stray_point,
            "the manifest and its generation disagree: 0 chunk(s) counted but missing, "
            "1 held but not counted",
            id="a point not counted",
        ),
    ],
)
def test_a_damaged_index_rebuilds_in_full_even_when_the_corpus_is_unchanged(
    scratch: Scratch, embedded: list[int], damage: Callable[[Scratch], Any], reason: str
) -> None:
    # With nothing changed, a run that skipped these checks would call the index up to
    # date while rag-query refused it: the two would point at each other (S2-6's review).
    scratch.ingest()
    damage(scratch)
    embedded.clear()
    report = scratch.ingest()
    assert report.rebuild is not None and reason in report.rebuild
    assert embedded == [7]
    assert scratch.ids() == scratch.listed()


def test_a_new_chunking_version_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The code that makes chunks changed, the config did not: without this, every
    # unchanged file would keep its old chunks for good (S2-6's second review).
    scratch.ingest()
    monkeypatch.setattr("rag_qa.ingest.CHUNKING_VERSION", 2)
    embedded.clear()
    report = scratch.ingest()
    assert report.rebuild == (
        "this version makes chunks differently (chunking version 1 then, 2 now)"
    )
    assert embedded == [7]
    assert scratch.manifest().chunking == 2


def test_a_manifest_without_its_chunking_version_rebuilds_in_full(
    scratch: Scratch, embedded: list[int]
) -> None:
    # Version 1 defaulted a missing chunking version to 1, for S2-6's first indexes. Every
    # version-2 manifest writes it, so one without it is not one this code wrote.
    scratch.ingest()
    edit_manifest(scratch.client, lambda m: m.pop("chunking"))
    embedded.clear()
    report = scratch.ingest()
    assert report.rebuild is not None and "chunking: Field required" in report.rebuild
    assert embedded == [7]


def test_a_manifest_that_counts_a_chunk_its_generation_lacks_rebuilds_in_full(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest()
    edit_manifest(scratch.client, lambda m: m["documents"]["notes.txt"].update(chunks=2))
    (scratch.corpus / "new.txt").write_text("A change, so the run reads the index.\n")
    embedded.clear()

    report = scratch.ingest()
    assert report.rebuild == (
        "the manifest and its generation disagree: 1 chunk(s) counted but missing, "
        "0 held but not counted"
    )
    assert embedded == [8]  # everything, once: checked before anything was embedded
    assert scratch.ids() == scratch.listed()


def test_a_generation_that_cannot_be_read_rebuilds_in_full_and_says_why(
    scratch: Scratch, embedded: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.ingest()

    def unreadable(store: Any, generation: str) -> Any:
        raise StoreError("Qdrant at http://qdrant.test answered 500: segment is corrupt")

    monkeypatch.setattr("rag_qa.ingest.open_generation", unreadable)
    (scratch.corpus / "new.txt").write_text("A change, so the run reads the index.\n")
    embedded.clear()

    report = scratch.ingest()
    assert report.rebuild is not None
    assert report.rebuild.startswith("the index cannot be read (StoreError: Qdrant at")
    assert str(tmp_path) not in report.rebuild
    assert embedded == [8]


def test_qdrant_gone_mid_run_stops_it_with_exit_2_and_no_rebuild(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # An outage says nothing about the index: it must not cost a rebuild (DEC-19).
    scratch.ingest()
    live = scratch.live()

    def gone(store: Any, generation: str) -> Any:
        raise StoreUnavailableError("cannot use Qdrant at http://qdrant.test (ConnectError)")

    monkeypatch.setattr("rag_qa.ingest.open_generation", gone)
    assert scratch.main() == EXIT_CANNOT_START
    assert "cannot use Qdrant at" in capsys.readouterr().err
    assert scratch.live() == live and scratch.generations() == [live]


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


def test_a_crash_before_the_switch_publishes_nothing_and_the_next_run_clears_it(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    # Story step 11: update_collection_aliases fails mid-update. The live generation is
    # untouched, and the next run's recovery removes the orphan.
    scratch.ingest()
    live = scratch.live()
    before = scratch.ids()
    (scratch.corpus / "new.txt").write_text("Added just before the crash.\n", encoding="utf-8")

    def crash(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated crash before the switch")

    real = scratch.client.update_collection_aliases
    monkeypatch.setattr(scratch.client, "update_collection_aliases", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        scratch.ingest()
    monkeypatch.setattr(scratch.client, "update_collection_aliases", real)

    assert scratch.live() == live
    assert scratch.ids(live) == before
    (orphan,) = [name for name in scratch.generations() if name != live]
    assert metadata(scratch.client, orphan)["complete"] is True  # built, never published
    # Queries still read the old generation, which never had new.txt: checked on what the
    # query side resolves, and on every chunk the live generation holds (S2-6's review).
    config = scratch.config()
    assert check_index(config)[0] == live
    assert build_query_pipeline(config).retrieve is not None
    every = scratch.chunks(live)
    assert len(every) == 7
    assert {doc.metadata["source"] for doc in every} == DOCUMENTS

    caplog.set_level(logging.INFO, logger="rag_qa.ingest")
    report = scratch.ingest()  # recovery first: the orphan goes
    assert f"Removed what an earlier run left behind: {orphan}" in caplog.text
    assert orphan not in scratch.generations()
    assert report.added == 1
    assert scratch.generations() == sorted([live, report.published or ""])


def test_a_failure_in_the_apply_phase_publishes_nothing(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.ingest()
    live = scratch.live()
    before = scratch.ids()
    (taken,) = scratch.manifest().chunk_ids_of("notes.txt")
    (scratch.corpus / "new.txt").write_text("Its ID will collide.\n", encoding="utf-8")
    # Two chunks under one ID: the upsert keeps one, so the count check refuses to publish.
    monkeypatch.setattr("rag_qa.ingest.chunk_id", lambda *args: taken)

    with pytest.raises(StoreError, match="nothing was published"):
        scratch.ingest()
    assert scratch.live() == live
    assert scratch.ids(live) == before
    (orphan,) = [name for name in scratch.generations() if name != live]
    assert metadata(scratch.client, orphan)["complete"] is False


def test_only_the_live_and_the_previous_generation_survive(scratch: Scratch) -> None:
    names = [scratch.ingest(rebuild=True).published for _ in range(3)]
    assert scratch.generations() == names[1:]  # the names sort in the order they were made
    assert scratch.live() == names[2]


def test_with_its_live_collection_lost_an_ingest_keeps_every_complete_generation(
    scratch: Scratch, caplog: Any
) -> None:
    # Story step 11 (design review 2): deleting the alias's target deletes the alias. The
    # next ingest must not take that as "nothing is live" and delete the previous
    # generation, by then the last usable copy: only an incomplete orphan goes.
    previous = scratch.ingest().published
    (scratch.corpus / "new.txt").write_text("A second generation.\n", encoding="utf-8")
    live = scratch.ingest().published
    orphan = "rag_qa--gen-29991231T235959999999Z-0badf00d"
    scratch.client.create_collection(orphan, vectors_config={}, metadata={"complete": False})
    scratch.client.delete_collection(live or "")
    assert not published(scratch.client)

    caplog.set_level(logging.INFO, logger="rag_qa.ingest")
    report = scratch.ingest()
    assert f"Removed what an earlier run left behind: {orphan}" in caplog.text
    assert report.rebuild == "there is no index yet"
    assert scratch.generations() == [previous, report.published]  # the previous survived


@pytest.mark.parametrize("taken", ["a collection", "an alias of something else"])
def test_something_in_the_alias_s_way_exits_2_and_is_left_alone(
    scratch: Scratch, capsys: Any, taken: str
) -> None:
    # S2-6's second review, for Qdrant: someone's own data under the alias's name is never
    # switched away from, written over or deleted.
    if taken == "a collection":
        scratch.client.create_collection("rag_qa", vectors_config={}, metadata={"theirs": 1})
        expected = "is a collection of its own"
    else:
        scratch.client.create_collection("theirs", vectors_config={})
        scratch.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name="theirs", alias_name="rag_qa")
                )
            ]
        )
        expected = "is an alias of 'theirs', which is not one of its generations"
    before = sorted(c.name for c in scratch.client.get_collections().collections)
    assert scratch.main() == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert expected in err and "RAG_QDRANT_COLLECTION" in err
    assert sorted(c.name for c in scratch.client.get_collections().collections) == before
    assert not scratch.state.joinpath("ingest.lock").exists()  # refused before the lock


def test_the_old_index_variable_is_refused_and_nothing_is_written(
    scratch: Scratch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    # Caveat 9's habit (DEC-19): a scratch ingest that set only RAG_VECTOR_STORE_PATH would
    # otherwise write the real index. Exit 2, with the migration.
    monkeypatch.setenv(ENV_VECTOR_STORE_PATH, str(tmp_path / "index"))
    assert scratch.main() == EXIT_CANNOT_START
    err = capsys.readouterr().err
    assert "RAG_VECTOR_STORE_PATH is set" in err and "RAG_QDRANT_COLLECTION" in err
    assert not published(scratch.client) and not (tmp_path / "index").exists()


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
    scratch.state.mkdir(parents=True, exist_ok=True)
    lock = scratch.state / "ingest.lock"
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
    assert not published(scratch.client)  # it never started


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
    docs = scratch.chunks()
    assert len(docs) == 7
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
        "LOUD.TXT",
        "README.md",
        "guide.pdf, p. 1",
        "guide.pdf, p. i",
        "guide.pdf, p. ii",
        "notes.txt",
        "report.docx",
    ]


#: ``sample_corpus``'s chunks under each CHUNKING_VERSION, as :func:`chunks_digest` gives.
SAMPLE_CORPUS_CHUNKS = {1: "556f8396d1dba0f5a1f48ff6a8075f78fa862cb5f84bc6c53117b7770b58162b"}


def chunks_digest(scratch: Scratch) -> str:
    """The sha256 of the live generation's chunks in walk order, as FAISS held them: text
    and metadata, less the file's hash and the run's time, which are not how chunks are
    made. Qdrant keeps points in ID order, so walk order is rebuilt from the manifest: each
    file as the walk finds it, then its chunks by index."""
    manifest = scratch.manifest()
    scan = scan_corpus(scratch.config(), IngestReport())
    ordered = [cid for file in scan.files for cid in manifest.chunk_ids_of(file.name)]
    by_id = {doc.id: doc for doc in scratch.chunks()}
    chunks = []
    for cid in ordered:
        doc = by_id[cid]
        kept = {
            key: value
            for key, value in doc.metadata.items()
            if key not in ("source_sha256", "ingested_at")
        }
        chunks.append([doc.page_content, kept])
    return hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()


def test_the_sample_corpus_chunks_as_the_chunking_version_says(scratch: Scratch) -> None:
    # The tripwire for CHUNKING_VERSION (second review). If the chunks changed, whether
    # by a change to the loaders or to ingest's metadata, or by a library upgrade, bump
    # CHUNKING_VERSION and pin this digest under the new version: every index then
    # rebuilds once, instead of keeping old chunks for every unchanged file.
    scratch.ingest()
    digest = chunks_digest(scratch)
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
    # vectors of another size, which the store would refuse, every run.
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
    vectors = scratch.client.get_collection(scratch.live()).config.params.vectors
    assert isinstance(vectors, dict) and vectors["dense"].size == 16


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
    err = capsys.readouterr().err
    # The log's lines, on stderr since S3-7, then the outcome's.
    assert "Nothing changed apart from the documents that could not be read." in err
    assert "up to date" not in err.split("Nothing changed apart")[0]
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
    # Paths outside the corpus and paths.state used to pass through: a module's file named
    # in an error, say, or the lock's own path.
    from rag_qa.ingest import describe_error

    config = scratch.config()
    moved = AttributeError(
        "Can't get attribute 'X' on <module 'langchain_core.documents.base' from "
        "'/home/someone/project/.venv/lib/python3.12/site-packages/langchain_core/documents/base.py'>"
    )
    described = describe_error(moved, config)
    assert "/home/someone" not in described
    assert "from 'base.py'>" in described
    lock = OSError(30, "Read-only file system", f"{config.paths.state}/ingest.lock")
    assert describe_error(lock, config) == (
        "OSError: [Errno 30] Read-only file system: 'ingest.lock'"
    )
    elsewhere = OSError(13, "Permission denied", "/var/data/secret.txt")  # a PermissionError
    assert describe_error(elsewhere, config) == (
        "PermissionError: [Errno 13] Permission denied: 'secret.txt'"
    )


# --- sparse vectors (S3-5, DEC-20) ---------------------------------------------------------

BM25 = "components.sparse_encoders.bm25"


def with_bm25(c: dict[str, Any]) -> None:
    """Config edit: the real config's BM25 encoder set for ingestion."""
    c["pipeline"]["ingestion"]["sparse_encoder"] = BM25


def bm25_with(**settings: Any) -> Edit:
    def edit(c: dict[str, Any]) -> None:
        with_bm25(c)
        c["components"]["sparse_encoders"]["bm25"].update(settings)

    return edit


def bm25_renamed(c: dict[str, Any]) -> None:
    entries = c["components"]["sparse_encoders"]
    entries["renamed"] = entries.pop("bm25")
    c["pipeline"]["ingestion"]["sparse_encoder"] = "components.sparse_encoders.renamed"


def sparse_points(scratch: Scratch, generation: str | None = None) -> dict[str, Any]:
    """Each point's sparse vector, as ``(indices, values)``; ``None`` without one."""
    points, _ = scratch.client.scroll(generation or scratch.live(), limit=10_000, with_vectors=True)
    found = {}
    for p in points:
        vector = p.vector.get("bm25") if isinstance(p.vector, dict) else None
        found[str(p.id)] = None if vector is None else (vector.indices, vector.values)
    return found


def test_with_a_sparse_encoder_every_chunk_gets_its_bm25_vector(scratch: Scratch) -> None:
    config = scratch.config(with_bm25)
    ingest(config)
    record = scratch.manifest().sparse
    assert record is not None and record.ref == BM25
    assert record.identity == spec_identity(config.component(BM25).spec())
    encoder = Bm25Encoder(k1=1.2, b=0.75, avg_len=100.76)
    texts = {doc.id: doc.page_content for doc in scratch.chunks()}
    held = sparse_points(scratch)
    assert held.keys() == texts.keys() == scratch.listed()
    for pid, text in texts.items():
        expected = encoder.encode_document(text)
        indices, values = held[pid]
        assert indices == list(expected.indices)
        assert values == pytest.approx(list(expected.values))


def test_without_one_the_index_has_no_sparse_vectors(scratch: Scratch) -> None:
    scratch.ingest()
    assert scratch.manifest().sparse is None
    assert all(vector is None for vector in sparse_points(scratch).values())


@pytest.mark.parametrize(
    ("first", "then", "reason"),
    [
        pytest.param(None, with_bm25, "a sparse encoder is now set", id="appearing"),
        pytest.param(with_bm25, None, "no sparse encoder is set now", id="going"),
        pytest.param(
            with_bm25,
            bm25_with(k1=1.5),
            "the sparse encoder is not the one that built the index",
            id="another spec",
        ),
    ],
)
def test_a_sparse_encoder_that_appears_goes_or_changes_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], first: Edit | None, then: Edit | None, reason: str
) -> None:
    # The generation's vector configuration changes, or every sparse vector does (DEC-20).
    scratch.ingest(first)
    embedded.clear()
    report = scratch.ingest(then)
    assert report.rebuild is not None and report.rebuild.startswith(reason)
    assert embedded == [7]
    # Every point has a sparse vector after the rebuild, or none does.
    assert all((vector is None) == (then is None) for vector in sparse_points(scratch).values())


def test_a_new_sparse_version_rebuilds_in_full(
    scratch: Scratch, embedded: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch.ingest(with_bm25)
    monkeypatch.setattr("rag_qa.manifest.SPARSE_VERSION", SPARSE_VERSION + 1)
    embedded.clear()
    report = scratch.ingest(with_bm25)
    assert report.rebuild == (
        f"this version makes sparse vectors differently (sparse version {SPARSE_VERSION} "
        f"then, {SPARSE_VERSION + 1} now)"
    )
    assert embedded == [7]
    sparse = scratch.manifest().sparse
    assert sparse is not None and sparse.version == SPARSE_VERSION + 1


def test_a_renamed_sparse_encoder_with_the_same_spec_rebuilds_nothing(
    scratch: Scratch, embedded: list[int]
) -> None:
    scratch.ingest(with_bm25)
    embedded.clear()
    report = scratch.ingest(bm25_renamed)
    assert (report.rebuild, report.published, embedded) == (None, None, [])


def test_an_update_of_a_hybrid_index_equals_a_rebuild(scratch: Scratch) -> None:
    # The copy carries the unchanged chunks' sparse vectors; the changed and added ones get
    # new ones. Qdrant applies the IDF at query time, so nothing else moves.
    scratch.ingest(with_bm25)
    (scratch.corpus / "notes.txt").write_text("Kestrel Island has forty staff.\n")
    (scratch.corpus / "terns.txt").write_text("Terns nest on Kestrel Island.\n")
    (scratch.corpus / "sub" / "deeper" / "LOUD.TXT").unlink()
    updated = scratch.ingest(with_bm25)
    assert updated.rebuild is None and (updated.added, updated.changed, updated.removed) == (
        1,
        1,
        1,
    )
    after_update = sparse_points(scratch)
    rebuilt = scratch.ingest(with_bm25, rebuild=True)
    assert rebuilt.rebuild == "--rebuild was given"
    after_rebuild = sparse_points(scratch)
    assert after_update.keys() == after_rebuild.keys()
    for pid, (indices, values) in after_rebuild.items():
        assert after_update[pid][0] == indices
        assert after_update[pid][1] == pytest.approx(values)


#: ``sample_corpus``'s chunks' BM25 vectors under each SPARSE_VERSION, as
#: :func:`sparse_digest` gives, with config.yaml's bm25 settings.
SAMPLE_CORPUS_SPARSE = {1: "026af369bdb1c0207b2fb5d21d78c496c9e5af8f54182f236245b3e560ce0007"}


def sparse_digest(scratch: Scratch) -> str:
    """The sha256 of the encoder's vectors for the live generation's chunks, in walk order:
    the encoder's own output, so the digest does not depend on how Qdrant stores floats."""
    manifest = scratch.manifest()
    scan = scan_corpus(scratch.config(), IngestReport())
    by_id = {doc.id: doc.page_content for doc in scratch.chunks()}
    encoder = Bm25Encoder(k1=1.2, b=0.75, avg_len=100.76)
    vectors = []
    for file in scan.files:
        for cid in manifest.chunk_ids_of(file.name):
            vector = encoder.encode_document(by_id[cid])
            vectors.append([list(vector.indices), list(vector.values)])
    return hashlib.sha256(json.dumps(vectors).encode()).hexdigest()


def test_the_sample_corpus_encodes_as_the_sparse_version_says(scratch: Scratch) -> None:
    # The tripwire for SPARSE_VERSION over real chunks, beside tests/test_sparse.py's probe
    # texts. If it moved, bump SPARSE_VERSION and pin this digest under the new version.
    scratch.ingest()
    digest = sparse_digest(scratch)
    assert SAMPLE_CORPUS_SPARSE.get(SPARSE_VERSION) == digest, (
        f"the sample corpus no longer encodes as SPARSE_VERSION {SPARSE_VERSION} did: bump it "
        f"in rag_qa/sparse.py, and pin {digest} under the new version"
    )
