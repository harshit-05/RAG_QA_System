"""rag-ingest's error policy and CLI surface (S1-4: ISS-05, NFR-7).

These run the real ``main()`` end to end — discovery, loading, splitting, embedding,
publishing a generation — with a deterministic fake embedder, so no model download.
RAG_DATA_PATH points at scratch, and the store is the test's own in-memory Qdrant
(conftest's ``qdrant``): the real index must never be touched by a test.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import pytest
from conftest import MakeConfig, published, use_fake_embedder
from qdrant_client import QdrantClient

from rag_qa.config import load_config
from rag_qa.ingest import (
    EXIT_CANNOT_START,
    EXIT_OK,
    EXIT_RUN_FAILED,
    IngestReport,
    discover_files,
    ingest,
    main,
    outcome,
    scan_corpus,
)
from rag_qa.loaders import PdfLoader
from rag_qa.observability import logs


@pytest.fixture
def run(make_config: MakeConfig, qdrant: QdrantClient, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run ``rag-ingest --config <scratch config>`` over ``corpus``; return (status, store)."""

    def run_ingest(corpus: Path, edit: Any = None) -> tuple[int, QdrantClient]:
        monkeypatch.setenv("RAG_DATA_PATH", str(corpus))

        def edits(c: dict[str, Any]) -> None:
            use_fake_embedder(c)
            if edit:
                edit(c)

        return main(["--config", str(make_config(edits))]), qdrant

    return run_ingest


# --- exit codes -------------------------------------------------------------------------


def test_clean_run_exits_0_and_saves_the_index(run: Any, sample_corpus: Path, capsys: Any) -> None:
    status, store = run(sample_corpus)
    assert status == EXIT_OK
    assert published(store)
    assert "Ingestion Complete" in capsys.readouterr().out


def test_one_corrupt_document_is_reported_does_not_abort_and_fails_the_run(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    # NFR-7: the other documents are still indexed; the run still fails.
    (sample_corpus / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    status, store = run(sample_corpus)
    out, err = capsys.readouterr()

    assert status == EXIT_RUN_FAILED
    assert published(store)  # rebuilt from the rest
    assert "Loaded 7 document pages/sections" in out  # 3 PDF pages + 4 single-doc files
    assert "Failed 1 item(s)." in out
    # The exception type is recorded. Which pypdf error it is (PdfStreamError here)
    # is pypdf's detail and can change between versions, so match the shape only.
    assert re.search(r"broken\.pdf: \w+Error: ", out)
    assert "could not be read" in err and "rebuilt without them" in err


def test_nothing_to_index_exits_1(run: Any, tmp_path: Path, capsys: Any) -> None:
    corpus = tmp_path / "only-images"
    corpus.mkdir()
    (corpus / "diagram.png").write_bytes(b"\x89PNG")
    status, store = run(corpus)
    assert status == EXIT_RUN_FAILED
    assert not published(store)
    assert "No documents were loaded" in capsys.readouterr().err


def test_invalid_config_exits_2_with_the_message_not_a_traceback(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    status, _ = run(
        sample_corpus, lambda c: c["components"].__setitem__("llmS", c["components"].pop("llms"))
    )
    err = capsys.readouterr().err
    assert status == EXIT_CANNOT_START
    assert "did you mean 'llms'" in err
    assert "Traceback" not in err


def test_missing_corpus_directory_exits_2(run: Any, tmp_path: Path, capsys: Any) -> None:
    status, _ = run(tmp_path / "nope")
    assert status == EXIT_CANNOT_START
    assert "Corpus directory not found" in capsys.readouterr().err


def test_an_unreadable_corpus_directory_exits_2_like_a_missing_one(
    run: Any, tmp_path: Path, unreadable: Any, capsys: Any
) -> None:
    # Before S2-8 the walk reported the root itself as "./: PermissionError" and exited 1,
    # as if one document were bad.
    corpus = tmp_path / "locked"
    corpus.mkdir()
    unreadable(corpus)
    status, _ = run(corpus)
    out, err = capsys.readouterr()
    assert status == EXIT_CANNOT_START
    assert f"Error: Corpus directory not readable: {corpus} (Permission denied; set by " in err
    assert "Traceback" not in err
    assert "./" not in out
    assert not (tmp_path / "var" / "ingest.lock").exists()  # refused before the lock


def test_a_loader_that_cannot_be_built_is_a_config_error_not_a_document_failure(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    # A kwarg typo breaks every .txt file the same way: stop and name it, rather
    # than report each text file as a "failed document".
    status, _ = run(
        sample_corpus, lambda c: c["components"]["loaders"]["txt"].update(encodng="utf-8")
    )
    err = capsys.readouterr().err
    assert status == EXIT_CANNOT_START
    assert "Cannot build the loader 'components.loaders.txt'" in err
    assert "encodng" in err


def test_help_does_no_work(tmp_path: Path, qdrant: QdrantClient, capsys: Any) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert not published(qdrant) and not (tmp_path / "var").exists()
    out = capsys.readouterr().out
    assert "--config" in out and "Exit status" in out
    # A crash also exits 1 (Python's default); the contract says so rather than
    # promising 1 means only "a document could not be read".
    assert "unexpected error" in " ".join(out.split())


def test_config_flag_beats_rag_config(
    run: Any, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_CONFIG", str(tmp_path / "does-not-exist.yaml"))
    status, _ = run(sample_corpus)  # passes --config explicitly
    assert status == EXIT_OK


def test_ctrl_c_is_not_swallowed_as_a_document_failure(
    sample_corpus: Path,
    make_config: MakeConfig,
    qdrant: QdrantClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(self: PdfLoader) -> list:
        raise KeyboardInterrupt

    monkeypatch.setattr(PdfLoader, "load", interrupted)
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    with pytest.raises(KeyboardInterrupt):
        ingest(load_config(make_config(use_fake_embedder)))
    assert not published(qdrant)  # and nothing was published


# --- what the walk passes over is reported, not silent ----------------------------------


def test_ignored_files_are_counted(sample_corpus: Path) -> None:
    # sample_corpus holds 3: a checkpoint copy, a hidden file, a Word lock file.
    assert discover_files(sample_corpus).ignored == 3


def test_symlinks_are_never_followed_and_are_listed(tmp_path: Path) -> None:
    corpus, outside = tmp_path / "corpus", tmp_path / "outside"
    (outside / "folder").mkdir(parents=True)
    corpus.mkdir()
    (outside / "secret.txt").write_text("outside the corpus\n")
    (outside / "folder" / "inner.txt").write_text("also outside\n")
    (corpus / "real.txt").write_text("in the corpus\n")
    (corpus / "file-link.txt").symlink_to(outside / "secret.txt")
    (corpus / "folder-link").symlink_to(outside / "folder")
    (corpus / "broken-link.txt").symlink_to(tmp_path / "missing.txt")

    listing = discover_files(corpus)
    assert [p.name for p in listing.files] == ["real.txt"]
    assert sorted(p.name for p in listing.symlinks) == [
        "broken-link.txt",
        "file-link.txt",
        "folder-link",
    ]


@pytest.fixture
def unreadable() -> Any:
    """Make folders unlistable (chmod 000) for one test, and restore them after."""
    if os.geteuid() == 0:
        pytest.skip("root can list any folder, so chmod 000 proves nothing")
    locked: list[Path] = []

    def lock(folder: Path) -> Path:
        folder.chmod(0)
        locked.append(folder)
        return folder

    yield lock
    for folder in locked:
        folder.chmod(0o755)  # or tmp_path cleanup cannot remove it


def test_an_unreadable_folder_is_listed_not_silently_skipped(
    sample_corpus: Path, unreadable: Any
) -> None:
    # Python 3.12's rglob passes over a folder it cannot open without a word, which
    # would drop everything inside it from the index unnoticed (NFR-7).
    (sample_corpus / "locked").mkdir()
    (sample_corpus / "locked" / "inside.txt").write_text("never seen\n")
    unreadable(sample_corpus / "locked")

    listing = discover_files(sample_corpus)
    assert [(p.name, error.split(":")[0]) for p, error in listing.unreadable] == [
        ("locked", "PermissionError")
    ]


def test_an_unreadable_folder_fails_the_run_and_the_rest_is_indexed(
    run: Any, sample_corpus: Path, unreadable: Any, capsys: Any
) -> None:
    (sample_corpus / "locked").mkdir()
    (sample_corpus / "locked" / "inside.txt").write_text("never seen\n")
    unreadable(sample_corpus / "locked")

    status, store = run(sample_corpus)
    out = capsys.readouterr().out
    assert status == EXIT_RUN_FAILED
    assert published(store)
    assert "Loaded 7 document pages/sections" in out
    assert re.search(r"locked/: PermissionError: ", out)


def test_an_unreadable_folder_inside_an_ignored_one_does_not_fail(
    sample_corpus: Path, unreadable: Any
) -> None:
    # .git objects or a checkpoint folder are not documents, readable or not.
    unreadable(sample_corpus / ".ipynb_checkpoints")
    assert discover_files(sample_corpus).unreadable == []


def test_the_summary_accounts_for_everything_not_indexed(
    sample_corpus: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sample_corpus / "link.md").symlink_to(sample_corpus / "README.md")
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    report = IngestReport()
    scan_corpus(load_config(make_config()), report)

    assert report.skipped == ["diagram.png"]
    assert report.symlinks == ["link.md"]
    assert report.ignored == 3  # a checkpoint folder (counted once), a hidden file, a lock file
    summary = report.summary()
    assert "Skipped 1 symlink(s) (not followed)." in summary
    assert "Ignored 3 hidden or lock file(s) and folder(s)." in summary


@pytest.mark.parametrize(
    ("size", "limit", "shown"),
    [
        pytest.param(
            25_370_930, 20, "25.4 MB, above the 20 MB limit", id="the largest corpus file"
        ),
        pytest.param(1_521_034, 1, "1.52 MB, above the 1 MB limit", id="three figures"),
        pytest.param(1_530, 0.001, "0.00153 MB, above the 0.001 MB limit", id="under a MB"),
        pytest.param(1_250_000_000, 50, "1,250 MB, above the 50 MB limit", id="over a GB"),
        pytest.param(5_000, 0.004, "0.005 MB, above the 0.004 MB limit", id="no trailing zero"),
        # Three figures would say 12.3: under the limit, on the line that says above it.
        pytest.param(
            12_345_001, 12.345, "12.35 MB, above the 12.345 MB limit", id="never reads as under"
        ),
        pytest.param(
            50_000_001, 50, "50.000001 MB, above the 50 MB limit", id="one byte over, exactly"
        ),
        pytest.param(15, 0.00001, "0.000015 MB, above the 0.00001 MB limit", id="no exponent"),
    ],
)
def test_a_file_above_the_limit_is_logged_with_its_size_and_the_limit(
    tmp_path: Path,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    size: int,
    limit: float,
    shown: str,
) -> None:
    corpus = tmp_path / "sizes"
    corpus.mkdir()
    with open(corpus / "big.pdf", "wb") as f:
        f.truncate(size)  # sparse, so instant: above the limit, it is never read
    monkeypatch.setenv("RAG_DATA_PATH", str(corpus))
    config = load_config(
        make_config(lambda c: c["pipeline"]["ingestion"].update(max_file_mb=limit))
    )
    report = IngestReport()
    with caplog.at_level(logging.INFO, logger="rag_qa.ingest"):
        scan_corpus(config, report)
    assert f"  - Skipped big.pdf ({shown})" in caplog.messages
    assert (report.too_large, report.found) == ([("big.pdf", size)], 0)


@pytest.mark.parametrize(
    ("report", "summary"),
    [
        pytest.param(IngestReport(), "", id="nothing at all"),
        pytest.param(
            IngestReport(found=3, unchanged=3, ignored=1),
            "Documents: 3 unchanged.\nIgnored 1 hidden or lock file(s) and folder(s).",
            id="nothing changed",
        ),
        pytest.param(
            IngestReport(
                documents=1,
                chunks=1,
                found=1,
                added=1,
                rebuild="there is no index yet",
                published="g",
            ),
            "Loaded 1 document page/section, split into 1 chunk.\n"
            "Rebuilt the index in full: there is no index yet.\n"
            "Documents: 1 added.",
            id="one text file",
        ),
        pytest.param(
            IngestReport(
                documents=4,
                chunks=9,
                found=5,
                added=1,
                changed=1,
                unchanged=2,
                kept=1,
                skipped=["a.png"],
                symlinks=["l.md"],
                failed=[("b.pdf", "PdfReadError: bad"), ("locked/", "PermissionError: no")],
                published="g",
            ),
            "Loaded 4 document pages/sections, split into 9 chunks.\n"
            "Documents: 1 added, 1 changed, 2 unchanged; 1 failed and kept their previous "
            "chunks.\n"
            "Skipped 1 file(s) with no configured loader.\n"
            "Skipped 1 symlink(s) (not followed).\n"
            "Failed 2 item(s).\n"
            "  - b.pdf: PdfReadError: bad\n"
            "  - locked/: PermissionError: no",
            id="an update with failures",
        ),
        pytest.param(
            IngestReport(found=1, unchanged=1, too_large=[("big.pdf", 60_200_000)], max_file_mb=50),
            "Documents: 1 unchanged.\nSkipped 1 file(s) above the 50 MB limit (max_file_mb).",
            id="a file above max_file_mb (S3-4)",
        ),
        pytest.param(
            IngestReport(too_large=[("big.pdf", 60_200_000)]),  # built by hand: no limit
            "Skipped 1 file(s) above the limit (max_file_mb).",
            id="a file above an unrecorded limit",
        ),
    ],
)
def test_the_summary_leaves_out_every_zero_count(report: IngestReport, summary: str) -> None:
    # "Skipped 0 symlink(s)" and its kind printed on every run (S1-4), and S2-6 added a
    # "Documents:" line of up to five zeros. "item(s)": a folder can fail too.
    assert report.summary() == summary


@pytest.mark.parametrize(
    ("report", "line"),
    [
        pytest.param(
            IngestReport(
                documents=561, chunks=1708, points=1708, load_s=9.8, embed_s=173.5, write_s=3.9
            ),
            "Throughput: 57.2 pages/s loaded, 9.84 chunks/s embedded, 438 points/s written.",
            id="a rebuild",
        ),
        pytest.param(
            IngestReport(points=1708, write_s=1.2),  # an update that only removed a document
            "Throughput: 1,423 points/s written.",
            id="only a write",
        ),
        pytest.param(
            IngestReport(documents=1, chunks=1, points=1, load_s=2.0, embed_s=4.0, write_s=8.0),
            "Throughput: 0.500 pages/s loaded, 0.250 chunks/s embedded, 0.125 points/s written.",
            id="below one a second",
        ),
    ],
)
def test_the_summary_reports_the_runs_throughput(report: IngestReport, line: str) -> None:
    # NFR-3 (S3-7): pages loaded, chunks embedded and points written per second, for the
    # phases that ran; three significant figures, never an exponent.
    assert line in report.summary().splitlines()


def test_one_document_of_one_chunk_is_logged_in_the_singular(
    tmp_path: Path,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The two plural slips S2-8 left (backlog): "Embedding 1 chunks..." and "Published …:
    # 1 documents, 1 chunks." (S3-4 found the second again).
    corpus = tmp_path / "one"
    corpus.mkdir()
    (corpus / "note.txt").write_text("A single short note.\n")
    monkeypatch.setenv("RAG_DATA_PATH", str(corpus))
    with caplog.at_level(logging.INFO, logger="rag_qa.ingest"):
        report = ingest(load_config(make_config(use_fake_embedder)))
    assert "Embedding 1 chunk..." in caplog.messages
    published = [r for r in caplog.records if r.getMessage().startswith("Published ")]
    (line,) = published
    assert line.getMessage() == f"Published {report.published}: 1 document, 1 chunk."
    # The same line carries NFR-3's numbers as fields, for the JSON log (S3-7).
    fields = line.__dict__
    assert (fields["documents"], fields["points"], fields["pages_loaded"]) == (1, 1, 1)
    assert (
        fields["chunks_embedded"] == 1 and fields["write_s"] >= 0
    )  # rounded to ms: a write can be 0.0
    assert {"pages_per_s", "chunks_per_s", "points_per_s"} <= fields.keys()
    assert report.points == 1 and report.load_s > 0 and report.embed_s > 0


def test_no_line_of_a_real_run_prints_a_zero_count(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    # Verification 4's grep, over a full rebuild, a run that changes nothing and an update.
    # Both streams: the summary is printed, and the progress is the log, on stderr (S3-7).
    outputs = []
    for change in [None, None, "added.txt"]:
        if change:
            (sample_corpus / change).write_text("One more document.\n")
        status, _ = run(sample_corpus)
        assert status == EXIT_OK
        captured = capsys.readouterr()
        outputs.append(captured.out + captured.err)
    for out in outputs:
        assert re.findall(r".* 0 .*", out) == []
    assert "Nothing changed" in outputs[1]
    assert "Updating the index: 1 added, 5 unchanged." in outputs[2]
    # A text file is one section, not "1 pages/sections" (found in S2-6).
    assert "    1 page/section\n" in outputs[0]
    assert re.search(r"\b1 (document )?pages/sections", "".join(outputs)) is None


def test_a_corrupt_pdf_prints_the_clean_error_without_pypdfs_own_warnings(
    run: Any,
    sample_corpus: Path,
    make_config: MakeConfig,
    caplog: pytest.LogCaptureFixture,
    capsys: Any,
) -> None:
    (sample_corpus / "broken.pdf").write_bytes(b"not a pdf at all\n")
    with caplog.at_level(logging.WARNING):
        status, _ = run(sample_corpus)
    assert status == EXIT_RUN_FAILED
    assert re.search(r"broken\.pdf: \w+Error: ", capsys.readouterr().out)
    assert [r.getMessage() for r in caplog.records if r.name.startswith("pypdf")] == []
    # For the whole process since S3-7, so the API's ingest jobs are quiet too.
    assert logging.getLogger("pypdf").level == logging.ERROR

    # The control: without rag-ingest's setting, the same file does log pypdf's warnings,
    # so the check above could have failed.
    logs.reset_logging()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ingest(load_config(make_config(use_fake_embedder)))
    assert "invalid pdf header: b'not a'" in [
        r.getMessage() for r in caplog.records if r.name.startswith("pypdf")
    ]


@pytest.mark.parametrize(
    ("report", "status", "line"),
    [
        (IngestReport(found=3, published="g"), EXIT_OK, None),
        (IngestReport(found=3), EXIT_OK, None),  # nothing changed
        (
            IngestReport(found=2, failed=[("a.pdf", "PdfReadError: bad")], published="g"),
            EXIT_RUN_FAILED,
            (
                "1 item(s) could not be read (listed above). The index was updated; the failed "
                "documents kept their previous chunks, if they had any, and are retried next run."
            ),
        ),
        (IngestReport(), EXIT_RUN_FAILED, "No documents were loaded."),
        (
            IngestReport(found=1, rebuild="--rebuild was given"),
            EXIT_RUN_FAILED,
            "Nothing was indexed: no document had any text.",
        ),
    ],
)
def test_outcome_is_the_rule_rag_ingest_exits_by(
    report: IngestReport, status: int, line: str | None
) -> None:
    # Shared with the API's ingest jobs, whose exit_code it sets (S2-7).
    assert outcome(report) == (status, line)
