"""rag-ingest's error policy and CLI surface (S1-4: ISS-05, NFR-7).

These run the real ``main()`` end to end — discovery, loading, splitting, embedding,
saving a FAISS index — with a deterministic fake embedder, so no model download.
Both RAG_DATA_PATH and RAG_VECTOR_STORE_PATH always point at scratch: the real
index must never be touched by a test.
"""

import re
from pathlib import Path
from typing import Any

import pytest
from conftest import MakeConfig

from rag_qa.config import load_config
from rag_qa.ingest import (
    EXIT_CANNOT_START,
    EXIT_OK,
    EXIT_RUN_FAILED,
    IngestReport,
    discover_files,
    load_documents,
    main,
)
from rag_qa.loaders import PdfLoader
from rag_qa.vectorstore import store_exists


def _fake_embedder(c: dict[str, Any]) -> None:
    c["components"]["embedders"]["fake"] = {
        "_target_": "langchain_core.embeddings.DeterministicFakeEmbedding", "size": 8
    }
    c["pipeline"]["ingestion"]["embedder"] = "components.embedders.fake"


@pytest.fixture
def run(make_config: MakeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run ``rag-ingest --config <scratch config>`` over ``corpus``; return (status, index dir)."""

    def run_ingest(corpus: Path, edit: Any = None) -> tuple[int, Path]:
        index = tmp_path / "index"
        monkeypatch.setenv("RAG_DATA_PATH", str(corpus))
        monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(index))

        def edits(c: dict[str, Any]) -> None:
            _fake_embedder(c)
            if edit:
                edit(c)

        return main(["--config", str(make_config(edits))]), index

    return run_ingest


# --- exit codes -------------------------------------------------------------------------


def test_clean_run_exits_0_and_saves_the_index(run: Any, sample_corpus: Path, capsys: Any) -> None:
    status, index = run(sample_corpus)
    assert status == EXIT_OK
    assert store_exists(index)
    assert "Ingestion Complete" in capsys.readouterr().out


def test_one_corrupt_document_is_reported_does_not_abort_and_fails_the_run(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    # NFR-7: the other documents are still indexed; the run still fails.
    (sample_corpus / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    status, index = run(sample_corpus)
    out, err = capsys.readouterr()

    assert status == EXIT_RUN_FAILED
    assert store_exists(index)                       # rebuilt from the rest
    assert "Loaded 7 document pages/sections" in out  # 3 PDF pages + 4 single-doc files
    assert "Failed 1 file(s)." in out
    # The exception type is recorded. Which pypdf error it is (PdfStreamError here)
    # is pypdf's detail and can change between versions, so match the shape only.
    assert re.search(r"broken\.pdf: \w+Error: ", out)
    assert "could not be read" in err and "rebuilt without them" in err


def test_nothing_to_index_exits_1(run: Any, tmp_path: Path, capsys: Any) -> None:
    corpus = tmp_path / "only-images"
    corpus.mkdir()
    (corpus / "diagram.png").write_bytes(b"\x89PNG")
    status, index = run(corpus)
    assert status == EXIT_RUN_FAILED
    assert not store_exists(index)
    assert "No documents were loaded" in capsys.readouterr().err


def test_invalid_config_exits_2_with_the_message_not_a_traceback(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    status, _ = run(sample_corpus, lambda c: c["components"].__setitem__("llmS", c["components"].pop("llms")))
    err = capsys.readouterr().err
    assert status == EXIT_CANNOT_START
    assert "did you mean 'llms'" in err
    assert "Traceback" not in err


def test_missing_corpus_directory_exits_2(run: Any, tmp_path: Path, capsys: Any) -> None:
    status, _ = run(tmp_path / "nope")
    assert status == EXIT_CANNOT_START
    assert "Corpus directory not found" in capsys.readouterr().err


def test_a_loader_that_cannot_be_built_is_a_config_error_not_a_document_failure(
    run: Any, sample_corpus: Path, capsys: Any
) -> None:
    # A kwarg typo breaks every .txt file the same way: stop and name it, rather
    # than report each text file as a "failed document".
    status, _ = run(sample_corpus, lambda c: c["components"]["loaders"]["txt"].update(encodng="utf-8"))
    err = capsys.readouterr().err
    assert status == EXIT_CANNOT_START
    assert "Cannot build the loader 'components.loaders.txt'" in err
    assert "encodng" in err


def test_help_does_no_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    index = tmp_path / "index"
    monkeypatch.setenv("RAG_VECTOR_STORE_PATH", str(index))
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert not index.exists()
    out = capsys.readouterr().out
    assert "--config" in out and "Exit status" in out


def test_config_flag_beats_rag_config(
    run: Any, sample_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_CONFIG", str(tmp_path / "does-not-exist.yaml"))
    status, _ = run(sample_corpus)  # passes --config explicitly
    assert status == EXIT_OK


def test_ctrl_c_is_not_swallowed_as_a_document_failure(
    sample_corpus: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(self: PdfLoader) -> list:
        raise KeyboardInterrupt

    monkeypatch.setattr(PdfLoader, "load", interrupted)
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    with pytest.raises(KeyboardInterrupt):
        load_documents(load_config(make_config()), IngestReport())


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
    assert sorted(p.name for p in listing.symlinks) == ["broken-link.txt", "file-link.txt", "folder-link"]


def test_the_summary_accounts_for_everything_not_indexed(
    sample_corpus: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sample_corpus / "link.md").symlink_to(sample_corpus / "README.md")
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    report = IngestReport()
    load_documents(load_config(make_config()), report)

    assert report.skipped == ["diagram.png"]
    assert report.symlinks == ["link.md"]
    assert report.ignored == 3
    summary = report.summary()
    assert "Skipped 1 symlink(s) (not followed)." in summary
    assert "Ignored 3 hidden or lock file(s)." in summary
