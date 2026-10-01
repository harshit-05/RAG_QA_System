"""Our loaders (S1-3, DEC-8): the metadata contract, parity with the loaders they
replace, and recursive corpus discovery."""

from pathlib import Path

import pytest
from conftest import REPO_ROOT, MakeConfig, write_docx, write_pdf

from rag_qa.chain import citation
from rag_qa.config import load_config
from rag_qa.ingest import IngestReport, discover_files, load_documents
from rag_qa.loaders import DocxLoader, PdfLoader, TextLoader
from rag_qa.registry import import_from_string

SMALL_REAL_PDF = (
    REPO_ROOT / "corpus"
    / "Batch22_SmartSurveillanceSystemsUsingYOLOv8AScalableApproachforCrowdandThreatDetection.pdf"
)

# --- the metadata contract (citations are built from it) ------------------------------


def test_pdf_loader_yields_one_document_per_page_with_labels(sample_corpus: Path) -> None:
    path = sample_corpus / "guide.pdf"
    docs = PdfLoader(path).load()

    assert [d.page_content for d in docs] == ["Preface text", "Contents text", "Chapter one text"]
    assert [d.metadata for d in docs] == [
        {"source": str(path), "page": page, "page_label": label, "total_pages": 3, "loader": "PdfLoader"}
        for page, label in [(0, "i"), (1, "ii"), (2, "1")]
    ]


def test_citations_use_the_printed_page_label(sample_corpus: Path) -> None:
    # The point of page_label: the preface's first page is cited "p. i", not "p. 1".
    docs = PdfLoader(sample_corpus / "guide.pdf").load()
    assert [citation(d) for d in docs] == ["guide.pdf, p. i", "guide.pdf, p. ii", "guide.pdf, p. 1"]


def test_docx_loader_reads_the_whole_document(sample_corpus: Path) -> None:
    path = sample_corpus / "sub" / "report.docx"
    (doc,) = DocxLoader(path).load()
    assert doc.page_content == "Report title\n\nReport body"
    assert doc.metadata == {"source": str(path), "loader": "DocxLoader"}
    assert citation(doc) == "report.docx"  # no page for docx


def test_text_loader_pins_utf8(tmp_path: Path) -> None:
    # Explicit, not the locale's default (which is what the old TextLoader used).
    # Changing LC_ALL mid-process would not prove this; the attribute does.
    path = tmp_path / "accents.md"
    path.write_text("Café — naïve résumé\n", encoding="utf-8")
    loader = TextLoader(path)
    assert loader.encoding == "utf-8"
    (doc,) = loader.load()
    assert doc.page_content == "Café — naïve résumé\n"
    assert doc.metadata == {"source": str(path), "loader": "TextLoader"}


def test_text_loader_encoding_is_configurable(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("Café".encode("latin-1"))
    assert TextLoader(path, encoding="latin-1").load()[0].page_content == "Café"
    with pytest.raises(UnicodeDecodeError):
        TextLoader(path).load()  # the default does not silently mis-decode


@pytest.mark.parametrize("name", ["PdfLoader", "DocxLoader", "TextLoader"])
def test_loaders_pass_the_target_allowlist(name: str) -> None:
    # Classes defined in rag_qa: all three S1-2 checks (prefix, defining module, class).
    assert import_from_string(f"rag_qa.loaders.{name}").__name__ == name


# --- parity with the langchain_community loaders being replaced -------------------------
# The oracle tests go when langchain-community leaves (DEC-5, Phase 3).


def _pdf_cases(sample: Path) -> list[Path]:
    return [sample / "guide.pdf", SMALL_REAL_PDF]


def test_pdf_loader_matches_pypdfloader(sample_corpus: Path) -> None:
    from langchain_community.document_loaders import PyPDFLoader

    for path in _pdf_cases(sample_corpus):
        ours, theirs = PdfLoader(path).load(), PyPDFLoader(str(path)).load()
        assert len(ours) == len(theirs), path.name
        for mine, old in zip(ours, theirs, strict=True):
            assert mine.page_content == old.page_content, (path.name, mine.metadata["page"])
            for key in ("source", "page", "page_label", "total_pages"):
                assert mine.metadata[key] == old.metadata[key], (path.name, key)


def test_docx_and_text_loaders_match_the_old_ones(sample_corpus: Path) -> None:
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_community.document_loaders import TextLoader as OldTextLoader

    docx = sample_corpus / "sub" / "report.docx"
    assert DocxLoader(docx).load()[0].page_content == Docx2txtLoader(str(docx)).load()[0].page_content
    for text in (sample_corpus / "notes.txt", sample_corpus / "README.md"):
        assert TextLoader(text).load()[0].page_content == OldTextLoader(str(text)).load()[0].page_content


# --- recursive discovery and the extension map (ISS-13, FR-2) --------------------------


def test_discovery_is_recursive_and_sorted(sample_corpus: Path) -> None:
    found = [str(p.relative_to(sample_corpus)) for p in discover_files(sample_corpus).files]
    assert found == [
        "README.md", "diagram.png", "guide.pdf", "notes.txt",
        "sub/deeper/LOUD.TXT", "sub/report.docx",
    ]


def test_discovery_ignores_hidden_paths_and_office_lock_files(sample_corpus: Path) -> None:
    # Recursion reaches folders the old flat listdir never did: a Jupyter checkpoint
    # copy would be indexed twice and crowd the top-k, and Word's ~$ lock file ends
    # in .docx, so it would count as a failed file while the document is open.
    found = {p.relative_to(sample_corpus).as_posix() for p in discover_files(sample_corpus).files}
    assert not found & {
        ".ipynb_checkpoints/notes-checkpoint.txt", ".draft.md", "sub/~$report.docx"
    }


def test_missing_corpus_directory_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Corpus directory not found.*RAG_DATA_PATH"):
        discover_files(tmp_path / "nope")


def test_load_documents_walks_subdirectories_and_uses_the_map(
    sample_corpus: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    report = IngestReport()
    docs = load_documents(load_config(make_config()), report)

    by_file = {}
    for doc in docs:
        by_file.setdefault(Path(doc.metadata["source"]).relative_to(sample_corpus).as_posix(), []).append(
            doc.metadata["loader"]
        )
    assert by_file == {
        "README.md": ["TextLoader"],                # .md maps to the txt loader
        "guide.pdf": ["PdfLoader"] * 3,
        "notes.txt": ["TextLoader"],
        "sub/deeper/LOUD.TXT": ["TextLoader"],      # two levels down, uppercase suffix
        "sub/report.docx": ["DocxLoader"],          # one level down
    }
    assert report.skipped == ["diagram.png"]
    assert report.failed == []


def test_a_new_extension_is_one_config_line(
    sample_corpus: Path, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What FR-1 promises: supporting a file type is config, not code.
    (sample_corpus / "notes.rst").write_text("Restructured text.\n", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_PATH", str(sample_corpus))
    path = make_config(lambda c: c["pipeline"]["ingestion"]["loaders"].update({".rst": "components.loaders.txt"}))
    report = IngestReport()
    docs = load_documents(load_config(path), report)
    assert any(d.metadata["source"].endswith("notes.rst") for d in docs)


def test_fixture_writers_round_trip(tmp_path: Path) -> None:
    # Guards the fixture itself: if write_pdf broke, the tests above would prove little.
    path = write_pdf(tmp_path / "x.pdf", ["only page"])
    assert PdfLoader(path).load()[0].metadata["page_label"] == "1"
    assert write_docx(tmp_path / "x.docx", ["para"]).stat().st_size > 0
