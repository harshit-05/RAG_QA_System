"""Shared fixtures. The suite is hermetic: no network, no Ollama, no model download.

Binary fixtures (PDF, DOCX) are generated at test time rather than committed —
CLAUDE.md: never commit binaries — and the generators are small enough to read.
"""

import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_qa.settings import ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CONFIG = REPO_ROOT / "config.yaml"

MakeConfig = Callable[..., Path]


def write_pdf(path: Path, texts: list[str], page_labels: str | None = None) -> Path:
    """A minimal valid PDF: one line of Helvetica text per page.

    ``page_labels`` is the body of the catalog's ``/PageLabels /Nums`` array, e.g.
    ``"0 << /S /r >> 2 << /S /D >>"`` labels pages i, ii, 1 — the case where the
    printed label and ``page + 1`` disagree, which is what citations depend on.
    Hand-built because pypdf can only add objects through a private API.
    """
    first_page = 4  # objects 1-3 are catalog, page tree, font; then page/content pairs
    page_refs = " ".join(f"{first_page + 2 * i} 0 R" for i in range(len(texts)))
    labels = f" /PageLabels << /Nums [{page_labels}] >>" if page_labels else ""
    bodies = [
        f"<< /Type /Catalog /Pages 2 0 R{labels} >>",
        f"<< /Type /Pages /Kids [{page_refs}] /Count {len(texts)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(texts):
        content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        bodies.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {first_page + 2 * i + 1} 0 R >>"
        )
        bodies.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(bodies, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref_at = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def write_docx(path: Path, paragraphs: list[str]) -> Path:
    """A minimal .docx: the three parts docx2txt and Word need, one run per paragraph."""
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    parts = {
        "[Content_Types].xml": (
            f'{xml}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
        ),
        "_rels/.rels": (
            f'{xml}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            'relationships/officeDocument" Target="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": (
            f'{xml}<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>"
        ),
    }
    with zipfile.ZipFile(path, "w") as docx:
        for name, content in parts.items():
            docx.writestr(name, content)
    return path


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """A small nested corpus covering every loader, recursion and the skip path::

        corpus/guide.pdf               3 pages, labelled i, ii, 1
        corpus/notes.txt
        corpus/README.md
        corpus/diagram.png             no loader mapped -> skipped
        corpus/sub/report.docx         one level down
        corpus/sub/deeper/LOUD.TXT     two levels down, uppercase suffix
    """
    root = tmp_path / "corpus"
    (root / "sub" / "deeper").mkdir(parents=True)
    write_pdf(
        root / "guide.pdf",
        ["Preface text", "Contents text", "Chapter one text"],
        page_labels="0 << /S /r >> 2 << /S /D >>",
    )
    (root / "notes.txt").write_text("Plain text notes.\n", encoding="utf-8")
    (root / "README.md").write_text("# Heading\n\nMarkdown body.\n", encoding="utf-8")
    (root / "diagram.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    write_docx(root / "sub" / "report.docx", ["Report title", "Report body"])
    (root / "sub" / "deeper" / "LOUD.TXT").write_text("Shouted text.\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported RAG_* variables must not leak into any test."""
    for name in (ENV_CONFIG, ENV_DATA_PATH, ENV_VECTOR_STORE_PATH):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def make_config(tmp_path: Path) -> MakeConfig:
    """Write a scratch config and return its path.

    By default it is the repo's real ``config.yaml``, optionally changed by ``edit``
    (a function mutating the loaded dict), so every malformed case is one small change
    to the real file rather than a hand-written config that drifts from it. ``text``
    writes raw content instead, for cases YAML can't express as a dict.
    """

    def make(edit: Callable[[dict[str, Any]], None] | None = None, *, text: str | None = None) -> Path:
        path = tmp_path / "config.yaml"
        if text is not None:
            path.write_text(text)
            return path
        data = yaml.safe_load(REAL_CONFIG.read_text())
        if edit is not None:
            edit(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return make
