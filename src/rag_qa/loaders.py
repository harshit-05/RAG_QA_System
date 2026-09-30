"""Document loaders: one file in, a list of ``Document``s out.

Our own code on ``pypdf`` and ``docx2txt``, replacing the ``langchain_community``
loaders (DEC-5 step 1 of 3, DEC-8). Those were thin wrappers over the same two
libraries, and the package is sunset, so owning ~40 lines beats depending on a
frozen one. Each loader is a class because a ``_target_`` must name a class
(:mod:`rag_qa.registry`, check 3) and is built by :func:`rag_qa.registry.build_object`
with ``file_path`` added.

**Metadata contract.** Citations are built from this (:func:`rag_qa.chain.citation`),
so it is load-bearing, and the PDF half reproduces what ``PyPDFLoader`` emitted:

* ``source`` — the file path, as given;
* ``page`` — 0-indexed page number (PDF only);
* ``page_label`` — the printed page label, e.g. ``iv`` in a preface (PDF only);
* ``total_pages`` — the PDF's page count (PDF only);
* ``loader`` — the loader's class name.

Deliberately dropped: the PDF's own document-info fields (producer, creator, dates,
author, title…) that ``PyPDFLoader`` copied onto every page. Nothing reads them; the
content hash and ingestion timestamp of SRS §7.3 arrive with the Phase 2 manifest.
"""

from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from pypdf import PdfReader


class _FileLoader:
    """A file path plus a ``load()`` that reads it."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = str(file_path)

    def _metadata(self, **extra: object) -> dict[str, object]:
        return {"source": self.file_path, **extra, "loader": type(self).__name__}

    def load(self) -> list[Document]:
        raise NotImplementedError


class PdfLoader(_FileLoader):
    """One ``Document`` per page, empty pages included (as ``PyPDFLoader`` did)."""

    def load(self) -> list[Document]:
        reader = PdfReader(self.file_path)
        labels = reader.page_labels
        total = len(reader.pages)
        return [
            Document(
                # "plain" is pypdf's legacy extraction and what PyPDFLoader used;
                # "layout" would change the text, hence the chunks, hence the index.
                page_content=page.extract_text(extraction_mode="plain").strip(),
                metadata=self._metadata(page=number, page_label=labels[number], total_pages=total),
            )
            for number, page in enumerate(reader.pages)
        ]


class DocxLoader(_FileLoader):
    """The whole document as one ``Document``. ``.doc`` is not supported (binary format)."""

    def load(self) -> list[Document]:
        return [Document(page_content=docx2txt.process(self.file_path), metadata=self._metadata())]


class TextLoader(_FileLoader):
    """The whole file as one ``Document``.

    UTF-8 by default and explicit: ``TextLoader`` used the locale's encoding, which is
    UTF-8 on this host but makes the index depend on where ingestion ran.
    """

    def __init__(self, file_path: str | Path, encoding: str = "utf-8") -> None:
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        text = Path(self.file_path).read_text(encoding=self.encoding)
        return [Document(page_content=text, metadata=self._metadata())]
