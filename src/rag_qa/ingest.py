"""Corpus ingestion: load, chunk, embed, persist.

Depends on components, config and the vector store only. It must never import
:mod:`rag_qa.chain` (ARCHITECTURE.md §0.2) — in v2 the dependency ran the wrong way,
which made ingestion drag the whole query stack in with it.

:func:`ingest` returns an :class:`IngestReport` instead of only printing, so the
Phase 2 ``/v1/ingest`` job-status endpoint can report the same numbers the CLI does.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from rag_qa.components import build_embedder, build_loader, build_splitter
from rag_qa.config import load_config
from rag_qa.schema import RagConfig
from rag_qa.vectorstore import create_store


@dataclass
class IngestReport:
    """What one ingestion run did. ``failed`` holds ``(filename, error)`` pairs."""

    documents: int = 0
    chunks: int = 0
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def summary(self):
        lines = [
            f"Loaded {self.documents} document pages/sections, split into {self.chunks} chunks.",
            f"Skipped {len(self.skipped)} file(s) with no configured loader.",
            f"Failed {len(self.failed)} file(s).",
        ]
        lines += [f"  - {name}: {error}" for name, error in self.failed]
        return "\n".join(lines)


def discover_files(data_path: Path) -> list[Path]:
    """Every file under the corpus directory, subdirectories included (ISS-13).

    Sorted by full path, which for a flat corpus is the old ``sorted(os.listdir())``
    order — so chunk order, and with it the index, is unchanged for existing corpora.
    """
    if not data_path.is_dir():
        # rglob on a missing directory yields nothing rather than raising, which
        # would read as "0 documents" instead of naming the real problem.
        raise FileNotFoundError(
            f"Corpus directory not found: {data_path} (set by paths.data or RAG_DATA_PATH)"
        )
    return sorted(path for path in data_path.rglob("*") if path.is_file())


def load_documents(config: RagConfig, report: IngestReport) -> list:
    """Load every file under the corpus with the loader its extension maps to.

    Files are named by their path relative to the corpus, so ``a/notes.md`` and
    ``b/notes.md`` stay distinguishable. Loader errors are recorded in
    ``report.failed`` and the run continues; the exit-code policy is S1-4 (ISS-05).
    """
    all_docs = []
    data_path = config.paths.data

    print(f"Loading documents from '{data_path}'...")
    for path in discover_files(data_path):
        name = str(path.relative_to(data_path))
        try:
            loader = build_loader(config, path)
            if loader is None:
                print(f"  - Skipped {name} (no loader configured for this file type)")
                report.skipped.append(name)
                continue
            print(f"  - Loading {name} with {type(loader).__name__}")
            docs = loader.load()
            all_docs.extend(docs)
            print(f"    {len(docs)} pages/sections")
        except Exception as e:  # noqa: BLE001  # S1-4 (ISS-05)
            print(f"    Error loading {name}: {e}")
            report.failed.append((name, str(e)))
    return all_docs


def ingest(config: RagConfig) -> IngestReport:
    """Build and persist the vector store from the configured ingestion pipeline."""
    report = IngestReport()

    documents = load_documents(config, report)
    report.documents = len(documents)
    if not documents:
        return report

    text_splitter = build_splitter(config)
    chunks = text_splitter.split_documents(documents)
    report.chunks = len(chunks)

    embeddings = build_embedder(config)
    # Embedding is the longest step of ingestion and is otherwise silent. Turn on
    # the embedder's own progress bar for this instance only: the query path
    # builds a separate instance, so rag-query stays quiet, and the config itself
    # is never modified. Embedders without the switch simply run without a bar.
    if hasattr(embeddings, "show_progress"):
        embeddings.show_progress = True
    print(f"Embedding {len(chunks)} chunks and saving the vector store...")
    create_store(chunks, embeddings, config.paths.vector_store)
    return report


def main(config_path=None):
    """CLI entry point (``rag-ingest``)."""
    print("--- Starting Document Ingestion Engine ---")
    config = load_config(config_path)
    report = ingest(config)

    print()
    print(report.summary())
    if not report.documents:
        print("Error: No documents were loaded. Exiting.")
        sys.exit(1)
    print(f"--- Ingestion Complete. Vector store saved at '{config.paths.vector_store}' ---")


if __name__ == "__main__":
    main()
