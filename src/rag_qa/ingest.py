"""Corpus ingestion: load, chunk, embed, persist.

Depends on the registry, config and vector store only. It must never import
:mod:`rag_qa.chain` (ARCHITECTURE.md §0.2) — in v2 the dependency ran the wrong way,
which made ingestion drag the whole query stack in with it.

:func:`ingest` returns an :class:`IngestReport` instead of only printing, so the
Phase 2 ``/v1/ingest`` job-status endpoint can report the same numbers the CLI does.
"""

import os
import sys
from dataclasses import dataclass, field

from rag_qa.config import load_config
from rag_qa.registry import build_object, import_from_string, resolve_ref
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


def load_documents(config, report):
    """Dynamically loads documents using loaders defined in the config.

    Loader errors are recorded in ``report.failed`` and the run continues; turning
    any failure into a non-zero exit is ISS-05 (Phase 1).
    """
    all_docs = []
    data_path = config["paths"]["data"]

    loader_configs = config["components"]["loaders"].values()

    print(f"Loading documents from '{data_path}'...")
    for filename in sorted(os.listdir(data_path)):
        file_path = os.path.join(data_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        loader_found = False
        for loader_config in loader_configs:
            if file_ext in loader_config.get("extensions", []):
                try:
                    loader_class_name = loader_config["_target_"].split('.')[-1]
                    print(f"  - Loading {filename} with {loader_class_name}")
                    LoaderClass = import_from_string(loader_config["_target_"])
                    loader = LoaderClass(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"    {len(docs)} pages/sections")
                except Exception as e:
                    print(f"    Error loading {filename}: {e}")
                    report.failed.append((filename, str(e)))
                loader_found = True
                break
        if not loader_found:
            print(f"  - Skipped {filename} (no loader configured for this file type)")
            report.skipped.append(filename)
    return all_docs


def ingest(config):
    """Build and persist the vector store from the configured ingestion pipeline."""
    report = IngestReport()
    ingestion_config = config["pipeline"]["ingestion"]

    documents = load_documents(config, report)
    report.documents = len(documents)
    if not documents:
        return report

    text_splitter = build_object(resolve_ref(config, ingestion_config["splitter"]))
    chunks = text_splitter.split_documents(documents)
    report.chunks = len(chunks)

    embeddings = build_object(resolve_ref(config, ingestion_config["embedder"]))
    # Embedding is the longest step of ingestion and is otherwise silent. Turn on
    # the embedder's own progress bar for this instance only: the query path
    # builds a separate instance, so rag-query stays quiet, and the config itself
    # is never modified. Embedders without the switch simply run without a bar.
    if hasattr(embeddings, "show_progress"):
        embeddings.show_progress = True
    print(f"Embedding {len(chunks)} chunks and saving the vector store...")
    create_store(chunks, embeddings, config)
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
    print(f"--- Ingestion Complete. Vector store saved at '{config['paths']['vector_store']}' ---")


if __name__ == "__main__":
    main()
