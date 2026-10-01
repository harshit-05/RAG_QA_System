"""Corpus ingestion: load, chunk, embed, persist.

Depends on components, config and the vector store only. It must never import
:mod:`rag_qa.chain` (ARCHITECTURE.md §0.2) — in v2 the dependency ran the wrong way,
which made ingestion drag the whole query stack in with it.

:func:`ingest` returns an :class:`IngestReport` instead of only printing, so the
Phase 2 ``/v1/ingest`` job-status endpoint can report the same numbers the CLI does.

**Error policy (S1-4, ISS-05, NFR-7).** Two kinds of failure, handled differently:

* a *configuration* problem — an invalid config, a missing corpus directory, a
  loader that cannot be built — is the same for every file, so the run stops at
  once with the problem named;
* a *document* problem — one corrupt, unreadable or mis-encoded file — is recorded
  and the run continues with the rest (NFR-7: one bad document must not abort the
  run). The run still fails at the end.

``rag-ingest`` exit codes: **0** everything indexed; **1** the run completed but a
document could not be read, or there was nothing to index; **2** the run could not
start (configuration problem, or a command-line usage error, as ``argparse`` uses).
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag_qa.components import build_embedder, build_loader, build_splitter
from rag_qa.config import ConfigError, load_config
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_CONFIG
from rag_qa.vectorstore import create_store

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_CANNOT_START = 2


@dataclass
class IngestReport:
    """What one ingestion run did.

    ``skipped`` and ``symlinks`` hold corpus-relative names; ``failed`` holds
    ``(name, error)`` pairs; ``ignored`` counts hidden and lock files, which are not
    listed one by one because a ``.git`` folder alone can hold thousands.
    """

    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    ignored: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Loaded {self.documents} document pages/sections, split into {self.chunks} chunks.",
            f"Skipped {len(self.skipped)} file(s) with no configured loader.",
            f"Skipped {len(self.symlinks)} symlink(s) (not followed).",
            f"Ignored {self.ignored} hidden or lock file(s).",
            f"Failed {len(self.failed)} file(s).",
        ]
        lines += [f"  - {name}: {error}" for name, error in self.failed]
        return "\n".join(lines)


def _ignored(relative: Path) -> bool:
    """Hidden files and folders, and Office lock files: never corpus documents.

    Recursion reaches folders the flat listing never did. A hidden folder such as
    ``.ipynb_checkpoints`` holds copies that would be indexed twice and crowd the
    top-k; ``.git`` holds nothing to index. Word's ``~$report.docx`` lock file ends
    in ``.docx`` but is not one, so it would count as a failed file while the
    document is open. LibreOffice's ``.~lock.*#`` is covered by the hidden rule.
    """
    return any(part.startswith(".") for part in relative.parts) or relative.name.startswith("~$")


@dataclass
class CorpusListing:
    """What the corpus walk found: files to load, symlinks passed over, ignored count."""

    files: list[Path]
    symlinks: list[Path]
    ignored: int


def discover_files(data_path: Path) -> CorpusListing:
    """Walk the corpus directory, subdirectories included (ISS-13).

    * Hidden paths and Office lock files are **ignored** (:func:`_ignored`) and only
      counted: they were never documents.
    * Symlinks — to files, to folders, or broken — are **never followed** and are
      listed. One rule, where before a linked file was read even from outside the
      corpus while a linked folder was not walked (Python 3.12's ``rglob`` does
      not descend into one). The corpus is the directory's real contents.
    * Everything else that is a file is returned, sorted by full path: for a flat
      corpus that is the old ``sorted(os.listdir())`` order, so chunk order, and
      with it the index, is unchanged for existing corpora.
    """
    if not data_path.is_dir():
        # rglob on a missing directory yields nothing rather than raising, which
        # would read as "0 documents" instead of naming the real problem.
        raise FileNotFoundError(
            f"Corpus directory not found: {data_path} (set by paths.data or RAG_DATA_PATH)"
        )
    files, symlinks, ignored = [], [], 0
    for path in sorted(data_path.rglob("*")):
        relative = path.relative_to(data_path)
        if _ignored(relative):
            if path.is_symlink() or not path.is_dir():
                ignored += 1  # count entries, not the folders that hold them
        elif path.is_symlink():
            symlinks.append(path)
        elif path.is_file():
            files.append(path)
    return CorpusListing(files=files, symlinks=symlinks, ignored=ignored)


def load_documents(config: RagConfig, report: IngestReport) -> list:
    """Load every file under the corpus with the loader its extension maps to.

    Files are named by their path relative to the corpus, so ``a/notes.md`` and
    ``b/notes.md`` stay distinguishable. See the module docstring for the error
    policy: a loader that cannot be built raises :class:`ConfigError`; a document
    that cannot be read is recorded in ``report.failed`` and the run continues.
    """
    all_docs = []
    data_path = config.paths.data

    print(f"Loading documents from '{data_path}'...")
    listing = discover_files(data_path)
    report.ignored = listing.ignored
    for link in listing.symlinks:
        name = str(link.relative_to(data_path))
        print(f"  - Skipped {name} (symlink, not followed)")
        report.symlinks.append(name)

    for path in listing.files:
        name = str(path.relative_to(data_path))
        try:
            loader = build_loader(config, path)
        except (ImportError, TypeError) as e:
            # Same for every file of this type: a config problem, not a document one.
            ref = config.pipeline.ingestion.loaders.get(path.suffix.lower())
            raise ConfigError(f"Cannot build the loader {ref!r} (needed for {name}): {e}") from e
        if loader is None:
            print(f"  - Skipped {name} (no loader configured for this file type)")
            report.skipped.append(name)
            continue

        print(f"  - Loading {name} with {type(loader).__name__}")
        try:
            docs = loader.load()
        # Deliberately broad (NFR-7): reading one file must not abort the run, and
        # a bad file raises from many unrelated families — pypdf's errors,
        # BadZipFile, KeyError, XML ParseError (a SyntaxError), UnicodeDecodeError,
        # OSError, all seen on real bad inputs (S1-4). It is not silent: the error
        # and its type are recorded, printed, and fail the run at the end.
        # KeyboardInterrupt and SystemExit are not Exceptions and still stop it.
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            print(f"    Error loading {name}: {error}")
            report.failed.append((name, error))
            continue
        all_docs.extend(docs)
        print(f"    {len(docs)} pages/sections")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Build the vector index from the corpus: load, chunk, embed, save.",
        epilog="Exit status: 0 all indexed; 1 a document could not be read or nothing "
        "was indexed; 2 could not start (configuration or usage error).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-ingest``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work

    print("--- Starting Document Ingestion Engine ---")
    try:
        config = load_config(args.config)
        report = ingest(config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    print()
    print(report.summary())
    if report.failed:
        saved = (
            f"The index at '{config.paths.vector_store}' was rebuilt without them."
            if report.documents
            else "Nothing was indexed."
        )
        print(f"Error: {len(report.failed)} file(s) could not be read (listed above). {saved}", file=sys.stderr)
        return EXIT_RUN_FAILED
    if not report.documents:
        print("Error: No documents were loaded.", file=sys.stderr)
        return EXIT_RUN_FAILED
    print(f"--- Ingestion Complete. Vector store saved at '{config.paths.vector_store}' ---")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
