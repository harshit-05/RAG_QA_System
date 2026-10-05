"""Corpus ingestion: walk, compare with the manifest, embed what changed, publish.

Depends on components, config, manifest and the vector store only. It must never import
:mod:`rag_qa.chain` (ARCHITECTURE.md §0.2) — in v2 the dependency ran the wrong way,
which made ingestion drag the whole query stack in with it.

:func:`ingest` returns an :class:`IngestReport` instead of only logging, so the Phase 2
``/v1/ingest`` job (S2-7) can report the same numbers the CLI does.

**Incremental (DEC-17, S2-6).** A run compares the corpus walk with the live generation's
manifest, and sorts each document into added, changed (its bytes or its loader's identity
differ), removed or unchanged. Only added and changed documents are loaded, split and
embedded: the *prepare* phase. The *apply* phase then loads the live generation into
memory, deletes the chunks of removed and changed documents, adds the new ones, and
publishes the result as a new generation with one atomic flip (:mod:`rag_qa.vectorstore`).
The live generation is never written to, so a crash or an exception publishes nothing.

A run rebuilds the index in full, and logs why, when there is no index or no usable
manifest (every v0.2 index), when the embedder's identity or the splitter changed, when
this code makes chunks differently (:data:`rag_qa.manifest.CHUNKING_VERSION`), when the
manifest disagrees with its index, or with ``--rebuild``.

**One writer.** A run holds an advisory ``flock(2)`` on ``<store>.lock`` from start to
finish. A second run, from the CLI or the API, exits 2: "another ingestion is running".

**Error policy (S1-4, ISS-05, NFR-7).** Two kinds of failure, handled differently:

* a *configuration* problem — an invalid config, a missing corpus directory, a
  loader that cannot be built — is the same for every file, so the run stops at
  once with the problem named;
* a *document* problem — one corrupt, unreadable or mis-encoded file, or a folder that
  cannot be listed — is recorded and the run continues with the rest (NFR-7). In an
  update the document keeps its previous chunks and manifest entry (DEC-17, which
  supersedes DEC-13 there); a full rebuild leaves it out. Either way the run fails at the
  end, and the next run tries the document again.

Failures and log lines carry no absolute path: S2-7 returns both over HTTP (DEC-18).

``rag-ingest`` exit codes: **0** the index matches the corpus; **1** the run completed but
a document or folder could not be read, or there was nothing to index — and also an
unexpected error, which exits 1 with a traceback (Python's own status for an unhandled
exception), so 1 alone does not say whether an index was published; **2** the run could
not start (configuration problem, another ingestion running, or a command-line usage
error, as ``argparse`` uses).
"""

import argparse
import fcntl
import logging
import os
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_qa.components import build_embedder, build_loader, build_splitter
from rag_qa.config import ConfigError, load_config
from rag_qa.manifest import (
    CHUNKING_VERSION,
    DocumentRecord,
    EmbedderRecord,
    FileState,
    Manifest,
    SplitterRecord,
    chunk_id,
    diff,
    embedder_identity,
    file_sha256,
    spec_identity,
)
from rag_qa.schema import RagConfig
from rag_qa.settings import ENV_CONFIG
from rag_qa.vectorstore import (
    Embedded,
    OpenGeneration,
    in_the_way,
    live_index,
    open_generation,
    recover,
    write_generation,
)

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_CANNOT_START = 2

#: Progress and document failures. ``rag-ingest`` prints it plainly to stdout; the API job
#: (S2-7) and Phase 3's structlog consume the same records.
logger = logging.getLogger(__name__)

#: The text an update embeds first, to check the embedder still makes vectors of the size
#: the index holds.
DIMENSION_PROBE = "dimension check"


@dataclass
class IngestReport:
    """What one ingestion run did.

    ``documents`` and ``chunks`` count the pages loaded and the chunks embedded by this
    run, so an update that changed nothing reports 0 of each. ``found`` counts the corpus
    files a loader is mapped to. ``added`` … ``unchanged`` sort those files against the
    manifest, and ``kept`` counts failed documents that kept their previous chunks.
    ``rebuild`` says why the run rebuilt the index in full, or set out to (``None`` for an
    update), and ``published`` names the generation it published (``None`` when it
    published nothing: then the previous index, if any, is still the live one).

    ``skipped`` and ``symlinks`` hold corpus-relative names, and ``failed`` holds
    ``(name, "Type: message")`` pairs with no absolute path in them. ``ignored`` counts
    hidden and lock files, and hidden folders once each: the walk never enters them, and
    a ``.git`` folder alone can hold thousands of files.
    """

    documents: int = 0
    chunks: int = 0
    found: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    kept: int = 0
    rebuild: str | None = None
    published: str | None = None
    skipped: list[str] = field(default_factory=list)
    symlinks: list[str] = field(default_factory=list)
    ignored: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Loaded {self.documents} document pages/sections, split into {self.chunks} chunks.",
            (
                f"Documents: {self.added} added, {self.changed} changed, {self.removed} "
                f"removed, {self.unchanged} unchanged; {self.kept} failed and kept their "
                f"previous chunks."
            ),
            f"Skipped {len(self.skipped)} file(s) with no configured loader.",
            f"Skipped {len(self.symlinks)} symlink(s) (not followed).",
            f"Ignored {self.ignored} hidden or lock file(s) and folder(s).",
            f"Failed {len(self.failed)} file(s).",
        ]
        if self.rebuild is not None and self.published is not None:
            lines.insert(1, f"Rebuilt the index in full: {self.rebuild}.")
        elif self.rebuild is not None:
            lines.insert(1, f"A full rebuild was due ({self.rebuild}), but published nothing.")
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


def _check_corpus(data_path: Path) -> None:
    if not data_path.is_dir():
        # A walk of a missing directory yields nothing rather than raising, which
        # would read as "0 documents" instead of naming the real problem.
        raise FileNotFoundError(
            f"Corpus directory not found: {data_path} (set by paths.data or RAG_DATA_PATH)"
        )


@dataclass
class CorpusListing:
    """What the corpus walk found: files to load, symlinks passed over, ignored count,
    and folders it could not list, each with the error."""

    files: list[Path]
    symlinks: list[Path]
    ignored: int
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


def discover_files(data_path: Path) -> CorpusListing:
    """Walk the corpus directory, subdirectories included (ISS-13).

    * Hidden paths and Office lock files are **ignored** (:func:`_ignored`) and only
      counted: they were never documents. An ignored folder is pruned from the walk
      and counted once; nothing inside it is read.
    * Symlinks — to files, to folders, or broken — are **never followed** and are
      listed. One rule, where before a linked file was read even from outside the
      corpus while a linked folder was not walked (Python 3.12's ``rglob`` does
      not descend into one). The corpus is the directory's real contents.
    * A folder that cannot be listed is **unreadable** and returned with its error:
      what is inside it cannot be read this time, so the run must fail (NFR-7).
      ``rglob`` would pass over it without a word.
    * Everything else that is a file is returned, sorted by full path: for a flat
      corpus that is the old ``sorted(os.listdir())`` order, so chunk order, and
      with it the index, is unchanged for existing corpora.
    """
    _check_corpus(data_path)
    files, symlinks, ignored, unreadable = [], [], 0, []

    def could_not_list(error: OSError) -> None:
        unreadable.append((Path(error.filename), f"{type(error).__name__}: {error.strerror}"))

    # follow_symlinks=False: a linked folder arrives in `names`, not `folders`, so it
    # is listed as a symlink below instead of walked.
    for folder, folders, names in data_path.walk(on_error=could_not_list):
        relative = folder.relative_to(data_path)
        # Hidden folders are pruned in place, so the walk never enters them (Path.walk is
        # top-down). Only hidden ones: the "~$" rule is for Word's lock *files*, and a
        # folder named that way was always walked.
        kept = [sub for sub in folders if not sub.startswith(".")]
        ignored += len(folders) - len(kept)
        folders[:] = kept
        for name in names:
            path = folder / name
            if _ignored(relative / name):
                ignored += 1
            elif path.is_symlink():
                symlinks.append(path)
            elif path.is_file():
                files.append(path)
    return CorpusListing(
        files=sorted(files),
        symlinks=sorted(symlinks),
        ignored=ignored,
        unreadable=sorted(unreadable),
    )


#: An absolute path inside an error message: a slash at the start, or after a space, a
#: quote, a bracket or "=", up to the next space, quote, bracket, comma or colon.
_ABSOLUTE_PATH = re.compile(r"(?<![^\s'\"(<=\[])/[^\s'\"<>()\[\],:]+")


def _describe(error: BaseException, config: RagConfig) -> str:
    """``Type: message`` for the report and the log, with no absolute path in it.

    An ``OSError``'s text names the path it failed on, a FAISS error the file it could not
    open, and an unpickling error the module file it read. Paths under the corpus, or
    under the folder that holds the index, become relative to it. Any other absolute
    path is cut to its last part. S2-7 returns failures over HTTP (DEC-18).
    """
    text = f"{type(error).__name__}: {error}"
    roots = {config.paths.data, config.paths.vector_store.parent}
    # Longest first, in case one root holds the other. A filesystem root is never stripped.
    for root in sorted(roots, key=lambda path: len(str(path)), reverse=True):
        if root != Path(root.anchor):
            text = text.replace(f"{root}{os.sep}", "").replace(str(root), root.name)
    return _ABSOLUTE_PATH.sub(lambda match: Path(match.group()).name or "/", text)


class IngestionRunningError(Exception):
    """Another ingestion holds the writer lock on this index (DEC-17)."""


@contextmanager
def _writer_lock(store: Path) -> Iterator[None]:
    """Hold the one-writer lock on ``store`` for the ``with`` body, or raise at once.

    An advisory ``flock(2)`` on ``<store>.lock``, beside the store and outside every
    generation, so no flip or recovery ever deletes it. It is ``flock``, never
    ``fcntl.lockf``: a ``flock`` lock belongs to the open file, so a second open in the
    same process (the API's ingest thread) is refused too, and closing some other
    descriptor of the file never drops it. POSIX ``lockf`` locks belong to the process,
    so they fail on both counts (verified in S2-6). Closing the file releases the lock, and
    so does the process ending, however it ends.
    """
    path = store.with_name(f"{store.name}.lock")
    with open(path, "a") as lock:  # "a": created if missing, never truncated
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise IngestionRunningError(
                f"another ingestion is running on this index ({path.name} is locked). "
                f"Wait for it to finish, then run rag-ingest again."
            ) from None
        yield


@dataclass(frozen=True)
class CorpusFile:
    """A corpus file a loader is mapped to."""

    path: Path  # absolute, for reading
    name: str  # corpus-relative and POSIX: the manifest key and its chunks' ``source``
    state: FileState


@dataclass(frozen=True)
class CorpusScan:
    """The corpus as one run sees it."""

    files: list[CorpusFile]  # in walk order, the order v0.2 loaded and embedded them in
    unhashed: frozenset[str]  # files a loader is mapped to whose bytes could not be read
    unlisted: tuple[str, ...]  # corpus-relative folders that could not be listed


def scan_corpus(config: RagConfig, report: IngestReport) -> CorpusScan:
    """Walk the corpus and fingerprint each file a loader is mapped to (DEC-17).

    A file's fingerprint is the sha256 of its bytes plus its loader's reference and
    identity. Whatever the walk passes over goes into ``report`` and the log: files with
    no loader, symlinks, ignored entries, and folders it cannot list. A file whose bytes
    cannot be read is a document failure, recorded here too.
    """
    data_path = config.paths.data
    listing = discover_files(data_path)
    report.ignored = listing.ignored
    for link in listing.symlinks:
        name = link.relative_to(data_path).as_posix()
        logger.info("  - Skipped %s (symlink, not followed)", name)
        report.symlinks.append(name)
    unlisted = []
    for folder, error in listing.unreadable:
        name = folder.relative_to(data_path).as_posix()
        logger.warning("  - Cannot list %s/: %s", name, error)
        report.failed.append((f"{name}/", error))  # the slash: a folder, not a file
        unlisted.append(name)

    loaders = config.pipeline.ingestion.loaders
    identities = {ref: spec_identity(config.component(ref).spec()) for ref in loaders.values()}
    files, unhashed = [], set()
    for path in listing.files:
        name = path.relative_to(data_path).as_posix()
        ref = loaders.get(path.suffix.lower())
        if ref is None:
            logger.info("  - Skipped %s (no loader configured for this file type)", name)
            report.skipped.append(name)
            continue
        try:
            name.encode("utf-8")  # the name goes into chunk IDs, the manifest and JSON
        except UnicodeEncodeError:
            # One such file must not abort the run (NFR-7): it fails on its own, named
            # with its undecodable bytes escaped so the report and the log can print it.
            shown = name.encode("utf-8", "backslashreplace").decode("utf-8")
            error = "UnicodeEncodeError: the file name is not valid UTF-8; rename the file"
            logger.warning("  - Cannot index %s: %s", shown, error)
            report.failed.append((shown, error))
            continue
        try:
            sha256 = file_sha256(path)
        except OSError as e:
            error = _describe(e, config)
            logger.warning("  - Cannot read %s: %s", name, error)
            report.failed.append((name, error))
            unhashed.add(name)
            continue
        files.append(CorpusFile(path, name, FileState(sha256, ref, identities[ref])))
    report.found = len(files) + len(unhashed)
    return CorpusScan(files=files, unhashed=frozenset(unhashed), unlisted=tuple(unlisted))


@dataclass(frozen=True)
class _Identities:
    """The embedder and splitter the config asks for, as the manifest records them."""

    embedder_ref: str
    embedder: str
    splitter: SplitterRecord

    @classmethod
    def of(cls, config: RagConfig) -> "_Identities":
        ingestion = config.pipeline.ingestion
        return cls(
            embedder_ref=ingestion.embedder,
            embedder=embedder_identity(config.component(ingestion.embedder).spec()),
            splitter=SplitterRecord(
                ref=ingestion.splitter,
                identity=spec_identity(config.component(ingestion.splitter).spec()),
            ),
        )

    def changed_since(self, manifest: Manifest) -> str | None:
        """Why ``manifest``'s index cannot be updated with these, or ``None``."""
        if manifest.embedder.identity != self.embedder:
            compared = _compared(manifest.embedder.ref, self.embedder_ref)
            return f"the embedder is not the one that built the index ({compared})"
        if manifest.splitter.identity != self.splitter.identity:
            compared = _compared(manifest.splitter.ref, self.splitter.ref)
            return f"the splitter is not the one that built the index ({compared})"
        if manifest.chunking != CHUNKING_VERSION:
            return (
                f"this version makes chunks differently (chunking version "
                f"{manifest.chunking} then, {CHUNKING_VERSION} now)"
            )
        return None


def _compared(then: str, now: str) -> str:
    """Two references, for a reason: the same name with another identity says so."""
    return f"{now}, whose identity has changed since" if then == now else f"{then} then, {now} now"


@dataclass
class _Plan:
    """What a run does, decided before any document is loaded."""

    load: list[CorpusFile]  # what to load, split and embed, in walk order
    live: Manifest | None = None  # what an update starts from; None in a full rebuild
    base: OpenGeneration | None = None  # the live generation in memory, for an update
    remove: tuple[str, ...] = ()  # documents whose chunks go
    hold: frozenset[str] = frozenset()  # documents left as they are: unreadable this run
    relabel: dict[str, str] = field(default_factory=dict)  # unchanged, but a renamed loader
    rebuild: str | None = None  # why the index is rebuilt in full; None for an update
    embeddings: Embeddings | None = None  # built already, when deciding needed it

    @property
    def idle(self) -> bool:
        """Whether an update has nothing to publish: no document to (re-)embed or remove,
        and no loader reference to rewrite."""
        return self.rebuild is None and not (self.load or self.remove or self.relabel)


def _plan(config: RagConfig, scan: CorpusScan, report: IngestReport, *, rebuild: bool) -> _Plan:
    """Decide between an update, a full rebuild and nothing to do (:attr:`_Plan.idle`)."""
    found = live_index(config.paths.vector_store)
    if rebuild:
        return _Plan(load=scan.files, rebuild="--rebuild was given")
    if isinstance(found, str):
        return _Plan(load=scan.files, rebuild=found)
    generation, live = found
    changed = _Identities.of(config).changed_since(live)
    if changed is not None:
        return _Plan(load=scan.files, rebuild=changed)
    # Opened on every update, a run that changes nothing included: an index that will not
    # open (a corrupt pickle) must be rebuilt, never reported up to date (second review).
    # It costs no model, only reading the index.
    base = _open_live(generation, live, config)
    if isinstance(base, str):
        return _Plan(load=scan.files, rebuild=base)

    files = {file.name: file.state for file in scan.files}
    changes = diff(live, files)
    hold = frozenset(
        name
        for name in live.documents
        if name in scan.unhashed or any(name.startswith(f"{f}/") for f in scan.unlisted)
    )
    remove = tuple(name for name in changes.removed if name not in hold)
    pending = {*changes.added, *changes.changed}
    relabel = {
        name: files[name].loader
        for name in changes.unchanged
        if live.documents[name].loader != files[name].loader
    }
    embeddings = None
    if pending:
        # The identity cannot see the weights: a model replaced under the same spec can
        # make vectors of another size, which FAISS would refuse with a bare assert on
        # every run. Probed before anything is prepared, so a rebuild embeds once.
        embeddings = build_embedder(config)
        size = len(embeddings.embed_documents([DIMENSION_PROBE])[0])
        if size != live.embedder.dimension:
            reason = (
                f"the embedder now makes {size}-dimension vectors, but the index holds "
                f"{live.embedder.dimension}-dimension ones"
            )
            return _Plan(load=scan.files, rebuild=reason, embeddings=embeddings)
    report.unchanged, report.kept = len(changes.unchanged), len(hold)
    if pending or remove or relabel:
        logger.info(
            "Updating the index: %d added, %d changed, %d removed, %d unchanged.",
            len(changes.added), len(changes.changed), len(remove), len(changes.unchanged),
        )
    return _Plan(
        load=[file for file in scan.files if file.name in pending],
        live=live,
        base=base,
        remove=remove,
        hold=hold,
        relabel=relabel,
        embeddings=embeddings,
    )


def _open_live(generation: Path, manifest: Manifest, config: RagConfig) -> OpenGeneration | str:
    """The live generation, loaded for an update, or why it cannot be updated in place.

    It must open, and hold exactly the chunks its manifest lists. Otherwise a ``delete``
    would report missing IDs, or chunks the manifest does not list would stay in the
    index for good. Checked before anything is embedded, so a rebuild never embeds twice,
    and the generation loaded here is the one the apply phase updates.
    """
    try:
        opened = open_generation(generation)
    # Deliberately broad: whatever stops the generation opening (a corrupt pickle, FAISS's
    # own errors, a count mismatch) means the same thing, a rebuild from the corpus.
    except Exception as e:  # noqa: BLE001
        return f"the index cannot be opened ({_describe(e, config)})"
    listed = manifest.chunk_ids()
    if opened.ids != listed:
        return (
            f"the manifest and its index disagree: {len(listed - opened.ids)} chunk(s) "
            f"listed but missing, {len(opened.ids - listed)} held but not listed"
        )
    entries = sum(len(record.chunk_ids) for record in manifest.documents.values())
    if entries != len(listed):
        # The same ID under two documents, or twice under one: deleting both would pop it
        # from the docstore twice and fail mid-apply (second review).
        return f"the manifest lists {entries - len(listed)} chunk ID(s) more than once"
    return opened


def _embed_with_progress(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    # Embedding is the longest step of ingestion and is otherwise silent. Turn on
    # the embedder's own progress bar for this instance only: the query path
    # builds a separate instance, so rag-query stays quiet, and the config itself
    # is never modified. Embedders without the switch simply run without a bar.
    # Here rather than at build time, so the one-text dimension probe draws none.
    if hasattr(embeddings, "show_progress"):
        embeddings.show_progress = True
    return embeddings.embed_documents(texts)


def _load(
    config: RagConfig, file: CorpusFile, report: IngestReport, ingested_at: str
) -> list[Document] | None:
    """One document's pages with the SRS §7.3 metadata, or ``None`` when it failed to load.

    The loaders keep their contract (``source`` is the path as given); this rewrites
    ``source`` to the corpus-relative name and adds ``source_sha256`` and ``ingested_at``.
    """
    try:
        loader = build_loader(config, file.path)
    except (ImportError, TypeError) as e:
        # Same for every file of this type: a config problem, not a document one.
        raise ConfigError(
            f"Cannot build the loader {file.state.loader!r} (needed for {file.name}): {e}"
        ) from e
    if loader is None:  # the scan mapped a loader to this suffix, so it cannot be None
        raise ConfigError(f"No loader is configured for {file.name}")

    logger.info("  - Loading %s with %s", file.name, type(loader).__name__)
    try:
        docs: list[Document] = loader.load()
    # Deliberately broad (NFR-7): reading one file must not abort the run, and
    # a bad file raises from many unrelated families — pypdf's errors,
    # BadZipFile, KeyError, XML ParseError (a SyntaxError), UnicodeDecodeError,
    # OSError, all seen on real bad inputs (S1-4). It is not silent: the error
    # and its type are recorded, logged, and fail the run at the end.
    # KeyboardInterrupt and SystemExit are not Exceptions and still stop it.
    except Exception as e:  # noqa: BLE001
        error = _describe(e, config)
        logger.warning("    Error loading %s: %s", file.name, error)
        report.failed.append((file.name, error))
        return None
    for doc in docs:
        doc.metadata.update(
            source=file.name, source_sha256=file.state.sha256, ingested_at=ingested_at
        )
    logger.info("    %d pages/sections", len(docs))
    return docs


@dataclass
class _Prepared:
    """The prepare phase's output: the new manifest entries and the chunks to embed."""

    records: dict[str, DocumentRecord] = field(default_factory=dict)
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    failed: set[str] = field(default_factory=set)


def _prepare(config: RagConfig, plan: _Plan, report: IngestReport) -> _Prepared:
    """Load and split each document to (re-)embed. Its chunks are embedded afterwards, in
    one call, as v0.2 embedded them (:func:`_update`)."""
    ingested_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    splitter = build_splitter(config)
    prepared = _Prepared()
    for file in plan.load:
        docs = _load(config, file, report, ingested_at)
        if docs is None:
            prepared.failed.add(file.name)
            continue
        report.documents += len(docs)
        chunks = splitter.split_documents(docs)
        ids = [chunk_id(file.name, file.state.sha256, i) for i in range(len(chunks))]
        prepared.ids += ids
        prepared.texts += [chunk.page_content for chunk in chunks]
        prepared.metadatas += [chunk.metadata for chunk in chunks]
        prepared.records[file.name] = DocumentRecord(
            sha256=file.state.sha256,
            loader=file.state.loader,
            loader_identity=file.state.loader_identity,
            chunk_ids=ids,
            ingested_at=ingested_at,
        )
    return prepared


def _update(config: RagConfig, plan: _Plan, report: IngestReport) -> None:
    """Prepare, embed, apply and publish what ``plan`` says; fill in ``report``."""
    store = config.paths.vector_store
    wanted = _Identities.of(config)
    live = plan.live
    prepared = _prepare(config, plan, report)
    report.chunks = len(prepared.texts)
    if live is None:  # a full rebuild
        if not prepared.texts:
            # A fresh index needs a chunk; the live one, if any, is left as it is.
            logger.info("Nothing to index: no document yielded any text.")
            return
        report.added = len(prepared.records)
        documents = dict(prepared.records)
        delete: list[str] = []
    else:
        replaced = [name for name in prepared.records if name in live.documents]
        report.kept += len(prepared.failed & live.documents.keys())
        if not prepared.records and not plan.remove and not plan.relabel:
            logger.info("Nothing to publish: every document to update failed.")
            return
        report.added = len(prepared.records) - len(replaced)
        report.changed = len(replaced)
        report.removed = len(plan.remove)
        documents = {
            # A renamed loader with the same spec: the chunks stand, the reference moves.
            name: record.model_copy(update={"loader": plan.relabel[name]})
            if name in plan.relabel
            else record
            for name, record in live.documents.items()
            if name not in plan.remove and name not in prepared.records
        } | prepared.records
        delete = [
            cid for name in (*plan.remove, *replaced) for cid in live.documents[name].chunk_ids
        ]

    vectors: list[list[float]] = []
    if prepared.texts:  # a run that only deletes never loads the embedding model
        logger.info("Embedding %d chunks...", len(prepared.texts))
        embeddings = plan.embeddings if plan.embeddings is not None else build_embedder(config)
        # One call over every chunk, in walk order: sentence-transformers sorts a call's
        # texts by length before batching, so these are the batches v0.2 embedded, and a
        # full rebuild reproduces its vectors exactly.
        vectors = _embed_with_progress(embeddings, prepared.texts)
    dimension = len(vectors[0]) if live is None else live.embedder.dimension
    manifest = Manifest(
        generation="",  # write_generation names it after the folder it writes
        chunking=CHUNKING_VERSION,
        embedder=EmbedderRecord(
            ref=wanted.embedder_ref, identity=wanted.embedder, dimension=dimension
        ),
        splitter=wanted.splitter,
        documents=dict(sorted(documents.items())),
    )
    published = write_generation(
        store,
        manifest,
        base=plan.base,
        delete=delete,
        add=Embedded(
            ids=prepared.ids,
            texts=prepared.texts,
            metadatas=prepared.metadatas,
            vectors=vectors,
        ),
    )
    report.published = published.generation
    logger.info(
        "Published %s: %d documents, %d chunks.",
        published.generation,
        len(manifest.documents),
        len(manifest.chunk_ids()),
    )
    if published.legacy is not None:
        logger.info(
            "Kept the v0.2 index beside it as %s; delete it once the new one works.",
            published.legacy,
        )
    try:
        recover(store)  # prune: the previous generation stays, older ones go
    except OSError as e:
        # Best effort: the new generation is live already, and the next run retries.
        logger.warning("Could not remove an old generation: %s", _describe(e, config))


def ingest(config: RagConfig, *, rebuild: bool = False) -> IngestReport:
    """Bring the index up to date with the corpus; see the module docstring.

    Raises ``FileNotFoundError`` (no corpus folder), ``FileExistsError`` (something at the
    index's path that is not an index, :func:`rag_qa.vectorstore.in_the_way`),
    :class:`IngestionRunningError` and :class:`ConfigError` before anything is published.
    Any other exception propagates, also with nothing published.
    """
    report = IngestReport()
    store = config.paths.vector_store
    _check_corpus(config.paths.data)  # before the lock, so a mistyped path writes nothing
    blocked = in_the_way(store)
    if blocked is not None:
        # The flip, or the v0.2 migration, would replace or rename it without a word: a
        # mis-set path can name someone's folder, or the corpus itself. Refuse instead.
        raise FileExistsError(
            f"'{store}' {blocked}. Move it, or set RAG_VECTOR_STORE_PATH to another location."
        )
    store.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(store):
        leftovers = recover(store)
        if leftovers:
            logger.info("Removed what an earlier run left behind: %s", ", ".join(leftovers))
        scan = scan_corpus(config, report)
        if not scan.files and not scan.unhashed:
            return report  # nothing to index: the live index, if any, is left as it is
        plan = _plan(config, scan, report, rebuild=rebuild)
        if plan.rebuild is not None:
            logger.info("Full rebuild: %s.", plan.rebuild)
            report.rebuild = plan.rebuild
        elif plan.idle:
            if report.failed:
                logger.info("Nothing changed apart from the documents that could not be read.")
            else:
                logger.info("Nothing changed: the index is up to date.")
            return report
        _update(config, plan, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Bring the vector index up to date with the corpus: only new and "
        "changed files are loaded, chunked and embedded.",
        epilog="Exit status: 0 the index matches the corpus; 1 a document or folder could "
        "not be read (an update keeps their previous chunks), nothing was indexed, or an "
        "unexpected error (shown with a traceback); 2 could not start (configuration or "
        "usage error, or another ingestion is running).",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help=f"config file to use (default: ${ENV_CONFIG}, then ./config.yaml)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-embed every document into a fresh index instead of updating it (an "
        "embedder or splitter change rebuilds in full anyway)",
    )
    return parser


def _failure_note(report: IngestReport) -> str:
    """What became of the index, for the error line after a partial failure."""
    if report.rebuild is not None or report.found == 0:
        return "The index was rebuilt without them." if report.published else "Nothing was indexed."
    changed = "The index was updated" if report.published else "Nothing else changed"
    # "If they had any": a new file that fails has no previous chunks to keep.
    return (
        f"{changed}; the failed documents kept their previous chunks, if they had any, and "
        f"are retried next run."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``rag-ingest``). Returns the exit status; see the module docstring."""
    args = build_parser().parse_args(argv)  # --help and usage errors exit here, before any work

    print("--- Starting Document Ingestion Engine ---")
    # The progress log, printed plainly so the output reads as it always did. Attached for
    # this run only, so repeated calls (the tests) never print a line twice.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        return _run(args)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        # The CLI's own line, not the log: it names the configured folder in full, which
        # the log and the report (both returned over HTTP in S2-7) never do.
        print(f"Loading documents from '{config.paths.data}'...")
        report = ingest(config, rebuild=args.rebuild)
    except (ConfigError, FileNotFoundError, FileExistsError, IngestionRunningError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_CANNOT_START

    print()
    print(report.summary())
    store = config.paths.vector_store
    if report.failed:
        print(
            f"Error: {len(report.failed)} file(s) could not be read (listed above). "
            f"{_failure_note(report)}",
            file=sys.stderr,
        )
        return EXIT_RUN_FAILED
    if not report.found:
        print("Error: No documents were loaded.", file=sys.stderr)
        return EXIT_RUN_FAILED
    if report.rebuild is not None and report.published is None:
        print("Error: Nothing was indexed: no document had any text.", file=sys.stderr)
        return EXIT_RUN_FAILED
    if report.published is None:
        print(f"--- Ingestion Complete. Nothing changed; the index at '{store}' is up to date ---")
    else:
        print(f"--- Ingestion Complete. Vector store saved at '{store}' ---")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
