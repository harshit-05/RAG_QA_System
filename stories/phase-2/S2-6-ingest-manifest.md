# S2-6: Ingest incrementally, tracked by a content-hash manifest

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-2 (the incremental SHOULD), ISS-14, NFR-4 (in-place update, still on FAISS), SRS §7.3; backlog: the embedder check on open, §7.3 metadata, walk pruning |
| **Depends on** | S2-1 (`build_retrieve` is where the query-side check lands) |
| **Model** | fable |
| **Plan-first** | yes |
| **Branch** | `feat/s2-6-ingest-manifest`. Risky: it rewrites how the index is written, and it is **breaking**: every existing index must be rebuilt once. The commit is `feat(ingest)!:`, with a `BREAKING-CHANGE:` trailer. |

## Goal

Four things become true:

- re-running `rag-ingest` re-embeds only what changed;
- a document that fails no longer drops out of the index;
- a crash never leaves a half-written index;
- a query refuses an index built with a different embedder, rather than comparing vectors
  that are not comparable.

These are the ingestion half of "make it a service". They are also what makes the API's
ingest job (S2-7) safe to expose. Design: ARCHITECTURE.md §2.1 DEC-17, manifest shape in
§2.4.

## Scope

- **`rag_qa/manifest.py`, pure Python with no LangChain.**
  - A streamed file sha256.
  - `spec_identity(spec)` and `embedder_identity(spec)`. The embedder identity is the spec
    minus `device`, `model_kwargs.device`, `show_progress`, `cache_folder` and
    `multi_process`.
  - **This story creates them, and S2-5 reuses them unchanged** (S2-6 now runs before
    S2-5). The committed tier-2 fingerprint is computed with them, and a second
    implementation would make that run stale.
  - `Manifest` load and save, atomically: a temp file, `fsync`, then `os.replace`. The
    manifest records its `generation`.
  - `diff(manifest, files) -> Changes(added, changed, removed, unchanged)`. Each file
    carries its sha256 **and its loader identity**, and a document is "changed" when
    either differs. Otherwise an edit to the extension map or a loader's spec would
    leave stale chunks forever.
  - `chunk_id(relative_path, sha256, index)` as `uuid5`. The path is in the key because
    two files with identical bytes would otherwise share IDs: FAISS rejects duplicates,
    and deleting one document would delete the other's chunks.
- **`vectorstore.py`, still the only FAISS caller** (ARCHITECTURE.md §2.1, DEC-17).
  - **Generations.** `paths.vector_store` becomes a symlink to
    `db_faiss.gen-<UTC timestamp>-<8 hex>`. A run writes a new generation, `fsync`s its
    files and the directory, then flips the symlink with one `os.replace` of a temporary
    link, and `fsync`s the parent.
  - **Apply all or nothing.** Load the live generation into memory, apply every delete
    and add by id, and save it as the new generation. Any exception aborts the run with
    nothing published. FAISS's own add is not atomic: vectors land before the docstore
    rejects a duplicate ID.
  - **Recovery, under the lock at start:** keep the symlink's target and the newest older
    generation, and delete every other generation and any stray temporary link.
  - **Migration:** if `db_faiss` is a real directory (v0.2), build a generation, rename
    the old directory to `db_faiss.v02-<timestamp>`, and create the symlink.
  - `store_exists` becomes manifest-aware, and follows the symlink.
- **`ingest.py`.**
  - **One writer.** An advisory `fcntl.flock` (flock(2), never `fcntl.lockf`, whose
    POSIX locks are per process and drop when any descriptor closes) on
    `vectorstore/db_faiss.lock`, outside the generations. A second run exits 2 with
    "another ingestion is running".
  - **What is embedded.** The run diffs the corpus walk against the manifest. It loads,
    splits and embeds only the added and changed documents (the prepare phase), then
    deletes the chunks of removed and changed ones and adds the new ones (the apply
    phase).
  - **Failures.** A document that fails in the prepare phase keeps its old chunks and its
    manifest entry. The store is never touched per document.
  - **Full rebuild** on any of:
    - no manifest, or an unknown manifest version;
    - a changed embedder identity;
    - a changed splitter identity;
    - a manifest that disagrees with its index (a `delete` reports missing IDs, or the
      generation cannot be opened), logged with the reason;
    - `--rebuild`.
  - **Metadata.** `source` becomes corpus-relative, and `source_sha256` and `ingested_at`
    are added. The loaders keep their contract, and `ingest` rewrites `source` after
    loading.
  - **Report.** `IngestReport` gains added, changed, removed, unchanged and kept counts.
  - **Logging.** Progress goes to the `rag_qa.ingest` logger, and `main()` attaches a
    plain stdout handler, so the output reads as before.
  - **No absolute paths in the report or the log.** `IngestReport.failed` stores
    `type: message`, with the corpus root and the store path made relative. An
    `OSError`'s text otherwise carries the absolute path. Log lines name paths
    relatively too. S2-7 returns both over HTTP.
  - **Walk.** It prunes ignored folders in place (`folders[:]`); `Path.walk` runs top-down.
  - **Exit codes stay 0, 1 and 2.** Exit 1 after a partial failure now reads "updated; the
    failed documents kept their previous chunks". Update DEC-13's wording to match.
- **The query side.**
  - `chain.build_retrieve` and `build_query_pipeline` check the manifest's embedder
    identity and dimension against the config.
  - On a mismatch, or with no manifest, they raise `IncompatibleIndexError`.
  - `rag-query` then exits 2 with "re-run rag-ingest", adding a `--rebuild` hint when the
    identity changed.
  - A `db_faiss` that is still a real directory (a v0.2 index) gets the same message.
- **README:** the one-time re-ingest for v0.2 indexes, and `--rebuild`.
- **Tests**, using `DeterministicFakeEmbedding` and temporary corpora:
  - The first run is full. A second run with nothing changed embeds nothing; count the
    embed calls.
  - Add one file, change one, delete one. Moving `.md` to another loader in the
    extension map re-embeds the `.md` files only.
  - A file that becomes corrupt keeps its chunks, and the run exits 1.
  - Each full-rebuild trigger, including a manifest that lists an ID its index lacks.
  - Identity: `minilm_cuda` and `minilm_cpu` give equal identities, and a different
    `model_name` does not.
  - **A crash before the flip:** raise after the new generation is written but before
    `os.replace`. The symlink still points at the old generation, queries are
    unaffected, and the next run deletes the orphan.
  - **A failure in the apply phase** (a forced duplicate ID): nothing is published, and
    the live generation's files are byte-identical before and after.
  - **The v0.2 migration:** a real `db_faiss` directory becomes a symlink, and the old
    directory is kept as `db_faiss.v02-*`.
  - Only the current and the previous generation survive a third run.
  - Lock contention exits 2, including from a second thread in the same process (the
    API's case).
  - An unreadable file's failure in the report and the log carries no absolute path.
  - The metadata keys, and that `source` is relative.
  - Two files with identical bytes in different folders both index, and deleting one
    leaves the other's chunks.
  - The walk never enters `.git`.
  - `rag-query` refuses a manifest-less index, and an index from a mismatched embedder.

## Out of scope

- API ingest jobs (S2-7).
- Qdrant, and snapshots and backup (Phase 3).
- Printing only nonzero summary lines, and the other polish (S2-8).
- Ingesting several corpora into one index.

## Verification

```bash
# 1. the gates
uv run ruff check && uv run mypy && uv run pytest --cov=rag_qa -q

# 2. on the real corpus, at scratch paths (caveat 9: set BOTH paths)
cp -r corpus <scratch>/corpus
export RAG_DATA_PATH=<scratch>/corpus RAG_VECTOR_STORE_PATH=<scratch>/index
time uv run rag-ingest                 # full: 1,708 chunks, the v0.1 count; ~2 min
time uv run rag-ingest                 # nothing changed: 0 embedded, seconds
printf 'Phase 2 incremental test.\n' > <scratch>/corpus/note.txt && uv run rag-ingest   # 1 added
rm <scratch>/corpus/note.txt && uv run rag-ingest                                       # 1 removed
readlink <scratch>/index; ls -d <scratch>/index.gen-*   # → a symlink to the newest generation; two generations kept

# 3. a failure keeps the document's previous chunks
cp <scratch>/corpus/2412.14140v2.pdf <scratch>/glider.bak
printf 'corrupt' > <scratch>/corpus/2412.14140v2.pdf
uv run rag-ingest; echo "exit $?"      # → 1, "kept their previous chunks"
mv <scratch>/glider.bak <scratch>/corpus/2412.14140v2.pdf
uv run python -c "import json; m = json.load(open('<scratch>/index/manifest.json')); print(len(m['documents']), m['embedder'])"

# 4. the query side refuses what it cannot use
uv run rag-query --config <scratch config with pipeline.ingestion.embedder: multilingual_mpnet_cpu>; echo "exit $?"   # → 2
unset RAG_DATA_PATH RAG_VECTOR_STORE_PATH
uv run rag-query; echo "exit $?"       # the real v0.2 index has no manifest → 2, "re-run rag-ingest"
uv run rag-ingest                      # the one-time rebuild of the real index (caveat 20)

# 5. one writer: start a second ingest while one runs → exit 2

# 6. CI
gh run list --limit 2                  # check and eval-retrieval green (CI ingests from scratch)
```

## Review notes for the human

- **The generation flip, the apply phase and the lock are the risky code.** Read the
  crash-before-flip test, the apply-failure test, and the recovery at startup. Check
  that `fsync` comes before the flip, and that nothing writes into the live generation.
- **Check the embedder identity's exclusions key by key.** One wrong exclusion means two
  embedders that really differ count as one. Then the guard passes, and queries silently
  compare incomparable vectors, which is the bug this story exists to prevent.
- **Check that citations do not change.** `citation()` reads only the file name, so
  making `source` relative must leave every printed citation as it was.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
