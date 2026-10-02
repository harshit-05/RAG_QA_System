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
  - **If S2-5 has already created these two functions, reuse them unchanged.** The
    committed tier-2 fingerprint is computed with them, and a second implementation makes
    that run stale.
  - `Manifest` load and save, atomically: a temp file, then `os.replace`.
  - `diff(manifest, files) -> Changes(added, changed, removed, unchanged)`.
  - `chunk_id(sha256, index)` as `uuid5`.
- **`vectorstore.py`, still the only FAISS caller.**
  - Open the live index for update, and apply deletes and adds by id.
  - Save into `<path>.staging/`, then swap through `<path>.previous`.
  - Repair leftovers when a run starts.
  - `store_exists` becomes manifest-aware.
- **`ingest.py`.**
  - **One writer.** An advisory `fcntl.flock` on `<path>.lock`. A second run exits 2 with
    "another ingestion is running".
  - **What is embedded.** The run diffs the corpus walk against the manifest. It loads,
    splits and embeds only the added and changed documents, and deletes the chunks of
    removed and changed ones.
  - **Failures.** A document that fails keeps its old chunks and its manifest entry.
  - **Full rebuild** on any of:
    - no manifest, or an unknown manifest version;
    - a changed embedder identity;
    - a changed splitter identity;
    - `--rebuild`.
  - **Metadata.** `source` becomes corpus-relative, and `source_sha256` and `ingested_at`
    are added. The loaders keep their contract, and `ingest` rewrites `source` after
    loading.
  - **Report.** `IngestReport` gains added, changed, removed, unchanged and kept counts.
  - **Logging.** Progress goes to the `rag_qa.ingest` logger, and `main()` attaches a
    plain stdout handler, so the output reads as before.
  - **Walk.** It prunes ignored folders in place (`folders[:]`); `Path.walk` runs top-down.
  - **Exit codes stay 0, 1 and 2.** Exit 1 after a partial failure now reads "updated; the
    failed documents kept their previous chunks". Update DEC-13's wording to match.
- **The query side.**
  - `chain.build_retrieve` and `build_query_pipeline` check the manifest's embedder
    identity and dimension against the config.
  - On a mismatch, or with no manifest, they raise `IncompatibleIndexError`.
  - `rag-query` then exits 2 with "re-run rag-ingest", adding a `--rebuild` hint when the
    identity changed.
  - It also names a leftover `.previous` or `.staging` state.
- **README:** the one-time re-ingest for v0.2 indexes, and `--rebuild`.
- **Tests**, using `DeterministicFakeEmbedding` and temporary corpora:
  - The first run is full. A second run with nothing changed embeds nothing; count the
    embed calls.
  - Add one file, change one, delete one.
  - A file that becomes corrupt keeps its chunks, and the run exits 1.
  - Each full-rebuild trigger.
  - Identity: `minilm_cuda` and `minilm_cpu` give equal identities, and a different
    `model_name` does not.
  - A crash between the two renames: raise after step 2. The next run repairs it.
  - Lock contention exits 2.
  - The metadata keys, and that `source` is relative.
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

- **The staged swap and the lock are the risky code.** Read the crash-between-renames
  test, and the repair at startup.
- **Check the embedder identity's exclusions key by key.** One wrong exclusion means two
  embedders that really differ count as one. Then the guard passes, and queries silently
  compare incomparable vectors, which is the bug this story exists to prevent.
- **Check that citations do not change.** `citation()` reads only the file name, so
  making `source` relative must leave every printed citation as it was.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
