# S2-6: Ingest incrementally, tracked by a content-hash manifest

| | |
| --- | --- |
| **Status** | Done 2026-10-06 (PR #12). Verification 1–6 run and shown, and two reviews (10 and 15 findings, all fixed). CI's first `eval-retrieval` run stalled with no log; its re-run reproduced tier 1 question by question. Deviation in one line: two maintainer decisions (`multi_process` kept in the identity; a full rebuild leaves failed documents out), and the reviews tightened what counts as the index (below). |
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

- **API facts, checked against the installed versions before any code.** The versions:
  langchain-core 1.6.3, langchain-community 0.4.2, text-splitters 1.1.2,
  langchain-huggingface 1.2.2, faiss-cpu 1.15.1, sentence-transformers 6.1.0 and
  Python 3.12.13.
  - **FAISS's add is not atomic.** A duplicate within one batch is refused before
    `index.add` (faiss.py:308). An ID that already exists is refused by the docstore
    (in_memory.py:26) only after `index.add` (faiss.py:313). `delete` checks for
    missing IDs before it changes anything, and keeps the rest in insertion order.
    Exact ties come back in insertion order.
  - **`from_embeddings` writes the same `index.faiss` as `from_documents`** for the
    same vectors.
  - **`SentenceTransformer.encode` sorts each call's inputs by length before
    batching** (model.py:925). So the full rebuild embeds every chunk in one call, in
    walk order, as v0.2 did.
  - **The flip:** `os.replace` swaps one symlink for another, and refuses with EISDIR
    over a real directory. `fsync` works on a read-only file descriptor and on a
    directory.
  - **`flock` refuses a second open in the same process and a second thread;
    `lockf` does not.** A negative control swapped `flock` for `lockf` in-process,
    and the thread test then failed.
  - **`Path.walk` honours in-place pruning.** The root logger has no handlers once
    the stack is loaded.
- **`multi_process` can change the vectors** (huggingface.py:125–127):
  `encode_multi_process(texts, pool)` is called without `encode_kwargs`. That makes the
  story's exclusion unsafe; decided below.
- **The config resolved the store's symlink at load.** `Paths.anchor` called
  `.resolve()`, so once `db_faiss` was a link, every caller was handed the generation
  folder. The query side read it as a v0.2 directory, and a second ingest would have
  "migrated" the live generation. The existing suite caught it the first time it met a
  link. Now the last part of `vector_store` is never resolved, and the folders above
  it are, as before.
- **A shared test fixture leaked between tests.** `use_fake_embedder` assigned
  conftest's module-level `FAKE_EMBEDDER` dict itself, so an edit to `size` changed it
  for every test after it. It now hands each config a copy.
- **Tier 1 did not move.** Recorded before any code on the real v0.2 index, and after
  each full S2-6 rebuild (twice, before and after the reviews):
  - 0.950, 0.7958 and 0.8583, identical question by question, retrieved lists
    included;
  - `index.faiss` byte-identical to v0.2's;
  - still byte-identical after a file was added and then deleted, since `delete`
    compacts in order.
- **Timings on this CPU, with HF offline:**
  - a full ingest of the real corpus: 113–155 s;
  - a run that changes nothing: under 1 s, including opening the live index;
  - one file added or removed: a few seconds.
- **Carry-over 3, CI.** `$RUNNER_TEMP/index` becomes a symlink beside `index.gen-*` and
  `index.lock`. The cache steps and the warm-up never touch it, and the eval step follows
  the link, so `ci.yml` is unchanged. Verification 2 runs exactly this shape locally;
  Verification 6 runs it on the runner.
- **For S2-7:**
  - Its ingest job calls `ingest(config, rebuild=…)`. `IngestionRunningError` maps to
    the 409, and `FileExistsError` is a misconfigured store path.
  - `IncompatibleIndexError` subclasses `NoIndexError`, and
    `build_query_pipeline(require_index=False)` turns either into `retrieve=None`.
  - `check_index` returns the generation it checked; the API compares that with
    `generation_path` to reload.
  - The ingest log is the `rag_qa.ingest` logger, path-free. The report's new fields
    are `found`, `added`, `changed`, `removed`, `unchanged`, `kept`, `rebuild` and
    `published`.
- **For S2-8:**
  - pypdf's own warnings still print above a failed PDF ("invalid pdf header"), seen
    in Verification 3.
  - The summary prints more zero-count lines now.
  - "1 pages/sections" for a text file.
- **For S2-9:** README's status line still lists incremental ingestion as planned. §2.8
  should fold in the refinements under Deviation.

## Deviation from plan

- **Model:** Opus 5.5 at max effort, not Fable, which the story names. Fable is
  unavailable from 2026-10-04 (CLAUDE.md, "Model routing"). The two reviews ran as
  planned.
- **Maintainer decisions (2026-10-05), recorded on DEC-17 in ARCHITECTURE.md:**
  - **`multi_process` stays in the embedder identity.** The exclusions are `device`,
    `model_kwargs.device`, `show_progress` and `cache_folder`.
  - **A full rebuild leaves failed documents out** ("rebuilt without them"). In an
    update they keep their chunks.
- **Refinements the design left open:**
  - **Generation stamps carry microseconds**, and a new stamp always sorts after the
    live one's.
  - **The manifest is checked against its index before anything is embedded.** That
    is the case in which a `delete` would report missing IDs.
  - **`IncompatibleIndexError` subclasses `NoIndexError`.**
  - **What else is kept, and what publishes nothing:**
    - documents under a folder that cannot be listed keep their chunks;
    - a run that changes nothing publishes nothing;
    - an empty corpus publishes nothing (v0.2's rule).
  - **Writing a generation never embeds.** A placeholder stands in for the embedder,
    so a run that only deletes never loads the model.
- **Outside the Scope list, each needed by the above:**
  - `schema.py`: the store symlink is no longer resolved at load.
  - `evaluation/retrieval.py`: checks before building the embedder.
  - `chain.build_rag_chain`: checks first.
  - The `cli.py` docstring and epilog.
  - Two `config.yaml` comments that became false.
  - README lines beyond the two required: the Quickstart exit codes, the GPU section,
    Configuration and the layout.
  - The module map in `rag_qa/__init__.py`, the conftest copy, and dated
    ARCHITECTURE.md notes on DEC-13 and DEC-17.
- **First review (2026-10-05): 10 findings, fixed:**
  - a damaged index (files missing) was reported up to date;
  - `require_index=False` could still raise;
  - the clock could reorder generations;
  - an empty `model_kwargs` is now dropped from the identity, which changed
    `minilm_cpu`'s, so every S2-6 index built before the fix rebuilt itself once;
  - `normpath` folded `..` after a symlinked folder;
  - "Updating" was logged before a rebuild;
  - the live index was loaded twice, and the model even for deletes;
  - one shared usable-index rule (`vectorstore.live_index`);
  - one missing-corpus check.

  The empty-corpus guard kept its behaviour; only its wording was corrected.
- **Second review (2026-10-05): 15 findings, fixed:**
  - **Two could lose data.**
    - Any real folder at the store path was renamed aside as "v0.2", which could be
      the corpus. Now only a folder holding exactly `index.faiss` and `index.pkl`
      counts, and anything else in the way exits 2.
    - Recovery compared the raw link text with folder names, so a link written as
      `./name`, absolute or with a trailing slash deleted the live and previous
      generations. The link is now read however it is written.
  - **Also fixed:**
    - a user's own symlink at the path was overwritten; now refused;
    - a corrupt pickle was reported up to date, and crashed queries. Every update now
      opens the live index, and the query side refuses instead of crashing;
    - paths outside the two roots leaked, such as a pickled module's file;
    - one non-UTF-8 file name aborted the run; now that file fails on its own;
    - a model whose vectors changed size under the same spec failed on a bare assert.
      A dimension probe now triggers a full rebuild;
    - `build_rag_chain`, and the dimension refusal, came after the LLM;
    - folders named `~$…` were pruned;
    - an ID listed twice passed the consistency check;
    - "Rebuilt the index in full" was printed when nothing was published;
    - new failed files were said to have kept chunks;
    - the parent was not fsynced before the flip;
    - renaming a loader entry left stale references in the manifest;
    - the crash test's query check could not fail.

  Each has a regression test.
- **The verification commands:**
  - Step 0 recorded the tier-1 baseline on the real v0.2 index before any code.
  - Verifications 2–5 were run once, then again after the reviews, which changed the
    identity. The second time, the real index refused queries as "minilm_cpu as it was
    then", and `rag-ingest` rebuilt it by itself.
  - The v0.2 refusal was shown the second time on a copy of the kept
    `db_faiss.v02-*`, because the real index had been migrated in the first pass.
  - Verification 5 ran as one command: the long run in the foreground, the second
    `rag-ingest` forked to start 20 s in.
  - One shown exit code was wrong and was corrected: inside `time ( … )`, zsh's
    `pipestatus` reports the subshell.
- **Verification 6, CI on PR #12 (run 37282757727).**
  - **The first attempt stalled.** `check` passed, but `eval-retrieval` stalled in "Ingest
    the corpus" until the job's 20-minute limit. GitHub kept no log at all
    (`BlobNotFound`); a step that only hangs still uploads its log on timeout.
  - **Not reproduced here.** A fresh clone of the pushed commit, with its own synced
    environment and CI's exact command and variables, ingested in 122–136 s, peaking at
    about 950 MB with 5 KB of log. The model cache had hit, and `check` ran at its usual
    speed.
  - **Re-run (maintainer's choice): green.** Ingest took 2m07s, everything ran offline on
    an exact cache hit, and the runner's tier-1 table matched this host's question by
    question (0.950, 0.796, 0.858).
  - **Read as a runner fault, not a code fault.** If it recurs, the next step is a
    step-level timeout and a faulthandler dump on the ingest step (backlog).
