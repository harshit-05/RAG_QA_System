# S3-2: Move the index to Qdrant: generations behind an alias

| | |
| --- | --- |
| **Status** | Done (2026-10-19), pending the merge: PR #20, with follow-up commits for both rounds of the second review. CI was green on `24c5780` (run 39283008142, both jobs). `a2b7d38` went red in Audit (run 39294075683) on aiohttp advisories published between the two runs; round 2's first commit bumps aiohttp, and the head, `5ab1dee`, is green (run 39375380128, both jobs). Verification 1–7 are run and shown below. Merge it with a merge commit. |
| **Closes** | DEC-3, DEC-19; NFR-4 (upsert in the store); SRS §11 (an interrupted ingestion never leaves the store half-written) |
| **Depends on** | S3-1 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-2-qdrant-store`. Risky: it replaces the store under every command, changes the config and the environment variables (breaking), and adds a service to run. |

## Goal

The index moves from a FAISS pickle to a Qdrant server, with every guarantee DEC-17 gave:

- an update is all or nothing, published by one atomic alias switch;
- a reader sees the old generation or the new one, never half;
- one recovery rule cleans up after a crash.

The manifest moves into the collection's metadata, so a query replica needs no shared
disk. Retrieval returns exactly what FAISS returned, and that is proven, so the committed
tier-2 run stays valid with no re-run. Design: ARCHITECTURE.md §3.1, DEC-19; interfaces in
§3.4.

## Scope

**The plan-first step decides** whether this splits into two stories. If it does, the seam is
the parity tool: step 1 lands first, as its own pull request, while FAISS is still the store,
and the rest follows. Not the store against the API's refresh, CI and the docs (design review
1): between those two, `main`'s API and CI would point at a store that is gone. It records the
decision here.

**Decided in the plan (2026-10-17, maintainer approved): no split.** The parity tool is about
100 lines, and a pull request of its own would add a whole review and close-out cycle to save
the main one almost nothing. It lands instead as the branch's first commit, made by the
maintainer at a checkpoint once it has run on FAISS, before any store code is written. So the
FAISS file comes from a committed tool, and no later commit has to stage part of a file.

1. **First commit, while FAISS is still the store: the parity tool.**
   `rag-eval retrieval --contexts-out FILE` writes, for all 25 golden questions,
   answerable or not, in id order:
   - the sha256 of the rendered context (`chain.format_docs` over what `pipeline.retrieve`
     returns, the rerank included);
   - the citations, in order.

   Then run it against the current FAISS index, and keep the file outside the repository
   somewhere a reboot does not clear (for example `~/dev/rag-s3-2/contexts-faiss.json`).
   Paste its sha256 here.

   **Run on FAISS (2026-10-17),** the live generation `db_faiss.gen-20261005T080811.306241Z-3e556052`
   (3 documents, 1,708 chunks), with HF offline:
   - `~/dev/rag-s3-2/contexts-faiss.json`: sha256
     `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc`, 25 questions (5
     unanswerable);
   - `~/dev/rag-s3-2/tier1-faiss.json` (`--json`, for the per-question comparison): sha256
     `e5dc2aff6815b9d96835b12bf999392db84300d09f1437a44cd2013bb7a4eb46`; hit rate 0.950, MRR
     0.7958, recall 0.8583, S2-4's figures.

   **Re-run with the reviewed tool (2026-10-19),** which retrieves through `ainvoke` as every
   answer does (Review, below): both files came out with the same two sha256s, so the files
   above are what the committed tool writes.
2. **Dependency:** `uv add "qdrant-client>=1.19.1"`. Record the resolved versions; no locked
   version should move (checked in the architecture pass).
3. **`compose.yaml`**, with one service, `qdrant`:
   - `qdrant/qdrant:v1.19.2-unprivileged`, pinned by its index digest (DEC-24's table). The
     plain image runs as root, and a volume that starts unprivileged never holds root's
     files (design review 2);
   - published on `127.0.0.1:6333` only, with a named volume and
     `restart: unless-stopped`.

   No API key on loopback yet; S3-9 adds it with the rest of the stack.
4. **Config and settings (breaking).**
   - **The store component:** `components.vector_stores.qdrant` (`_target_:
     rag_qa.vectorstore.QdrantStore`, `url`, `collection`, `timeout_s`), referenced by
     `pipeline.vector_store`.
   - **Collection names** must match `[a-z0-9_]{1,48}`.
   - **`paths.state`** (default `var/`) holds the writer lock (`var/ingest.lock`).
   - **New environment variables:** `RAG_QDRANT_URL`, `RAG_QDRANT_COLLECTION`,
     `RAG_QDRANT_API_KEY` and `RAG_STATE_PATH`, read in `settings.py` only.
   - **Refused at load, exit 2, with the migration in the message:** `paths.vector_store`,
     and `RAG_VECTOR_STORE_PATH` while it is set. Caveat 9's habit must fail loudly, not
     write the real index.
   - `vectorstore/` and `var/` are in `.gitignore`.
5. **Manifest version 2** (`manifest.py`), read from and written to the collection's
   metadata.
   - Per document, `chunks: n` replaces `chunk_ids`, and `Manifest.chunk_ids()` derives
     them through `chunk_id`.
   - It adds `store: {distance: cosine, exact: true}`.
   - **The whole shape of §3.4, now** (design review 1): `sparse: null`, and each document's
     `splitter` set to the default splitter's identity (S3-4 makes it per extension). The
     records are strict, so a field added in a later story would make every version-2
     manifest unreadable.
   - Version 1 is no longer read: it only ever lived beside a FAISS index.
   - **Under one metadata key, written whole, beside `complete`** (DEC-19, design review 2).
     A metadata update merges top-level keys, a `null` deletes one, and a nested object is
     replaced whole, so no stale field can survive inside the manifest.
   - **Never read in a hot path.** `GET /collections/{name}` returns the whole manifest, so
     the refresh compares `/aliases`, counts come from `count`, and the manifest is read once
     per new generation. A request is limited to 32 MiB, about 100,000 documents.
6. **`vectorstore.py`, rewritten on `qdrant-client`** (the seam).
   - **What it provides:** `QdrantStore`, `live_index`, `in_the_way`, `open_generation`,
     `write_generation`, `recover`, and a dense `retriever`, as in §3.4.
   - **Generations** are collections `<collection>--gen-<stamp>-<hex>`, with a dense vector
     named `dense` (cosine), a payload index on `metadata.source`, and the manifest in
     `metadata`.
   - **An update** copies the live generation without the deleted points, upserts the new
     ones (`wait=True`), checks the count, marks the generation `complete: true`, then
     switches the alias with one `update_collection_aliases` call (delete plus create).
   - **Recovery,** under the lock, keeps the alias's target and the newest older generation.
     **With no alias, it deletes only incomplete generations** (DEC-19, design review 2):
     Qdrant deletes an alias with its target, so a lost live collection would otherwise cost
     the previous one too. A generation is complete when its metadata says so and its
     manifest names the collection itself. `rag-query`'s "no index" message then names the
     newest complete generation, and the README line that points the alias at it.
   - **In the way:** a real collection named like the alias, or an alias pointing outside
     the pattern, exits 2 and is never renamed. It is told from `get_collections()`, which
     omits aliases, and `get_aliases()`, never from `collection_exists`, which resolves an
     alias to its target (checked, design review 2).
   - **Search** is `exact=True`, and **our code makes every cut** (DEC-19, design review 2).
     The server decides a tie at a limit by the collection's layout, which changes with the
     insertion order and across restarts. So every search asks for more than the cut,
     extends while the point at the cut ties with the next, orders by score, then point ID,
     and cuts. The real corpus has no tie at the cut today, so parity is unaffected.
   - **Documents come back as FAISS returned them:** the same `page_content` and metadata,
     except `ingested_at`, which the re-ingest sets anew (Discovered). No `_id` or
     `_collection_name` keys, since the prompt and `SourceRef` read the metadata.
   - **It is the only module importing `qdrant_client`.**
7. **Ingest** (`ingest.py`): the writer lock moves to `paths.state`. `_open_live` checks the
   generation's point IDs against the manifest's derived ones. Every DEC-17 rule keeps
   holding: a failed document keeps its chunks, and an identity change rebuilds.
8. **The query side** (`chain.py`).
   - `check_index`, `open_retrieval` and `live_generation` go through the store, and the
     retriever binds to the generation it resolved, never to the alias.
   - **Readers resolve the alias again for each question** (DEC-19, design review 1):
     `rag-query` before each question, `rag-eval` before each item, as the API does per
     request. Two switches delete a generation, and a REPL session can outlive two ingests.
     A search whose collection has gone resolves once more. The server answers that search
     with a 404, mid-search too and never a 500, so the 404 is the signal, and it is never
     retried (design review 2).
   - **v0.3's folder:** with `vectorstore/db_faiss*` still on disk, `rag-query`'s "no index"
     message says that v0.3's index is no longer read.
9. **The API** (`api/state.py`): `refresh()` compares the alias's target with the generation
   served, so a CLI ingest is picked up as before.
10. **Qdrant first, everywhere.** `rag-ingest`, `rag-query`, `rag-eval` and `rag-serve` check
    Qdrant before anything else. If it is unreachable: one message, exit 2, and
    `Is Qdrant running? docker compose up -d qdrant`. A new exception from shared functions
    means finding every caller with grep (caveat 28).
11. **Tests.**
    - A fixture shares one `QdrantClient(":memory:")` between ingest and the query side
      within a test.
    - The FAISS-specific tests are rewritten for the store.
    - The architecture test gains: only `vectorstore` imports `qdrant_client`.
    - A hermetic test makes `update_collection_aliases` fail mid-update: the live generation
      is untouched, and the next recovery removes the orphan. The moment between the new
      collection and the switch lasts a second or two, too short to hit by hand.
    - With no alias, recovery keeps every complete generation and deletes the rest: a test
      deletes the live collection, then runs an ingest (design review 2).
    - The cut rule: points tied at the cut give the same result whatever the insertion
      order, hermetically, and under `qdrant_server` from two collections built in
      different orders (design review 2).
    - A `qdrant_server` marker (in `pyproject.toml`'s pytest settings) covers tests that
      need the real server: the alias switch and recovery, and the metadata round trip.
      They are skipped unless `RAG_TEST_QDRANT_URL` is set.
12. **CI** (needs the `workflow` scope). `eval-retrieval` gains `services: qdrant`, the same
    image by digest, and:
    - ingests into it (`RAG_QDRANT_URL`, `RAG_QDRANT_COLLECTION=ci`);
    - runs tier 1 as before;
    - runs `pytest -m qdrant_server` against it.
13. **Docs.**
    - **README:** Requirements gain Docker, or Qdrant's static binary. The Quickstart starts
      `docker compose up -d qdrant` before `rag-ingest`. A new "Upgrading from v0.3": start
      Qdrant, re-ingest, delete `vectorstore/`, and set `RAG_QDRANT_COLLECTION` where
      `RAG_VECTOR_STORE_PATH` was used. Configuration and layout are updated, and the pickle
      limitation goes. It also gains the line that points the alias at a named generation,
      which the "no index" message cites (design review 2).
    - **CLAUDE.md:** the environment facts mention the Qdrant container.
14. **The commit that changes the config** carries `!` and a `BREAKING-CHANGE:` trailer.

## Out of scope

- Removing `faiss-cpu` and `langchain-community` (S3-3). They stay installed here, which is
  what lets step 1 run.
- Sparse vectors, hybrid retrieval and filters (S3-5).
- Timeouts beyond the client's own, retries and breakers (S3-6).
- The full compose stack, the Qdrant API key, backups (S3-9, S3-10).

## Verification

```bash
docker compose up -d qdrant && curl -s 127.0.0.1:6333/readyz

# 1. parity: the same contexts as FAISS, for all 25 questions (step 1's file)
uv run rag-ingest
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-2/contexts-qdrant.json
diff ~/dev/rag-s3-2/contexts-faiss.json ~/dev/rag-s3-2/contexts-qdrant.json && echo IDENTICAL
# tier 1 question by question equals S2-4's: hit rate 0.950, MRR 0.7958, recall 0.8583
# and twice on Qdrant: an exact search must not vary between runs (DEC-20 has how a cut can)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-2/contexts-qdrant-2.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-2/contexts-qdrant-2.json && echo STABLE
# S3-3 to S3-10 diff their own contexts against contexts-qdrant.json (§3.5)

# 2. nothing hashed moved
uv run rag-eval check; echo "exit $?"              # 0

# 3. the unit suite is hermetic: no Docker, no network, no model (caveat 23)
docker compose stop qdrant
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
docker compose start qdrant
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q

# 4. a crash mid-update publishes nothing, and the next run cleans up. kill -9 a
#    `rag-ingest --rebuild` while it embeds (about 2 minutes, an easy window); the moment
#    between the new collection and the switch is the hermetic test's (step 11). Then:
curl -s 127.0.0.1:6333/aliases                     # still the old generation
uv run rag-ingest                                  # "Removed what an earlier run left behind"
uv run rag-query <<< "What is GLIDER?"            # answers from the live generation

# 5. refusals: Qdrant down, the old variable, the old key
docker compose stop qdrant; uv run rag-query; echo "exit $?"      # 2, with the hint
docker compose start qdrant
RAG_VECTOR_STORE_PATH=/tmp/x uv run rag-ingest; echo "exit $?"  # 2, with the migration

# 6. the API picks up a CLI ingest without a restart (foreground terminal, tee -i):
#    `rag-ingest --rebuild` makes a new generation without touching the corpus, and
#    /v1/health names it

# 7. CI green on the PR: check, and eval-retrieval with the Qdrant service
gh pr checks
```

## Review notes for the human

- **`write_generation` and `recover`:** the alias switch is one request, the new generation
  is complete before it, and recovery can never delete the target or the newest older
  generation. A crash at each step should leave the live generation intact.
- **Readers bind to a generation, not to the alias.** Check `open_retrieval` and the API's
  refresh, and that `rag-query` resolves the alias again before each question.
- **Parity:** if step 1's diff is not empty, the story must name the question and the tie,
  and the re-run moves to S3-11b; nothing else changes.
- **The in-memory fixture** must share one client between ingest and the query side, or the
  tests pass against an empty store.
- **Every cut is ours, and a lost alias costs no complete generation** (design review 2):
  read the cut rule's extension loop, and `recover`'s branch for no alias.

## Discovered

- **`ingested_at` will differ from FAISS's** (second review). Every chunk's metadata carries
  it, set once per ingest run (`ingest.py`, `_prepare`). Qdrant's index is a
  fresh ingest, so step 6's "the same metadata" cannot hold for that one key. Design review
  2's payloads came back equal because it copied FAISS's own payloads; it did not
  re-ingest. A test comparing metadata with FAISS's must leave `ingested_at` out. The
  contexts digest is unaffected: the prompt never shows it.
- **Two questions share a context:** `cade7-venue` and `unanswerable-cade8-venue` get the
  same five chunks, so the FAISS file has 24 distinct digests for 25 questions. Expected:
  the unanswerable one is a near-miss of the answerable one.

**Phases B to H (2026-10-19).** Checked against the installed source of `qdrant-client`
1.19.1 and a probe run in local mode and on the compose server, v1.19.2, before the store
was written:

- **The resolve:** `uv add "qdrant-client>=1.19.1"` added qdrant-client 1.19.1, grpcio
  1.84.0, protobuf 7.36.2, portalocker 3.2.0, h2 4.4.1, hpack 4.2.0 and hyperframe 6.1.0,
  plus pywin32 312 behind portalocker's Windows-only marker (in the lock, not installed).
  **No locked version moved:** 8 packages added, none removed or changed.
- **The image:** `qdrant/qdrant:v1.19.2-unprivileged` by DEC-24's digest runs as uid 1000,
  and stores in `/qdrant/storage`, which that user owns, so a new named volume does too. It
  has no curl or wget: readiness is asked from outside (compose has no healthcheck; CI's
  wait step curls `/readyz` from the runner).
- **The client:**
  - `QdrantClient(url=…)` with the default `check_compatibility=True` starts a thread that
    asks the server's version, and warns when it is down, before any command could say so
    properly. The store turns it off; `QdrantStore.check()` asks `info()` itself, and
    refuses a server older than 1.16, which has no collection metadata.
  - An API key with an `http://` URL keeps HTTP and warns "insecure connection"; with no
    scheme a key would switch to HTTPS. For S3-9, which adds the key on the compose network.
  - `timeout` is whole seconds, rounded up.
  - **Errors:** no answer at all is `ResponseHandlingException` (its `.source` is httpx's
    error); an HTTP answer is `UnexpectedResponse` with `.status_code`; local mode raises
    `ValueError("Collection … not found")`. `vectorstore` translates all three, since no
    other module may import `qdrant_client` to catch them.
- **Local mode is not the server,** in four ways that matter to tests:
  - **A failing alias batch is applied halfway:** local mode deleted the alias, then
    failed the create, and left no alias. The server kept the alias as it was. So the
    hermetic switch-failure test injects the failure before the call, and the atomicity is
    a `qdrant_server` test.
  - A metadata key set to `null` stays as `None` in local mode; the server deletes it.
  - `delete_collection` of a missing collection returns `True` in local mode, `False` on
    the server; both count as deleted.
  - Vectors come back at float64 from a first upsert and float32 after a copy, so the
    hermetic copy test compares to float32 precision; the bit-exact copy is a server test.
  - It warns that payload indexes and `search_params` have no effect: filtered by message
    in `pyproject.toml`.
- **Copying is bit-exact on the server.** 2,000 float32 unit vectors of 384 dimensions,
  stored and then copied twice by scroll and upsert, came back with every bit and every
  top-20 order unchanged: cosine stores an already unit-length vector as it is. So the
  store adds no drift to a copy. The embedder does vary, by about 1e-7 with the batch a
  text is embedded in (second review: 1 of 555 vectors differed by 5e-8 between an update
  and a rebuild of the same corpus), so a copy is not bit-equal to a fresh ingest of the
  same text. A difference that size could reorder only chunks whose scores tie to 1e-7;
  that probe compared vectors, not ranks.
- **The same on both:** metadata merges top-level keys and replaces a nested object whole;
  the first publish's delete-plus-create is accepted with no alias; `get_collections()`
  omits aliases while `collection_exists` resolves them; deleting the target deletes the
  alias; scroll returns string IDs in sorted order; a payload keeps `page` an int.
- **Design choices made in implementation** (none re-opens §3.1; each is in the code's
  docstrings):
  - **A generation is created after the embedding,** as §3.4's `write_generation(…, add:
    Embedded)` implies. A kill while embedding therefore leaves nothing behind (Deviation).
  - **A retriever takes `k` only.** FAISS's other `as_retriever` options (`search_type`
    other than similarity, `fetch_k`, `score_threshold`…) would be silently ignored by the
    store, so the schema refuses them at load. The real config's retriever dump, which tier
    2 hashes, is unchanged. Part of the breaking change.
  - **`rag-eval generate` holds a generation it moves to against its corpus scan:** the
    alias is resolved per question (step 8), and an ingest of a changed corpus mid-run
    would otherwise mix answers from two corpora under one fingerprint (the maintainer's
    2026-10-06 rule). It stops with `StaleIndexError`, resumable.
  - **The API goes not-ready when the alias cannot be read** (`NOT_SEEN`), answers 503
    rather than 500, and serves again on the first request after Qdrant returns. S3-6 adds
    the breaker and `Retry-After`.
  - **`rag-eval check` and `score` never touch Qdrant:** CI's hermetic `check` job has no
    server, and neither command reads the index.
  - **The writer lock is per host, whatever the collection:** a scratch ingest blocks the
    real one, as DEC-19's "one writer per host" says.
  - **The CHUNKING_VERSION tripwire** now rebuilds walk order from the manifest, since
    Qdrant keeps points by ID: it gave the same pinned digest, `556f8396…`, so the chunks
    are what FAISS held.
  - **The deprecation gate now requires the sunset notice to be absent:** no module imports
    `langchain_community` since the store moved. A stronger guard for S3-3.
- **Verification 3 found a test that was not hermetic:** `test_api_server.py`'s Ctrl-C test
  runs `rag-serve` in a child process, which cannot see the fixture, so its Qdrant check
  reached the real server on localhost:6333. It passed only while Qdrant ran, and failed
  with it stopped. The child now patches `connect` to an in-memory client, as the
  deprecation gate's child does.
- **Out of scope, from CI's annotations (run 39283008142):** "The ubuntu-latest label will
  migrate to Ubuntu 26 beginning October 19, 2026". Both jobs run on `ubuntu-latest`; the
  Python is uv's own, but Docker, the service container and the system tools come from the
  image. A backlog line, for S3-9, which pins the CI images.

## Verification record (2026-10-19)

All on this host, Qdrant v1.19.2 from `compose.yaml`; the output is in the session.

1. **Parity.** `docker compose up -d qdrant`, `readyz` "all shards are ready". The first
   `rag-ingest` published `rag_qa--gen-20261019T124332028607Z-5aadb820`: 3 documents, 1,708
   chunks (149, 1,507 and 52, as FAISS's manifest), in 2 min 3 s, exit 0.
   - `contexts-qdrant.json` against `contexts-faiss.json`: **IDENTICAL**, both sha256
     `5a36e2f9d00bbdaf49edaf07b8ba6af6f0e2ed52a5cf4c2e16c39aa66a074cbc`.
   - A second run, `contexts-qdrant-2.json`: **STABLE**.
   - Tier 1: hit rate 0.950, MRR 0.7958, recall 0.8583; `tier1-qdrant.json` (from the
     second run, `--json` added) equals `tier1-faiss.json` whole: 20 of 20 questions, the
     same chunks in the same order.
2. `rag-eval check`: **exit 0**, all six fingerprint parts `ok`.
3. Qdrant stopped: `HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q`:
   **987 passed, 8 skipped, 5 xfailed**, coverage 98.36%, the HF folder empty. (The first
   such run failed one test: Discovered, last item.) Qdrant started:
   `pytest -m qdrant_server`: **8 passed**, and the server left with no test collection.
4. **The crash drill, twice**, each `kill -9` sent to the Python process's own PID
   (`.venv/bin/rag-ingest`: killing `uv run` kills only its wrapper):
   - **(a) as written, while it embeds** (batch 9 of 54): exit 137; the alias and the
     collections unchanged, nothing left behind. The next `rag-ingest`: "Nothing changed:
     the index is up to date", exit 0, and no "Removed …" line, since no collection had
     been created (Deviation).
   - **(b) timed** by polling `/collections` every 10 ms until the new generation appeared:
     killed with `rag_qa--gen-20261019T132644350728Z-578e4951` created, 0 points,
     `complete: false`; the alias unchanged. The next `rag-ingest`: **"Removed what an
     earlier run left behind: rag_qa--gen-20261019T132644350728Z-578e4951"**, then "Nothing
     changed", exit 0, and only the live generation left.
   - `rag-query <<< "What is GLIDER?"` (maintainer's terminal, mistral): answered from the
     live generation, sources 2412.14140v2.pdf pp. 9, 8, 6, 1, 1.
5. **Refusals.** Qdrant stopped: `rag-query` exit **2**, `Error: cannot use Qdrant at
   http://localhost:6333 (ConnectError: [Errno 111] Connection refused). Is Qdrant running?
   docker compose up -d qdrant`. `RAG_VECTOR_STORE_PATH=/tmp/x rag-ingest`: exit **2**, the
   migration message, and `/tmp/x` never created.
6. **The API follows a CLI ingest.** `rag-serve` (maintainer's terminal, `tee -i`) served
   `…5aadb820`; `rag-ingest --rebuild` published `rag_qa--gen-20261019T153834837765Z-7f20891e`;
   `/v1/health` then named it with no restart, and the server logged "Serving index
   rag_qa--gen-20261019T153834837765Z-7f20891e (3 documents)". `/v1/documents` gave 149,
   1,507 and 52 chunks. Ctrl-C: "Application shutdown complete", no traceback.
7. `gh pr checks` on PR #20, head `24c5780` (run 39283008142): **check pass** (3m2s:
   983 passed, 12 skipped, which are S3-1's 4 plus the 8 `qdrant_server` tests, 5 xfailed,
   coverage 98.36%, `rag-eval check` matches) and **eval-retrieval pass** (2m11s: Qdrant
   ready, `ci--gen-20261019T161223486877Z-6d96bfc0` published with 3 documents and 1,708
   chunks, tier 1 0.950 / 0.796 / 0.858, the server tests 8 passed).

Gates on the working tree: `ruff format --check`, `ruff check` and `mypy` clean;
`uv lock --check` clean.

## Deviation from plan

- **Model:** ran on Opus 5.5 at max effort, where the story says fable, which is unavailable
  (CLAUDE.md, "Model routing"). The second review (also Opus 5.5) is the two-review habit's
  substitute for a stronger model.
- **A generation is created after the embedding,** not before, as §3.4's
  `write_generation(…, add: Embedded)` implies. So verification 4's crash drill, as
  written ("kill -9 while it embeds, then the next run prints Removed …"), finds nothing to
  remove: a kill while embedding leaves no collection. Drill (b) times a kill into the
  window after the collection exists, which is the case the recovery line is for.
- **A retriever takes `k` only,** and the schema refuses FAISS's other options at load. The
  plan did not say; the alternative was to ignore them silently. Part of the breaking
  change, and in the README's upgrade guide.
- **No split,** as the plan decided (2026-10-17): the parity tool is the branch's first
  commit.
- **Added by the second review:** a test that every search asks for an exact scan.
- **Added by its round 2:** aiohttp 3.14.5 in the lock (a dependency move outside the plan,
  for advisories published mid-PR); `QdrantStore.check()` refusing an answer that is not
  Qdrant's; a failed switch read back.

## Review (2026-10-19)

**Second review of phase A, the parity tool (Opus 5.5): no blockers; 2 findings, 1 wording
fix and 2 nits, fixed before the first commit.** It checked the tool's three claims against
the real answer path, not against its own code.

- **Checked and sound:**
  - **The digest is what the prompt gets.** All 25 golden questions went through the real
    `stream_answer` with a fake model recording its messages: each got the config's prompt
    rendered with the context whose sha256 the FAISS file holds. 0 mismatches.
  - **The FAISS run reproduces.** From the working tree, HF offline, `contexts-faiss.json`
    and `tier1-faiss.json` came out byte-identical to the recorded sha256s (34 s), from the
    live generation the story names.
  - **Without the flag, nothing extra is retrieved.** The code skips unanswerable questions,
    a test counts the calls, and a run with the flag gave the same tier-1 JSON.
  - **Scores:** FAISS's retriever attaches none; the chunks' only score is the reranker's,
    which came out bit-identical with its candidates reversed. Leaving it out is right
    because the prompt never shows it.
  - **The gates:** ruff, format and mypy clean; `rag-eval check` exit 0.
- **Findings, fixed:**
  - **The tool retrieved with `invoke`; every answer retrieves with `ainvoke`**
    (`stream_answer`, so `rag-query`, the API and tier-2 `generate`). Equal on FAISS, but a
    Qdrant retriever with an async path of its own would have been proven on a path no
    answer takes. `evaluate_retrieval` now awaits `ainvoke` in one `asyncio.run`; step 1,
    re-run with it, gave the same two sha256s.
  - **`test_contexts_out_digests_exactly_what_was_retrieved` could not fail on its own:** it
    checked `context_record`'s definition. Replaced by
    `test_a_digest_is_of_the_context_stream_answer_gives_the_model`, which runs the real
    answer path. A new `test_tier_1_retrieves_through_ainvoke_as_every_answer_does` guards
    the first fix. Each failed on a scratch copy with its guard broken (tier 1 back on
    `invoke`; a step between retrieval and `format_docs` in `stream_answer`), and only that
    one.
  - **Wording:** the no-score test blamed "FAISS's L2, Qdrant's cosine", which no chunk
    carries; it now says the prompt shows no score.
  - **Nits:** `CONTEXTS_VERSION` moved beside `context_record` in `retrieval.py`; a
    redundant `str()` in the sort key went.
- **After the fixes:** **899 passed, 5 xfailed**, coverage 98.38%; ruff, format and mypy
  clean; `rag-eval check` exit 0.
- **Recorded for phase B:** `ingested_at` (Discovered), and step 6's wording.

## Second review of the whole story (2026-10-19, Opus 5.5)

**No blockers; 1 test gap, 3 doc findings and 2 notes. The gap and the findings are fixed
in the follow-up commit.** Scratch work ran on a throwaway Qdrant v1.19.2 container
(removed afterwards); the real `rag_qa` alias and collections were only read, and are as
they were.

- **Reproduced:**
  - **Parity:** two runs on the live index wrote `contexts-faiss.json`'s sha256
    (`5a36e2f9…`) both times; tier 1 equals `tier1-faiss.json` (`e5dc2aff…`).
  - **The hermetic suite** with `RAG_QDRANT_URL` at a dead port and an empty HF cache:
    987 passed, 8 skipped, 5 xfailed. The 8 server tests passed and left no collection.
  - **Update equals rebuild** on a scratch corpus (first ingest, change + add + remove,
    no-op, rebuild): points equal the manifest's counts and derived IDs at every step; a
    512-chunk copy across two upsert batches is exact; the previous generation is kept;
    ids, texts and metadata (minus `ingested_at`) match a rebuild, vectors to 1e-6.
  - **Attacks that held:** a reader bound to a generation three rebuilds had deleted
    re-resolved the alias and answered, searching the lost generation once; a real
    collection named like the alias stopped `rag-ingest` and `rag-query` (exit 2) and was
    left alone; after the live collection was deleted, `rag-query` named the newest
    complete generation and the README's `curl` restored service; a Qdrant stopped
    mid-rebuild gave exit 2 and left nothing; with Qdrant unreachable, `rag-ingest`,
    `rag-query`, `rag-eval retrieval|generate` and `rag-serve` exit 2 with the hint, and
    `rag-eval check` exits 0; the API went 503 (not 500), logged once, recovered on the
    first request after Qdrant returned, and followed a rebuild with no restart.
  - **The cut rule:** 40 exactly tied 384-dimension copies straddling the cut, three
    insertion orders and a server restart: the server's own cut at the same limit gave 6
    distinct answers, ours one. Their scores were bit-identical.
  - **Library and server claims,** against the installed client and Qdrant's own files:
    `check_compatibility` starts a thread; `timeout` is rounded up; a key over `http://`
    warns; collection metadata is v1.16.0's (release notes); the request limit is 32 MiB
    (`config.yaml` at v1.19.2). The lock adds 8 packages and moves none.
  - **Mutation battery,** 15 broken guards in a scratch copy: 13 killed by the committed
    tests (the cut's widening and tie order, recovery with and without an alias, the count
    check, the manifest-name check, the in-the-way check, the alias re-resolve and the
    lost-generation retry, the architecture rule, the complete mark, the alias operations'
    order). Two survived: see below.
- **Findings, fixed:**
  - **Nothing tested that search is exact:** with `exact=False` every hermetic and server
    test passed, since local mode ignores the flag and a small collection is scanned in
    full whatever it says. This index (about 2.6 MB of vectors) is far below the server's
    indexing threshold; past it the search would turn approximate, and break DEC-19, the
    cut rule and the manifest's `store.exact`. `test_every_search_asks_the_server_for_an_exact_scan`
    spies on the request, and fails with `exact=False`.
  - **"A copy can never drift from a fresh ingest"** overstated: the store adds no drift,
    but the embedder varies by about 1e-7 with batch composition (1 of 555 vectors, 5e-8).
    Discovered says so.
  - **README "Upgrading from v0.2"** promised a `db_faiss.v02-*` folder that is no longer
    made; it now points to "Upgrading from v0.3".
  - **Deviation** was empty, and Status said "In progress"; both are filled.
- **Notes, no change:**
  - **The other survivor is an equivalent mutant:** on v1.19.2 a bare `create_alias` on an
    existing name repoints it atomically, so the switch's delete is redundant today. It is
    harmless, and the order matters (create then delete loses the alias): that mutation is
    killed.
  - **The Qdrant URL reaches API clients** inside an ingest job's error text
    (`describe_error` passes it through). It is not a path, so DEC-18's rule holds. S3-8's
    hardening, once S3-9 puts Qdrant on an internal network, should decide whether to
    redact it (a backlog line).
- **After the fixes:** ruff, format and mypy clean.

## Second review, round 2 (2026-10-20, Opus 5.5)

**No blockers; 1 concurrency finding and 2 exit-code findings, fixed on the PR, and the
cause of CI's red run, fixed too.** Run on `a2b7d38`, with the earlier records treated as
claims to check. Probes ran on a throwaway Qdrant v1.19.2 (`review-s3-2-qdrant`, removed
afterwards); the real `rag_qa` alias was only read, and still names `…7f20891e`.

- **Why CI went red on the first round's follow-up (run 39294075683).** The Audit step, not
  the change: `a2b7d38` and its two parents touch docs and one test, no dependency.
  - Three advisories against aiohttp 3.14.3 were published at 17:07–17:08 UTC on
    2026-10-19, between the green run on `24c5780` (16:07) and the red one (17:26):
    CVE-2026-111064 (high: no request/header timeout), GHSA-2g87-pp6c-29x5 (high:
    unbounded memory/CPU) and CVE-2026-111063 (low: the cookie Secure attribute).
  - CI's audit, run on `main`'s lock, `24c5780`'s and `a2b7d38`'s, fails alike on each,
    so `main` was red too for its next run.
  - aiohttp reaches the main environment only through `langchain-community` (S3-3 removes
    it). 3.14.5 (2026-10-11) fixes all three. `uv lock --upgrade-package aiohttp` moves
    aiohttp alone, and the main environment then audits clean. The `eval` extra keeps its
    four known findings (langchain, langgraph-sdk, diskcache, ragas: S3-9).
  - The audit policy allows no ignore for a fixed advisory, and S2-5's multidict bump
    (`2dd2010`) is the precedent: a `build(deps)` commit on the story's PR.
  - `eval-retrieval` was skipped, not failed: it `needs: check`, which predates S3-2.
- **Finding, documented: the writer lock excludes only writers that share `var/`.** The lock
  is a file in `paths.state`, the index a URL and a collection, so DEC-19's "one writer per
  host" holds per state folder. A second checkout on this host (S2-9's fresh clone, S3-12's
  coming one) defaults to the same `rag_qa` alias at localhost:6333, under its own lock.
  Two writers, one held inside its write window, varying only the lock's folder:
  - **the same folder:** the second was refused, "another ingestion is running";
  - **different folders:** the second's start-of-run recovery deleted the first's
    in-progress generation, published, and the first died on a raw 404 `StoreError`, a
    traceback and exit 1 under `rag-ingest`;
  - **forced, the worse order:** the first switched to its generation just before the
    second's recovery deleted it. The first reported "published", the alias was deleted with
    its target, and no alias existed for the rest of the second run (34 of 34 reads, every
    50 ms), until its full rebuild published. About 2 minutes on the real corpus.

  Docs now say so: caveat 36, a dated note in DEC-19, `ingest.py`'s docstrings, CLAUDE.md,
  README, and S3-12's fresh-clone run. The backlog's several-hosts line names a lock Qdrant
  holds as the code fix (the maintainer's call: docs now).
- **Finding, fixed: a wrong `RAG_QDRANT_URL` that reaches another HTTP server crashed.**
  Story step 10 promises one message and exit 2, and `rag-serve` documents 0 and 2 only.
  With the right host and a wrong path (a 404), or Ollama's port (a body that is not JSON),
  `rag-ingest`, `rag-query` and `rag-serve` gave a traceback and exit 1, and `rag-eval
  retrieval` exit 2 by its crash path, with no hint. JSON of another shape, which the
  client wraps, said "Is Qdrant running?" while it ran. `check()` now refuses any answer
  that is not Qdrant's, naming the URL to check: all four commands exit 2 on both URLs, no
  traceback. A test runs the real client against a loopback server for each of the three
  answers.
- **Finding, fixed: a switch the server applied, but whose answer was lost, said "published
  nothing".** Its read timeout became `StoreUnavailableError`: exit 2 and "Is Qdrant
  running?", though the alias had moved. `write_generation` now reads the alias back after a
  failed switch; a run whose alias names its generation is published. On the throwaway, such
  a run now reports its generation. Tests: applied-then-lost is published; not applied
  still fails, with the read-back answering and failing.
- **Checked and sound:**
  - the hermetic suite on `a2b7d38` (988 passed, 8 skipped, 5 xfailed, the HF cache
    empty), and the server tests on the throwaway (8 passed, nothing left);
  - the first round's exact-scan test fails, alone, with `exact=False` in a scratch copy;
    its README pointer and anchor are right;
  - the image digest is the registry's for the tag today, as in `compose.yaml`, `ci.yml`
    and DEC-24;
  - the cut rule's stop condition, the one-request switch, recovery's two branches, readers
    bound to a generation, `generate`'s check of a generation it moves to, and the API's
    not-ready refresh, read against the review notes;
  - a bad `RAG_QDRANT_COLLECTION` is a clean exit 2, and a URL with no scheme works;
  - all 10 commits: headers, wrapped bodies, parseable trailers, the `BREAKING-CHANGE:`
    trailer naming the retriever change, and no attribution in them or the PR body.
- **Note, no change:** during a hung Qdrant each API request pays two client timeouts (the
  refresh reads the alias, then again under its lock), up to 10 s each. S3-6's breaker owns
  it.
- **After the fixes:** ruff, format and mypy clean; `uv lock --check` clean; **994 passed, 8
  skipped, 5 xfailed**, coverage 98.37%, with a dead Qdrant URL and an empty HF cache; the
  server tests 8 passed; `rag-eval check` exit 0, every part ok. The 4 new tests of new
  behaviour fail on `a2b7d38`'s `vectorstore.py`; the 2 guards pass on both.
