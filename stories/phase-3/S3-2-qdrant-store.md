# S3-2: Move the index to Qdrant: generations behind an alias

| | |
| --- | --- |
| **Status** | Todo |
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

1. **First commit, while FAISS is still the store: the parity tool.**
   `rag-eval retrieval --contexts-out FILE` writes, for all 25 golden questions,
   answerable or not, in id order:
   - the sha256 of the rendered context (`chain.format_docs` over what `pipeline.retrieve`
     returns, the rerank included);
   - the citations, in order.

   Then run it against the current FAISS index, and keep the file outside the repository
   somewhere a reboot does not clear (for example `~/dev/rag-s3-2/contexts-faiss.json`).
   Paste its sha256 here.
2. **Dependency:** `uv add "qdrant-client>=1.19.1"`. Record the resolved versions; no locked
   version should move (checked in the architecture pass).
3. **`compose.yaml`**, with one service, `qdrant`:
   - `qdrant/qdrant:v1.19.2`, pinned by digest;
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
6. **`vectorstore.py`, rewritten on `qdrant-client`** (the seam).
   - **What it provides:** `QdrantStore`, `live_index`, `in_the_way`, `open_generation`,
     `write_generation`, `recover`, and a dense `retriever`, as in §3.4.
   - **Generations** are collections `<collection>--gen-<stamp>-<hex>`, with a dense vector
     named `dense` (cosine), a payload index on `metadata.source`, and the manifest in
     `metadata`.
   - **An update** copies the live generation without the deleted points, upserts the new
     ones (`wait=True`), checks the count, then switches the alias with one
     `update_collection_aliases` call (delete plus create).
   - **Recovery,** under the lock, keeps the alias's target and the newest older generation.
   - **In the way:** a real collection named like the alias, or an alias pointing outside
     the pattern, exits 2 and is never renamed.
   - **Search** is `exact=True`.
   - **Documents come back as FAISS returned them:** the same `page_content` and metadata.
     No `_id` or `_collection_name` keys, since the prompt and `SourceRef` read the
     metadata.
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
     A search whose collection has gone resolves once more.
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
      limitation goes.
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

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
