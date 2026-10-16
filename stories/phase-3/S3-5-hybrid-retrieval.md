# S3-5: Hybrid retrieval and source filters, measured by tier 1

| | |
| --- | --- |
| **Status** | Todo |
| **Closes** | FR-3 (lexical retrieval fused by RRF; metadata filtering); DEC-20 |
| **Depends on** | S3-4 |
| **Model** | fable (runs on Opus 5.5 at max effort while Fable is unavailable; record it under Deviation) |
| **Plan-first** | yes |
| **Branch** | `feat/s3-5-hybrid-retrieval`. Moderate risk: new retrieval code and identity rules. The real config does not change, so no answer can. |

## Goal

FR-3's two SHOULDs.

- **Hybrid retrieval.** BM25 sparse vectors beside the dense ones, fused by our own RRF,
  then reranked by the cross-encoder.
- **Metadata filtering.** A question can be limited to named documents.

Tier 1 measures hybrid retrieval against dense on the golden set, with S2-4's discipline:
question by question, on a scratch config and collection. The measurements become S3-11's
input. **The real config keeps dense retrieval and sets no sparse encoder**, so neither tier
2's `query` part nor its `ingestion` part moves, and the switch, if tier 1 earns it, rides
S3-11b's one re-run (DEC-26). Design: ARCHITECTURE.md §3.1, DEC-20; §3.3, §3.4.

## Scope

- **`rag_qa/sparse.py`: `Bm25Encoder`**, standard library only (an architecture test holds
  it to that).
  - **Tokens:** lowercase Unicode word tokens, minus a fixed English stopword list, with no
    stemming. Each token becomes a uint32 through blake2b (a 4-byte digest).
  - **Document weights:** `tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))`. **Query
    weights:** 1 per distinct token.
  - **Output:** sparse vectors as our own `(indices, values)` type; `vectorstore` converts
    them for Qdrant.
  - **Settings:** `k1` (1.2), `b` (0.75), and `avg_len`. Measure `avg_len` first, as the
    mean token count of the corpus's chunks, and fix it in `config.yaml`.
  - **Versioning:** `SPARSE_VERSION`, pinned by a test over the sample corpus as
    `CHUNKING_VERSION` is.
- **Schema.**
  - The new component kind `sparse_encoders`, and `pipeline.ingestion.sparse_encoder`
    (optional).
  - `RetrieverSpec` gains `mode` (`dense` when absent, or `hybrid`), `prefetch` and `rrf_k`.
    Unset, each stays out of `kwargs()`, so `vector_search`'s dump does not change.
  - `search_type` narrows to `similarity`.
  - A `hybrid` retriever requires an ingestion sparse encoder (a load error naming both).
- **Ingest and the store.**
  - With a sparse encoder set, the prepare phase also encodes each chunk.
  - **The generation** gains a sparse vector `bm25` with `modifier=IDF`, and the copy step
    carries sparse vectors over.
  - **The manifest** records the encoder's identity (its spec and `SPARSE_VERSION`). A
    changed identity, or one appearing or disappearing, rebuilds in full: the vector
    configuration changes.
- **The hybrid query** (DEC-20, as design review 2 changed it):
  - two `query_points` calls, dense (exact) and sparse, each with the filter and each cut to
    `prefetch` (40) by S3-2's cut rule. No cut is left to the server, which decides a tie by
    the collection's layout: review 1's whole-union fusion still let each prefetch cut its
    own list there;
  - fused in `vectorstore` by our own RRF: `1 / (rrf_k + rank)` summed over the lists a point
    is in, with one-based ranks and `rrf_k` always explicit (60). Qdrant's own fusion, and its
    default constant of 2 (caveat 31), are no longer used;
  - then ordered by score, ties by point ID, and cut to `reranker_candidates`.
- **The `sources` filter.**
  - **The retrieval half** takes a `RetrievalQuery(question, sources=())`; settle the exact
    type in the plan, with tier 1's call sites in mind.
  - **`stream_answer(pipeline, question, *, sources=())`.**
  - **HTTP:** `QueryRequest` gains an optional `sources`: at most 50 corpus-relative paths,
    each in the served manifest, else 422 naming the unknown ones. It stays
    `extra="forbid"`.
  - **CLI:** `rag-query --source PATH`, repeatable, checked against the manifest the same way
    at start-up.
  - **In Qdrant:** a `MatchAny` on the indexed `metadata.source`, in each branch's search
    (and as the query filter in dense mode).
- **The fingerprint.** `ingestion` includes the sparse encoder's identity only when one is
  set: an absent key is left out, so v0.3's hash holds. Unit tests show the real config's
  six parts unchanged, and the scratch config's `ingestion` and `query` parts moved.
- **Config.** Add `components.sparse_encoders.bm25` (with the measured `avg_len`) and
  `components.retrievers.hybrid`, both referenced by nothing in `pipeline`. Comment them
  with tier 1's table, as S2-4 commented the reranker.
- **Tier 1, measured and recorded here,** on a scratch config that differs from the real one
  only in `retriever: components.retrievers.hybrid` and `sparse_encoder`, ingested into a
  scratch collection:
  - dense against hybrid, both reranked to 5: hit rate, MRR, recall, and question by
    question (wins, losses, ties), as S2-4 did;
  - `rrf_k` 60 against one alternative, and `prefetch` 20 against 40;
  - the reach: how often an expected page is among the fused candidates before the rerank;
  - **the `min_score` table, for S3-11:** for cutoffs off, −4, −2, 0 and 2, on the
    configured retrieval, the answerable questions' hit rate and recall, and the chunks
    that survive on answerable and on unanswerable questions;
  - a recommendation: hybrid on or off, and a `min_score` value.
- **Server tests** (`qdrant_server`):
  - a hybrid query and a filtered query on the real server;
  - a fused list with a tie across its two lists, cut where the tie falls;
  - a tie at a branch's own cut, which must give the same points from two collections built
    in different insertion orders (design review 2). Within one process the server always
    agrees with itself, so two runs there prove nothing.
- **README:** the `sources` filter (HTTP and CLI) and what hybrid retrieval is, marked as
  measured but not yet on.

## Out of scope

- Switching the real config to hybrid, setting the sparse encoder there, or setting
  `min_score`: S3-11 chooses, and S3-11b commits the choice with the re-run.
- Stemming (a backlog line); weighted RRF (unless the table makes a case, recorded for
  S3-11).

## Verification

```bash
uv run rag-eval check; echo "exit $?"                     # 0: nothing hashed moved
uv run rag-ingest                                         # the real index: nothing changed
uv run rag-eval retrieval                                 # unchanged: 0.950 / 0.7958 / 0.8583
# no answer changed: the same contexts as S3-2's on Qdrant (§3.5, design review 1)
uv run rag-eval retrieval --contexts-out ~/dev/rag-s3-5/contexts.json
diff ~/dev/rag-s3-2/contexts-qdrant.json ~/dev/rag-s3-5/contexts.json && echo IDENTICAL
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 uv run pytest --cov=rag_qa -q
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -m qdrant_server -q
uv run ruff check && uv run ruff format --check && uv run mypy

# tier 1, dense against hybrid, on a scratch collection (caveat 9)
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-ingest --config <scratch hybrid config>
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-eval retrieval --config <scratch hybrid config> \
  --contexts-out ~/dev/rag-s3-5/hybrid-1.json
docker compose restart qdrant   # the second run after a restart, where the server's own tie
                                # order changes (design review 2)
RAG_QDRANT_COLLECTION=s3_5_hybrid uv run rag-eval retrieval --config <scratch hybrid config> \
  --contexts-out ~/dev/rag-s3-5/hybrid-2.json
diff ~/dev/rag-s3-5/hybrid-1.json ~/dev/rag-s3-5/hybrid-2.json && echo STABLE   # the cut rule
# ... the variants and the min_score cutoffs; every table pasted here
# clean up: deleting the scratch generations removes their alias too
curl -s 127.0.0.1:6333/collections | jq -r '.result.collections[].name' | grep '^s3_5_hybrid--gen-' \
  | xargs -I{} curl -s -X DELETE 127.0.0.1:6333/collections/{}

# the filter, over HTTP (rag-serve in a foreground terminal, logged with tee -i)
curl -sN -X POST 127.0.0.1:8000/v1/query -H 'content-type: application/json' \
  -d '{"question": "What is GLIDER?", "sources": ["2412.14140v2.pdf"]}' | head -3   # one file cited
curl -s -o /dev/null -w '%{http_code}\n' -X POST 127.0.0.1:8000/v1/query \
  -H 'content-type: application/json' -d '{"question": "x", "sources": ["nope.pdf"]}'  # 422
gh pr checks
```

## Review notes for the human

- **The BM25 arithmetic:** compare one chunk's weights with the formula by hand, and check
  that the query side sends 1 per distinct token. The IDF must come from Qdrant, not twice.
- **Our code makes every cut, and the fusion** (design review 2): two searches, each cut by
  S3-2's rule, fused by our RRF with `rrf_k` explicit, then ordered by score, then point ID.
- **The identity rules:** the sparse identity is in the manifest always, and in the
  fingerprint only when set.
- **Tier 1's table** is a decision input: make sure the scratch config differs from the real
  one only where it says.

## Discovered

(Filled during implementation.)

## Deviation from plan

(Filled at close-out.)
